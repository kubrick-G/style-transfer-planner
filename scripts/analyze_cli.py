#!/usr/bin/env python3
"""Analyze one or more videos and print/save the Style Card.

    python scripts/analyze_cli.py video.mp4 [more.mp4 ...] [--mock-gemini] [--no-cache] [--json]

Needs GEMINI_API_KEY (env var or .env) for real Gemini-judged fields. Without it, code-measured fields still work
and Gemini-judged fields are reported as unavailable. --mock-gemini uses FAKE labels for offline wiring tests.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.gemini_client import GeminiClient  # noqa: E402
from core.mock_gemini import mock_backend  # noqa: E402
from core.pipeline import DEFAULT_CACHE_DIR, analyze_video  # noqa: E402
from core.schema import StyleCard  # noqa: E402
from core.utils import AnalysisError  # noqa: E402


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    os.environ.setdefault(k.strip(), v)


def _f(v, nd=2, unit=""):
    return "n/a" if v is None else (f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}")


def format_summary(card: StyleCard) -> str:
    o, m = card.overall, card.meta
    L = []
    if m.gemini_mode == "mock":
        L.append("!! MOCK GEMINI: Gemini-judged fields below are FAKE placeholders !!")
    elif m.gemini_mode == "unavailable":
        L.append("!! No GEMINI_API_KEY: Gemini-judged fields are unavailable (code-measured fields are real) !!")
    L.append(f"video      {card.video.duration_s}s  {card.video.fps}fps  {card.video.resolution[0]}x{card.video.resolution[1]}"
             f"  sha256={card.video.sha256[:12]}")
    L.append(f"pacing     {o.pacing.n_scenes} scenes | avg shot {_f(o.pacing.avg_shot_len_s, 2, 's')} | "
             f"{_f(o.pacing.cuts_per_min, 1)} cuts/min | curve: {o.pacing.pacing_curve}")
    a = o.audio
    L.append(f"audio      has_audio={a.has_audio} bpm={_f(a.bpm, 1)} beat_sync={_f(a.beat_sync_score)} "
             f"energy={a.energy} genre={a.genre_guess}")
    c = o.color
    L.append(f"color      grade={c.grade} contrast={c.contrast} sat={c.saturation} warmth={c.warmth} "
             f"vignette={c.vignette} grain={c.grain}  palette={' '.join(c.palette)}")
    t = o.text
    fs = t.font_style
    L.append(f"text       present={t.present} anim={t.animation} case={t.case} pos={t.position} color={t.color} "
             f"font={(fs.category, fs.weight, fs.width, fs.geometry) if fs else None}")
    L.append(f"transit.   {', '.join(f'{x.type} x{x.count}' for x in o.transitions) or 'n/a'}")
    L.append(f"           caption chunking={t.chunking} variety={t.variety} changes/s={t.changes_per_s} region={t.region or None}")
    if t.description:
        L.append(f"           text treatment: {t.description}")
    L.append(f"camera     move={o.camera.dominant_movement} shots={o.camera.shot_types} layout={o.layout} mood={o.mood}")
    L.append("scenes:")
    for s in card.scenes:
        L.append(f"  #{s.index:<2} {s.start_s:6.2f}-{s.end_s:6.2f}s {s.role:<4} energy={s.energy:<6} "
                 f"motion={_f(s.motion_magnitude)} cam={s.camera_motion} shot={s.shot_type} "
                 f"in={s.transition_in} text={s.has_text} on_beat={s.cut_on_beat}")
    cp = card.coupling
    L.append(f"coupling   cuts_on_beat={_f(cp.cuts_on_beat_ratio)} text~pacing={cp.text_energy_matches_pacing} "
             f"trans_on_peaks={_f(cp.transitions_on_music_peaks_ratio)} grade_consistent={cp.grade_consistent_across_scenes}")
    for n in cp.notes:
        L.append(f"           - {n}")
    L.append("confidence " + "  ".join(f"{k}={v:.2f}" for k, v in card.confidence.items()))
    L.append(f"timing     total={m.analysis_seconds}s  " + "  ".join(f"{k}={v}s" for k, v in m.stage_seconds.items()))
    calls = m.gemini_calls
    if calls:
        L.append(f"gemini     model={m.gemini_model or m.gemini_mode} {sum(c.ok for c in calls)}/{len(calls)} calls ok, "
                 f"tokens in/out={sum(c.prompt_tokens or 0 for c in calls)}/{sum(c.output_tokens or 0 for c in calls)}")
    failed = [c for c in calls if not c.ok]
    for c in failed[:3]:
        L.append(f"gemini FAIL {c.name}: {(c.error or '')[:170]}")
    if len(failed) > 3:
        L.append(f"gemini FAIL ... and {len(failed) - 3} more failed calls (see meta.gemini_calls in the JSON)")
    for d in m.degraded:
        L.append(f"degraded   {d.stage}: {d.reason}")
    for n in m.notes:
        L.append(f"note       {n}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze videos into a Style Card.")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--mock-gemini", action="store_true", help="use FAKE Gemini labels (offline wiring test)")
    ap.add_argument("--no-cache", action="store_true", help="ignore any cached result and re-analyse (the new result is still saved)")
    ap.add_argument("--json", action="store_true", help="print the full JSON instead of the summary")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    _load_dotenv()
    client = GeminiClient(backend=mock_backend, mode="mock") if a.mock_gemini else GeminiClient()
    rc = 0
    for v in a.videos:
        print(f"\n=== {v}")
        prog = None if a.quiet else (lambda s, f: print(f"  [{int(f * 100):3d}%] {s}", file=sys.stderr))
        t0 = time.perf_counter()
        try:
            card = analyze_video(v, client=client, cache_dir=Path(a.cache_dir), use_cache=not a.no_cache, progress=prog)
        except AnalysisError as e:
            print(f"ERROR: {e}")
            rc = 2
            continue
        out = Path(a.out_dir) / f"{Path(v).stem}.stylecard.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(card.model_dump_json(indent=2))
        print(card.model_dump_json(indent=2) if a.json else format_summary(card))
        print(f"\n{'CACHE HIT' if card.meta.cache_hit else 'fresh analysis'} in {time.perf_counter() - t0:.2f}s -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
