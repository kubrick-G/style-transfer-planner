#!/usr/bin/env python3
"""Apply a reference's style to a target clip.

    python scripts/apply_cli.py --reference outputs/opusjake.stylecard.json --target data/examples/stoic.mp4 \\
        --apply pacing,audio_rhythm,transitions,color,text --out outputs/stoic_in_opusjake_style.mp4

Needs no Gemini calls (the reference card is already analysed; the target is measured by code only).
Attributes: pacing, audio_rhythm, transitions, color, text.  Writes the video (if anything renders) and
<out>.plan.json with the full editing plan, fallback log and style-match report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.apply_engine import DEFAULT_ZOOM, apply_style  # noqa: E402
from core.bundles import ATTRIBUTES  # noqa: E402
from core.schema import StyleCard  # noqa: E402
from core.gemini_client import GeminiClient  # noqa: E402
from core.transcript import load_words_json  # noqa: E402
from core.utils import AnalysisError  # noqa: E402


def summary(res) -> str:
    p, L = res.plan, []
    L.append(f"reference  sha256={p.reference_sha[:12]}   target={p.target_name}")
    L.append(f"selected   {', '.join(p.selected) or '(none)'}")
    for w in p.warnings:
        L.append(f"WARNING    [{w.attribute}] {w.message}" + (f"  -> add '{w.suggest_add}'" if w.suggest_add else ""))
    L.append("plan:")
    for it in p.items:
        L.append(f"  [{it.status:<8}] {it.attribute:<13} {it.action}")
        for k, v in it.detail.items():
            if k in ("placements", "zoom_levels", "cut_times_s"):
                continue
            L.append(f"               {k}: {v}")
        if it.attribute == "pacing":
            L.append(f"               cuts at: {it.detail['cut_times_s']}")
        if it.attribute == "transitions":
            L.append("               " + ", ".join(f"{x['at_s']}s:{x['rendered_as']}" for x in it.detail["placements"]))
        if it.note:
            L.append(f"               note: {it.note}")
    if p.fallbacks:
        L.append("fallback log:")
        for f in p.fallbacks:
            L.append(f"  [{f.status:<8}] {f.attribute}: '{f.reference_value}' -> '{f.chosen}'"
                     + (f" (score {f.score})" if f.score is not None else ""))
            L.append(f"               {f.explanation}")
    m = res.match
    if m:
        L.append("style match (measured on the rendered output):")
        if m.color_distance_before is not None:
            L.append(f"  color distance to reference: {m.color_distance_before} -> {m.color_distance_after}  (lower is closer)")
            for k in m.color_reference:
                L.append(f"    {k:<18} reference={m.color_reference[k]:>7}  before={m.color_before[k]:>7}  after={m.color_after[k]:>7}")
        L.append(f"  shots: {m.scheduled_cuts + 1} scheduled (avg {m.avg_shot_len_target_s}s; reference avg "
                 f"{m.avg_shot_len_reference_s}s); the scene detector finds {m.detected_scenes_after} in the output "
                 f"(jump cuts with small framing changes may not all register)")
    L.append(f"output     {res.output_path or 'nothing to render for this selection (plan only)'}   ({res.seconds}s)")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="path to a .stylecard.json made by analyze_cli.py")
    ap.add_argument("--target", required=True, help="the clip to restyle")
    ap.add_argument("--apply", default=",".join(ATTRIBUTES), help=f"comma list of: {', '.join(ATTRIBUTES)}")
    ap.add_argument("--out", help="output mp4 (default outputs/<target>_styled.mp4)")
    ap.add_argument("--strength", type=float, default=0.8, help="color match strength 0..1 (default 0.8)")
    ap.add_argument("--zoom", type=float, default=DEFAULT_ZOOM, help=f"framing change for jump cuts (default {DEFAULT_ZOOM})")
    ap.add_argument("--transcript", help="JSON with word timings [{text,start_s,end_s}, ...] to use for captions "
                                         "instead of transcribing with Gemini")
    ap.add_argument("--caption-offset", type=float, default=0.0,
                    help="shift all caption timing by this many seconds (negative = earlier), e.g. -0.1")
    ap.add_argument("--auto-sync", action="store_true",
                    help="measure the transcript against the audio and shift all captions by the estimated global lag "
                         "(only when the estimate is confident)")
    ap.add_argument("--words-per-caption", type=int, help="override the automatic words-per-caption (1..6)")
    ap.add_argument("--debug", action="store_true",
                    help="also write <out>_debug.mp4 (shot/framing burned in, red flash at each cut) and <out>_cuts.png")
    a = ap.parse_args(argv)
    ref = Path(a.reference)
    if not ref.exists():
        print(f"ERROR: reference card not found: {ref}\nRun: python scripts/analyze_cli.py <reference video> first.")
        return 2
    sel = [s.strip() for s in a.apply.split(",") if s.strip()]
    out = Path(a.out) if a.out else Path("outputs") / f"{Path(a.target).stem}_styled.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        card = StyleCard.model_validate_json(ref.read_text())
        words = load_words_json(a.transcript) if a.transcript else None
        client = None
        if words is None and "text" in sel:
            from scripts.analyze_cli import _load_dotenv
            _load_dotenv()
            c = GeminiClient()
            client = c if c.available else None
        res = apply_style(card, a.target, sel, str(out), strength=a.strength, zoom=a.zoom, client=client,
                          transcript=words, caption_offset_s=a.caption_offset,
                          words_per_caption=a.words_per_caption, auto_sync=a.auto_sync)
    except (AnalysisError, ValueError) as e:
        print(f"ERROR: {e}")
        return 2
    plan_path = out.with_suffix(".plan.json")
    plan_path.write_text(res.model_dump_json(indent=2))
    print(summary(res))
    txt = next((i for i in res.plan.items if i.attribute == "text"), None)
    if res.transcript and txt and txt.detail.get("transcript_source") == "gemini":
        tpath = out.with_suffix(".transcript.json")
        tpath.write_text(json.dumps([w.model_dump() for w in res.transcript], indent=1), encoding="utf-8")
        print(f"transcript {tpath}   (fix any word or time, then re-render with --transcript {tpath})")
    if a.debug and res.output_path:
        from core.debug_overlay import annotate_cuts, cut_sheet
        dbg = annotate_cuts(res.output_path, res.plan.cut_times_s, res.plan.segment_zooms, out.with_name(out.stem + "_debug.mp4"))
        png = cut_sheet(res.output_path, res.plan.cut_times_s, res.plan.segment_zooms, out.with_name(out.stem + "_cuts.png"))
        print(f"debug video  {dbg or 'no cuts to mark'}\ndebug sheet  {png or 'no cuts to show'}")
    print(f"plan json  {plan_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
