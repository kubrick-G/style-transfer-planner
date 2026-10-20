"""Apply engine: borrow selected style attributes from a reference Style Card and apply them to a target clip.

What is REALLY rendered (FFmpeg):
  color    - matches the target's brightness / contrast / saturation / warmth to the reference's measured values
  pacing   - cuts the target on the reference's shot-length rhythm (a continuous take gets visible JUMP CUTS made by
             alternating framing; audio is untouched, nothing is removed)
  audio_rhythm - snaps those cuts to the target's beat grid
What is only PLANNED (a concrete, specific plan; not rendered in this version): captions/text, non-trivial transitions,
layout. Everything unsupported is logged as a fallback with a reason; nothing blocks the workflow.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import analyze_audio, extract_wav
from .captions import CaptionStyle, chunk_words, hex_to_rgb, pick_chunk_size, render_captions
from .bundles import check_selection
from .color import _mean_stats, _small, frame_stats
from .font_files import load_font
from .fonts import resolve_font
from .frames import FrameReader, sample_times
from .gemini_client import GeminiClient
from .plan import ApplyResult, EditingPlan, FallbackEntry, PlanItem, StyleMatch
from .scenes import detect_scenes
from .schema import StyleCard
from .sync_check import analyze_sync, corrected_words, extract_audio
from .transcript import TranscriptWord, transcribe
from .utils import MAX_DURATION_S, InvalidVideoError, VideoTooLongError, ffprobe_info, run, temp_workdir

RENDERED_TRANSITIONS = {"hard cut", "zoom punch-in"}
DEFAULT_ZOOM = 1.15
PUNCH_BOOST = 1.12
MIN_SHOT_S = 1.0
MAX_SHOT_S = 10.0
SNAP_TOL_S = 0.30
PREVIEW_MAX_SIDE = 720
STAT_KEYS = ("saturation", "contrast", "brightness", "warmth_shadows", "warmth_highlights")
STAT_TOLERANCE = {"saturation": 0.20, "contrast": 0.10, "brightness": 0.20, "warmth_shadows": 0.10, "warmth_highlights": 0.10}


# --------------------------------------------------------------------------- target analysis (code only, no Gemini)
@dataclass
class TargetInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool
    beat_times: list[float] = field(default_factory=list)
    bpm: Optional[float] = None
    color_stats: dict[str, float] = field(default_factory=dict)


def measure_color_stats(path, n: int = 12) -> dict[str, float]:
    with FrameReader(path) as r:
        dur = r.n_frames / r.fps if r.fps else 0
        stats = []
        for t in sample_times(max(dur, 0.1), hz=1.0, max_n=n):
            fr = r.frame_at(t)
            if fr is not None:
                stats.append(frame_stats(_small(fr)))
    if not stats:
        raise RuntimeError("could not read any frames to measure color")
    return _mean_stats(stats)


def measure_all_frames(path, max_frames: int = 240) -> dict[str, float]:
    """Mean color statistics over EVERY frame (thinned evenly if there are more than max_frames).
    Used for the tiny calibration probe, where midpoint sampling would alias with the cut cadence."""
    with FrameReader(path) as r:
        n = max(1, r.n_frames)
        step = max(1, int(np.ceil(n / max_frames)))
        stats = []
        for i, (_, fr) in enumerate(r.iter_frames(r.fps / step)):
            stats.append(frame_stats(_small(fr)))
    if not stats:
        raise RuntimeError("probe had no frames")
    return _mean_stats(stats)


def analyze_target(path) -> TargetInfo:
    if not Path(path).is_file():
        raise InvalidVideoError(f"Target video not found: {path}\n"
                                f"Check the path (tip: list files with  Get-ChildItem -Recurse data -Filter *.mp4 ).")
    info = ffprobe_info(path)
    if info["duration_s"] > MAX_DURATION_S:
        raise VideoTooLongError(f"The target is {info['duration_s']:.0f}s; the limit is {MAX_DURATION_S:.0f}s.")
    beats, bpm = [], None
    if info["has_audio"]:
        with temp_workdir() as tmp:
            try:
                a = analyze_audio(extract_wav(path, tmp / "a.wav"))
                beats, bpm = a.beat_times, a.bpm
            except Exception:
                pass
    return TargetInfo(info["duration_s"], info["fps"], info["width"] - info["width"] % 2, info["height"] - info["height"] % 2,
                      info["has_audio"], beats, bpm, measure_color_stats(path))


# ------------------------------------------------------------------------------------------------- color
@dataclass
class ColorAdjust:
    contrast: float
    brightness: float
    saturation: float
    rs: float
    bs: float
    rh: float
    bh: float

    @property
    def ffmpeg(self) -> str:
        return (f"eq=contrast={self.contrast:.3f}:brightness={self.brightness:.3f}:saturation={self.saturation:.3f},"
                f"colorbalance=rs={self.rs:.3f}:bs={self.bs:.3f}:rh={self.rh:.3f}:bh={self.bh:.3f}")

    def params(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def color_adjust(ref: dict[str, float], tgt: dict[str, float], strength: float = 0.8) -> ColorAdjust:
    """Move the target's measured color statistics toward the reference's. strength 0 = no change, 1 = full match."""
    s = _clip(strength, 0.0, 1.0)
    sat = 1 + (_clip(ref["saturation"] / max(tgt["saturation"], 1e-3), 0.4, 2.0) - 1) * s
    con = 1 + (_clip(ref["contrast"] / max(tgt["contrast"], 1e-3), 0.6, 1.6) - 1) * s
    m_t = tgt["brightness"]
    m_goal = m_t + (ref["brightness"] - m_t) * s
    bri = _clip(m_goal - ((m_t - 0.5) * con + 0.5), -0.30, 0.30)

    def shift(key: str) -> float:
        return _clip((ref[key] - tgt[key]) * 0.5 * s, -0.30, 0.30)

    dh, ds = shift("warmth_highlights"), shift("warmth_shadows")
    return ColorAdjust(con, bri, sat, ds, -ds, dh, -dh)


def color_distance(stats: dict[str, float], ref: dict[str, float]) -> float:
    """Mean tolerance-normalised absolute difference across the measured color statistics (lower = closer)."""
    return round(float(np.mean([abs(stats[k] - ref[k]) / STAT_TOLERANCE[k] for k in STAT_KEYS])), 3)


def _goal_stats(ref: dict[str, float], tgt: dict[str, float], strength: float) -> dict[str, float]:
    s = _clip(strength, 0.0, 1.0)
    return {k: tgt[k] + (ref[k] - tgt[k]) * s for k in STAT_KEYS}


def _refine(adj: ColorAdjust, got: dict[str, float], goal: dict[str, float]) -> ColorAdjust:
    """One correction step from the measured residual (the warmth filter also changes saturation, so a one-shot
    formula under-/over-shoots; measuring and correcting converges)."""
    sat = _clip(adj.saturation * _clip(goal["saturation"] / max(got["saturation"], 1e-3), 0.5, 2.0), 0.0, 3.0)
    con = _clip(adj.contrast * _clip(goal["contrast"] / max(got["contrast"], 1e-3), 0.7, 1.4), 0.5, 2.0)
    bri = _clip(adj.brightness + (goal["brightness"] - got["brightness"]), -0.45, 0.45)
    rh = _clip(adj.rh + (goal["warmth_highlights"] - got["warmth_highlights"]) * 0.5, -0.45, 0.45)
    rs = _clip(adj.rs + (goal["warmth_shadows"] - got["warmth_shadows"]) * 0.5, -0.45, 0.45)
    return ColorAdjust(con, bri, sat, rs, -rs, rh, -rh)


def calibrate_pipeline(target, info: TargetInfo, cuts: list[float], zooms: list[float], ref: dict[str, float],
                       strength: float = 0.8, iterations: int = 10, good_enough: float = 0.15
                       ) -> tuple[ColorAdjust, list[float], dict[str, float]]:
    """Closed-loop color match that renders the REAL pipeline (same cuts, same zooms, same filter chain) on a tiny
    1-fps copy of the target, so whatever those steps do to the statistics is seen by the loop.

    Returns (best adjustment, distance-to-goal history, 'before' stats = the same pipeline WITHOUT the grade).
    Falls back to the open-loop estimate on the original measurements if the probe cannot be rendered."""
    try:
        with temp_workdir() as tmp:
            small = tmp / "small.mp4"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(target), "-an", "-vf", "scale=320:-2:flags=bicubic",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", "-pix_fmt", "yuv420p", str(small)], timeout=180)
            si = ffprobe_info(small)
            sinfo = TargetInfo(si["duration_s"], si["fps"], si["width"] - si["width"] % 2, si["height"] - si["height"] % 2, False)
            n = {"i": 0}

            def probe(adj: Optional[ColorAdjust]) -> dict[str, float]:
                n["i"] += 1
                out = tmp / f"p{n['i']}.mp4"
                render(small, out, sinfo, [c for c in cuts if c < sinfo.duration_s - 0.05], zooms, adj, crf=16,
                       max_side=320, fps=2, preset="ultrafast", with_audio=False)
                return measure_all_frames(out)

            base = probe(None)
            goal = _goal_stats(ref, base, strength)
            adj = color_adjust(ref, base, strength)
            best, best_d, hist = adj, None, []
            for i in range(iterations + 1):
                got = probe(adj)
                d = color_distance(got, goal)
                hist.append(d)
                if best_d is None or d < best_d:
                    best, best_d = adj, d
                if d <= good_enough or i == iterations:
                    break
                adj = _refine(adj, got, goal)
            return best, hist, base
    except Exception:
        return color_adjust(ref, info.color_stats, strength), [], info.color_stats


def calibrate_grade(target, ref: dict[str, float], tgt: dict[str, float], strength: float = 0.8,
                    iterations: int = 10, good_enough: float = 0.15) -> tuple[ColorAdjust, list[float]]:
    """Color-only calibration (no cuts/zoom): kept for callers and tests; see calibrate_pipeline."""
    try:
        d = ffprobe_info(target)
    except Exception:
        return color_adjust(ref, tgt, strength), []
    info = TargetInfo(d["duration_s"], d["fps"], d["width"], d["height"], False, color_stats=tgt)
    adj, hist, _ = calibrate_pipeline(target, info, [], [1.0], ref, strength, iterations, good_enough)
    return adj, hist


# ------------------------------------------------------------------------------------------------ pacing
def plan_cuts(ref_lengths: list[float], duration: float, beats: Optional[list[float]] = None, *,
              min_len: float = MIN_SHOT_S, max_len: float = MAX_SHOT_S, snap_tol: float = SNAP_TOL_S
              ) -> tuple[list[float], int]:
    """Cut times for the target following the reference's shot-length sequence (cycled). Optionally snapped to beats.
    Returns (cut_times, n_snapped). Shots are kept within [min_len, max_len] and away from the very end."""
    lengths = [max(min_len, min(max_len, L)) for L in ref_lengths if L > 0.05] or [4.0]
    cuts, t, i = [], 0.0, 0
    while True:
        t += lengths[i % len(lengths)]
        if t > duration - min_len:
            break
        cuts.append(round(t, 3))
        i += 1
    snapped = 0
    if beats and cuts:
        b = np.asarray(beats)
        out, prev = [], 0.0
        for c in cuts:
            nearest = float(b[np.argmin(np.abs(b - c))])
            new = nearest if abs(nearest - c) <= snap_tol else c
            if new - prev >= min_len and new <= duration - min_len:
                snapped += int(new != c)
                c = new
            out.append(round(c, 3))
            prev = c
        cuts = out
    return cuts, snapped


def segment_zooms(cut_types: list[str], zoom: float) -> list[float]:
    """Framing per segment. Alternating 1.0/zoom makes jump cuts visible on a continuous take; a 'zoom punch-in'
    cut gets a stronger punch."""
    z = [1.0]
    for t in cut_types:
        if t == "zoom punch-in":
            z.append(round(zoom * PUNCH_BOOST, 3) if z[-1] < zoom * PUNCH_BOOST - 1e-6 else 1.0)
        else:
            z.append(1.0 if z[-1] > 1.0 else round(zoom, 3))
    return z


# ------------------------------------------------------------------------------------------------ render
def render(target, out_path, info: TargetInfo, cuts: list[float], zooms: list[float],
           grade: Optional[ColorAdjust], crf: int = 20, *, max_side: int = PREVIEW_MAX_SIDE, fps: Optional[int] = None,
           preset: str = "veryfast", with_audio: bool = True) -> Path:
    W, H, D = info.width, info.height, info.duration_s
    bounds = [0.0] + list(cuts) + [D]
    n = len(bounds) - 1
    parts = []
    if n > 1:
        parts.append(f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n)))
        for i in range(n):
            z = zooms[i]
            # even crop size; setsar=1 on EVERY segment (scale silently changes SAR, and concat rejects mixed SARs)
            zoom = ("" if z <= 1.0001 else
                    f",crop=trunc(iw/{z}/2)*2:trunc(ih/{z}/2)*2:(in_w-out_w)/2:(in_h-out_h)/2,"
                    f"scale={W}:{H}:flags=bicubic")
            parts.append(f"[s{i}]trim=start={bounds[i]:.3f}:end={bounds[i + 1]:.3f},setpts=PTS-STARTPTS{zoom},setsar=1[o{i}]")
        parts.append("".join(f"[o{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cat]")
        src = "[cat]"
    else:
        src = "[0:v]"
    chain = [f"fps={fps}"] if fps else []
    if grade:
        chain.append(grade.ffmpeg)
    chain.append("scale='if(gt(iw,ih),min(%d,iw),-2)':'if(gt(iw,ih),-2,min(%d,ih))',setsar=1,format=yuv420p"
                 % (max_side, max_side))
    parts.append(f"{src}{','.join(chain)}[vout]")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(target), "-filter_complex", ";".join(parts), "-map", "[vout]"]
    cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"] if (info.has_audio and with_audio) else ["-an"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-movflags", "+faststart", str(out_path)]
    run(cmd, timeout=600)
    return Path(out_path)


# ------------------------------------------------------------------------------------------------- plan
def _fb(attr: str, ref_value: str, chosen: str, status: str, explanation: str, score=None) -> FallbackEntry:
    return FallbackEntry(attribute=attr, reference_value=ref_value, chosen=chosen, status=status,
                         score=score, explanation=explanation)


def _transition_choice(ref_type: str) -> tuple[str, str, Optional[FallbackEntry]]:
    """-> (type used in the render, plan status, fallback log entry or None)."""
    if ref_type in RENDERED_TRANSITIONS:
        return ref_type, "applied", None
    why = ("unclassified soft change in the reference" if ref_type == "other"
           else f"the renderer in this version only produces hard cuts and zoom punch-ins, not '{ref_type}'")
    return "hard cut", "fallback", _fb("transition", ref_type, "hard cut", "fallback",
                                       f"{why}; a hard cut is used. The plan keeps '{ref_type}' for an editor that supports it.")


def apply_style(reference: StyleCard, target_path, selected: set[str] | list[str], out_path: Optional[str] = None, *,
                strength: float = 0.8, zoom: float = DEFAULT_ZOOM, snap_beats: Optional[bool] = None,
                client: Optional[GeminiClient] = None, transcript: Optional[list[TranscriptWord]] = None,
                caption_offset_s: float = 0.0, words_per_caption: Optional[int] = None,
                auto_sync: bool = False, timeout_s: Optional[float] = None) -> ApplyResult:
    t0 = time.perf_counter()

    def _check_timeout():
        if timeout_s is not None and (time.perf_counter() - t0) > timeout_s:
            raise AnalysisError(f"Style transfer exceeded the time budget of {timeout_s:.0f}s")

    sel = set(selected)
    warnings = check_selection(sel, reference)
    info = analyze_target(target_path)
    _check_timeout()
    o = reference.overall
    items: list[PlanItem] = []
    fallbacks: list[FallbackEntry] = []
    cuts: list[float] = []
    zooms: list[float] = [1.0]
    grade: Optional[ColorAdjust] = None
    before_stats: Optional[dict[str, float]] = None
    caption_job = None
    used_words: Optional[list[TranscriptWord]] = None

    # ---- pacing + audio rhythm + transitions (they share the cut schedule)
    ref_lengths = [s.end_s - s.start_s for s in reference.scenes]
    if "pacing" in sel:
        use_beats = ("audio_rhythm" in sel) if snap_beats is None else snap_beats
        beats = info.beat_times if (use_beats and info.beat_times) else None
        cuts, snapped = plan_cuts(ref_lengths, info.duration_s, beats)
        ref_trans = [s.transition_in for s in reference.scenes[1:] if s.transition_in] or ["hard cut"]
        kinds = []
        for k in range(len(cuts)):
            rtype = ref_trans[k % len(ref_trans)] if "transitions" in sel else "hard cut"
            used, status, fb = _transition_choice(rtype)
            kinds.append((rtype, used, status))
            if fb and "transitions" in sel and not any(f.reference_value == rtype and f.attribute == "transition" for f in fallbacks):
                fallbacks.append(fb)
        zooms = segment_zooms([k[1] for k in kinds], zoom)
        avg = round(info.duration_s / (len(cuts) + 1), 2)
        items.append(PlanItem(attribute="pacing", action="jump-cut schedule", status="applied",
                              detail={"cut_times_s": cuts, "shots": len(cuts) + 1, "avg_shot_len_s": avg,
                                      "reference_avg_shot_len_s": o.pacing.avg_shot_len_s,
                                      "zoom_levels": zooms},
                              note="The target is one continuous take, so cuts are made visible by alternating framing "
                                   "(jump cuts). Audio is untouched; no content is removed."))
        if "audio_rhythm" in sel:
            if beats:
                items.append(PlanItem(attribute="audio_rhythm", action="snap cuts to the target's beat grid",
                                      status="applied", detail={"target_bpm": info.bpm, "snapped": snapped,
                                                                "of": len(cuts), "tolerance_s": SNAP_TOL_S}))
            else:
                items.append(PlanItem(attribute="audio_rhythm", action="snap cuts to beats", status="fallback",
                                      note="The target has no usable beat grid; exact reference timings are used."))
                fallbacks.append(_fb("audio_rhythm", f"{o.audio.bpm} BPM", "unsnapped timings", "fallback",
                                     "The target has no detectable beat, so cuts keep the reference's timings."))
        if "transitions" in sel:
            items.append(PlanItem(
                attribute="transitions", action="place transitions on the new cuts", status="applied",
                detail={"placements": [{"at_s": c, "reference_type": r, "rendered_as": u, "status": s}
                                       for c, (r, u, s) in zip(cuts, kinds)]},
                note="Transition mix mirrors the reference's order."))
    else:
        for a in ("audio_rhythm", "transitions"):
            if a in sel:
                items.append(PlanItem(attribute=a, action="skipped", status="skipped",
                                      note="Needs Pacing: there are no new cuts to attach it to."))

    # ---- color
    if "color" in sel:
        _check_timeout()
        ref_stats = o.color.stats
        if ref_stats and all(k in ref_stats for k in STAT_KEYS):
            grade, calib_hist, before_stats = calibrate_pipeline(target_path, info, cuts, zooms, ref_stats, strength)
            items.append(PlanItem(attribute="color", action="match brightness/contrast/saturation/warmth to reference",
                                  status="applied", detail={"ffmpeg": grade.ffmpeg, "adjust": grade.params(),
                                                            "strength": strength, "reference_grade": o.color.grade,
                                                            "calibration_distance_to_goal": calib_hist,
                                                            "calibrated_on": "same cuts, zoom and filter chain at low resolution"},
                                  note="Vignette and grain are not applied (detectors are unreliable on dark footage)."))
        else:
            items.append(PlanItem(attribute="color", action="skipped", status="skipped",
                                  note="The reference card has no measured color statistics."))

    # ---- text: rendered as captions when a transcript is available, otherwise a concrete plan
    if "text" in sel:
        tx = o.text
        if not tx.present:
            items.append(PlanItem(attribute="text", action="none", status="skipped",
                                  note="The reference has no detected on-screen text."))
        else:
            font = resolve_font(tx.font_style)
            fallbacks.append(_fb("font", ", ".join(x for x in [tx.font_style.category, tx.font_style.weight,
                                                               tx.font_style.geometry] if x) if tx.font_style else "unknown",
                                 f"{font.name} {font.weight}", font.status, font.explanation, font.score))
            reg = tx.region
            detail = {"font": font.name, "font_weight": font.weight, "font_url": font.url, "case": tx.case,
                      "color": tx.color, "box_or_outline": tx.outline_or_shadow, "animation": tx.animation or "none",
                      "chunking": tx.chunking, "caption_changes_per_s": tx.changes_per_s}
            pos = {"x_center": 0.5, "y_center": 0.78}
            size_rel = 0.05
            if reg:
                pos = {"x_center": round((reg[0] + reg[2]) / 2, 3), "y_center": round((reg[1] + reg[3]) / 2, 3)}
                size_rel = round((reg[3] - reg[1]) * 1.1, 3)
                detail.update(position=pos, size_rel_frame_height=size_rel)
            varied = tx.variety == "varied" or tx.position == "mixed"
            words, tsource = transcript, ("file" if transcript else None)
            if words is None and client is not None and info.has_audio:
                words = transcribe(client, target_path, info.duration_s)
                tsource = "gemini" if words else None
            status, note = ("fallback" if varied else "planned"), ""
            if varied:
                note = ("Reference typography is varied/kinetic; only its dominant treatment is described"
                        + (" and rendered. " if words else ". "))
                fallbacks.append(_fb("text", "varied kinetic typography", "dominant treatment only", "fallback",
                                     "Fonts, sizes, colors and positions change throughout the reference; only the dominant look is used."))
            sync_info = None
            if words and info.has_audio:
                try:
                    with temp_workdir() as stmp:
                        rep_sync = analyze_sync(extract_audio(target_path, stmp / "s.wav"), words)
                    sync_info = {"estimated_lag_s": rep_sync.global_lag_s, "confidence": rep_sync.global_confidence,
                                 "trustworthy": rep_sync.trustworthy,
                                 "suspect_windows": [(w.start_s, w.end_s, w.lag_s) for w in rep_sync.deviating_windows()][:6]}
                    if auto_sync:
                        words, applied = corrected_words(words, rep_sync)
                        sync_info["auto_sync"] = applied
                except Exception as e:                         # a failed check must never block the render
                    sync_info = {"error": str(e)[:120]}
            if words:
                used_words = words
                if caption_offset_s:
                    words = [TranscriptWord(text=w.text, start_s=max(0.0, w.start_s + caption_offset_s),
                                            end_s=max(0.06, w.end_s + caption_offset_s)) for w in words]
                lf = load_font(font.name, font.weight or "bold", font.category)
                chunk = words_per_caption or pick_chunk_size(words, tx.changes_per_s, tx.chunking)
                caps = chunk_words(words, chunk, tx.case)
                caption_job = (caps, CaptionStyle(size_rel=size_rel, color=hex_to_rgb(tx.color),
                                                  box=tx.outline_or_shadow or "none", x_center=pos["x_center"],
                                                  y_center=pos["y_center"], case=tx.case), lf)
                detail.update(captions=len(caps), words_per_caption=chunk, transcript_source=tsource,
                              font_file=lf.used, font_file_source=lf.source,
                              words_per_caption_from="override" if words_per_caption else "reference caption rate/label",
                              caption_offset_s=caption_offset_s, sync_check=sync_info)
                if sync_info and sync_info.get("trustworthy") and abs(sync_info["estimated_lag_s"]) >= 0.1 and not sync_info.get("auto_sync", {}).get("applied"):
                    lag = sync_info["estimated_lag_s"]
                    note += (f"Timing check: the transcript looks {'late' if lag < 0 else 'early'} by about {abs(lag):.2f}s "
                             f"relative to the audio (confidence {sync_info['confidence']}). Try --auto-sync or "
                             f"--caption-offset {lag:+.2f}. ")
                if lf.is_fallback:
                    fallbacks.append(_fb("font-file", f"{font.name} {font.weight}", lf.used, "fallback",
                                         f"The font file for {font.name} {font.weight} was not available "
                                         f"(source: {lf.source}); rendering with {lf.used}."))
                if not reg:
                    note += "Caption position/size unknown, so a lower-third default is used. "
                    fallbacks.append(_fb("text-position", "unknown", "lower third", "fallback",
                                         "The reference card has no usable caption region; captions are placed at 78% height."))
                if (tx.animation or "none") != "none":
                    note += f"Animation '{tx.animation}' is not rendered; captions swap instantly. "
                    fallbacks.append(_fb("text-animation", tx.animation, "none", "fallback",
                                         "This version renders captions without entrance animation."))
                if not varied:
                    status = "applied"
                note += ("Rendered. Caption timing comes from word timestamps (approximate); "
                         "check the transcript if the wording looks off.")
            else:
                note += ("Planned, not rendered: no transcript available (set GEMINI_API_KEY for automatic "
                         "transcription, or pass --transcript words.json).")
            items.append(PlanItem(attribute="text", action="caption style" + (" (rendered)" if words else " spec"),
                                  status=status, detail=detail, note=note.strip()))

    # ---- layout (never rendered)
    if o.layout in ("split-screen", "picture-in-picture") and sel:
        items.append(PlanItem(attribute="layout", action="reference layout noted", status="fallback",
                              detail={"reference_layout": o.layout},
                              note="Layout transfer (panels / insets) is not implemented; the target keeps its full-frame layout."))
        fallbacks.append(_fb("layout", o.layout, "full-frame (unchanged)", "fallback",
                             "This version does not rebuild split-screen or picture-in-picture layouts."))

    plan = EditingPlan(reference_sha=reference.video.sha256, target_name=Path(target_path).name, selected=sorted(sel),
                       items=items, fallbacks=fallbacks, warnings=warnings, cut_times_s=cuts, segment_zooms=zooms)
    output, match = None, None
    if out_path and (cuts or grade is not None or caption_job):
        _check_timeout()
        if caption_job:
            caps, cstyle, lf = caption_job
            with temp_workdir() as tmp:
                src = target_path
                if cuts or grade is not None:
                    src = render(target_path, tmp / "main.mp4", info, cuts, zooms, grade)
                render_captions(src, caps, cstyle, lf, out_path)
            output = str(out_path)
        else:
            output = str(render(target_path, out_path, info, cuts, zooms, grade))
        match = style_match(reference, info, output, len(cuts), before=before_stats)
    return ApplyResult(plan=plan, output_path=output, match=match, transcript=used_words,
                       seconds=round(time.perf_counter() - t0, 2))


def style_match(reference: StyleCard, info: TargetInfo, output_path, scheduled_cuts: int,
                before: Optional[dict[str, float]] = None) -> StyleMatch:
    ref_stats = {k: reference.overall.color.stats.get(k) for k in STAT_KEYS}
    have_ref = all(v is not None for v in ref_stats.values())
    after = measure_color_stats(output_path)
    m = StyleMatch(color_after={k: round(after[k], 3) for k in STAT_KEYS},
                   color_before={k: round((before or info.color_stats)[k], 3) for k in STAT_KEYS},
                   avg_shot_len_reference_s=reference.overall.pacing.avg_shot_len_s, scheduled_cuts=scheduled_cuts)
    if have_ref:
        m.color_reference = {k: round(v, 3) for k, v in ref_stats.items()}
        m.color_distance_before = color_distance(before or info.color_stats, ref_stats)
        m.color_distance_after = color_distance(after, ref_stats)
    pinfo = ffprobe_info(output_path)
    scenes = detect_scenes(output_path, pinfo["duration_s"], pinfo["fps"])
    m.detected_scenes_after = len(scenes)
    m.avg_shot_len_target_s = round(pinfo["duration_s"] / (scheduled_cuts + 1), 2)
    return m
