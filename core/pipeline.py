"""Orchestrator: analyze_video(path) -> StyleCard.

Principle: CODE measures anything objective/numeric/timing; GEMINI only makes subjective judgments.
Every non-core stage is wrapped so a failure degrades that attribute (null + confidence 0) instead of crashing.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import prompts
from .audio import AudioResult, analyze_audio, beat_sync_score, extract_wav
from .color import ColorResult, analyze_color
from .coupling import compute_coupling
from .frames import FrameReader, jpeg_bytes
from .gemini_client import GeminiClient, Media
from .motion import analyze_motion
from .scenes import SceneSpan, detect_scenes, pacing_stats, relative_energy_labels, scene_energy
from .schema import (CAMERA_MOVE_VOCAB, GRADE_VOCAB, LAYOUT_VOCAB, SCHEMA_VERSION, SHOT_TYPE_VOCAB, Audio, Camera, Color, Degraded,
                     GradeResponse, Meta, Overall, Pacing, Scene, SceneLabelsResponse, StyleCard, TextStyle,
                     VideoInfo, VideoOverviewResponse, normalize_choice)
from .text_style import caption_activity, detect_text_frames, extract_text_style
from .transitions import aggregate, classify_transitions
from .utils import (MAX_DURATION_S, InvalidVideoError, VideoTooLongError, ffprobe_info, log, run, sha256_file,
                    stage_timer, temp_workdir)

STAGES = ["probe", "scenes", "audio", "color", "motion", "transitions", "text", "overview", "coupling", "finalize"]
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
SCENE_LABEL_BATCH = 8
OVERVIEW_MAX_BYTES = 18 * 1024 * 1024

Progress = Callable[[str, float], None]


# ----------------------------------------------------------------------- cache
def _cache_path(sha: str, cache_dir: Path, mode: str) -> Path:
    d = cache_dir / "mock" if mode == "mock" else cache_dir
    return d / f"{sha}.json"


def load_cached(sha: str, cache_dir: Path, mode: str = "live") -> Optional[StyleCard]:
    p = _cache_path(sha, cache_dir, mode)
    if not p.exists():
        return None
    try:
        card = StyleCard.model_validate_json(p.read_text())
    except Exception:
        return None
    return card if card.meta.schema_version == SCHEMA_VERSION else None


def save_cached(card: StyleCard, cache_dir: Path, mode: str) -> Path:
    p = _cache_path(card.video.sha256, cache_dir, mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(card.model_dump_json(indent=2))
    return p


# ---------------------------------------------------------------------- ffmpeg
def make_proxy(src, out, has_audio: bool) -> Path:
    """Normalise to H.264 mp4, constant frame rate, long side <= 720 px (analysis proxy)."""
    vf = "scale='if(gt(iw,ih),min(720,iw),-2)':'if(gt(iw,ih),-2,min(720,ih))'"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-fps_mode", "cfr"]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if has_audio else ["-an"]
    run(cmd + ["-movflags", "+faststart", str(out)], timeout=300)
    return Path(out)


def make_overview_clip(proxy, out, has_audio: bool) -> Path:
    vf = "scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)'"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(proxy), "-vf", vf, "-r", "12", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "48k"] if has_audio else ["-an"]
    run(cmd + ["-movflags", "+faststart", str(out)], timeout=300)
    return Path(out)


# ------------------------------------------------------------- gemini sub-steps
def _grade(client: GeminiClient, reader: FrameReader, color: ColorResult):
    stats = {k: round(v, 3) for k, v in color.stats.items()}
    stats.update({"contrast_label": color.contrast, "saturation_label": color.saturation,
                  "warmth_label": color.warmth, "palette": color.palette})
    contents: list = [prompts.GRADE.format(stats=stats, grades=", ".join(GRADE_VOCAB))]
    for t in color.representative_times:
        fr = reader.frame_at(t)
        if fr is not None:
            contents.append(Media(jpeg_bytes(fr, max_side=512), "image/jpeg"))
    return client.generate_json("color_grade", contents, GradeResponse)


def _label_scenes(client: GeminiClient, reader: FrameReader, scenes: list[SceneSpan], motion) -> tuple[dict, Optional[float], bool]:
    out: dict[int, tuple[Optional[str], Optional[str]]] = {}
    confs, ok = [], True
    for b in range(0, len(scenes), SCENE_LABEL_BATCH):
        chunk = scenes[b:b + SCENE_LABEL_BATCH]
        contents: list = [prompts.SCENE_LABELS.format(n=len(chunk), shots=", ".join(SHOT_TYPE_VOCAB),
                                                      moves=", ".join(CAMERA_MOVE_VOCAB))]
        used = 0
        for local, sc in enumerate(chunk):
            fr = reader.frame_at((sc.start_s + sc.end_s) / 2)
            if fr is None:
                continue
            contents += [f"Scene {local} (flow hint: {motion[sc.index].hint}, {sc.length:.1f}s)",
                         Media(jpeg_bytes(fr, max_side=448), "image/jpeg")]
            used += 1
        if not used:
            continue
        resp = client.generate_json(f"scene_labels[{b}:{b + len(chunk)}]", contents, SceneLabelsResponse)
        if resp is None:
            ok = False
            continue
        confs.append(resp.confidence)
        for item in resp.scenes:
            if 0 <= item.index < len(chunk):
                out[chunk[item.index].index] = (normalize_choice(item.shot_type, SHOT_TYPE_VOCAB, "other"),
                                                normalize_choice(item.camera_motion, CAMERA_MOVE_VOCAB, "other"))
    return out, (float(np.mean(confs)) if confs else None), ok and bool(confs)


def _overview(client: GeminiClient, proxy: Path, workdir: Path, has_audio: bool):
    clip = make_overview_clip(proxy, workdir / "overview.mp4", has_audio)
    data = clip.read_bytes()
    if len(data) > OVERVIEW_MAX_BYTES:
        raise RuntimeError("overview clip too large to send inline")
    prompt = prompts.OVERVIEW.format(layouts=", ".join(LAYOUT_VOCAB))
    return client.generate_json("overview", [prompt, Media(data, "video/mp4")], VideoOverviewResponse)


# --------------------------------------------------------------------- main
def analyze_video(video_path, *, client: Optional[GeminiClient] = None, cache_dir: Optional[Path] = None,
                  use_cache: bool = True, progress: Optional[Progress] = None) -> StyleCard:
    path = Path(video_path)
    if not path.is_file():
        raise InvalidVideoError(f"File not found: {path}")
    client = client or GeminiClient()
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    t_total = time.perf_counter()
    calls_before = len(client.calls)
    prog = progress or (lambda s, f: None)

    prog("probe", 0.0)
    sha = sha256_file(path)
    if use_cache:
        hit = load_cached(sha, cache_dir, client.mode)
        if hit is not None:
            hit.meta.cache_hit = True
            prog("finalize", 1.0)
            return hit

    timings: dict[str, float] = {}
    degraded: list[Degraded] = []
    notes: list[str] = []

    def degrade(stage: str, reason: str) -> None:
        log.warning("stage %s degraded: %s", stage, reason)
        degraded.append(Degraded(stage=stage, reason=str(reason)[:200]))

    def step(stage: str) -> None:
        prog(stage, STAGES.index(stage) / len(STAGES))

    with temp_workdir() as tmp:
        # ---- probe + proxy (core: failures raise friendly errors)
        step("probe")
        with stage_timer(timings, "probe"):
            info = ffprobe_info(path)
            if info["duration_s"] > MAX_DURATION_S:
                raise VideoTooLongError(
                    f"This video is {info['duration_s']:.0f}s long; the limit is {MAX_DURATION_S:.0f}s. "
                    f"Please trim it to a short clip.")
            proxy = make_proxy(path, tmp / "proxy.mp4", info["has_audio"])
            pinfo = ffprobe_info(proxy)
            duration, fps = pinfo["duration_s"], pinfo["fps"]
        video = VideoInfo(duration_s=round(duration, 3), fps=round(fps, 3),
                          resolution=(info["width"], info["height"]), sha256=sha)

        with FrameReader(proxy) as reader:
            # ---- scenes + pacing (core)
            step("scenes")
            with stage_timer(timings, "scenes"):
                scenes = detect_scenes(proxy, duration, fps)
                pace = pacing_stats(scenes, duration)
            cut_times = [s.start_s for s in scenes[1:]]

            # ---- audio
            step("audio")
            audio_res = AudioResult()
            with stage_timer(timings, "audio"):
                if info["has_audio"]:
                    try:
                        audio_res = analyze_audio(extract_wav(proxy, tmp / "audio.wav"))
                    except Exception as e:
                        degrade("audio", e)
                else:
                    notes.append("No audio track: audio fields are null and beat-based coupling is skipped.")
            sync = beat_sync_score(cut_times, audio_res.beat_times)

            # ---- color
            step("color")
            color_res: Optional[ColorResult] = None
            grade_resp = None
            with stage_timer(timings, "color"):
                try:
                    color_res = analyze_color(reader, scenes, duration)
                    if client.available:
                        grade_resp = _grade(client, reader, color_res)
                        if grade_resp is None:
                            degrade("color.grade", "Gemini failed; using measured-stats heuristic grade")
                    else:
                        degrade("color.grade", "Gemini unavailable; using measured-stats heuristic grade")
                except Exception as e:
                    degrade("color", e)

            # ---- motion (+ Gemini shot/camera labels)
            step("motion")
            motion = None
            labels: dict = {}
            labels_conf, labels_ok = None, False
            with stage_timer(timings, "motion"):
                try:
                    motion = analyze_motion(reader, scenes)
                    if client.available:
                        labels, labels_conf, labels_ok = _label_scenes(client, reader, scenes, motion)
                        if not labels_ok:
                            degrade("scene_labels", "Gemini shot/camera labels unavailable; using optical-flow hints only")
                    else:
                        degrade("scene_labels", "Gemini unavailable; using optical-flow hints only")
                except Exception as e:
                    degrade("motion", e)

            # ---- transitions
            step("transitions")
            cuts, tlabels, tconf, tok = [], {}, None, False
            with stage_timer(timings, "transitions"):
                try:
                    cuts, tlabels, tconf, tok = classify_transitions(client, reader, scenes, notes)
                    if cuts and not tok:
                        degrade("transitions", "Gemini unavailable/failed; only code-based 'hard cut' vs 'other' labels")
                except Exception as e:
                    degrade("transitions", e)

            # ---- text
            step("text")
            text = TextStyle()
            text_det, text_conf = None, None
            with stage_timer(timings, "text"):
                if client.available:
                    try:
                        text_det = detect_text_frames(client, reader, duration)
                        if text_det is None:
                            degrade("text", "Gemini text detection failed")
                        else:
                            ts, text_conf = extract_text_style(client, reader, text_det, duration)
                            if ts is None:
                                degrade("text.style", "Gemini text-style call failed")
                                text = TextStyle(present=True)
                            else:
                                text = ts
                                region = text_det.region
                                if text_det.scattered:
                                    text.position = "mixed"
                                    notes.append("Text is placed all over the frame (kinetic typography): a single "
                                                 "caption region is not meaningful, so caption pacing was not measured.")
                                elif not region and text_det.present:
                                    sample = text_det.bboxes[:2]
                                    notes.append("Gemini flagged text frames but gave no usable caption box, so caption "
                                                 f"activity was not measured (raw boxes seen: {sample or 'none'}).")
                                if region:
                                    try:
                                        text.region = region
                                        text.changes_per_s = caption_activity(reader, region, duration)
                                    except Exception as e:
                                        degrade("text.activity", e)
                    except Exception as e:
                        degrade("text", e)
                else:
                    degrade("text", "Gemini unavailable; text presence and style unknown")

            # ---- overview
            step("overview")
            overview = None
            with stage_timer(timings, "overview"):
                if client.available:
                    try:
                        overview = _overview(client, proxy, tmp, info["has_audio"])
                        if overview is None:
                            degrade("overview", "Gemini overview failed; mood/genre unknown")
                    except Exception as e:
                        degrade("overview", e)
                else:
                    degrade("overview", "Gemini unavailable; mood/genre unknown")

        # ---- assemble scenes
        step("coupling")
        energies = [scene_energy(sc.length,
                                 motion[sc.index].magnitude if motion else None,
                                 audio_res.level_between(sc.start_s, sc.end_s) if audio_res.rms.size else None)
                    for sc in scenes]
        elabels = relative_energy_labels([e[0] for e in energies])
        scene_models: list[Scene] = []
        for sc in scenes:
            m = motion[sc.index] if motion else None
            score, elabel = energies[sc.index][0], elabels[sc.index]
            shot, cam = labels.get(sc.index, (None, m.hint if m else None))
            if shot == "screen-recording":
                cam = "static"          # camera motion is meaningless for a screen recording
            tin = tlabels.get(sc.index - 1) if sc.index > 0 else None
            tout = tlabels.get(sc.index) if sc.index < len(scenes) - 1 else None
            on_beat = None
            if sc.index > 0 and audio_res.beat_times:
                on_beat = bool(beat_sync_score([sc.start_s], audio_res.beat_times) == 1.0)
            scene_models.append(Scene(
                index=sc.index, start_s=sc.start_s, end_s=sc.end_s, role=sc.role, energy=elabel, energy_score=score,
                shot_type=shot, camera_motion=cam, motion_magnitude=m.magnitude if m else None,
                transition_in=tin, transition_out=tout,
                dominant_colors=color_res.scenes[sc.index].dominant_colors if color_res else [],
                has_text=(text_det.has_text_between(sc.start_s, sc.end_s) if text_det else None),
                cut_on_beat=on_beat))

        # ---- overall
        color_model = Color()
        if color_res:
            grade = (normalize_choice(grade_resp.grade, GRADE_VOCAB, "other") if grade_resp else color_res.heuristic_grade)
            color_model = Color(palette=color_res.palette, grade=grade, contrast=color_res.contrast,
                                saturation=color_res.saturation, warmth=color_res.warmth,
                                grain=grade_resp.grain if grade_resp else None, vignette=color_res.vignette,
                                stats={k: round(v, 4) for k, v in color_res.stats.items()})
        weights: Counter = Counter()
        shots: Counter = Counter()
        for s, sc in zip(scene_models, scenes):
            if s.camera_motion:
                weights[s.camera_motion] += sc.length
            if s.shot_type:
                shots[s.shot_type] += sc.length
        camera = Camera(dominant_movement=weights.most_common(1)[0][0] if weights else None,
                        shot_types=[k for k, _ in shots.most_common(3)])
        audio_model = Audio(
            has_audio=info["has_audio"], bpm=audio_res.bpm, beat_sync_score=sync,
            genre_guess=overview.genre_guess if overview else None, energy=audio_res.energy,
            beat_times=audio_res.beat_times, energy_curve=audio_res.energy_curve)
        overall = Overall(
            pacing=Pacing(**pace), audio=audio_model, color=color_model, text=text,
            transitions=aggregate(tlabels), camera=camera,
            mood=overview.mood if overview else None,
            layout=normalize_choice(overview.layout, LAYOUT_VOCAB) if overview else None,
            structure_notes=(overview.structure_notes or None) if overview else None)

        # ---- coupling
        coupling = compute_coupling(
            cut_times=cut_times, beat_times=audio_res.beat_times, peak_times=audio_res.peak_times,
            transition_labels=tlabels, scene_stats=[c.stats for c in color_res.scenes] if color_res else [],
            cuts_per_min=pace["cuts_per_min"], text_animation=text.animation, scenes=scenes,
            avg_len=pace["avg_shot_len_s"], pacing_curve=pace["pacing_curve"])

        # ---- confidence (self-reported / heuristic, NOT calibrated) + provenance
        conf: dict[str, float] = {"pacing": 0.95 if len(scenes) >= 3 else 0.6}
        if not info["has_audio"]:
            conf["audio"] = 0.0
        elif audio_res.bpm is None:
            conf["audio"] = 0.4
        else:
            conf["audio"] = 0.9 if (audio_res.beat_strength or 0) >= 1.3 else 0.65
        conf["color"] = (0.6 + 0.25 * bool(grade_resp)) if color_res else 0.0
        conf["text"] = min(0.9, text_conf) if text_conf is not None else (0.8 if (text_det and not text_det.present) else 0.0)
        if cuts:
            conf["transitions"] = min(0.9, tconf) if (tok and tconf is not None) else 0.4
        conf["camera"] = min(0.9, labels_conf) if (labels_ok and labels_conf is not None) else (0.4 if motion else 0.0)
        conf["mood"] = min(0.9, overview.confidence) if overview else 0.0
        provenance = {
            "measured_by_code": ["pacing", "audio.bpm", "audio.beat_sync_score", "audio.energy", "color.palette",
                                 "color.stats/contrast/saturation/warmth/vignette", "scene.motion_magnitude",
                                 "scene.energy", "scene.dominant_colors", "coupling"],
            "judged_by_gemini": ["color.grade", "color.grain", "scene.shot_type", "scene.camera_motion",
                                 "transitions.type", "text.*", "mood", "audio.genre_guess", "structure_notes"],
        }
        if client.mode == "mock":
            notes.append("MOCK MODE: Gemini-judged fields are fake placeholders, not real analysis.")

        card = StyleCard(
            video=video, overall=overall, scenes=scene_models, coupling=coupling, confidence=conf,
            provenance=provenance,
            meta=Meta(gemini_mode=client.mode, gemini_model=client.model if client.mode == "live" else None,
                      analysis_seconds=round(time.perf_counter() - t_total, 2), stage_seconds=timings,
                      gemini_calls=list(client.calls[calls_before:]), degraded=degraded, notes=notes))
        prog("finalize", 1.0)

    # Only cache fully-successful results, so a degraded run never hides a later good one.
    # (use_cache=False means "ignore the existing cache and re-analyse"; the fresh result is still saved.)
    if not degraded:
        save_cached(card, cache_dir, client.mode)
    return card
