"""Shot boundaries (PySceneDetect), pacing statistics and per-scene energy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scenedetect import ContentDetector, detect

CONTENT_THRESHOLD = 27.0   # PySceneDetect default; lower = more sensitive
MIN_SCENE_LEN_S = 0.4      # ignore "scenes" shorter than this (flashes, compression glitches)


def _secs(t) -> float:
    return float(t.seconds) if hasattr(t, "seconds") else float(t.get_seconds())


@dataclass
class SceneSpan:
    index: int
    start_s: float
    end_s: float
    role: str

    @property
    def length(self) -> float:
        return self.end_s - self.start_s


def merge_short_spans(spans: list[tuple[float, float]], min_len: float) -> list[tuple[float, float]]:
    """Merge any span shorter than min_len into its previous neighbour (or the next one for the first span)."""
    spans = list(spans)
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for i, (a, b) in enumerate(spans):
            if b - a < min_len:
                if i > 0:
                    spans[i - 1] = (spans[i - 1][0], b)
                else:
                    spans[1] = (a, spans[1][1])
                del spans[i]
                changed = True
                break
    return spans


def detect_scenes(path, duration_s: float, fps: float,
                  threshold: float = CONTENT_THRESHOLD, min_len_s: float = MIN_SCENE_LEN_S) -> list[SceneSpan]:
    min_frames = max(1, int(round(min_len_s * fps)))
    raw = detect(str(path), ContentDetector(threshold=threshold, min_scene_len=min_frames))
    bounds = [(_secs(s), _secs(e)) for s, e in raw]
    if not bounds:
        bounds = [(0.0, duration_s)]
    # make contiguous and clamp to the real duration, then absorb tiny fragments (e.g. a 0.07 s tail)
    starts = [0.0] + [b[0] for b in bounds[1:]]
    ends = starts[1:] + [duration_s]
    merged = merge_short_spans(list(zip(starts, ends)), min_len_s)
    spans = []
    for i, (a, b) in enumerate(merged):
        role = "hook" if i == 0 else ("end" if i == len(merged) - 1 else "body")
        spans.append(SceneSpan(i, round(a, 3), round(b, 3), role))
    return spans


def pacing_stats(scenes: list[SceneSpan], duration_s: float) -> dict:
    n = len(scenes)
    cut_times = [s.start_s for s in scenes[1:]]
    avg = duration_s / n if n else None
    cpm = (n - 1) / duration_s * 60.0 if duration_s > 0 else None
    return {
        "avg_shot_len_s": round(avg, 3) if avg else None,
        "cuts_per_min": round(cpm, 2) if cpm is not None else None,
        "pacing_curve": pacing_curve(cut_times, duration_s),
        "n_scenes": n,
    }


def pacing_curve(cut_times: list[float], duration_s: float) -> str:
    """Coarse shape of cut density across thirds of the video."""
    if len(cut_times) < 3 or duration_s <= 0:
        return "too-few-cuts"
    edges = [0, duration_s / 3, 2 * duration_s / 3, duration_s + 1e-6]
    counts = [sum(1 for c in cut_times if edges[i] <= c < edges[i + 1]) for i in range(3)]
    mean = sum(counts) / 3
    if mean == 0:
        return "too-few-cuts"
    hi, lo = 1.3 * mean, 0.7 * mean
    r1, r2, r3 = counts
    if all(lo <= c <= hi for c in counts):
        return "steady"
    if r1 >= hi and r3 <= mean:
        return "fast-start-then-slower"
    if r3 >= hi and r1 <= mean:
        return "slow-start-then-faster"
    if r2 >= hi:
        return "fast-middle"
    if r2 <= lo:
        return "slow-middle"
    return "uneven"


def scene_energy(length_s: float, motion: Optional[float], audio_level: Optional[float]) -> tuple[float, str]:
    """Code-measured scene energy: motion + short shot length + audio loudness (0..1 each)."""
    shortness = max(0.0, min(1.0, 1.0 - length_s / 4.0))
    m = 0.0 if motion is None else motion
    if audio_level is None:
        score = 0.6 * m + 0.4 * shortness
    else:
        score = 0.4 * m + 0.3 * audio_level + 0.3 * shortness
    label = "low" if score < 0.33 else ("medium" if score < 0.66 else "high")
    return round(score, 3), label


def relative_energy_labels(scores: list[float]) -> list[str]:
    """Label each scene low/medium/high relative to the OTHER scenes in the same video (terciles of its own range).

    Falls back to absolute thresholds when there are <3 scenes or the scores barely differ (a uniformly calm or
    uniformly busy video shouldn't be stretched into low/medium/high)."""
    def absolute(x: float) -> str:
        return "low" if x < 0.33 else ("medium" if x < 0.66 else "high")

    if len(scores) < 3 or max(scores) - min(scores) < 0.15:
        return [absolute(x) for x in scores]
    lo, hi = min(scores), max(scores)
    return ["low" if (x - lo) / (hi - lo) < 1 / 3 else ("medium" if (x - lo) / (hi - lo) < 2 / 3 else "high")
            for x in scores]
