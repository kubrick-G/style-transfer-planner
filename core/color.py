"""Color measurements (code): palette, saturation/contrast/warmth stats, vignette, heuristic grade."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .frames import FrameReader, resize_max_side, sample_times
from .scenes import SceneSpan
from .utils import hex_color

# NOTE: thresholds are reasonable defaults, not calibrated on a labelled dataset.
SMALL_W = 96
MAX_POOL = 40
KMEANS_PIXELS = 20000
CONTRAST_LOW, CONTRAST_HIGH = 0.15, 0.27
SAT_LOW, SAT_HIGH = 0.20, 0.40
VIGNETTE_RATIO = 0.80


@dataclass
class SceneColor:
    dominant_colors: list[str]
    stats: dict[str, float]


@dataclass
class ColorResult:
    palette: list[str]
    stats: dict[str, float]
    contrast: str
    saturation: str
    warmth: str
    vignette: bool
    heuristic_grade: str
    scenes: list[SceneColor] = field(default_factory=list)
    representative_times: list[float] = field(default_factory=list)


def _small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (SMALL_W, max(1, int(h * SMALL_W / w))), interpolation=cv2.INTER_AREA)


def frame_stats(bgr: np.ndarray) -> dict[str, float]:
    f = bgr.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0)
    p5, p25, p75, p95 = np.percentile(lum, [5, 25, 75, 95])
    sh, hi = lum <= p25, lum >= p75
    h, w = lum.shape
    ph, pw = max(1, int(h * 0.12)), max(1, int(w * 0.12))
    corners = np.mean([lum[:ph, :pw].mean(), lum[:ph, -pw:].mean(), lum[-ph:, :pw].mean(), lum[-ph:, -pw:].mean()])
    center = lum[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)].mean()
    return {
        "saturation": sat,
        "contrast": float(lum.std()),
        "brightness": float(lum.mean()),
        "warmth_shadows": float((r - b)[sh].mean()),
        "warmth_highlights": float((r - b)[hi].mean()),
        "black_level": float(p5),
        "white_level": float(p95),
        "vignette_ratio": float(corners / (center + 1e-6)),
    }


def _mean_stats(stats: list[dict]) -> dict[str, float]:
    return {k: round(float(np.mean([s[k] for s in stats])), 4) for k in stats[0]}


def _kmeans_colors(pixels_rgb: np.ndarray, k: int) -> list[str]:
    if len(pixels_rgb) > KMEANS_PIXELS:
        idx = np.random.default_rng(0).choice(len(pixels_rgb), KMEANS_PIXELS, replace=False)
        pixels_rgb = pixels_rgb[idx]
    k = max(1, min(k, len(np.unique(pixels_rgb, axis=0))))
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pixels_rgb)
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    return [hex_color(*km.cluster_centers_[i]) for i in order]


def label_contrast(c: float) -> str:
    return "low" if c < CONTRAST_LOW else ("high" if c > CONTRAST_HIGH else "medium")


def label_saturation(s: float) -> str:
    return "low" if s < SAT_LOW else ("high" if s > SAT_HIGH else "medium")


def label_warmth(ws: float, wh: float) -> str:
    if wh > 0.05 and ws < -0.02:
        return "warm-highlights-cool-shadows"
    if wh > 0.05 and ws > 0.05:
        return "warm"
    if wh < -0.05 and ws < -0.05:
        return "cool"
    return "neutral"


def heuristic_grade(s: dict[str, float]) -> str:
    """Fallback grade name from measurements alone (used if Gemini is unavailable)."""
    if s["saturation"] < 0.06:
        return "black-and-white"
    if s["warmth_highlights"] > 0.05 and s["warmth_shadows"] < -0.02 and s["saturation"] >= 0.2:
        return "teal-orange"
    if s["black_level"] > 0.12 and s["contrast"] < 0.20:
        return "faded/matte"
    if s["contrast"] > CONTRAST_HIGH:
        return "high-contrast"
    return "natural"


def analyze_color(reader: FrameReader, scenes: list[SceneSpan], duration_s: float) -> ColorResult:
    pool: list[tuple[float, np.ndarray]] = []
    for t in sample_times(duration_s, hz=1.0, max_n=MAX_POOL):
        fr = reader.frame_at(t)
        if fr is not None:
            pool.append((t, _small(fr)))
    mids: list[Optional[tuple[float, np.ndarray]]] = []
    for sc in scenes:
        t = (sc.start_s + sc.end_s) / 2
        fr = reader.frame_at(t)
        mids.append((t, _small(fr)) if fr is not None else None)
    allf = pool + [m for m in mids if m is not None]
    if not allf:
        raise RuntimeError("no frames could be read for color analysis")

    stats_list = [frame_stats(f) for _, f in allf]
    stats = _mean_stats(stats_list)
    px = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2RGB).reshape(-1, 3) for _, f in allf])
    palette = _kmeans_colors(px, 5)

    scene_colors = []
    for sc, mid in zip(scenes, mids):
        frames = [f for t, f in pool if sc.start_s <= t < sc.end_s] + ([mid[1]] if mid else [])
        if not frames:
            frames = [f for _, f in allf[:1]]
        rgb = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2RGB).reshape(-1, 3) for f in frames])
        scene_colors.append(SceneColor(_kmeans_colors(rgb, 3), _mean_stats([frame_stats(f) for f in frames])))

    n = 3
    rep = [round(duration_s * (i + 0.5) / n, 3) for i in range(n)]
    return ColorResult(
        palette=palette, stats=stats,
        contrast=label_contrast(stats["contrast"]), saturation=label_saturation(stats["saturation"]),
        warmth=label_warmth(stats["warmth_shadows"], stats["warmth_highlights"]),
        vignette=bool(stats["vignette_ratio"] < VIGNETTE_RATIO),
        heuristic_grade=heuristic_grade(stats), scenes=scene_colors, representative_times=rep,
    )
