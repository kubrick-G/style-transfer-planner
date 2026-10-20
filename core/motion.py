"""Optical-flow motion per scene (code): magnitude 0..1 and a rough camera-motion hint."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .frames import FrameReader
from .scenes import SceneSpan

FLOW_W = 160
PAIR_DT = 1 / 6          # seconds between the two frames of a flow pair
MAX_PAIRS = 8
MAG_REF = 0.03           # mean flow (fraction of frame width per PAIR_DT) that maps to magnitude 1.0
STATIC_MAG = 0.003
RAD_MIN = 0.003


@dataclass
class MotionResult:
    magnitude: float
    hint: str
    mean_mag: float


def _gray_small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (FLOW_W, max(8, int(h * FLOW_W / w))), interpolation=cv2.INTER_AREA)


def _pair_times(sc: SceneSpan, fps: float) -> list[tuple[float, float]]:
    length = sc.length
    dt = PAIR_DT
    if length < dt + 0.1:
        dt = max(2.0 / fps, length * 0.5)
    lo, hi = sc.start_s + 0.03, sc.end_s - 0.03 - dt
    if hi <= lo:
        return [(sc.start_s + 0.01, sc.start_s + 0.01 + dt)] if length > dt + 0.02 else []
    n = min(MAX_PAIRS, max(1, int((hi - lo) / dt)))
    starts = np.linspace(lo, hi, n)
    return [(float(s), float(s + dt)) for s in starts]


def analyze_motion(reader: FrameReader, scenes: list[SceneSpan]) -> list[MotionResult]:
    results = []
    for sc in scenes:
        mags, dxs, dys, rads = [], [], [], []
        for t0, t1 in _pair_times(sc, reader.fps):
            a, b = reader.frame_at(t0), reader.frame_at(t1)
            if a is None or b is None:
                continue
            ga, gb = _gray_small(a), _gray_small(b)
            flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 2, 15, 2, 5, 1.2, 0)
            h, w = ga.shape
            fx, fy = flow[..., 0] / w, flow[..., 1] / w
            ys, xs = np.mgrid[0:h, 0:w]
            rx, ry = xs - w / 2.0, ys - h / 2.0
            rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
            mags.append(float(np.sqrt(fx ** 2 + fy ** 2).mean()))
            dxs.append(float(fx.mean()))
            dys.append(float(fy.mean()))
            rads.append(float(((fx * rx + fy * ry) / rn).mean()))
        if not mags:
            results.append(MotionResult(0.0, "other", 0.0))
            continue
        mm = float(np.mean(mags))
        mdx, mdy, mrad = float(np.mean(dxs)), float(np.mean(dys)), float(np.mean(rads))
        jitter = float(np.mean([abs(x - mdx) + abs(y - mdy) for x, y in zip(dxs, dys)]))
        if mm < STATIC_MAG:
            hint = "static"
        elif len(mags) >= 3 and jitter > 0.6 * mm and mm > 0.006:
            hint = "handheld"
        elif abs(mrad) >= max(abs(mdx), abs(mdy)) and abs(mrad) > RAD_MIN:
            hint = "push-in" if mrad > 0 else "pull-out"
        elif abs(mdx) >= abs(mdy) and abs(mdx) > RAD_MIN:
            hint = "pan-right" if mdx < 0 else "pan-left"   # content moves left when camera pans right
        elif abs(mdy) > RAD_MIN:
            hint = "tilt-up" if mdy > 0 else "tilt-down"    # content moves down when camera tilts up
        else:
            hint = "other"
        results.append(MotionResult(round(float(np.clip(mm / MAG_REF, 0, 1)), 3), hint, mm))
    return results
