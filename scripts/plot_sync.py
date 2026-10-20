#!/usr/bin/env python3
"""Draw the audio's speech energy with the transcript's words placed at their transcript times, so you can SEE a timing offset.

    python scripts/plot_sync.py outputs/uploads/clip.mp4 transcript.json            # writes outputs/<clip>.sync.png

Each row is 12 s. Filled shape = audio speech energy (bursts are words, valleys are pauses). Grey bands = pauses found in the
audio. Coloured boxes = where the TRANSCRIPT says each word is. If the boxes sit in the valleys while the bursts are just before
them, the captions are late by roughly that gap. No extra dependencies (OpenCV only).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sync_check import HOP_S, extract_audio, speech_envelope  # noqa: E402
from core.transcript import load_words_json  # noqa: E402

PX_PER_S = 150
ROW_S = 12.0
LEFT = 40
ROW_H = 230
COLORS = [(80, 160, 255), (120, 220, 120), (255, 170, 80), (210, 120, 230)]


def pauses(env: np.ndarray, thr: float = 0.18, min_s: float = 0.12) -> list[tuple[float, float]]:
    out, start = [], None
    for i, v in enumerate(env):
        if v < thr and start is None:
            start = i
        elif v >= thr and start is not None:
            if (i - start) * HOP_S >= min_s:
                out.append((start * HOP_S, i * HOP_S))
            start = None
    if start is not None and (len(env) - start) * HOP_S >= min_s:
        out.append((start * HOP_S, len(env) * HOP_S))
    return out


def draw(env: np.ndarray, words, title: str) -> np.ndarray:
    dur = max(len(env) * HOP_S, max((w.end_s for w in words), default=0.0))
    rows = int(np.ceil(dur / ROW_S))
    W = LEFT + int(ROW_S * PX_PER_S) + 20
    img = np.full((rows * ROW_H + 40, W, 3), 24, np.uint8)
    cv2.putText(img, title, (LEFT, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    pz = pauses(env)
    for r in range(rows):
        t0 = r * ROW_S
        y0 = 40 + r * ROW_H
        x = lambda t: LEFT + int((t - t0) * PX_PER_S)
        for a, b in pz:                                           # pauses in the audio (grey bands)
            if b > t0 and a < t0 + ROW_S:
                cv2.rectangle(img, (max(LEFT, x(a)), y0), (min(W - 20, x(b)), y0 + 100), (58, 58, 58), -1)
        for s in range(int(ROW_S) + 1):                           # second grid + labels
            xs = x(t0 + s)
            cv2.line(img, (xs, y0), (xs, y0 + ROW_H - 20), (48, 48, 48), 1)
            cv2.putText(img, f"{t0 + s:g}s", (xs + 2, y0 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        pts = [(x(i * HOP_S), y0 + 98 - int(v * 90)) for i, v in enumerate(env)
               if t0 <= i * HOP_S < t0 + ROW_S]
        if len(pts) > 1:
            poly = np.array([(pts[0][0], y0 + 98)] + pts + [(pts[-1][0], y0 + 98)], np.int32)
            cv2.fillPoly(img, [poly], (200, 150, 70))
        for k, w in enumerate(words):                             # transcript words, alternating heights
            if w.end_s < t0 or w.start_s >= t0 + ROW_S:
                continue
            c = COLORS[k % len(COLORS)]
            yy = y0 + 125 + (k % 3) * 24
            cv2.rectangle(img, (x(w.start_s), yy), (max(x(w.end_s), x(w.start_s) + 3), yy + 18), c, -1)
            cv2.putText(img, w.text.strip(",."), (x(w.start_s), yy + 34 if k % 3 == 2 else yy - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("transcript")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    for p in (a.video, a.transcript):
        if not Path(p).exists():
            print(f"ERROR: not found: {p}")
            return 2
    words = load_words_json(a.transcript)
    with tempfile.TemporaryDirectory() as td:
        env = speech_envelope(extract_audio(a.video, Path(td) / "a.wav"))
    out = Path(a.out) if a.out else Path("outputs") / f"{Path(a.video).stem}.sync.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw(env, words, f"{Path(a.video).name}: audio energy (filled) vs transcript words (boxes)"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
