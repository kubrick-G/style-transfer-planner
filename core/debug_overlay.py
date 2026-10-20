"""Debug aids so the applied edits are VISIBLE: a labelled copy of the output video and a before/after strip per cut."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils import run, temp_workdir


def _label(img, text, org, scale):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def annotate_cuts(video_path, cut_times: list[float], zooms: list[float], out_path, flash_s: float = 0.25) -> Optional[Path]:
    """Copy of the video with 'shot k/N, framing xZ' burned in and a red border flashing at every cut. Audio kept."""
    if not cut_times:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_shots = len(cut_times) + 1
    scale, border = max(0.5, H / 900), max(4, H // 70)
    with temp_workdir() as tmp:
        silent = tmp / "silent.mp4"
        vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = i / fps
            seg = sum(1 for c in cut_times if c <= t)
            _label(fr, f"shot {seg + 1}/{n_shots}   framing x{zooms[min(seg, len(zooms) - 1)]:.2f}   t={t:.1f}s",
                   (int(W * 0.03), int(H * 0.06)), scale)
            if any(c <= t < c + flash_s for c in cut_times):
                cv2.rectangle(fr, (0, 0), (W - 1, H - 1), (0, 0, 255), border)
                _label(fr, "CUT", (int(W * 0.03), int(H * 0.06) + int(40 * scale)), scale * 1.2)
            vw.write(fr)
            i += 1
        vw.release()
        cap.release()
        run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(video_path), "-map", "0:v", "-map", "1:a?",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
             str(out_path)], timeout=600)
    return Path(out_path)


def cut_sheet(video_path, cut_times: list[float], zooms: list[float], out_png, tile_h: int = 260,
              gap_s: float = 0.15) -> Optional[Path]:
    """One row per cut: the frame just BEFORE | just AFTER. If the two look the same, the cut is invisible."""
    if not cut_times:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    for k, c in enumerate(cut_times):
        pair = []
        for t, tag in ((max(0.0, c - gap_s), "before"), (c + gap_s, "after")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                break
            h, w = fr.shape[:2]
            tile = cv2.resize(fr, (int(w * tile_h / h), tile_h), interpolation=cv2.INTER_AREA)
            z = zooms[k] if tag == "before" else zooms[min(k + 1, len(zooms) - 1)]
            _label(tile, f"cut {k + 1} @ {c:.2f}s  {tag}  x{z:.2f}", (6, 20), 0.5)
            pair.append(tile)
        if len(pair) == 2:
            rows.append(np.hstack(pair))
    cap.release()
    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for r in rows]
    per_col = 3
    while len(rows) % per_col:
        rows.append(np.full_like(rows[0], 30))
    grid = np.vstack([np.hstack(rows[i:i + per_col]) for i in range(0, len(rows), per_col)])
    cv2.imwrite(str(out_png), grid)
    return Path(out_png)
