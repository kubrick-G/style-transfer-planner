"""Frame access and image helpers (OpenCV)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .utils import InvalidVideoError


class FrameReader:
    """Random-access frame reader. Intended for short, constant-frame-rate proxies."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise InvalidVideoError("Could not open the video for frame analysis.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frame_index(self, idx: int) -> Optional[np.ndarray]:
        if self.n_frames > 0:
            idx = max(0, min(int(idx), self.n_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        return fr if ok else None

    def frame_at(self, t: float) -> Optional[np.ndarray]:
        return self.frame_index(int(round(t * self.fps)))

    def iter_frames(self, sample_fps: float):
        """Yield (t, frame) at ~sample_fps in one sequential pass (much faster than many seeks)."""
        step = max(1, int(round(self.fps / sample_fps)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while self.cap.grab():
            if i % step == 0:
                ok, fr = self.cap.retrieve()
                if ok:
                    yield i / self.fps, fr
            i += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sample_times(duration: float, hz: float = 1.0, max_n: Optional[int] = None) -> list[float]:
    """Evenly spaced timestamps (bin midpoints) at ~hz, thinned evenly to at most max_n."""
    n = max(1, int(duration * hz))
    if max_n and n > max_n:
        n = max_n
    step = duration / n
    return [min(duration - 1e-3, (i + 0.5) * step) for i in range(n)]


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max_side / float(max(h, w))
    if s >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def jpeg_bytes(frame: np.ndarray, max_side: int = 512, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", resize_max_side(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def montage_rows(rows: list[list[np.ndarray]], labels: list[str], cell_w: int = 200) -> np.ndarray:
    """Stack rows of frames (one row per label). All frames are resized to cell_w wide."""
    built = []
    for row, label in zip(rows, labels):
        cells = []
        for f in row:
            h, w = f.shape[:2]
            cells.append(cv2.resize(f, (cell_w, max(1, int(h * cell_w / w))), interpolation=cv2.INTER_AREA))
        hh = min(c.shape[0] for c in cells)
        r = np.hstack([c[:hh] for c in cells])
        cv2.putText(r, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(r, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        built.append(r)
    width = max(b.shape[1] for b in built)
    built = [cv2.copyMakeBorder(b, 0, 2, 0, width - b.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for b in built]
    return np.vstack(built)
