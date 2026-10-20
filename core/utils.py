"""Shared helpers: hashing, subprocess/ffprobe wrappers, errors, timing, temp dirs."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("stp")

MAX_DURATION_S = 60.0


class AnalysisError(Exception):
    """A friendly, user-facing analysis error (safe to show in a UI)."""


class InvalidVideoError(AnalysisError):
    pass


class VideoTooLongError(AnalysisError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str], timeout: float = 300) -> subprocess.CompletedProcess:
    """Run a command; raise AnalysisError with a short stderr tail on failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        raise AnalysisError(f"Required tool not found: {cmd[0]} (is FFmpeg installed?)") from e
    except subprocess.TimeoutExpired as e:
        raise AnalysisError(f"{cmd[0]} timed out after {timeout:.0f}s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise AnalysisError(f"{cmd[0]} failed: {' | '.join(tail)}")
    return proc


def parse_fps(s: Optional[str]) -> float:
    if not s or s in ("0/0", "0"):
        return 0.0
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(s)
    except ValueError:
        return 0.0


def ffprobe_info(path: str | Path) -> dict:
    """Return duration_s, fps, width, height (as displayed, rotation-aware), has_audio."""
    try:
        proc = run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            timeout=60,
        )
    except AnalysisError as e:
        if str(e).startswith("Required tool not found"):
            raise
        raise InvalidVideoError("Could not read this file as a video (it may be corrupt or not a video).") from e
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise InvalidVideoError("Could not read this file as a video.") from e
    streams = data.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None:
        raise InvalidVideoError("No video stream found. Is this really a video file?")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fps = parse_fps(vs.get("avg_frame_rate")) or parse_fps(vs.get("r_frame_rate")) or 30.0
    dur = None
    for src in (data.get("format", {}), vs):
        try:
            dur = float(src.get("duration"))
            break
        except (TypeError, ValueError):
            continue
    if not dur or dur <= 0:
        raise InvalidVideoError("Could not determine the video's duration.")
    w, h = int(vs.get("width", 0)), int(vs.get("height", 0))
    rot = 0
    try:
        rot = int(float((vs.get("tags") or {}).get("rotate", 0)))
    except (TypeError, ValueError):
        pass
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rot = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
    if abs(rot) % 180 == 90:
        w, h = h, w
    return {"duration_s": dur, "fps": fps, "width": w, "height": h, "has_audio": has_audio}


@contextmanager
def stage_timer(store: dict, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[name] = round(time.perf_counter() - t0, 3)


@contextmanager
def temp_workdir(prefix: str = "stp_") -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def hex_color(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, round(v)))) for v in (r, g, b)))
