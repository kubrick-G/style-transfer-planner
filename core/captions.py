"""Caption rendering: chunk a word-timed transcript into captions and burn them into a video in the reference's style.

Rendered attributes: font (via core.font_files), size, color, case, position, background pill / outline / shadow,
chunk size (words per caption). Text ANIMATION is not rendered (captions appear and swap instantly).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .font_files import LoadedFont, pil_font
from .transcript import TranscriptWord
from .utils import run, temp_workdir

import json as _json
import subprocess as _subprocess

MIN_CAPTION_S = 0.25
TAIL_S = 0.12             # keep a caption on screen briefly after its last word (if the next one hasn't started)
MAX_CHUNK = 6
MAX_WIDTH_FRAC = 0.94
PREVIEW_MAX_SIDE = 720


@dataclass
class Caption:
    text: str
    start_s: float
    end_s: float


@dataclass
class CaptionStyle:
    size_rel: float = 0.05                # caption box height as a fraction of frame height
    font_px: int = 0                      # filled in at render time from the real frame height
    color: tuple[int, int, int] = (255, 255, 255)
    box: str = "none"                     # background-box | outline | shadow | outline+shadow | none
    x_center: float = 0.5
    y_center: float = 0.78
    case: Optional[str] = None


def hex_to_rgb(hex_color: Optional[str], default=(255, 255, 255)) -> tuple[int, int, int]:
    if not hex_color or not hex_color.startswith("#") or len(hex_color) != 7:
        return default
    try:
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    except ValueError:
        return default


def apply_case(text: str, case: Optional[str]) -> str:
    if case == "uppercase":
        return text.upper()
    if case == "lowercase":
        return text.lower()
    if case == "title":
        return text.title()
    if case == "sentence":
        return text[:1].upper() + text[1:].lower()
    return text


def pick_chunk_size(words: list[TranscriptWord], changes_per_s: Optional[float], chunking: Optional[str]) -> int:
    """Words per caption. Prefer the MEASURED caption rate of the reference; else the labelled chunking."""
    if changes_per_s and 0.3 <= changes_per_s <= 4.0 and len(words) >= 3:
        span = max(1.0, words[-1].end_s - words[0].start_s)
        wps = len(words) / span
        return int(max(1, min(MAX_CHUNK, round(wps / changes_per_s))))
    return {"single-word": 1, "short-chunks": 2, "full-lines": 6, "mixed": 3}.get(chunking or "", 2)


def chunk_words(words: list[TranscriptWord], size: int, case: Optional[str] = None) -> list[Caption]:
    size = max(1, size)
    caps = []
    for i in range(0, len(words), size):
        grp = words[i:i + size]
        caps.append(Caption(apply_case(" ".join(w.text for w in grp), case), grp[0].start_s, grp[-1].end_s))
    for k, c in enumerate(caps):
        nxt = caps[k + 1].start_s if k + 1 < len(caps) else float("inf")
        c.end_s = max(min(c.end_s + TAIL_S, nxt), c.start_s + MIN_CAPTION_S) if nxt - c.start_s >= MIN_CAPTION_S else nxt
    return caps


def _overlay(text: str, style: CaptionStyle, font, W: int, H: int):
    """-> (bgr patch, alpha 0..1 patch, x0, y0)"""
    sw = max(2, style.font_px // 14) if "outline" in style.box else 0
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), text, font=font, stroke_width=sw)
    boxed = style.box == "background-box"
    padx = int(style.font_px * (0.45 if boxed else 0.15))
    pady = int(style.font_px * (0.28 if boxed else 0.12))
    off = max(2, style.font_px // 16) if "shadow" in style.box else 0
    w, h = (r - l) + 2 * padx + off, (b - t) + 2 * pady + off
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if boxed:
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=int(style.font_px * 0.3), fill=(0, 0, 0, 175))
    if off:
        d.text((padx - l + off, pady - t + off), text, font=font, fill=(0, 0, 0, 200), stroke_width=sw)
    d.text((padx - l, pady - t), text, font=font, fill=style.color + (255,), stroke_width=sw, stroke_fill=(0, 0, 0, 255))
    arr = np.asarray(img)
    bgr, alpha = arr[..., :3][..., ::-1].copy(), (arr[..., 3] / 255.0).astype(np.float32)
    if w > MAX_WIDTH_FRAC * W:                      # never let a caption run off the frame
        s = MAX_WIDTH_FRAC * W / w
        size = (max(1, int(w * s)), max(1, int(h * s)))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        h, w = alpha.shape
    x0 = int(min(max(0, style.x_center * W - w / 2), W - w))
    y0 = int(min(max(0, style.y_center * H - h / 2), H - h))
    return bgr, alpha, x0, y0


def _frame_rates(path) -> tuple[float, float]:
    """(r_frame_rate, avg_frame_rate) of the first video stream; they differ for variable-frame-rate files."""
    try:
        out = _subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(path)],
                              capture_output=True, text=True, timeout=60).stdout
        st = _json.loads(out)["streams"][0]

        def f(x):
            a, _, b = str(x).partition("/")
            return float(a) / float(b) if b and float(b) else float(a or 0)
        return f(st.get("r_frame_rate")), f(st.get("avg_frame_rate"))
    except Exception:
        return 0.0, 0.0


def is_variable_frame_rate(path, tol: float = 0.01) -> bool:
    r, a = _frame_rates(path)
    return r > 0 and a > 0 and abs(r - a) / r > tol


def ensure_constant_frame_rate(video_in, workdir: Path) -> Path:
    """Variable-frame-rate sources must be made constant BEFORE a frame-by-frame pass: writing frames at the average
    rate would time-warp the picture against the (untouched) audio. Real timestamps are kept (frames are duplicated)."""
    if not is_variable_frame_rate(video_in):
        return Path(video_in)
    r, _ = _frame_rates(video_in)
    out = workdir / "cfr.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video_in), "-an", "-fps_mode", "cfr", "-r", f"{r:g}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], timeout=900)
    return out


def render_captions(video_in, captions: list[Caption], style: CaptionStyle, loaded_font: LoadedFont, out_path) -> Path:
    """Burn captions into video_in (frame by frame) and write out_path (H.264 + original audio, long side <= 720)."""
    work = temp_workdir()
    tmp_cfr = work.__enter__()
    try:
        return _render_captions(video_in, ensure_constant_frame_rate(video_in, tmp_cfr), captions, style, loaded_font, out_path)
    finally:
        work.__exit__(None, None, None)


def _render_captions(video_in, frames_src, captions, style, loaded_font, out_path) -> Path:
    cap = cv2.VideoCapture(str(frames_src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {frames_src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    caps = sorted(captions, key=lambda c: c.start_s)
    style = CaptionStyle(**{**style.__dict__, "font_px": max(10, int(style.size_rel * H * 0.70))})
    font = pil_font(loaded_font, style.font_px)
    cache: dict[str, tuple] = {}
    with temp_workdir() as tmp:
        silent = tmp / "silent.mp4"
        vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        i, ci = 0, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = i / fps
            while ci < len(caps) and caps[ci].end_s <= t:
                ci += 1
            if ci < len(caps) and caps[ci].start_s <= t < caps[ci].end_s:
                txt = caps[ci].text
                if txt not in cache:
                    cache[txt] = _overlay(txt, style, font, W, H)
                bgr, alpha, x0, y0 = cache[txt]
                h, w = alpha.shape
                roi = fr[y0:y0 + h, x0:x0 + w].astype(np.float32)
                a = alpha[..., None]
                fr[y0:y0 + h, x0:x0 + w] = (roi * (1 - a) + bgr.astype(np.float32) * a).astype(np.uint8)
            vw.write(fr)
            i += 1
        vw.release()
        cap.release()
        vf = ("scale='if(gt(iw,ih),min(%d,iw),-2)':'if(gt(iw,ih),-2,min(%d,ih))',setsar=1,format=yuv420p"
              % (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(video_in), "-map", "0:v", "-map", "1:a?",
             "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "copy", "-shortest",
             "-movflags", "+faststart", str(out_path)], timeout=900)
    return Path(out_path)
