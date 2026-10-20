"""Generate synthetic test videos with KNOWN properties (BPM, cut times, text, shake, audio on/off).

These are NOT a substitute for real reference videos; they exist so the code-measured parts of the pipeline can be
verified against ground truth (e.g. 'cuts every 2 beats at 120 BPM' -> beat_sync_score ~ 1.0).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

SR = 22050
HUES = {"warm": [8, 18, 28, 12], "cool": [88, 100, 84, 95], "mixed": [0, 47, 94, 141]}


CAPTION_WORDS = ["ONE", "WAVE", "ZIP", "MOON", "EXTRA", "JOY", "QUICK", "LAMP"]


def _frame(w, h, t, seg, seg_start, palette, speed, text, shake, rng, caption=None):
    hues = HUES[palette]
    hue = hues[seg % len(hues)] if palette != "mixed" else (47 * seg) % 180
    val, sat = (225, 150) if seg % 2 == 0 else (110, 210)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0], hsv[..., 1] = hue, sat
    hsv[..., 2] = np.clip(val + np.linspace(-25, 25, h)[:, None], 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cx = int(w * (0.5 + 0.30 * np.sin(2 * np.pi * speed * t)))
    cy = int(h * (0.5 + 0.12 * np.cos(2 * np.pi * speed * t * 0.7)))
    cv2.circle(img, (cx, cy), int(w * 0.12), (245, 245, 245), -1)
    if text and seg % 2 == 1:
        words = text.split()
        n = min(len(words), int((t - seg_start) / 0.25) + 1)
        shown = " ".join(words[:n])
        org = (int(w * 0.06), int(h * 0.80))
        cv2.putText(img, shown, org, cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, shown, org, cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
    if caption:
        org = (int(w * 0.20), int(h * 0.90))
        cv2.putText(img, caption, org, cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(img, caption, org, cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    if shake:
        M = np.float32([[1, 0, rng.uniform(-shake, shake)], [0, 1, rng.uniform(-shake, shake)]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def _audio(duration, bpm, seed):
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    t = np.arange(n) / SR
    y = 0.04 * np.sin(2 * np.pi * 220 * t)
    beat = 60.0 / bpm
    k = 0
    while k * beat < duration:
        i0 = int(k * beat * SR)
        m = min(int(0.12 * SR), n - i0)
        tt = np.arange(m) / SR
        burst = np.exp(-tt / 0.03) * (0.55 * np.sin(2 * np.pi * 80 * tt) + 0.35 * rng.standard_normal(m))
        y[i0:i0 + m] += burst
        k += 1
    return np.clip(y, -1, 1).astype(np.float32)


def make_synthetic(out, *, duration=15.0, fps=24, bpm=120, cut_every_beats=2, cut_offset_s=0.0, audio=True,
                   text=None, shake=0.0, palette="warm", speed=0.6, size=(360, 640), seed=0,
                   caption_every_s=None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cut_period = cut_every_beats * 60.0 / bpm
    cuts = [cut_offset_s + k * cut_period for k in range(1, 400) if cut_offset_s + k * cut_period < duration - 0.3]
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.mp4"
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for i in range(int(duration * fps)):
            t = i / fps
            seg = sum(1 for c in cuts if c <= t)
            seg_start = cuts[seg - 1] if seg > 0 else 0.0
            cap = CAPTION_WORDS[int(t / caption_every_s) % len(CAPTION_WORDS)] if caption_every_s else None
            vw.write(_frame(w, h, t, seg, seg_start, palette, speed, text, shake, rng, cap))
        vw.release()
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(raw)]
        if audio:
            wav = Path(td) / "a.wav"
            sf.write(str(wav), _audio(duration, bpm, seed), SR)
            cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "128k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(out)]
        subprocess.run(cmd, check=True)
    return out


# name -> kwargs; ground truth is encoded in the parameters
SYNTHETIC_SET = {
    "fast_on_beat_warm":  dict(duration=20, bpm=128, cut_every_beats=2, palette="warm", text="MAKE IT COUNT NOW", speed=1.2),
    "slow_off_beat_cool": dict(duration=20, bpm=90, cut_every_beats=8, cut_offset_s=0.2, palette="cool", speed=0.3),
    "no_audio_mixed":     dict(duration=15, audio=False, bpm=100, cut_every_beats=3, palette="mixed", speed=0.5),
    "handheld_text":      dict(duration=18, bpm=110, cut_every_beats=4, palette="warm", text="STAY HANDHELD ALWAYS", shake=7, speed=0.6),
    "fast_short_mixed":   dict(duration=15, bpm=140, cut_every_beats=2, palette="mixed", speed=1.5),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/examples/synthetic")
    a = ap.parse_args(argv)
    for name, kw in SYNTHETIC_SET.items():
        p = make_synthetic(Path(a.out_dir) / f"{name}.mp4", **kw)
        print("wrote", p)


if __name__ == "__main__":
    main()
