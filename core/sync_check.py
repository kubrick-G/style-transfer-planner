"""Estimate whether a word-timed transcript is EARLY or LATE relative to the audio, globally and in short windows.

Idea: where the transcript says someone is speaking, the audio should carry speech energy. Slide the transcript's
speech/pause pattern against the audio energy envelope and take the shift with the best match. A late transcript
(captions appear after the words are spoken) gives a NEGATIVE lag: shift the captions earlier by that amount.

This is a statistical estimate (needs pauses between phrases to lock on; weak on continuous speech, music-heavy or noisy
audio), so every estimate carries a confidence and corrections are opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .transcript import TranscriptWord, clean_words
from .utils import run

SR = 16000
HOP_S = 0.01
MAX_LAG_S = 1.0
WINDOW_S = 3.0
STEP_S = 1.0
MIN_WORDS = 5
MIN_CONF = 0.12            # a single WINDOW's confidence must reach this to count as evidence
TRUST_CONF = 0.30          # the GLOBAL confidence needed before we call an estimate trustworthy (real speech is messier than
                           # synthetic speech: 0.21 with wildly disagreeing windows was seen on a real clip)
AGREE_TOL_S = 0.30         # confident windows must agree with the global lag to within this
AGREE_FRACTION = 0.6
EDGE_FRACTION = 0.9        # a lag this close to the edge of the search range is probably "off the end", not a real answer
APPLY_MIN_LAG_S = 0.06     # ignore corrections smaller than this


@dataclass
class WindowEstimate:
    center_s: float
    start_s: float
    end_s: float
    lag_s: float
    confidence: float
    n_words: int


@dataclass
class SyncReport:
    global_lag_s: float
    global_confidence: float
    windows: list[WindowEstimate] = field(default_factory=list)
    duration_s: float = 0.0
    max_lag_s: float = MAX_LAG_S

    def distrust_reasons(self) -> list[str]:
        reasons = []
        if self.global_confidence < TRUST_CONF:
            reasons.append(f"low confidence ({self.global_confidence:.2f} < {TRUST_CONF})")
        if abs(self.global_lag_s) >= EDGE_FRACTION * self.max_lag_s:
            reasons.append(f"the best lag ({self.global_lag_s:+.2f}s) is at the edge of the ±{self.max_lag_s:g}s search range "
                           f"(try a wider --max-lag)")
        good = [w for w in self.windows if w.confidence >= MIN_CONF]
        if len(good) >= 3:
            agree = sum(abs(w.lag_s - self.global_lag_s) <= AGREE_TOL_S for w in good) / len(good)
            if agree < AGREE_FRACTION:
                spread = f"{min(w.lag_s for w in good):+.2f}..{max(w.lag_s for w in good):+.2f}s"
                reasons.append(f"the windows disagree with each other ({spread}); a uniformly shifted transcript would not")
        return reasons

    @property
    def trustworthy(self) -> bool:
        return not self.distrust_reasons()

    def deviating_windows(self, tol_s: float = 0.15) -> list[WindowEstimate]:
        return [w for w in self.windows if w.confidence >= MIN_CONF and abs(w.lag_s - self.global_lag_s) > tol_s]


def extract_audio(video_path, wav_path) -> Path:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
         str(wav_path)], timeout=180)
    return Path(wav_path)


def speech_envelope(wav_path) -> np.ndarray:
    """Speech-presence signal in 0..1 at 100 Hz (log energy, floored at the noise level)."""
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=int(0.03 * SR), hop_length=int(HOP_S * SR))[0]
    env = np.log(rms + 1e-4)
    lo, hi = np.percentile(env, 20), np.percentile(env, 95)
    return np.clip((env - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def word_indicator(words: list[TranscriptWord], n: int) -> np.ndarray:
    ind = np.zeros(n, np.float32)
    for w in words:
        a, b = int(round(w.start_s / HOP_S)), int(round(w.end_s / HOP_S))
        ind[max(0, a):max(0, min(n, b)) + 1] = 1.0
    k = np.ones(5, np.float32) / 5                         # words boundaries are approximate: soften them
    return np.convolve(ind, k, mode="same")


def best_lag(env: np.ndarray, ind: np.ndarray, i0: int, i1: int, max_lag: int) -> tuple[float, float]:
    """Lag (seconds) at which the transcript pattern best matches env[i0:i1], and a confidence (contrast)."""
    pad = np.zeros(max_lag, np.float32)
    ind_p = np.concatenate([pad, ind, pad])
    a = env[i0:i1] - env[i0:i1].mean()
    na = np.linalg.norm(a)
    if na < 1e-6 or i1 - i0 < 20:
        return 0.0, 0.0
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros(len(lags))
    for k, tau in enumerate(lags):
        b = ind_p[max_lag + i0 - tau: max_lag + i1 - tau]
        b = b - b.mean()
        nb = np.linalg.norm(b)
        corr[k] = float(a @ b / (na * nb)) if nb > 1e-6 else 0.0
    j = int(np.argmax(corr))
    far = np.abs(lags - lags[j]) > int(0.3 / HOP_S)
    contrast = float(corr[j] - (corr[far].mean() if far.any() else 0.0))
    return float(lags[j]) * HOP_S, contrast


def analyze_sync(wav_path, words: list[TranscriptWord], max_lag_s: float = MAX_LAG_S) -> SyncReport:
    env = speech_envelope(wav_path)
    n = len(env)
    duration = n * HOP_S
    ind = word_indicator(words, n)
    max_lag = int(max_lag_s / HOP_S)
    g_lag, g_conf = best_lag(env, ind, 0, n, max_lag)
    wins = []
    half = WINDOW_S / 2
    c = half
    while c <= duration - half + 1e-9:
        i0, i1 = int((c - half) / HOP_S), int((c + half) / HOP_S)
        inside = [w for w in words if c - half <= (w.start_s + w.end_s) / 2 <= c + half]
        if len(inside) >= MIN_WORDS:
            lag, conf = best_lag(env, ind, i0, min(i1, n), max_lag)
            wins.append(WindowEstimate(round(c, 2), round(c - half, 2), round(c + half, 2), round(lag, 3), round(conf, 3), len(inside)))
        c += STEP_S
    return SyncReport(round(g_lag, 3), round(g_conf, 3), wins, round(duration, 2), max_lag_s)


def corrected_words(words: list[TranscriptWord], report: SyncReport) -> tuple[list[TranscriptWord], dict]:
    """Shift EVERY word by the global lag. Only when the estimate is trustworthy and large enough to matter.

    Local (per-window) correction was tried and rejected: on 12 random synthetic late runs it helped about a third of
    the time and made one case worse, because a short run of late words gives a short window too little signal.
    Windows are therefore diagnostic only (see SyncReport.deviating_windows)."""
    lag = report.global_lag_s
    if not report.trustworthy:
        return [w.model_copy() for w in words], {"applied": False, "reason": "estimate not trustworthy: " + "; ".join(report.distrust_reasons()),
                                                 "lag_s": lag, "confidence": report.global_confidence}
    if abs(lag) < APPLY_MIN_LAG_S:
        return [w.model_copy() for w in words], {"applied": False, "reason": "already in sync (within 60 ms)",
                                                 "lag_s": lag, "confidence": report.global_confidence}
    out = [TranscriptWord(text=w.text, start_s=max(0.0, w.start_s + lag), end_s=max(0.06, w.end_s + lag)) for w in words]
    out = clean_words(out, max(max((w.end_s for w in out), default=0.0) + 1.0, report.duration_s))
    return out, {"applied": True, "lag_s": lag, "confidence": report.global_confidence}
