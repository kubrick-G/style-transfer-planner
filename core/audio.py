"""Audio rhythm: BPM, beat grid, beat-sync of cuts, music peaks, energy curve (librosa)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from scipy.signal import find_peaks

from .utils import run

SR = 22050
HOP = 512
BEAT_TOLERANCE_S = 0.08      # a cut "lands on the beat" if within +/- 80 ms of a beat
ENERGY_CURVE_HZ = 4.0
MIN_BEATS_FOR_BPM = 4


@dataclass
class AudioResult:
    bpm: Optional[float] = None
    beat_times: list[float] = field(default_factory=list)
    peak_times: list[float] = field(default_factory=list)
    energy_curve: list[float] = field(default_factory=list)
    rms_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms_db_mean: Optional[float] = None
    beat_strength: Optional[float] = None
    energy: Optional[str] = None

    def level_between(self, t0: float, t1: float) -> Optional[float]:
        """Mean loudness in [t0, t1) relative to the loudest moment (0..1)."""
        if self.rms.size == 0 or self.rms.max() <= 0:
            return None
        m = (self.rms_times >= t0) & (self.rms_times < t1)
        if not m.any():
            return None
        return float(self.rms[m].mean() / self.rms.max())


def extract_wav(video_path, wav_path) -> Path:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "wav", str(wav_path)], timeout=120)
    return Path(wav_path)


def analyze_audio(wav_path) -> AudioResult:
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    if y.size < SR or float(np.max(np.abs(y))) < 1e-4:
        return AudioResult(energy="low", rms_db_mean=-90.0)
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP)
    beat_times = [round(float(t), 3) for t in librosa.frames_to_time(beats, sr=SR, hop_length=HOP)]
    bpm = float(np.atleast_1d(tempo)[0]) if len(beat_times) >= MIN_BEATS_FOR_BPM else None
    strength = None
    if len(beats) and onset.mean() > 0:
        strength = float(onset[beats].mean() / (onset.mean() + 1e-9))

    on = onset / (onset.max() + 1e-9)
    pk, _ = find_peaks(on, height=float(np.percentile(on, 75)), distance=max(1, int(0.5 * SR / HOP)))
    peak_times = [round(float(t), 3) for t in librosa.frames_to_time(pk, sr=SR, hop_length=HOP)]

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)
    bins = int(np.ceil(rms_t[-1] * ENERGY_CURVE_HZ)) + 1
    curve = []
    for i in range(bins):
        m = (rms_t >= i / ENERGY_CURVE_HZ) & (rms_t < (i + 1) / ENERGY_CURVE_HZ)
        curve.append(float(rms[m].mean()) if m.any() else 0.0)
    mx = max(curve) or 1.0
    curve = [round(c / mx, 3) for c in curve]
    db = float(20 * np.log10(rms.mean() + 1e-9))
    energy = "high" if db > -18 else ("medium" if db > -30 else "low")
    return AudioResult(bpm=round(bpm, 1) if bpm else None, beat_times=beat_times, peak_times=peak_times,
                       energy_curve=curve, rms_times=rms_t, rms=rms, rms_db_mean=round(db, 1),
                       beat_strength=strength, energy=energy)


def beat_sync_score(cut_times: list[float], beat_times: list[float],
                    tol: float = BEAT_TOLERANCE_S) -> Optional[float]:
    """Fraction of cuts within +/- tol seconds of a beat. None if it can't be measured."""
    cuts = [c for c in cut_times if c > 0]
    if not cuts or not beat_times:
        return None
    b = np.asarray(beat_times)
    hits = sum(1 for c in cuts if np.min(np.abs(b - c)) <= tol)
    return round(hits / len(cuts), 3)
