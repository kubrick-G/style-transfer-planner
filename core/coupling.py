"""Coupling: relationships between attributes (a style's 'feel' lives in the combination)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .audio import BEAT_TOLERANCE_S, beat_sync_score
from .schema import Coupling

FAST_CPM, SLOW_CPM = 30.0, 15.0
PEAK_TOLERANCE_S = 0.10
ENERGETIC_ANIMS = {"pop/scale", "bounce", "word-by-word", "glitch", "karaoke-highlight"}
CALM_ANIMS = {"fade", "slide-in"}            # "none" is neutral: no entrance effect says nothing about energy
CONSISTENT_STD = {"saturation": 0.12, "contrast": 0.08, "warmth_highlights": 0.12, "warmth_shadows": 0.12}


def text_energy_matches_pacing(cuts_per_min: Optional[float], animation: Optional[str]) -> Optional[bool]:
    """True/False when pacing is clearly fast or slow and the animation is clearly energetic or calm; else None."""
    if cuts_per_min is None or animation is None:
        return None
    if cuts_per_min >= FAST_CPM:
        return True if animation in ENERGETIC_ANIMS else (False if animation in CALM_ANIMS else None)
    if cuts_per_min <= SLOW_CPM:
        return True if animation in CALM_ANIMS else (False if animation in ENERGETIC_ANIMS else None)
    return None


def transitions_on_peaks_ratio(effect_cut_times: list[float], peak_times: list[float],
                               tol: float = PEAK_TOLERANCE_S) -> Optional[float]:
    if not effect_cut_times or not peak_times:
        return None
    p = np.asarray(peak_times)
    return round(sum(1 for c in effect_cut_times if np.min(np.abs(p - c)) <= tol) / len(effect_cut_times), 3)


def grade_consistent(scene_stats: list[dict[str, float]]) -> Optional[bool]:
    if len(scene_stats) < 2:
        return None
    return all(float(np.std([s[k] for s in scene_stats])) <= lim for k, lim in CONSISTENT_STD.items())


def make_notes(scenes, transition_labels: dict[int, str], avg_len: Optional[float],
               beat_ratio: Optional[float], pacing_curve: Optional[str]) -> list[str]:
    notes = []
    if beat_ratio is not None:
        if beat_ratio >= 0.7:
            notes.append(f"Cuts land on the beat {round(beat_ratio * 100)}% of the time: the edit is rhythm-driven.")
        elif beat_ratio <= 0.3:
            notes.append(f"Only {round(beat_ratio * 100)}% of cuts land on the beat: cuts are not tied to the music.")
    if avg_len:
        by_type: dict[str, list[bool]] = {}
        for k, t in transition_labels.items():
            if t in ("hard cut", "other") or k + 1 >= len(scenes):
                continue
            by_type.setdefault(t, []).append(max(scenes[k].length, scenes[k + 1].length) < avg_len)
        for t, flags in by_type.items():
            if len(flags) >= 2 and all(flags):
                notes.append(f"All {len(flags)} '{t}' transitions sit between shorter-than-average shots: "
                             f"they belong with fast pacing.")
    if pacing_curve in ("fast-start-then-slower", "slow-start-then-faster", "fast-middle", "slow-middle"):
        notes.append(f"Pacing shape: {pacing_curve.replace('-', ' ')}.")
    return notes


def compute_coupling(*, cut_times: list[float], beat_times: list[float], peak_times: list[float],
                     transition_labels: dict[int, str], scene_stats: list[dict[str, float]],
                     cuts_per_min: Optional[float], text_animation: Optional[str], scenes,
                     avg_len: Optional[float], pacing_curve: Optional[str]) -> Coupling:
    ratio = beat_sync_score(cut_times, beat_times, BEAT_TOLERANCE_S)
    effect_times = [scenes[k + 1].start_s for k, t in transition_labels.items()
                    if t not in ("hard cut", "other") and k + 1 < len(scenes)]
    return Coupling(
        cuts_on_beat_ratio=ratio,
        text_energy_matches_pacing=text_energy_matches_pacing(cuts_per_min, text_animation),
        transitions_on_music_peaks_ratio=transitions_on_peaks_ratio(effect_times, peak_times),
        grade_consistent_across_scenes=grade_consistent(scene_stats),
        notes=make_notes(scenes, transition_labels, avg_len, ratio, pacing_curve),
    )
