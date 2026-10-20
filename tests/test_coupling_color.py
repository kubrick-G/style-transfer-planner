import numpy as np

from core.color import frame_stats, heuristic_grade, label_contrast, label_saturation, label_warmth
from core.coupling import (compute_coupling, grade_consistent, text_energy_matches_pacing,
                           transitions_on_peaks_ratio)
from core.scenes import SceneSpan


def test_text_energy_matches_pacing_rules():
    assert text_energy_matches_pacing(40, "word-by-word") is True
    assert text_energy_matches_pacing(40, "fade") is False
    assert text_energy_matches_pacing(8, "fade") is True
    assert text_energy_matches_pacing(8, "bounce") is False
    assert text_energy_matches_pacing(22, "bounce") is None      # medium pacing: undecided
    assert text_energy_matches_pacing(40, "typewriter") is None  # neutral animation
    assert text_energy_matches_pacing(None, "fade") is None


def test_transitions_on_peaks_and_grade_consistency():
    assert transitions_on_peaks_ratio([1.0, 2.0], [1.05, 5.0]) == 0.5
    assert transitions_on_peaks_ratio([], [1.0]) is None
    same = {"saturation": 0.4, "contrast": 0.2, "warmth_highlights": 0.1, "warmth_shadows": -0.05}
    other = dict(same, saturation=0.9)
    assert grade_consistent([same, dict(same)]) is True
    assert grade_consistent([same, other]) is False
    assert grade_consistent([same]) is None


def test_compute_coupling_and_notes():
    scenes = [SceneSpan(i, i * 1.0, (i + 1) * 1.0, "body") for i in range(5)]
    labels = {0: "whip pan", 1: "whip pan", 2: "hard cut", 3: "whip pan"}
    c = compute_coupling(cut_times=[1, 2, 3, 4], beat_times=[1, 2, 3, 4], peak_times=[1.0, 2.0, 4.0],
                         transition_labels=labels, scene_stats=[], cuts_per_min=48.0, text_animation="bounce",
                         scenes=scenes, avg_len=2.0, pacing_curve="steady")
    assert c.cuts_on_beat_ratio == 1.0 and c.text_energy_matches_pacing is True
    assert c.transitions_on_music_peaks_ratio == 1.0           # whip pans at 1.0, 2.0, 4.0 all on peaks
    assert c.grade_consistent_across_scenes is None
    assert any("whip pan" in n for n in c.notes) and any("rhythm-driven" in n for n in c.notes)


def test_color_labels_and_stats():
    gray = np.full((40, 40, 3), 128, np.uint8)
    s = frame_stats(gray)
    assert s["saturation"] < 0.01 and s["contrast"] < 0.01 and abs(s["vignette_ratio"] - 1) < 0.01
    assert heuristic_grade(s) == "black-and-white"
    assert label_contrast(0.1) == "low" and label_contrast(0.3) == "high"
    assert label_saturation(0.1) == "low" and label_saturation(0.5) == "high"
    assert label_warmth(-0.1, 0.1) == "warm-highlights-cool-shadows"
    assert label_warmth(0.0, 0.0) == "neutral"


def test_warm_frame_reads_warm():
    bgr = np.zeros((40, 40, 3), np.uint8)
    bgr[..., 2], bgr[..., 1], bgr[..., 0] = 220, 140, 40      # orange
    s = frame_stats(bgr)
    assert s["warmth_highlights"] > 0.3 and s["saturation"] > 0.6
