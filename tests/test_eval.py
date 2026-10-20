import math
import pytest
from pathlib import Path

from eval.run_eval import evaluate_labels, format_results_markdown, match_cut_times
from core.schema import (
    Audio,
    Color,
    Coupling,
    FontStyle,
    Overall,
    Pacing,
    Scene,
    StyleCard,
    TextStyle,
    VideoInfo,
)


def make_dummy_card(bpm=120.0, layout="full-frame", grade="warm-contrast", cuts=None):
    scenes = [
        Scene(
            index=0,
            start_s=0.0,
            end_s=cuts[0] if cuts else 2.0,
            role="hook",
            energy="high",
            shot_type="medium",
            camera_motion="static",
            transition_in=None,
            has_text=True,
            cut_on_beat=True,
            motion_magnitude=0.1,
            dominant_colors=["#111111"],
        )
    ]
    if cuts:
        for i, c in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else c + 2.0
            scenes.append(
                Scene(
                    index=i + 1,
                    start_s=c,
                    end_s=end,
                    role="body",
                    energy="medium",
                    shot_type="close-up",
                    camera_motion="static",
                    transition_in="hard cut",
                    has_text=False,
                    cut_on_beat=True,
                    motion_magnitude=0.2,
                    dominant_colors=["#222222"],
                )
            )

    return StyleCard(
        video=VideoInfo(duration_s=10.0, fps=24.0, resolution=(360, 640), sha256="abc123456789"),
        overall=Overall(
            layout=layout,
            mood="energetic",
            pacing=Pacing(n_scenes=len(scenes), avg_shot_len_s=2.0, cuts_per_min=30.0, pacing_curve="steady"),
            audio=Audio(bpm=bpm, beat_sync_score=0.9, energy="high", beat_times=[]),
            color=Color(grade=grade, contrast="high", saturation="medium", warmth="warm", palette=["#ffffff"], stats={}),
            text=TextStyle(
                present=True,
                font_style=FontStyle(category="sans-serif", weight="bold", width="normal", geometry="geometric", mood=[]),
                case="uppercase",
                color="#ffffff",
                outline_or_shadow="none",
                animation="none",
                position="lower-third",
                chunking="short-chunks",
                variety="consistent",
            ),
            camera={"dominant_movement": "static", "shot_types": ["medium", "close-up"]},
            transitions=[{"type": "hard cut", "count": len(scenes) - 1}],
        ),
        coupling=Coupling(),
        scenes=scenes,
    )



def test_match_cut_times():
    truth = [1.0, 2.0, 3.0]
    pred = [1.05, 2.02, 3.8]  # third cut is outside 0.20s tolerance

    mae, matched, total = match_cut_times(truth, pred, tol_s=0.20)
    assert matched == 2
    assert total == 3
    assert abs(mae - ((0.05 + 0.02) / 2)) < 1e-4


def test_evaluate_labels_and_formatting():
    card = make_dummy_card(bpm=120.0, layout="full-frame", grade="warm-contrast", cuts=[2.0, 4.0])
    cards = {"clip1": card}

    truth = [
        {
            "clip": "clip1.mp4",
            "layout": "full-frame",
            "color_grade": "warm-contrast",
            "caption_position": "lower-third",
            "text_animation": "none",
            "font_category": "sans-serif",
            "bpm": 122.0,
            "cut_times_s": [2.05, 4.01],
            "transition_types": ["hard cut", "hard cut"],
        }
    ]

    metrics = evaluate_labels(truth, cards)
    assert metrics["n_clips"] == 1
    assert metrics["correct"]["layout"] == 1
    assert metrics["correct"]["color_grade"] == 1
    assert metrics["correct"]["caption_position"] == 1
    assert abs(metrics["bpm_mae"] - 2.0) < 1e-4
    assert metrics["cut_mae"] < 0.10

    report = format_results_markdown(metrics)
    assert "Evaluation Results: Style Transfer Planner" in report
    assert "n=1: indicative only" in report
    assert "100.0%" in report
