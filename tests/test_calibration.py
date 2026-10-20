import subprocess

import cv2
import numpy as np
import pytest

from core.apply_engine import (STAT_KEYS, _goal_stats, analyze_target, apply_style, calibrate_pipeline, color_distance,
                               measure_all_frames, measure_color_stats, segment_zooms)
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video

TOL = {"saturation": .2, "contrast": .1, "brightness": .2, "warmth_shadows": .1, "warmth_highlights": .1}


@pytest.fixture(scope="module")
def radial(tmp_path_factory):
    """Centre pale and desaturated, edges saturated: zooming into the centre changes the colour statistics."""
    td = tmp_path_factory.mktemp("radial")
    W, H, fps, D = 640, 360, 24, 12
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2).clip(0, 1)
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = 20, (50 + 150 * r).astype(np.uint8), (190 - 70 * r).astype(np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    raw = td / "raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for _ in range(D * fps):
        vw.write(frame)
    vw.release()
    out = td / "t.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def moderate_ref(radial, vid_beats, tmp_path_factory):
    card = analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path_factory.mktemp("c"))
    t = measure_color_stats(radial)
    card.overall.color.stats = dict(t, brightness=t["brightness"] - 0.18, saturation=t["saturation"] - 0.12,
                                    contrast=t["contrast"] + 0.03, warmth_shadows=t["warmth_shadows"] - 0.03,
                                    warmth_highlights=t["warmth_highlights"] - 0.02)
    return card


def test_more_iterations_never_hurt_and_usually_help(radial, moderate_ref):
    info = analyze_target(radial)
    ref = moderate_ref.overall.color.stats
    zooms = segment_zooms(["hard cut"], 1.15)
    _, h3, _ = calibrate_pipeline(radial, info, [4.0], zooms, ref, 0.8, iterations=3, good_enough=0.0)
    _, h10, _ = calibrate_pipeline(radial, info, [4.0], zooms, ref, 0.8, iterations=10, good_enough=0.0)
    assert min(h10) <= min(h3) + 1e-9 and min(h10) < 0.25


def test_calibration_stops_early_once_close_enough(radial, moderate_ref):
    info = analyze_target(radial)
    _, hist, _ = calibrate_pipeline(radial, info, [], [1.0], moderate_ref.overall.color.stats, 0.8, iterations=10, good_enough=5.0)
    assert len(hist) == 1                                          # already within the (very loose) threshold


@pytest.mark.parametrize("zoom", [1.15, 1.40])
def test_calibration_predicts_what_the_real_render_measures(radial, moderate_ref, tmp_path, zoom):
    """The probe renders the SAME cuts/zooms/filter chain, so its prediction must match the final render."""
    res = apply_style(moderate_ref, radial, ["pacing", "color"], str(tmp_path / "o.mp4"), zoom=zoom)
    item = next(i for i in res.plan.items if i.attribute == "color")
    predicted = min(item.detail["calibration_distance_to_goal"])
    ref = moderate_ref.overall.color.stats
    goal = _goal_stats({k: ref[k] for k in STAT_KEYS}, res.match.color_before, 0.8)
    real = float(np.mean([abs(res.match.color_after[k] - goal[k]) / TOL[k] for k in STAT_KEYS]))
    assert abs(predicted - real) < 0.15
    assert item.detail["calibrated_on"].startswith("same cuts")
    assert res.match.color_distance_after < res.match.color_distance_before


def test_before_stats_come_from_the_same_pipeline_without_the_grade(radial, moderate_ref, tmp_path):
    res = apply_style(moderate_ref, radial, ["pacing", "color"], str(tmp_path / "o.mp4"), zoom=1.4)
    original = measure_color_stats(radial)
    assert any(abs(res.match.color_before[k] - original[k]) > 0.003 for k in STAT_KEYS)   # zoomed baseline, not the raw clip


def test_probe_measurement_sees_every_frame_not_an_aliased_subset(tmp_path):
    """Alternating bright/dark frames: midpoint sampling of a 1-fps video rounds to every 2nd frame and sees only one kind."""
    fps, path = 2, tmp_path / "alt.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(20):
        vw.write(np.full((64, 64, 3), 230 if i % 2 == 0 else 20, np.uint8))
    vw.release()
    assert 0.35 < measure_all_frames(path)["brightness"] < 0.65          # sees both kinds, averages to mid-gray
