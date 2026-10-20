import json

from core.coupling import text_energy_matches_pacing
from core.frames import FrameReader
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video
from core.scenes import relative_energy_labels
from core.schema import LAYOUT_VOCAB, TextStyle
from core.text_style import TextDetection, caption_activity

CAPTION_REGION = [0.15, 0.82, 0.95, 0.95]


def test_caption_activity_matches_known_swap_rate(vid_caps):
    with FrameReader(vid_caps) as r:
        rate = caption_activity(r, CAPTION_REGION, 8.0)
    assert 1.5 <= rate <= 2.4            # 15 swaps in 8 s = ~1.9/s


def test_caption_activity_is_zero_without_captions(vid_static):
    with FrameReader(vid_static) as r:
        assert caption_activity(r, CAPTION_REGION, 6.0) < 0.2


def test_caption_activity_handles_bad_regions(vid_caps):
    with FrameReader(vid_caps) as r:
        assert caption_activity(r, None, 8.0) is None
        assert caption_activity(r, [0.1, 0.1], 8.0) is None
        assert caption_activity(r, [0.5, 0.5, 0.5, 0.5], 8.0) is None      # zero-size region


def test_text_detection_region_is_median_and_ignores_bad_boxes():
    d = TextDetection(flagged_times=[1, 2, 3],
                      bboxes=[[0.1, 0.8, 0.9, 0.9], [0.2, 0.8, 0.8, 0.9], [0.3, 0.7, 0.7, 0.95], [], [0.9, 0.2, 0.1, 0.1]])
    assert d.region == [0.2, 0.8, 0.8, 0.9]
    assert TextDetection().region is None


def test_relative_energy_spreads_a_calm_video_but_not_a_flat_one():
    labels = relative_energy_labels([0.05, 0.12, 0.30, 0.31, 0.5])
    assert labels[0] == "low" and labels[-1] == "high" and "medium" in labels
    assert relative_energy_labels([0.10, 0.12, 0.15, 0.11]) == ["low"] * 4       # barely different -> absolute
    assert relative_energy_labels([0.9, 0.95]) == ["high", "high"]               # <3 scenes -> absolute


def test_no_animation_is_neutral_for_pacing_coupling():
    assert text_energy_matches_pacing(8, "none") is None
    assert text_energy_matches_pacing(40, "none") is None
    assert text_energy_matches_pacing(8, "fade") is True


def test_pipeline_records_layout_chunking_and_caption_region(vid_beats, tmp_path):
    card = analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path)
    assert card.overall.layout in LAYOUT_VOCAB
    t = card.overall.text
    assert t.chunking == "single-word" and t.region == [0.1, 0.7, 0.9, 0.85]
    assert t.changes_per_s is not None and t.changes_per_s >= 0
    assert not card.meta.degraded


def test_camera_motion_is_forced_static_for_screen_recordings(vid_beats, tmp_path):
    def backend(contents, model):
        if model.__name__ == "SceneLabelsResponse":
            n = next(int(c.split("N_ITEMS=")[1].split()[0]) for c in contents if isinstance(c, str) and "N_ITEMS=" in c)
            return json.dumps({"scenes": [{"index": i, "shot_type": "screen-recording" if i % 2 else "medium",
                                           "camera_motion": "handheld"} for i in range(n)], "confidence": 0.9}), (0, 0)
        return mock_backend(contents, model)

    card = analyze_video(vid_beats, client=GeminiClient(backend=backend, min_interval_s=0), cache_dir=tmp_path)
    for s in card.scenes:
        assert s.camera_motion == ("static" if s.shot_type == "screen-recording" else "handheld")


def test_new_text_fields_default_to_empty():
    t = TextStyle()
    assert t.chunking is None and t.changes_per_s is None and t.region == []
