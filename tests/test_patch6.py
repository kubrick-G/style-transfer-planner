from core.schema import SHOT_TYPE_VOCAB, VARIETY_VOCAB, StyleCard, TextStyle, normalize_choice
from core.text_style import TextDetection


def scattered_boxes():
    return [[0.05, 0.6, 0.35, 0.75], [0.6, 0.1, 0.95, 0.3], [0.3, 0.4, 0.7, 0.6], [0.1, 0.1, 0.4, 0.3], [0.55, 0.7, 0.9, 0.9]]


def test_scattered_boxes_give_no_region():
    d = TextDetection(flagged_times=[1, 2, 3, 4, 5], bboxes=scattered_boxes())
    assert d.scattered is True and d.region is None


def test_consistent_boxes_still_give_a_region():
    d = TextDetection(flagged_times=[1, 2, 3, 4], bboxes=[[0.24, 0.38, 0.76, 0.42], [0.30, 0.38, 0.70, 0.42],
                                                         [0.22, 0.375, 0.78, 0.425], [0.28, 0.38, 0.72, 0.42]])
    assert d.scattered is False and d.region is not None and 0.37 < d.region[1] < 0.39


def test_too_few_boxes_are_never_called_scattered():
    d = TextDetection(flagged_times=[1, 2], bboxes=scattered_boxes()[:2])
    assert d.scattered is False


def test_has_text_is_tri_state():
    d = TextDetection(flagged_times=[2.5], checked_times=[0.5, 1.5, 2.5, 8.5])
    assert d.has_text_between(2.0, 3.0) is True        # text seen
    assert d.has_text_between(0.0, 1.0) is False       # sampled, no text
    assert d.has_text_between(4.0, 7.0) is None        # nothing sampled in this scene -> unknown, not "no text"


def test_new_vocab_and_fields():
    assert "motion-graphic" in SHOT_TYPE_VOCAB
    assert normalize_choice("Motion Graphic card", SHOT_TYPE_VOCAB) == "motion-graphic"
    assert normalize_choice("Varied", VARIETY_VOCAB) == "varied"
    t = TextStyle()
    assert t.variety is None and t.description is None


def test_pipeline_fills_variety_and_tri_state_text(vid_beats, tmp_path):
    from core.gemini_client import GeminiClient
    from core.mock_gemini import mock_backend
    from core.pipeline import analyze_video
    card = analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path)
    assert card.overall.text.variety == "consistent" and card.overall.text.description
    assert all(s.has_text in (True, False, None) for s in card.scenes)
    assert card.meta.schema_version == "1.2"
