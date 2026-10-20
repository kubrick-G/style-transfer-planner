from core.scenes import merge_short_spans
from core.text_style import TextDetection, normalize_bbox


def test_merge_drops_tail_fragment():
    assert merge_short_spans([(0, 5.0), (5.0, 5.07)], 0.4) == [(0, 5.07)]


def test_merge_absorbs_middle_and_leading_fragments():
    assert merge_short_spans([(0, 3), (3, 3.1), (3.1, 6)], 0.4) == [(0, 3.1), (3.1, 6)]
    assert merge_short_spans([(0, 0.1), (0.1, 4)], 0.4) == [(0, 4)]


def test_merge_leaves_normal_scenes_and_single_span_alone():
    assert merge_short_spans([(0, 1), (1, 2), (2, 3)], 0.4) == [(0, 1), (1, 2), (2, 3)]
    assert merge_short_spans([(0, 0.2)], 0.4) == [(0, 0.2)]


def test_merge_keeps_spans_contiguous():
    out = merge_short_spans([(0, 1), (1, 1.1), (1.1, 1.2), (1.2, 4), (4, 4.05)], 0.4)
    assert out[0][0] == 0 and out[-1][1] == 4.05
    assert all(a[1] == b[0] for a, b in zip(out, out[1:]))
    assert all(b - a >= 0.4 for a, b in out)


def test_bbox_accepts_unit_and_thousand_scales():
    assert normalize_bbox([0.24, 0.38, 0.76, 0.42]) == [0.24, 0.38, 0.76, 0.42]
    assert normalize_bbox([241, 377, 758, 421]) == [0.241, 0.377, 0.758, 0.421]


def test_bbox_rejects_junk():
    assert normalize_bbox([]) is None
    assert normalize_bbox([0.5, 0.5, 0.4, 0.6]) is None          # x1 < x0
    assert normalize_bbox([10, 10, 5000, 20]) is None            # not a known scale
    assert normalize_bbox(["a", 1, 2, 3]) is None
    assert normalize_bbox([0.1, 0.1, 0.1, 0.5]) is None          # zero width


def test_detection_region_uses_normalised_boxes():
    d = TextDetection(flagged_times=[1, 2], bboxes=[[200, 800, 800, 900], [0.2, 0.8, 0.8, 0.9]])
    assert d.region == [0.2, 0.8, 0.8, 0.9]
