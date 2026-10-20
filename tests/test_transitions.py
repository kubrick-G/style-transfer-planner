from core.transitions import reconcile


def test_hard_cut_claim_on_soft_change_is_overruled():
    label, why = reconcile("hard cut", abrupt=False)
    assert label == "other" and "not single-frame" in why


def test_crossfade_claim_on_one_frame_change_is_overruled():
    assert reconcile("crossfade", abrupt=True)[0] == "hard cut"
    assert reconcile("slide/push", abrupt=True)[0] == "hard cut"


def test_consistent_labels_are_left_alone():
    assert reconcile("hard cut", abrupt=True) == ("hard cut", None)
    assert reconcile("crossfade", abrupt=False) == ("crossfade", None)
    assert reconcile("whip pan", abrupt=True) == ("whip pan", None)      # effects can be abrupt; not contradicted
    assert reconcile("glitch", abrupt=False) == ("glitch", None)
