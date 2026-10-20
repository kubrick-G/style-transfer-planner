from core.schema import (GRADE_VOCAB, TEXT_ANIMATION_VOCAB, TRANSITION_VOCAB, StyleCard, normalize_choice,
                         normalize_hex)


def test_normalize_exact_alias_and_substring():
    assert normalize_choice("Whip-pan", TRANSITION_VOCAB) == "whip pan"
    assert normalize_choice("cross dissolve", TRANSITION_VOCAB) == "crossfade"
    assert normalize_choice("Word by word", TEXT_ANIMATION_VOCAB) == "word-by-word"
    assert normalize_choice("a teal and orange look", GRADE_VOCAB) == "teal-orange"


def test_normalize_default_and_bad_input():
    assert normalize_choice("zzz", TRANSITION_VOCAB, "other") == "other"
    assert normalize_choice("zzz", TRANSITION_VOCAB) is None
    assert normalize_choice(None, TRANSITION_VOCAB, "other") == "other"
    assert normalize_choice("", TRANSITION_VOCAB) is None


def test_normalize_hex():
    assert normalize_hex("text is #FFaa00 ok") == "#ffaa00"
    assert normalize_hex("white") is None
    assert normalize_hex(None) is None


def test_style_card_roundtrip_and_ignores_extra_fields():
    raw = {"video": {"duration_s": 1, "fps": 24, "resolution": [10, 20], "sha256": "x"},
           "overall": {}, "scenes": [], "coupling": {}, "surprise": 1}
    card = StyleCard.model_validate(raw)
    again = StyleCard.model_validate_json(card.model_dump_json())
    assert again.video.resolution == (10, 20)
    assert again.overall.audio.has_audio is False and again.meta.schema_version
