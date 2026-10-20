import json
import subprocess
from pathlib import Path

import pytest

from core.apply_engine import (STAT_KEYS, apply_style, calibrate_grade, color_adjust, color_distance, measure_color_stats, plan_cuts,
                               segment_zooms)
from core.bundles import ATTRIBUTES, check_selection
from core.fonts import load_catalog, resolve_font, score_font
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video
from core.schema import FontStyle
from core.utils import ffprobe_info


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def ref_card(vid_beats, tmp_path_factory):
    client = GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0)
    return analyze_video(vid_beats, client=client, cache_dir=tmp_path_factory.mktemp("c"))


# --------------------------------------------------------------------- fonts
def test_catalog_is_well_formed():
    cat = load_catalog()
    assert len(cat) >= 40
    for f in cat:
        assert {"name", "category", "weights", "width", "geometry", "mood", "license"} <= set(f)
        assert f["weights"] and "OFL" in f["license"]
    assert len({f["name"] for f in cat}) == len(cat)


def test_mono_slab_bold_resolves_to_a_mono_font():
    c = resolve_font(FontStyle(category="mono", weight="bold", width="normal", geometry="slab", mood=["tech"]))
    assert c.status == "close" and c.category == "mono" and c.weight == "bold" and c.score >= 0.8
    assert "not an identification" in c.explanation


def test_script_and_serif_resolve_to_their_categories():
    assert resolve_font(FontStyle(category="script", weight="regular")).category == "script"
    assert resolve_font(FontStyle(category="serif", weight="bold", geometry="slab")).category == "serif"


def test_unknown_or_missing_attributes_fall_back_with_a_reason():
    for fs in (None, FontStyle()):
        c = resolve_font(fs)
        assert c.status == "fallback" and c.name == "Montserrat" and "default" in c.explanation


def test_no_close_match_falls_back_to_default():
    tiny = [{"name": "OnlyFont", "category": "serif", "weights": ["regular"], "width": "normal",
             "geometry": "humanist", "mood": []}]
    c = resolve_font(FontStyle(category="mono", weight="bold"), catalog=tiny)
    assert c.status == "fallback" and c.name == "Montserrat" and "OnlyFont" in c.explanation


def test_weight_falls_back_to_nearest_available():
    c = resolve_font(FontStyle(category="serif", weight="light", geometry="humanist"))
    assert c.weight in load_catalog()[[f["name"] for f in load_catalog()].index(c.name)]["weights"]


def test_score_is_renormalised_over_provided_attributes():
    f = next(x for x in load_catalog() if x["name"] == "IBM Plex Mono")
    assert score_font(FontStyle(category="mono"), f) == 1.0
    assert score_font(FontStyle(category="sans"), f) == 0.0


# ------------------------------------------------------------------- bundles
def test_warnings_for_dependent_attributes(ref_card):
    w = {x.attribute: x for x in check_selection({"transitions"}, ref_card)}
    assert w["transitions"].suggest_add == "pacing"
    assert any(x.attribute == "audio_rhythm" for x in check_selection({"audio_rhythm"}, ref_card))
    assert check_selection({"pacing", "audio_rhythm", "transitions", "color"}, ref_card) == []


def test_pacing_without_rhythm_warns_when_reference_is_beat_driven(ref_card):
    assert (ref_card.coupling.cuts_on_beat_ratio or 0) >= 0.6
    assert any(x.suggest_add == "audio_rhythm" for x in check_selection({"pacing"}, ref_card))


def test_unknown_attribute_is_rejected(ref_card):
    with pytest.raises(ValueError):
        check_selection({"pacing", "sparkles"}, ref_card)
    assert set(ATTRIBUTES) == {"pacing", "audio_rhythm", "transitions", "color", "text"}


# --------------------------------------------------------------------- pacing
def test_plan_cuts_follows_and_cycles_the_reference_rhythm():
    cuts, _ = plan_cuts([2.0, 3.0], 12.0)
    assert cuts == [2.0, 5.0, 7.0, 10.0]                       # 2,3,2,3 then 12 is too close to the end
    assert plan_cuts([5.0], 4.0)[0] == []                      # target shorter than one shot -> no cuts


def test_plan_cuts_respects_min_length_and_end_margin():
    cuts, _ = plan_cuts([0.2, 0.3, 5.0], 20.0, min_len=1.0)
    assert cuts[0] == 1.0 and all(b - a >= 1.0 - 1e-9 for a, b in zip(cuts, cuts[1:])) and cuts[-1] <= 19.0


def test_plan_cuts_snaps_to_beats_only_within_tolerance():
    beats = [i * 0.5 for i in range(40)]
    cuts, snapped = plan_cuts([2.1, 2.2], 12.0, beats)
    assert len(cuts) == 5 and snapped == 5                      # every cut is within 0.3 s of a beat
    assert all(any(abs(c - b) < 1e-6 for b in beats) for c in cuts)
    far = plan_cuts([2.25], 12.0, [0.0, 10.0])
    assert far[1] == 0                                          # no beat within 0.3 s -> unchanged


def test_segment_zooms_alternate_and_punch():
    assert segment_zooms(["hard cut"] * 4, 1.1) == [1.0, 1.1, 1.0, 1.1, 1.0]
    z = segment_zooms(["hard cut", "zoom punch-in"], 1.1)
    assert z[0] == 1.0 and z[2] > z[1]


# ---------------------------------------------------------------------- color
STATS = {"saturation": 0.4, "contrast": 0.2, "brightness": 0.5, "warmth_shadows": 0.0, "warmth_highlights": 0.0}


def test_color_adjust_is_identity_for_equal_stats_or_zero_strength():
    a = color_adjust(STATS, STATS, 1.0)
    assert (a.saturation, a.contrast, a.brightness) == (1.0, 1.0, 0.0) and a.rh == a.rs == 0.0
    dark = dict(STATS, brightness=0.2, saturation=0.8)
    z = color_adjust(STATS, dark, 0.0)
    assert (z.saturation, z.contrast, z.brightness) == (1.0, 1.0, 0.0)


def test_color_adjust_moves_the_right_way_and_is_bounded():
    dark_saturated = dict(STATS, brightness=0.15, saturation=0.9, contrast=0.1, warmth_highlights=0.2)
    a = color_adjust(STATS, dark_saturated, 1.0)
    assert a.brightness > 0 and a.saturation < 1 and a.contrast > 1 and a.rh < 0 and a.bh > 0
    extreme = color_adjust(dict(STATS, saturation=0.9, brightness=0.95), dict(STATS, saturation=0.01, brightness=0.0), 1.0)
    assert extreme.saturation <= 2.0 and abs(extreme.brightness) <= 0.30
    assert "eq=" in a.ffmpeg and "colorbalance=" in a.ffmpeg


def test_color_distance_is_zero_for_identical_stats():
    assert color_distance(STATS, STATS) == 0.0
    assert color_distance(dict(STATS, saturation=0.8), STATS) > 0


# --------------------------------------------------------- end-to-end rendering
def streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.split()
    return out


def test_render_color_and_pacing_on_a_target_without_audio(ref_card, vid_static, tmp_path):
    out = tmp_path / "o.mp4"
    res = apply_style(ref_card, vid_static, ["pacing", "color"], str(out))
    info = ffprobe_info(out)
    assert out.exists() and abs(info["duration_s"] - 6.0) < 0.4 and max(info["width"], info["height"]) <= 720
    assert "audio" not in streams(out)
    assert res.plan.cut_times_s and len(res.plan.segment_zooms) == len(res.plan.cut_times_s) + 1
    m = res.match
    assert m.color_distance_after < m.color_distance_before        # the grade really moved toward the reference
    assert {i.attribute: i.status for i in res.plan.items} == {"pacing": "applied", "color": "applied"}


def test_audio_is_preserved_and_cuts_snap_to_beats(ref_card, vid_beats, tmp_path):
    out = tmp_path / "o.mp4"
    res = apply_style(ref_card, vid_beats, ["pacing", "audio_rhythm"], str(out))
    assert "audio" in streams(out) and abs(ffprobe_info(out)["duration_s"] - 10.0) < 0.4
    item = next(i for i in res.plan.items if i.attribute == "audio_rhythm")
    assert item.status == "applied" and item.detail["snapped"] >= 0 and item.detail["of"] == len(res.plan.cut_times_s)


def test_audio_rhythm_on_a_beatless_target_falls_back_and_says_so(ref_card, vid_static, tmp_path):
    res = apply_style(ref_card, vid_static, ["pacing", "audio_rhythm"], str(tmp_path / "o.mp4"))
    item = next(i for i in res.plan.items if i.attribute == "audio_rhythm")
    assert item.status == "fallback" and any(f.attribute == "audio_rhythm" for f in res.plan.fallbacks)


def test_transitions_without_pacing_is_skipped_with_a_warning(ref_card, vid_static, tmp_path):
    res = apply_style(ref_card, vid_static, ["transitions"], str(tmp_path / "o.mp4"))
    assert res.output_path is None and res.plan.items[0].status == "skipped"
    assert any(w.suggest_add == "pacing" for w in res.plan.warnings)


def test_unsupported_transitions_fall_back_to_a_hard_cut(ref_card, vid_static, tmp_path):
    card = ref_card.model_copy(deep=True)
    for s in card.scenes[1:]:
        s.transition_in = "whip pan"
    res = apply_style(card, vid_static, ["pacing", "transitions"], str(tmp_path / "o.mp4"))
    tr = next(i for i in res.plan.items if i.attribute == "transitions")
    assert all(p["rendered_as"] == "hard cut" and p["status"] == "fallback" and p["reference_type"] == "whip pan"
               for p in tr.detail["placements"])
    fb = [f for f in res.plan.fallbacks if f.attribute == "transition"]
    assert len(fb) == 1 and "whip pan" in fb[0].explanation


def test_text_only_plans_without_rendering_and_logs_the_font_match(ref_card, vid_static, tmp_path):
    res = apply_style(ref_card, vid_static, ["text"], str(tmp_path / "o.mp4"))
    assert res.output_path is None
    item = res.plan.items[0]
    assert item.attribute == "text" and item.status == "planned" and item.detail["font"]
    assert item.detail["position"]["y_center"] > 0 and "not rendered" in item.note
    assert any(f.attribute == "font" and f.status in ("close", "fallback") for f in res.plan.fallbacks)


def test_varied_typography_is_flagged_as_a_fallback(ref_card, vid_static, tmp_path):
    card = ref_card.model_copy(deep=True)
    card.overall.text.variety, card.overall.text.region = "varied", []
    res = apply_style(card, vid_static, ["text"], str(tmp_path / "o.mp4"))
    assert res.plan.items[0].status == "fallback" and "dominant" in res.plan.items[0].note
    assert any(w.attribute == "text" and "kinetic" in w.message for w in res.plan.warnings)


def test_split_screen_layout_is_reported_not_silently_ignored(ref_card, vid_static, tmp_path):
    card = ref_card.model_copy(deep=True)
    card.overall.layout = "split-screen"
    res = apply_style(card, vid_static, ["color"], str(tmp_path / "o.mp4"))
    assert any(i.attribute == "layout" and i.status == "fallback" for i in res.plan.items)


def test_measured_stats_cover_all_keys(vid_beats):
    assert set(measure_color_stats(vid_beats)) >= set(STAT_KEYS)


def test_cli_end_to_end(ref_card, vid_static, tmp_path, capsys):
    from scripts.apply_cli import main
    card_path = tmp_path / "ref.stylecard.json"
    card_path.write_text(ref_card.model_dump_json())
    out = tmp_path / "styled.mp4"
    rc = main(["--reference", str(card_path), "--target", str(vid_static), "--apply", "pacing,color,text",
               "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and out.exists() and "fallback log:" in text and "style match" in text
    plan = json.loads(out.with_suffix(".plan.json").read_text())
    assert plan["output_path"] and plan["plan"]["items"]
    assert main(["--reference", str(tmp_path / "nope.json"), "--target", str(vid_static)]) == 2


def test_closed_loop_calibration_never_gets_worse_and_usually_improves(ref_card, vid_static):
    ref = {k: ref_card.overall.color.stats[k] for k in STAT_KEYS}
    tgt = measure_color_stats(vid_static)
    adj, hist = calibrate_grade(vid_static, ref, tgt, 0.8)
    assert len(hist) >= 2 and min(hist) == hist[hist.index(min(hist))] and min(hist) <= hist[0] + 1e-9
    assert "eq=" in adj.ffmpeg


def test_calibration_falls_back_to_the_open_loop_estimate_on_failure(ref_card, tmp_path):
    ref = {k: ref_card.overall.color.stats[k] for k in STAT_KEYS}
    adj, hist = calibrate_grade(tmp_path / "missing.mp4", ref, dict(ref, brightness=0.2), 0.8)
    assert hist == [] and adj.params() == color_adjust(ref, dict(ref, brightness=0.2), 0.8).params()


def test_plan_records_calibration(ref_card, vid_static, tmp_path):
    res = apply_style(ref_card, vid_static, ["color"], str(tmp_path / "o.mp4"))
    item = res.plan.items[0]
    assert item.detail["calibration_distance_to_goal"] and res.match.color_distance_after < res.match.color_distance_before


# ----------------------------------------------------------------- debug aids
def test_debug_overlay_video_and_sheet(ref_card, vid_beats, tmp_path):
    import cv2
    from core.debug_overlay import annotate_cuts, cut_sheet
    out = tmp_path / "o.mp4"
    res = apply_style(ref_card, vid_beats, ["pacing"], str(out))
    cuts, zooms = res.plan.cut_times_s, res.plan.segment_zooms
    dbg = annotate_cuts(out, cuts, zooms, tmp_path / "dbg.mp4")
    png = cut_sheet(out, cuts, zooms, tmp_path / "sheet.png")
    assert dbg.exists() and abs(ffprobe_info(dbg)["duration_s"] - ffprobe_info(out)["duration_s"]) < 0.4
    assert "audio" in streams(dbg)                                  # audio survives the annotation pass
    img = cv2.imread(str(png))
    assert img is not None and img.shape[0] > 100 and img.shape[1] > 300


def test_debug_aids_do_nothing_without_cuts(vid_static, tmp_path):
    from core.debug_overlay import annotate_cuts, cut_sheet
    assert annotate_cuts(vid_static, [], [1.0], tmp_path / "a.mp4") is None
    assert cut_sheet(vid_static, [], [1.0], tmp_path / "a.png") is None


def test_jump_cuts_visibly_change_the_framing(ref_card, vid_static, tmp_path):
    """The point of the feature: frames just before and after a cut must differ substantially."""
    import cv2
    import numpy as np
    out = tmp_path / "o.mp4"
    res = apply_style(ref_card, vid_static, ["pacing"], str(out))
    c = res.plan.cut_times_s[0]
    cap = cv2.VideoCapture(str(out))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for t in (c - 0.15, c + 0.15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        frames.append(cap.read()[1].astype(np.float32))
    base = []
    cap2 = cv2.VideoCapture(str(vid_static))
    for t in (c - 0.15, c + 0.15):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(t * cap2.get(cv2.CAP_PROP_FPS)))
        base.append(cv2.resize(cap2.read()[1], (frames[0].shape[1], frames[0].shape[0])).astype(np.float32))
    changed = float(np.abs(frames[0] - frames[1]).mean())
    original = float(np.abs(base[0] - base[1]).mean())
    assert changed > original + 1.0                                 # the edit adds a visible change beyond the source


def test_missing_target_says_not_found_not_corrupt(ref_card, tmp_path):
    from core.utils import InvalidVideoError
    with pytest.raises(InvalidVideoError) as e:
        apply_style(ref_card, tmp_path / "nope.mp4", ["color"], str(tmp_path / "o.mp4"))
    assert "not found" in str(e.value) and "corrupt" not in str(e.value)
