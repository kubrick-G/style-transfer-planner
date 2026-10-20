import json

import pytest

from core.plan import EditingPlan, FallbackEntry, PlanItem
from core.ui_helpers import (attribute_rows, card_metrics, fallback_frame, font_preview_html, list_cards, list_videos,
                             palette_html, plan_frame, safe_stem, scene_frame, timeline_chart)
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video


@pytest.fixture(scope="module")
def card(vid_beats, tmp_path_factory):
    return analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path_factory.mktemp("c"))


@pytest.fixture(scope="module")
def silent_card(vid_static, tmp_path_factory):
    return analyze_video(vid_static, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path_factory.mktemp("c2"))


def test_listing_cards_and_videos(tmp_path):
    (tmp_path / "b.stylecard.json").write_text("{}")
    (tmp_path / "a.stylecard.json").write_text("{}")
    (tmp_path / "ignore.json").write_text("{}")
    assert list(list_cards(tmp_path)) == ["a", "b"] and list_cards(tmp_path / "nope") == {}
    for n in ("x.mp4", "y.MOV", "notes.txt"):
        (tmp_path / n).write_text("")
    assert list(list_videos(tmp_path)) == ["x.mp4", "y.MOV"]


def test_safe_stem_strips_unsafe_characters():
    assert safe_stem("My Video (final).mp4") == "My_Video__final" and safe_stem("...") == "video"


def test_metrics_and_attribute_rows_cover_the_card(card):
    m = dict((l, v) for l, v, _ in card_metrics(card))
    assert m["Scenes"] == "10" and m["BPM"] != "n/a" and m["Cuts on beat"].endswith("%")
    rows = {r["attribute"]: r for r in attribute_rows(card)}
    assert {"Pacing", "Audio rhythm", "Color", "Transitions", "Text", "Camera / shots", "Layout / mood"} <= set(rows)
    assert all("measured by" in r for r in rows.values())


def test_timeline_chart_builds_a_valid_spec(card, silent_card):
    spec = timeline_chart(card).to_dict()
    assert len(spec["vconcat"]) == 2 and len(scene_frame(card)) == 10
    assert json.dumps(spec)                                              # serialisable for the browser
    quiet = timeline_chart(silent_card).to_dict()                        # no audio: no beats, no energy curve
    assert len(quiet["vconcat"]) == 2


def test_palette_and_font_preview_escape_input():
    assert palette_html(["#ffffff", '"><script>']).count("<script>") == 0
    h = font_preview_html("IBM Plex Mono", "bold", "<b>hi</b>", "#ffffff", True)
    assert "family=IBM+Plex+Mono:wght@700" in h and "&lt;b&gt;" in h and "<b>" not in h and "rgba(0,0,0,.68)" in h
    assert "wght@400" in font_preview_html("Lato", "regular", "x", None, False) and "text-shadow" in font_preview_html("Lato", "bold", "x", "bad", False)


def test_plan_and_fallback_frames():
    plan = EditingPlan(reference_sha="x", target_name="t.mp4", selected=["text"], items=[
        PlanItem(attribute="text", action="caption style (rendered)", status="applied",
                 detail={"font": "IBM Plex Mono", "captions": 45, "ignored": 1}, note="n")],
        fallbacks=[FallbackEntry(attribute="font", reference_value="mono", chosen="IBM Plex Mono bold", status="close",
                                 score=0.97, explanation="e")])
    pf, ff = plan_frame(plan), fallback_frame(plan)
    assert pf.iloc[0]["attribute"] == "text" and "captions=45" in pf.iloc[0]["details"] and "ignored" not in pf.iloc[0]["details"]
    assert ff.iloc[0]["used instead"] == "IBM Plex Mono bold" and ff.iloc[0][""] == "🟢"
