import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.gemini_client as gc
import core.pipeline as pl
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video
from core.schema import FontStyle

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def demo_card(vid_beats, tmp_path_factory):
    card = analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path_factory.mktemp("c"))
    card.meta.gemini_mode, card.meta.notes = "live", []                  # present it as a normal card
    t = card.overall.text
    t.font_style = FontStyle(category="mono", weight="bold", width="normal", geometry="slab", mood=["tech"])
    t.case, t.color, t.outline_or_shadow, t.animation = "uppercase", "#ffffff", "background-box", "none"
    t.chunking, t.changes_per_s, t.region, t.variety = "short-chunks", 1.94, [0.241, 0.377, 0.758, 0.421], "consistent"
    return card


@pytest.fixture
def env(monkeypatch, tmp_path, demo_card, vid_static):
    cards, vids, outs = tmp_path / "cards", tmp_path / "videos", tmp_path / "out"
    for d in (cards, vids, outs):
        d.mkdir()
    (cards / "demo.stylecard.json").write_text(demo_card.model_dump_json())
    shutil.copy(vid_static, vids / "target.mp4")
    monkeypatch.setenv("STP_CARDS_DIR", str(cards))
    monkeypatch.setenv("STP_VIDEOS_DIR", str(vids))
    monkeypatch.setenv("STP_OUT_DIR", str(outs))
    monkeypatch.setenv("GEMINI_API_KEY", "")             # blank beats .env (setdefault) so no test ever calls Gemini
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    return {"cards": cards, "videos": vids, "out": outs}


def run_app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=240).run()


def test_app_loads_with_and_without_data(env, monkeypatch, tmp_path):
    at = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["① Inputs", "② Style card", "③ Choose what to borrow", "④ Result"]
    assert at.selectbox(key="ref_choice").value == "demo" and len(at.metric) >= 5      # style card tab populated
    monkeypatch.setenv("STP_CARDS_DIR", str(tmp_path / "nothing"))
    monkeypatch.setenv("STP_DEMO_CARDS_DIR", str(tmp_path / "nothing"))
    empty = run_app()
    assert not empty.exception and empty.selectbox(key="ref_choice").options == ["➕ Analyze a new reference video…"]


def test_style_card_tab_shows_attributes_and_font_match(env):
    at = run_app()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "IBM Plex Mono" in text and "Closest open-license font" in text
    assert any("No `GEMINI_API_KEY`" in w.value for w in at.sidebar.warning)


def test_dependency_warning_and_one_click_fix(env):
    at = run_app()
    at.toggle(key="sel_pacing").set_value(False).run()
    at.toggle(key="sel_audio_rhythm").set_value(False).run()
    assert any("Transitions are placed on the cuts" in w.value for w in at.warning)
    at.button(key="add_transitions_pacing").click().run()
    assert at.toggle(key="sel_pacing").value is True
    assert not any("Transitions are placed on the cuts" in w.value for w in at.warning)


def test_apply_style_end_to_end_through_the_ui(env):
    at = run_app()
    at.selectbox(key="tgt_choice").set_value("target.mp4").run()
    at.toggle(key="sel_text").set_value(False).run()                         # no transcript source in this test
    at.button(key="apply_btn").click().run()
    assert not at.exception and not at.error
    r = at.session_state["result"]
    out = Path(r["res"].output_path)
    assert out.exists() and out.parent == env["out"] and out.name == "target__in__demo_style.mp4"
    assert {i.attribute for i in r["res"].plan.items} >= {"pacing", "audio_rhythm", "transitions", "color"}
    assert r["res"].match.color_distance_after < r["res"].match.color_distance_before
    assert any("Finished" in s.value for s in at.success)


def test_apply_is_disabled_until_a_target_and_an_attribute_are_chosen(env):
    at = run_app()
    assert at.button(key="apply_btn").disabled is True                       # no target chosen
    at.selectbox(key="tgt_choice").set_value("target.mp4").run()
    assert at.button(key="apply_btn").disabled is False
    for a in ("pacing", "audio_rhythm", "transitions", "color", "text"):
        at.toggle(key=f"sel_{a}").set_value(False).run()
    assert at.button(key="apply_btn").disabled is True                       # nothing selected


def test_text_without_a_transcript_stays_a_plan_in_the_ui(env):
    at = run_app()
    at.selectbox(key="tgt_choice").set_value("target.mp4").run()
    for a in ("pacing", "audio_rhythm", "transitions", "color"):
        at.toggle(key=f"sel_{a}").set_value(False).run()
    at.button(key="apply_btn").click().run()
    res = at.session_state["result"]["res"]
    assert not at.exception and res.output_path is None and res.plan.items[0].status == "planned"


def test_analysing_a_new_reference_saves_a_card_and_selects_it(env, monkeypatch, demo_card):
    class FakeClient:
        available, model, mode = True, "fake-model", "live"

        def __init__(self, *args, **kwargs):
            self.api_key = "fake-key"

    seen = {}

    def fake_analyze(path, client=None, progress=None, **kw):
        seen["path"] = Path(path).name
        progress and progress("probe", 1.0)
        return demo_card

    monkeypatch.setattr(gc, "GeminiClient", FakeClient)
    import core.usage_cap as uc
    monkeypatch.setattr(uc, "CappedGeminiClient", FakeClient)
    monkeypatch.setattr(pl, "analyze_video", fake_analyze)
    at = run_app()
    at.selectbox(key="ref_choice").set_value("➕ Analyze a new reference video…").run()
    at.selectbox(key="an_src").set_value("target.mp4").run()
    at.button(key="an_btn").click().run()
    assert not at.exception and seen["path"] == "target.mp4"
    assert (env["cards"] / "target.stylecard.json").exists()
    assert at.selectbox(key="ref_choice").value == "target"


def test_zero_key_demo_flow(monkeypatch, tmp_path):
    # Pure zero-key first-run: empty user outputs, no API key, using bundled demo assets
    empty_cards = tmp_path / "user_cards"
    empty_vids = tmp_path / "user_vids"
    out_dir = tmp_path / "out"
    for d in (empty_cards, empty_vids, out_dir):
        d.mkdir()

    monkeypatch.setenv("STP_CARDS_DIR", str(empty_cards))
    monkeypatch.setenv("STP_VIDEOS_DIR", str(empty_vids))
    monkeypatch.setenv("STP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    at = run_app()
    assert not at.exception
    # Demo cards must be present and selectable
    ref_opts = at.selectbox(key="ref_choice").options
    assert any("(demo)" in opt for opt in ref_opts)
    demo_ref = next(opt for opt in ref_opts if "(demo)" in opt)
    at.selectbox(key="ref_choice").set_value(demo_ref).run()

    # Demo clips must be present and selectable
    tgt_opts = at.selectbox(key="tgt_choice").options
    assert any("(demo)" in opt for opt in tgt_opts)
    demo_tgt = next(opt for opt in tgt_opts if "(demo)" in opt)
    at.selectbox(key="tgt_choice").set_value(demo_tgt).run()

    # Apply style without Gemini API (using bundled demo transcript if speech clip)
    at.button(key="apply_btn").click().run()
    assert not at.exception and not at.error
    res_data = at.session_state.get("result")
    assert res_data is not None
    assert Path(res_data["res"].output_path).exists()
    assert any("Finished" in s.value for s in at.success)


def test_sidebar_usage_cap_display_and_exhausted_state(env, monkeypatch, tmp_path):
    cap_file = tmp_path / "test_cap.json"
    monkeypatch.setenv("STP_USAGE_CAP_FILE", str(cap_file))
    monkeypatch.setenv("STP_DAILY_GEMINI_LIMIT", "10")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    at = run_app()
    assert not at.exception
    # Verify metric in sidebar
    m = next(m for m in at.sidebar.metric if "Gemini requests left today" in m.label)
    assert m.value == "10 / 10"

    # Now exhaust the cap
    from core.usage_cap import UsageCap
    c = UsageCap(path=cap_file, daily_limit=10)
    assert c.charge(10) is True

    at_exhausted = run_app()
    assert not at_exhausted.exception
    m_ex = next(m for m in at_exhausted.sidebar.metric if "Gemini requests left today" in m.label)
    assert m_ex.value == "0 / 10"
    assert any("Daily limit reached" in err.value for err in at_exhausted.sidebar.error)


def test_app_public_mode_hardening(env, monkeypatch):
    monkeypatch.setenv("STP_PUBLIC_MODE", "1")
    monkeypatch.delenv("STP_ALLOW_ANALYSIS", raising=False)

    at = run_app()
    assert not at.exception
    # Privacy notice in sidebar
    assert any("Privacy Notice" in info.value for info in at.sidebar.info)
    # Analysis is allowed by default so users can upload their reference video
    ref_opts = at.selectbox(key="ref_choice").options
    assert "➕ Analyze a new reference video…" in ref_opts

    # Disabling analysis explicitly hides it
    monkeypatch.setenv("STP_ALLOW_ANALYSIS", "0")
    at2 = run_app()
    assert not at2.exception
    assert "➕ Analyze a new reference video…" not in at2.selectbox(key="ref_choice").options


def test_switch_to_upload_reference_button(env):
    at = run_app()
    assert not at.exception
    assert at.selectbox(key="ref_choice").value == "demo"
    at.button(key="switch_to_upload").click().run()
    assert not at.exception
    assert at.selectbox(key="ref_choice").value == "➕ Analyze a new reference video…"



