"""Style Transfer Planner: borrow the editing style of a reference video and apply it to your own clip.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.apply_engine import DEFAULT_ZOOM, apply_style  # noqa: E402
from core.bundles import ATTRIBUTES, BUNDLES, check_selection  # noqa: E402
from core.debug_overlay import annotate_cuts, cut_sheet  # noqa: E402
from core.fonts import resolve_font  # noqa: E402
from core.gemini_client import GeminiClient  # noqa: E402
from core.pipeline import analyze_video  # noqa: E402
from core.transcript import TranscriptWord, clean_words  # noqa: E402
from core.ui_helpers import (VIDEO_EXTS, attribute_rows, card_metrics, cards_dir, fallback_frame, font_preview_html,  # noqa: E402
                             list_cards, list_transcripts, list_videos, load_card, palette_html, plan_frame, results_dir, safe_stem,
                             scene_frame, timeline_chart, videos_dir)
from core.usage_cap import CappedGeminiClient, UsageCap  # noqa: E402
from core.utils import AnalysisError  # noqa: E402
from scripts.analyze_cli import _load_dotenv  # noqa: E402

_load_dotenv()
st.set_page_config(page_title="Style Transfer Planner", page_icon="🎬", layout="wide")
ss = st.session_state
NEW = "➕ Analyze a new reference video…"
ATTR_INFO = {
    "pacing": ("Pacing", "Cut your clip on the reference's shot-length rhythm (visible jump cuts; audio untouched)."),
    "audio_rhythm": ("Audio rhythm", "Snap those cuts to the beat of your clip's audio."),
    "transitions": ("Transitions", "Mirror the reference's transition mix on the new cuts (hard cut and zoom punch-in render; others fall back)."),
    "color": ("Color", "Match brightness, contrast, saturation and warmth to the reference."),
    "text": ("Text / captions", "Burn captions in the reference's font, size, color, position and box (needs a transcript)."),
}


def _find_video_for_ref(ref: str, vids: dict[str, Path]) -> Optional[Path]:
    clean = ref.replace(" (demo)", "").replace(" [demo]", "").strip()
    candidates = [ref, clean, f"{ref}.mp4", f"{clean}.mp4", f"{clean}.mp4 (demo)"]
    for c in candidates:
        if c in vids:
            return vids[c]
    return None


def chart(c):
    try:
        st.altair_chart(c, width="stretch")
    except TypeError:                                   # older Streamlit
        st.altair_chart(c, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def enable(attr: str) -> None:
    ss[f"sel_{attr}"] = True


def save_upload(upload, subdir: str) -> Path:
    d = results_dir() / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / Path(upload.name).name
    if not p.exists() or p.stat().st_size != upload.size:
        p.write_bytes(upload.getbuffer())
    return p


# ------------------------------------------------------------------------------------------------ state
if "_pending_ref" in ss:                                # set by a finished analysis, applied BEFORE the widget exists
    ss["ref_choice"] = ss.pop("_pending_ref")
ss.setdefault("result", None)
for _a in ATTRIBUTES:
    ss.setdefault(f"sel_{_a}", True)

cap = UsageCap()
probe = CappedGeminiClient(usage_cap=cap)
has_key = bool(probe.api_key)
rem = cap.remaining()
gemini_ok = probe.available

with st.sidebar:
    st.title("🎬 Style Transfer Planner")
    st.caption("Borrow the editing style of a reference video and apply it to your own clip.")
    st.markdown("**How it works**\n1. Pick a reference whose style you like (its *style card* is measured by code + Gemini).\n"
                "2. Pick your clip.\n3. Choose which attributes to borrow; the app warns about combinations that may feel off.\n"
                "4. Apply, then review the result, the plan and the fallback log.")
    st.metric("Gemini requests left today", f"{rem} / {cap.daily_limit}")
    if not has_key:
        st.warning("No `GEMINI_API_KEY` found (put it in `.env`). You can still apply styles from existing cards; "
                   "analysing new references and auto-captions need Gemini.")
    elif rem <= 0:
        st.error(f"Gemini requests left today: 0. Daily limit reached ({cap.daily_limit}/{cap.daily_limit}). "
                 "**Disabled:** Analysing new reference videos and auto-captions are disabled until UTC midnight. "
                 "**Still works:** Applying styles from existing or demo cards, color match, pacing cuts, beat sync, and captions from uploaded or bundled transcripts.")
    else:
        st.success(f"Gemini key found · model `{probe.model}`")
    st.caption("Analysing a new reference uses ~7-10 Gemini requests (cached afterwards). Auto-captions use 1. "
               "Everything else runs locally.")

tab_in, tab_card, tab_borrow, tab_result = st.tabs(["① Inputs", "② Style card", "③ Choose what to borrow", "④ Result"])
cards = list_cards()
videos = list_videos()

# ------------------------------------------------------------------------------------------- ① inputs
with tab_in:
    left, right = st.columns(2)
    with left:
        st.subheader("Reference · the style to copy")
        options = list(cards) + [NEW]
        ref_choice = st.selectbox("Reference style card", options, key="ref_choice",
                                  help="Cards are saved in outputs/ by analyze_cli.py or by this app.")
        if ref_choice == NEW:
            st.caption("Pick a video to analyse, or upload one (max 60 s).")
            src_name = st.selectbox("Video to analyse", list(videos) or ["(no videos in data/examples)"], key="an_src")
            up = st.file_uploader("…or upload a reference", type=[e[1:] for e in VIDEO_EXTS], key="an_upload")
            src = save_upload(up, "uploads") if up else videos.get(src_name)
            if st.button("Analyze with Gemini", type="primary", key="an_btn", disabled=not (gemini_ok and src)):
                with st.status("Analysing the reference…", expanded=True) as status:
                    bar = st.progress(0.0, text="starting")
                    try:
                        card_new = analyze_video(src, client=CappedGeminiClient(usage_cap=cap),
                                                 progress=lambda stage, f: bar.progress(min(f, 1.0), text=stage))
                        out = cards_dir() / f"{safe_stem(src.name)}.stylecard.json"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(card_new.model_dump_json(indent=2), encoding="utf-8")
                        ss["_pending_ref"] = safe_stem(src.name)
                        status.update(label="Done", state="complete")
                    except (AnalysisError, ValueError, RuntimeError) as e:
                        status.update(label="Analysis failed", state="error")
                        st.error(str(e))
                if "_pending_ref" in ss:
                    st.rerun()
            if not gemini_ok:
                st.info("Analysis needs a Gemini key. Add `GEMINI_API_KEY` to `.env` and restart.")
        else:
            rv = _find_video_for_ref(ref_choice, videos)
            if rv:
                st.video(str(rv))
    with right:
        st.subheader("Target · your clip")
        t_choice = st.selectbox("Pick a clip", ["(none)"] + list(videos), key="tgt_choice")
        t_up = st.file_uploader("…or upload your clip (max 60 s)", type=[e[1:] for e in VIDEO_EXTS], key="tgt_upload")
        target_path = save_upload(t_up, "uploads") if t_up else (videos.get(t_choice) if t_choice != "(none)" else None)
        if target_path:
            st.video(str(target_path))

card = load_card(cards[ref_choice]) if ref_choice in cards else None

# --------------------------------------------------------------------------------------- ② style card
with tab_card:
    if card is None:
        st.info("Choose or analyse a reference in the first tab.")
    else:
        m = card.meta
        if m.gemini_mode == "mock":
            st.error("This card was made with the MOCK backend: its AI-judged fields are fake placeholders.")
        for d in m.degraded:
            st.warning(f"Not available for this card: **{d.stage}**: {d.reason}")
        for col, (label, value, hint) in zip(st.columns(5), card_metrics(card)):
            col.metric(label, value, help=hint)
        chart(timeline_chart(card))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Color**")
            st.markdown(palette_html(card.overall.color.palette), unsafe_allow_html=True)
            col = card.overall.color
            st.caption(f"Grade: **{col.grade}** · contrast {col.contrast} · saturation {col.saturation} · {col.warmth}")
            if card.overall.transitions:
                st.markdown("**Transitions**")
                st.bar_chart(pd.DataFrame({"count": {t.type: t.count for t in card.overall.transitions}}))
        with c2:
            st.markdown("**Text style**")
            tx = card.overall.text
            if tx.present and tx.font_style:
                choice = resolve_font(tx.font_style)
                sample = "REVERSE ENGINEERS" if tx.case == "uppercase" else "reverse engineers"
                st.markdown(font_preview_html(choice.name, choice.weight, sample, tx.color,
                                              tx.outline_or_shadow == "background-box"), unsafe_allow_html=True)
                st.caption(f"Closest open-license font: **{choice.name} {choice.weight}** ({choice.status}, score {choice.score:.2f}). "
                           f"{choice.explanation}")
                if tx.description:
                    st.caption(f"Treatment: {tx.description}")
                if tx.variety == "varied" or tx.position == "mixed":
                    st.warning("This reference uses varied, kinetic typography. Only its dominant look can be described.")
            else:
                st.caption("No on-screen text detected (or unavailable)." if tx.present is False else "Text analysis unavailable.")
        st.markdown("**Attributes, confidence and how each was measured**")
        st.caption("Confidence values are heuristic or self-reported by the model, not calibrated probabilities.")
        table(pd.DataFrame(attribute_rows(card)))
        with st.expander("Scene-by-scene"):
            table(scene_frame(card))
        with st.expander("Notes and raw JSON"):
            for n in m.notes:
                st.write("• " + n)
            st.download_button("Download the style card (JSON)", card.model_dump_json(indent=2),
                               file_name=f"{ref_choice}.stylecard.json", mime="application/json")
            st.json(json.loads(card.model_dump_json()), expanded=False)

# ------------------------------------------------------------------------------- ③ choose what to borrow
with tab_borrow:
    if card is None:
        st.info("Choose or analyse a reference in the first tab.")
    else:
        st.markdown("Attributes are grouped into **bundles** because a style's feel comes from the combination. "
                    "Pick any mix; the app warns when a choice may look off on its own.")
        for col, (bundle, attrs) in zip(st.columns(3), BUNDLES.items()):
            with col:
                st.markdown(f"**{bundle}**")
                for a in attrs:
                    st.toggle(ATTR_INFO[a][0], key=f"sel_{a}", help=ATTR_INFO[a][1])
        selected = [a for a in ATTRIBUTES if ss.get(f"sel_{a}")]
        for w in check_selection(selected, card):
            st.warning(f"**{ATTR_INFO[w.attribute][0]}**: {w.message}")
            if w.suggest_add and not ss.get(f"sel_{w.suggest_add}"):
                st.button(f"Add {ATTR_INFO[w.suggest_add][0]}", key=f"add_{w.attribute}_{w.suggest_add}",
                          on_click=enable, args=(w.suggest_add,))
        s1, s2 = st.columns(2)
        strength = s1.slider("Color match strength", 0.0, 1.0, 0.8, 0.05, help="0 = no change, 1 = full match to the reference")
        zoom = s2.slider("Jump-cut framing change", 1.05, 1.40, float(DEFAULT_ZOOM), 0.01,
                         help="How much the framing steps in at each cut (1.15 = 15% punch-in)")
        words = None
        with st.expander("Advanced (captions, debugging)"):
            a1, a2 = st.columns(2)
            offset = a1.number_input("Caption timing shift (s)", -2.0, 2.0, 0.0, 0.05, help="Negative = earlier")
            wpc = a2.number_input("Words per caption (0 = automatic)", 0, 6, 0)
            bundled_tr = list_transcripts()
            tr_options = ["(none)"] + list(bundled_tr.keys())
            default_tr_idx = 0
            if target_path:
                target_stem = safe_stem(target_path.name)
                for idx, name in enumerate(tr_options):
                    if target_stem in name:
                        default_tr_idx = idx
                        break
            chosen_demo_tr = st.selectbox("Bundled demo transcript (optional)", tr_options, index=default_tr_idx, key="demo_tr_choice")
            tup = st.file_uploader("…or upload transcript JSON", type=["json"], key="tr_upload")
            if tup:
                try:
                    raw = json.loads(tup.getvalue().decode("utf-8"))
                    items = raw["words"] if isinstance(raw, dict) else raw
                    words = clean_words([TranscriptWord(**i) for i in items], 3600.0)
                    st.caption(f"Loaded {len(words)} words from upload.")
                except Exception as e:
                    st.error(f"Could not read that transcript: {e}")
            elif chosen_demo_tr != "(none)":
                try:
                    raw = json.loads(bundled_tr[chosen_demo_tr].read_text(encoding="utf-8"))
                    items = raw["words"] if isinstance(raw, dict) else raw
                    words = clean_words([TranscriptWord(**i) for i in items], 3600.0)
                    st.caption(f"Loaded {len(words)} words from demo transcript: `{chosen_demo_tr}`.")
                except Exception as e:
                    st.error(f"Could not read demo transcript: {e}")

            auto_sync = st.checkbox("Auto-correct caption timing (experimental)",
                                    help="Measures the transcript against the audio and shifts ALL captions by the estimated lag, "
                                         "only when the estimate is confident. It cannot fix a short run of late words.")
            debug = st.checkbox("Also create a debug video (red flash at every cut) and a before/after strip per cut")
        if "text" in selected and words is None:
            if rem <= 0 and has_key:
                st.info("Daily Gemini limit reached: auto-captions are disabled. Select a bundled demo transcript or upload one.")
            elif gemini_ok:
                st.info("Captions need a transcript: the app will transcribe with Gemini (1 request).")
            else:
                st.info("No Gemini key and no transcript: the text attribute will be a plan only, not rendered.")
        if target_path is None:
            st.info("Pick a target clip in the first tab.")
        if st.button("Apply style", type="primary", key="apply_btn", disabled=not (target_path and selected)):
            out_dir = results_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{safe_stem(target_path.name)}__in__{ref_choice}_style.mp4"
            with st.status("Applying the style…", expanded=True) as status:
                st.write("Measuring the clip, planning cuts, matching color and rendering (about a minute)…")
                try:
                    client = None
                    if "text" in selected and words is None:
                        c = CappedGeminiClient(usage_cap=cap)
                        client = c if c.available else None
                    res = apply_style(card, target_path, selected, str(out), strength=strength, zoom=zoom, client=client,
                                      transcript=words, caption_offset_s=float(offset), words_per_caption=int(wpc) or None,
                                      auto_sync=bool(auto_sync))
                    dbg_video = dbg_sheet = None
                    if debug and res.output_path and res.plan.cut_times_s:
                        dbg_video = annotate_cuts(res.output_path, res.plan.cut_times_s, res.plan.segment_zooms,
                                                  out.with_name(out.stem + "_debug.mp4"))
                        dbg_sheet = cut_sheet(res.output_path, res.plan.cut_times_s, res.plan.segment_zooms,
                                              out.with_name(out.stem + "_cuts.png"))
                    ss["result"] = {"res": res, "ref": ref_choice, "target": target_path, "debug_video": dbg_video,
                                    "debug_sheet": dbg_sheet}
                    status.update(label="Done. Open the Result tab ④", state="complete")
                except (AnalysisError, ValueError, RuntimeError) as e:
                    status.update(label="Failed", state="error")
                    st.error(str(e))
            if ss.get("result"):
                st.success("Finished. Open the **④ Result** tab to review it.")

# ------------------------------------------------------------------------------------------- ④ result
with tab_result:
    r = ss.get("result")
    if not r:
        st.info("Nothing yet. Apply a style in the third tab.")
    else:
        res, plan = r["res"], r["res"].plan
        st.caption(f"Reference **{r['ref']}** → target **{Path(r['target']).name}** · {res.seconds}s")
        if res.output_path:
            b, a = st.columns(2)
            b.markdown("**Before**")
            b.video(str(r["target"]))
            a.markdown("**After**")
            a.video(res.output_path)
        else:
            st.warning("Nothing was rendered for this selection (for example, text without a transcript). The plan below says what would be done.")
        if res.match and res.match.color_distance_before is not None:
            mt = res.match
            k1, k2, k3 = st.columns(3)
            k1.metric("Color distance to reference", mt.color_distance_after,
                      delta=round(mt.color_distance_after - mt.color_distance_before, 3), delta_color="inverse",
                      help="Lower is closer to the reference's measured color statistics")
            k2.metric("Shots", mt.scheduled_cuts + 1, help=f"Reference averages {mt.avg_shot_len_reference_s}s per shot")
            k3.metric("Avg shot (target)", f"{mt.avg_shot_len_target_s}s")
            with st.expander("Color statistics: reference vs before vs after"):
                table(pd.DataFrame({"reference": mt.color_reference, "before": mt.color_before, "after": mt.color_after}).reset_index(names="metric"))
        for w in plan.warnings:
            st.warning(f"**{ATTR_INFO[w.attribute][0]}**: {w.message}")
        st.markdown("#### What was done")
        table(plan_frame(plan))
        if plan.fallbacks:
            st.markdown("#### Fallback log: what could not be copied exactly, and what was used instead")
            table(fallback_frame(plan))
        with st.expander("Plan details (per attribute)"):
            for it in plan.items:
                st.markdown(f"**{it.attribute}** · {it.status} · {it.action}")
                if it.note:
                    st.caption(it.note)
                st.json(it.detail, expanded=False)
        d1, d2, d3 = st.columns(3)
        if res.output_path:
            d1.download_button("Download the video", Path(res.output_path).read_bytes(),
                               file_name=Path(res.output_path).name, mime="video/mp4")
        d2.download_button("Download the plan (JSON)", res.model_dump_json(indent=2), file_name="plan.json", mime="application/json")
        if res.transcript:
            d3.download_button("Download the transcript (JSON)", json.dumps([w.model_dump() for w in res.transcript], indent=1),
                               file_name="transcript.json", mime="application/json")
        if r.get("debug_video"):
            with st.expander("Debug: where the cuts are", expanded=True):
                st.video(str(r["debug_video"]))
                if r.get("debug_sheet"):
                    st.image(str(r["debug_sheet"]), caption="Frame just before | just after each cut")
