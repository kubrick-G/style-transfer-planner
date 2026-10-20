"""Pure helpers for the Streamlit app (no Streamlit imports): directories, card/plan tables, charts, previews."""
from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd

from .plan import EditingPlan
from .schema import StyleCard

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
WEIGHT_CSS = {"light": 300, "regular": 400, "medium": 500, "bold": 700, "black": 900}
STATUS_ICON = {"applied": "✅", "planned": "📝", "fallback": "⚠️", "skipped": "⏭️", "close": "🟢", "exact": "🟢"}
ENERGY_COLORS = {"low": "#4c78a8", "medium": "#f2b134", "high": "#e4572e"}


def cards_dir() -> Path:
    return Path(os.environ.get("STP_CARDS_DIR", ROOT / "outputs"))


def demo_cards_dir() -> Path:
    return Path(os.environ.get("STP_DEMO_CARDS_DIR", ROOT / "data" / "demo" / "cards"))


def videos_dir() -> Path:
    return Path(os.environ.get("STP_VIDEOS_DIR", ROOT / "data" / "examples"))


def demo_clips_dir() -> Path:
    return Path(os.environ.get("STP_DEMO_CLIPS_DIR", ROOT / "data" / "demo" / "clips"))


def demo_transcripts_dir() -> Path:
    return Path(os.environ.get("STP_DEMO_TRANSCRIPTS_DIR", ROOT / "data" / "demo" / "transcripts"))


def results_dir() -> Path:
    return Path(os.environ.get("STP_OUT_DIR", ROOT / "outputs"))


def list_cards(d: Optional[Path] = None) -> dict[str, Path]:
    if d is not None:
        p = Path(d)
        return {p.name[: -len(".stylecard.json")]: p for p in sorted(p.glob("*.stylecard.json"))} if p.exists() else {}

    u_dir, dm_dir = cards_dir(), demo_cards_dir()
    user_cards = {p.name[: -len(".stylecard.json")]: p for p in sorted(u_dir.glob("*.stylecard.json"))} if u_dir.exists() else {}
    demo_cards = {p.name[: -len(".stylecard.json")]: p for p in sorted(dm_dir.glob("*.stylecard.json"))} if dm_dir.exists() else {}

    res = dict(user_cards)
    for stem, path in demo_cards.items():
        if stem not in user_cards:
            res[f"{stem} (demo)"] = path
    return res


def list_videos(d: Optional[Path] = None) -> dict[str, Path]:
    if d is not None:
        p = Path(d)
        if not p.exists():
            return {}
        return {p.name: p for p in sorted(p.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS}

    u_dir, dm_dir = videos_dir(), demo_clips_dir()
    user_videos = {p.name: p for p in sorted(u_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS} if u_dir.exists() else {}
    demo_videos = {p.name: p for p in sorted(dm_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS} if dm_dir.exists() else {}

    res = dict(user_videos)
    for name, path in demo_videos.items():
        stem = Path(name).stem
        # user video overrides on name or stem clash
        if name not in user_videos and not any(Path(uv).stem == stem for uv in user_videos):
            res[f"{name} (demo)"] = path
    return res


def list_transcripts(d: Optional[Path] = None) -> dict[str, Path]:
    p = Path(d or demo_transcripts_dir())
    if not p.exists():
        return {}
    return {f.name: f for f in sorted(p.glob("*.json"))}


def load_card(path: Path) -> StyleCard:
    return StyleCard.model_validate_json(Path(path).read_text(encoding="utf-8"))


def safe_stem(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(name).stem)[:60].strip("_")
    return cleaned or "video"


# ------------------------------------------------------------------------------------------------ card views
def card_metrics(card: StyleCard) -> list[tuple[str, str, str]]:
    p, a = card.overall.pacing, card.overall.audio
    f = lambda v, nd=1, u="": "n/a" if v is None else f"{v:.{nd}f}{u}"
    return [
        ("Scenes", str(p.n_scenes), "Shots detected in the reference"),
        ("Avg shot", f(p.avg_shot_len_s, 2, " s"), "Average shot length"),
        ("Cuts / min", f(p.cuts_per_min), f"Pacing shape: {p.pacing_curve}"),
        ("BPM", f(a.bpm), "Tempo of the music (measured)"),
        ("Cuts on beat", "n/a" if a.beat_sync_score is None else f"{round(a.beat_sync_score * 100)}%",
         "Share of cuts landing within 80 ms of a beat"),
    ]


def attribute_rows(card: StyleCard) -> list[dict]:
    o, c = card.overall, card.confidence
    t, col = o.text, o.color
    trans = ", ".join(f"{x.type} ×{x.count}" for x in o.transitions) or "n/a"
    fs = t.font_style
    font = ", ".join(x for x in [fs.category, fs.weight, fs.width, fs.geometry] if x) if fs else "n/a"
    text_val = ("none detected" if t.present is False else "unknown" if t.present is None else
                f"{font} · {t.case or '?'} · {t.outline_or_shadow or '?'} · chunking {t.chunking or '?'} · variety {t.variety or '?'}")
    return [
        {"attribute": "Pacing", "value": f"{o.pacing.n_scenes} scenes, avg {o.pacing.avg_shot_len_s}s, {o.pacing.cuts_per_min} cuts/min",
         "confidence": c.get("pacing"), "measured by": "code"},
        {"attribute": "Audio rhythm", "value": f"{o.audio.bpm} BPM, {o.audio.beat_sync_score} beat sync, energy {o.audio.energy}",
         "confidence": c.get("audio"), "measured by": "code (genre: AI)"},
        {"attribute": "Color", "value": f"{col.grade} · contrast {col.contrast} · saturation {col.saturation} · {col.warmth}",
         "confidence": c.get("color"), "measured by": "code stats + AI grade name"},
        {"attribute": "Transitions", "value": trans, "confidence": c.get("transitions"), "measured by": "code (single-frame test) + AI type"},
        {"attribute": "Text", "value": text_val, "confidence": c.get("text"), "measured by": "AI (caption rate: code)"},
        {"attribute": "Camera / shots", "value": f"{o.camera.dominant_movement} · {', '.join(o.camera.shot_types) or 'n/a'}",
         "confidence": c.get("camera"), "measured by": "code flow hint + AI"},
        {"attribute": "Layout / mood", "value": f"{o.layout} · {o.mood}", "confidence": c.get("mood"), "measured by": "AI"},
    ]


def scene_frame(card: StyleCard) -> pd.DataFrame:
    return pd.DataFrame([{
        "scene": s.index, "start": s.start_s, "end": s.end_s, "role": s.role, "energy": s.energy or "n/a",
        "shot": s.shot_type, "camera": s.camera_motion, "transition in": s.transition_in, "text": s.has_text,
        "on beat": s.cut_on_beat, "motion": s.motion_magnitude, "colors": " ".join(s.dominant_colors),
    } for s in card.scenes])


def timeline_chart(card: StyleCard) -> alt.VConcatChart:
    """Scenes coloured by energy, beat grid as faint lines, and the audio energy curve underneath."""
    dur = card.video.duration_s
    sc = scene_frame(card).assign(lane="scenes")
    x = alt.X("start:Q", title=None, scale=alt.Scale(domain=[0, dur]))
    scenes = alt.Chart(sc).mark_rect(stroke="#111", strokeWidth=1).encode(
        x=x, x2="end:Q", y=alt.Y("lane:N", title=None, axis=None),
        color=alt.Color("energy:N", scale=alt.Scale(domain=list(ENERGY_COLORS), range=list(ENERGY_COLORS.values())),
                        legend=alt.Legend(title="scene energy", orient="right")),
        tooltip=["scene", "start", "end", "role", "energy", "shot", "transition in"])
    layers = [scenes]
    beats = card.overall.audio.beat_times
    if beats:
        layers.append(alt.Chart(pd.DataFrame({"t": beats})).mark_rule(opacity=0.18, color="white")
                      .encode(x=alt.X("t:Q", scale=alt.Scale(domain=[0, dur]))))
    top = alt.layer(*layers).properties(height=70, title="Scenes (colour = energy) and beat grid")
    curve = card.overall.audio.energy_curve
    hz = card.overall.audio.energy_curve_hz or 4.0
    ef = pd.DataFrame({"t": [i / hz for i in range(len(curve))], "energy": curve}) if curve else pd.DataFrame({"t": [0.0], "energy": [0.0]})
    bottom = alt.Chart(ef).mark_area(opacity=0.6, color="#4c78a8").encode(
        x=alt.X("t:Q", title="seconds", scale=alt.Scale(domain=[0, dur])), y=alt.Y("energy:Q", title="audio", axis=None)
    ).properties(height=60)
    return alt.vconcat(top, bottom).resolve_scale(color="independent")


def palette_html(colors: list[str]) -> str:
    cells = "".join(
        f'<div style="flex:1;min-width:70px;text-align:center"><div style="height:46px;background:{html.escape(c)};'
        f'border-radius:6px;border:1px solid #444"></div><div style="font:12px monospace;margin-top:4px">{html.escape(c)}</div></div>'
        for c in colors)
    return f'<div style="display:flex;gap:8px">{cells}</div>'


def font_preview_html(font_name: str, weight: Optional[str], text: str, color: Optional[str], boxed: bool) -> str:
    w = WEIGHT_CSS.get(weight or "bold", 700)
    fam = html.escape(font_name)
    url = font_name.replace(" ", "+")
    col = color if color and color.startswith("#") and len(color) == 7 else "#ffffff"
    box = "background:rgba(0,0,0,.68);padding:.3em .65em;border-radius:.4em;" if boxed else "text-shadow:0 0 4px #000;"
    return (f"<style>@import url('https://fonts.googleapis.com/css2?family={url}:wght@{w}&display=swap');</style>"
            f"<div style='padding:22px;background:#2b2b2b;border-radius:8px;text-align:center'>"
            f"<span style=\"font-family:'{fam}',monospace;font-weight:{w};color:{col};font-size:28px;{box}\">{html.escape(text)}</span></div>")


# ------------------------------------------------------------------------------------------------ plan views
def _short(detail: dict) -> str:
    keep = ["shots", "avg_shot_len_s", "target_bpm", "snapped", "of", "strength", "font", "font_weight", "captions",
            "words_per_caption", "transcript_source", "case", "reference_layout"]
    return " · ".join(f"{k}={detail[k]}" for k in keep if k in detail and detail[k] is not None)


def plan_frame(plan: EditingPlan) -> pd.DataFrame:
    return pd.DataFrame([{"": STATUS_ICON.get(i.status, ""), "attribute": i.attribute, "status": i.status,
                          "what happened": i.action, "details": _short(i.detail), "note": i.note or ""} for i in plan.items])


def fallback_frame(plan: EditingPlan) -> pd.DataFrame:
    return pd.DataFrame([{"": STATUS_ICON.get(f.status, ""), "attribute": f.attribute, "status": f.status,
                          "reference": f.reference_value, "used instead": f.chosen,
                          "score": f.score, "why": f.explanation} for f in plan.fallbacks])
