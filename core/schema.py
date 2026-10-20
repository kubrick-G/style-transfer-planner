"""Pydantic models: the Style Card, plus the response models Gemini must fill in.

Design notes
- Almost everything is Optional: a failed/unavailable analysis stage yields null + confidence 0
  instead of crashing the whole pipeline.
- Subjective labels come from fixed vocabularies; `normalize_choice` snaps near-miss model output
  ("Whip-pan", "cross dissolve") onto the vocabulary so a slightly-off label doesn't fail validation.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.2"

# ---------------------------------------------------------------- vocabularies
TRANSITION_VOCAB = ["hard cut", "crossfade", "whip pan", "zoom punch-in", "glitch",
                    "slide/push", "flash", "match cut", "jump cut", "other"]
TEXT_ANIMATION_VOCAB = ["none", "fade", "pop/scale", "typewriter", "word-by-word",
                        "slide-in", "bounce", "glitch", "karaoke-highlight"]
FONT_CATEGORY_VOCAB = ["sans", "serif", "script", "display", "handwritten", "mono"]
FONT_WEIGHT_VOCAB = ["light", "regular", "medium", "bold", "black"]
FONT_WIDTH_VOCAB = ["condensed", "normal", "wide"]
FONT_GEOMETRY_VOCAB = ["geometric", "humanist", "grotesque", "slab"]
GRADE_VOCAB = ["teal-orange", "faded/matte", "high-contrast", "black-and-white", "natural", "other"]
SHOT_TYPE_VOCAB = ["close-up", "medium", "wide", "over-the-shoulder", "screen-recording", "motion-graphic", "other"]
CAMERA_MOVE_VOCAB = ["static", "pan-left", "pan-right", "tilt-up", "tilt-down",
                     "push-in", "pull-out", "handheld", "other"]
TEXT_POSITION_VOCAB = ["top", "upper-third", "center", "lower-third", "bottom", "mixed"]
TEXT_CASE_VOCAB = ["uppercase", "lowercase", "title", "sentence", "mixed"]
OUTLINE_VOCAB = ["none", "outline", "shadow", "outline+shadow", "background-box"]
ENERGY_VOCAB = ["low", "medium", "high"]
LAYOUT_VOCAB = ["full-frame", "split-screen", "picture-in-picture", "other"]
CHUNKING_VOCAB = ["single-word", "short-chunks", "full-lines", "mixed"]
VARIETY_VOCAB = ["consistent", "varied"]

_ALIASES = {
    "whip": "whip pan", "whippan": "whip pan", "swishpan": "whip pan", "swish": "whip pan",
    "dissolve": "crossfade", "crossdissolve": "crossfade", "fade": "crossfade",
    "cut": "hard cut", "straightcut": "hard cut", "none": "hard cut",
    "zoom": "zoom punch-in", "punchin": "zoom punch-in", "zoomin": "zoom punch-in",
    "push": "slide/push", "slide": "slide/push", "wipe": "slide/push",
    "flashframe": "flash", "whiteflash": "flash",
    "pop": "pop/scale", "scale": "pop/scale", "popin": "pop/scale",
    "wordbyword": "word-by-word", "perword": "word-by-word",
    "slidein": "slide-in", "karaoke": "karaoke-highlight", "highlight": "karaoke-highlight",
    "sansserif": "sans", "monospace": "mono", "handwriting": "handwritten", "bw": "black-and-white",
    "blackwhite": "black-and-white", "monochrome": "black-and-white", "matte": "faded/matte",
    "faded": "faded/matte", "tealandorange": "teal-orange", "tealorange": "teal-orange",
    "closeup": "close-up", "medium": "medium", "mediumshot": "medium", "wideshot": "wide",
    "handheldshake": "handheld", "shake": "handheld", "pushin": "push-in", "pullout": "pull-out",
    "dollyin": "push-in", "dollyout": "pull-out",
    "bold": "bold", "semibold": "bold", "extrabold": "black", "heavy": "black", "thin": "light",
    "normalwidth": "normal", "regularwidth": "normal",
}


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def normalize_choice(value, vocab: list[str], default: Optional[str] = None) -> Optional[str]:
    """Snap a free-text label onto `vocab`. Returns `default` if nothing matches."""
    if not isinstance(value, str) or not value.strip():
        return default
    c = _canon(value)
    canon_map = {_canon(v): v for v in vocab}
    if c in canon_map:
        return canon_map[c]
    al = _ALIASES.get(c)
    if al in vocab:
        return al
    # phrase containing a vocab term or a known alias: the longest match wins
    best, best_len = None, 0
    for cv, v in canon_map.items():
        if len(cv) >= 4 and (cv in c or c in cv) and len(cv) > best_len:
            best, best_len = v, len(cv)
    for key, mapped in _ALIASES.items():
        if mapped in vocab and len(key) >= 4 and key in c and len(key) > best_len:
            best, best_len = mapped, len(key)
    return best if best is not None else default


def normalize_hex(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    m = re.search(r"#?([0-9a-fA-F]{6})\b", value)
    return f"#{m.group(1).lower()}" if m else None


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ------------------------------------------------------------------ Style Card
class VideoInfo(_M):
    duration_s: float
    fps: float
    resolution: tuple[int, int]
    sha256: str


class Pacing(_M):
    avg_shot_len_s: Optional[float] = None
    cuts_per_min: Optional[float] = None
    pacing_curve: Optional[str] = None
    n_scenes: int = 1


class Audio(_M):
    has_audio: bool = False
    bpm: Optional[float] = None
    beat_sync_score: Optional[float] = None
    genre_guess: Optional[str] = None
    energy: Optional[str] = None
    beat_times: list[float] = Field(default_factory=list)
    energy_curve: list[float] = Field(default_factory=list)  # normalised 0..1, 4 samples/second
    energy_curve_hz: float = 4.0


class Color(_M):
    palette: list[str] = Field(default_factory=list)
    grade: Optional[str] = None
    contrast: Optional[str] = None
    saturation: Optional[str] = None
    warmth: Optional[str] = None
    grain: Optional[bool] = None
    vignette: Optional[bool] = None
    stats: dict[str, float] = Field(default_factory=dict)


class FontStyle(_M):
    category: Optional[str] = None
    weight: Optional[str] = None
    width: Optional[str] = None
    geometry: Optional[str] = None
    mood: list[str] = Field(default_factory=list)


class TextStyle(_M):
    present: Optional[bool] = None
    font_style: Optional[FontStyle] = None
    case: Optional[str] = None
    position: Optional[str] = None
    color: Optional[str] = None
    outline_or_shadow: Optional[str] = None
    animation: Optional[str] = None
    chunking: Optional[str] = None                 # how much text is on screen at once
    variety: Optional[str] = None                  # "consistent" or "varied" (kinetic typography: style changes a lot)
    description: Optional[str] = None              # one-sentence overall text treatment
    changes_per_s: Optional[float] = None          # MEASURED caption change events per second (experimental)
    region: list[float] = Field(default_factory=list)  # normalised [x0,y0,x1,y1] of the caption area


class TransitionCount(_M):
    type: str
    count: int


class Camera(_M):
    dominant_movement: Optional[str] = None
    shot_types: list[str] = Field(default_factory=list)


class Overall(_M):
    pacing: Pacing = Field(default_factory=Pacing)
    audio: Audio = Field(default_factory=Audio)
    color: Color = Field(default_factory=Color)
    text: TextStyle = Field(default_factory=TextStyle)
    transitions: list[TransitionCount] = Field(default_factory=list)
    camera: Camera = Field(default_factory=Camera)
    mood: Optional[str] = None
    layout: Optional[str] = None
    structure_notes: Optional[str] = None


class Scene(_M):
    index: int
    start_s: float
    end_s: float
    role: str
    energy: Optional[str] = None
    energy_score: Optional[float] = None
    shot_type: Optional[str] = None
    camera_motion: Optional[str] = None
    motion_magnitude: Optional[float] = None
    transition_in: Optional[str] = None
    transition_out: Optional[str] = None
    dominant_colors: list[str] = Field(default_factory=list)
    has_text: Optional[bool] = None
    cut_on_beat: Optional[bool] = None


class Coupling(_M):
    cuts_on_beat_ratio: Optional[float] = None
    text_energy_matches_pacing: Optional[bool] = None
    transitions_on_music_peaks_ratio: Optional[float] = None
    grade_consistent_across_scenes: Optional[bool] = None
    notes: list[str] = Field(default_factory=list)


class GeminiCallLog(_M):
    name: str
    ok: bool
    attempts: int = 1
    seconds: float = 0.0
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None


class Degraded(_M):
    stage: str
    reason: str


class Meta(_M):
    schema_version: str = SCHEMA_VERSION
    gemini_mode: str = "unavailable"  # live | mock | unavailable
    gemini_model: Optional[str] = None
    analysis_seconds: float = 0.0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    gemini_calls: list[GeminiCallLog] = Field(default_factory=list)
    degraded: list[Degraded] = Field(default_factory=list)
    cache_hit: bool = False
    notes: list[str] = Field(default_factory=list)


class StyleCard(_M):
    video: VideoInfo
    overall: Overall
    scenes: list[Scene]
    coupling: Coupling
    confidence: dict[str, float] = Field(default_factory=dict)
    provenance: dict[str, list[str]] = Field(default_factory=dict)
    meta: Meta = Field(default_factory=Meta)


# ------------------------------------------- Gemini response models (LLM output)
class GradeResponse(_M):
    grade: str
    grain: bool = False
    confidence: float = 0.7


class SceneLabel(_M):
    index: int
    shot_type: str
    camera_motion: str


class SceneLabelsResponse(_M):
    scenes: list[SceneLabel]
    confidence: float = 0.7


class TransitionLabel(_M):
    index: int
    type: str


class TransitionsResponse(_M):
    transitions: list[TransitionLabel]
    confidence: float = 0.7


class TextFrameFlag(_M):
    index: int
    has_text: bool
    bbox: list[float] = Field(default_factory=list)  # normalised [x0, y0, x1, y1]


class TextPresenceResponse(_M):
    frames: list[TextFrameFlag]


class TextStyleResponse(_M):
    font_category: str
    weight: str
    width: str
    geometry: str
    mood: list[str] = Field(default_factory=list)
    case: str
    position: str
    color: str
    outline_or_shadow: str
    animation: str
    chunking: str = ""
    variety: str = ""
    description: str = ""
    confidence: float = 0.6


class VideoOverviewResponse(_M):
    mood: str
    genre_guess: str
    layout: str = ""
    structure_ok: bool = True
    structure_notes: str = ""
    confidence: float = 0.7
