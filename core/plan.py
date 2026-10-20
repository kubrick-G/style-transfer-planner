"""Stage 2 data models: the editing plan, the fallback log, warnings and the style-match report."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .transcript import TranscriptWord


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FallbackEntry(_M):
    attribute: str
    reference_value: str
    chosen: str
    status: Literal["exact", "close", "fallback"]
    score: Optional[float] = None
    explanation: str


class PlanItem(_M):
    attribute: str
    action: str
    status: Literal["applied", "planned", "fallback", "skipped"]
    detail: dict[str, Any] = Field(default_factory=dict)
    note: Optional[str] = None


class StyleWarning(_M):
    attribute: str
    message: str
    suggest_add: Optional[str] = None


class StyleMatch(_M):
    color_reference: dict[str, float] = Field(default_factory=dict)
    color_before: dict[str, float] = Field(default_factory=dict)
    color_after: dict[str, float] = Field(default_factory=dict)
    color_distance_before: Optional[float] = None
    color_distance_after: Optional[float] = None
    avg_shot_len_reference_s: Optional[float] = None
    avg_shot_len_target_s: Optional[float] = None
    scheduled_cuts: int = 0
    detected_scenes_after: Optional[int] = None


class EditingPlan(_M):
    reference_sha: str
    target_name: str
    selected: list[str]
    items: list[PlanItem] = Field(default_factory=list)
    fallbacks: list[FallbackEntry] = Field(default_factory=list)
    warnings: list[StyleWarning] = Field(default_factory=list)
    cut_times_s: list[float] = Field(default_factory=list)
    segment_zooms: list[float] = Field(default_factory=list)


class ApplyResult(_M):
    plan: EditingPlan
    output_path: Optional[str] = None
    match: Optional[StyleMatch] = None
    transcript: Optional[list[TranscriptWord]] = None   # words used for captions (unshifted)
    seconds: float = 0.0
