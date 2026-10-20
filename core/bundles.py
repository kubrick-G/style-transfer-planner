"""Attribute bundles and data-driven dependency warnings.

A style's feel lives in the COMBINATION of attributes, so choosing one without its partners can look wrong.
Rules are plain data (BUNDLES, RULES); they can use the reference card's measured `coupling` values.
"""
from __future__ import annotations

from typing import Optional

from .plan import StyleWarning
from .schema import StyleCard

ATTRIBUTES = ["pacing", "audio_rhythm", "transitions", "color", "text"]
BUNDLES = {
    "Rhythm": ["pacing", "audio_rhythm", "transitions"],
    "Look": ["color"],
    "Text": ["text"],
}


def _fast(card: StyleCard) -> bool:
    c = card.overall.pacing.cuts_per_min
    return c is not None and c >= 30


def _beat_driven(card: StyleCard) -> Optional[float]:
    r = card.coupling.cuts_on_beat_ratio
    return r if (r is not None and r >= 0.6) else None


# each rule: (attribute it is about, condition(selected, card) -> bool, message(card) -> str, attribute to suggest)
RULES = [
    ("transitions", lambda s, c: "transitions" in s and "pacing" not in s,
     lambda c: "Transitions are placed on the cuts that Pacing creates. Without Pacing there are no new cuts, "
               "so nothing will change.", "pacing"),
    ("audio_rhythm", lambda s, c: "audio_rhythm" in s and "pacing" not in s,
     lambda c: "Audio rhythm only moves where cuts land. Without Pacing there are no cuts to move.", "pacing"),
    ("pacing", lambda s, c: "pacing" in s and "audio_rhythm" not in s and _beat_driven(c) is not None,
     lambda c: f"In the reference, {round(_beat_driven(c) * 100)}% of cuts land on the beat. Without Audio rhythm the "
               f"new cuts will ignore the music and may feel off.", "audio_rhythm"),
    ("text", lambda s, c: "text" in s and "pacing" not in s and _fast(c),
     lambda c: f"The reference cuts fast ({c.overall.pacing.cuts_per_min:.0f} cuts/min) and its text energy is tuned "
               f"to that. Without Pacing, the text may feel out of step.", "pacing"),
    ("text", lambda s, c: "text" in s and (c.overall.text.variety == "varied" or c.overall.text.position == "mixed"),
     lambda c: "The reference uses varied, kinetic typography (style and position change throughout). Only its "
               "dominant treatment can be described; the full look will not be reproduced.", None),
    ("color", lambda s, c: "color" in s and c.overall.color.grade in (None, "other"),
     lambda c: "The reference's color grade could not be named clearly; the match relies on measured numbers only.",
     None),
]


def check_selection(selected: set[str] | list[str], card: StyleCard) -> list[StyleWarning]:
    sel = set(selected)
    unknown = sel - set(ATTRIBUTES)
    if unknown:
        raise ValueError(f"Unknown attributes: {sorted(unknown)}. Choose from {ATTRIBUTES}.")
    return [StyleWarning(attribute=a, message=msg(card), suggest_add=add)
            for a, cond, msg, add in RULES if cond(sel, card)]
