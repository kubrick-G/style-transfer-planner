"""Font fallback resolver.

The pipeline never identifies a font by NAME (that is unreliable from compressed video). It extracts ATTRIBUTES
(category, weight, width, geometry, mood). This module maps those attributes onto a catalog of open-license Google
Fonts and reports HOW good the match is, so the user always gets a result plus a transparent explanation:

  close    - a catalog font of the right category matched well
  fallback - nothing matched well; a safe default is used and the explanation says why
  exact    - reserved: would require identifying the original font, which we deliberately do not attempt
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schema import FontStyle

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "fonts_catalog.json"
WEIGHT_ORDER = ["light", "regular", "medium", "bold", "black"]
W_CATEGORY, W_GEOMETRY, W_WIDTH, W_WEIGHT, W_MOOD = 0.40, 0.20, 0.15, 0.15, 0.10
CLOSE_THRESHOLD = 0.60
DEFAULT_FONT, DEFAULT_WEIGHT = "Montserrat", "bold"

_cache: Optional[list[dict]] = None


@dataclass
class FontChoice:
    name: str
    weight: Optional[str]
    score: float
    status: str
    explanation: str
    category: Optional[str] = None

    @property
    def url(self) -> str:
        return "https://fonts.google.com/specimen/" + self.name.replace(" ", "+")


def load_catalog(path: Optional[Path] = None) -> list[dict]:
    global _cache
    if path is None and _cache is not None:
        return _cache
    data = json.loads(Path(path or CATALOG_PATH).read_text())["fonts"]
    if path is None:
        _cache = data
    return data


def _weight_score(ref: str, available: list[str]) -> float:
    if ref in available:
        return 1.0
    if ref in WEIGHT_ORDER:
        i = WEIGHT_ORDER.index(ref)
        near = {WEIGHT_ORDER[j] for j in (i - 1, i + 1) if 0 <= j < len(WEIGHT_ORDER)}
        if near & set(available):
            return 0.5
    return 0.0


def _nearest_weight(ref: Optional[str], available: list[str]) -> str:
    if ref in available:
        return ref
    if ref in WEIGHT_ORDER:
        i = WEIGHT_ORDER.index(ref)
        return min(available, key=lambda w: abs(WEIGHT_ORDER.index(w) - i))
    return "bold" if "bold" in available else available[0]


def score_font(ref: FontStyle, font: dict) -> float:
    """Weighted attribute match in 0..1, renormalised over the attributes the reference actually provides."""
    parts, total = 0.0, 0.0
    if ref.category:
        total += W_CATEGORY
        parts += W_CATEGORY * (ref.category == font["category"])
    if ref.geometry:
        total += W_GEOMETRY
        parts += W_GEOMETRY * (ref.geometry == font["geometry"])
    if ref.width:
        total += W_WIDTH
        parts += W_WIDTH * (ref.width == font["width"])
    if ref.weight:
        total += W_WEIGHT
        parts += W_WEIGHT * _weight_score(ref.weight, font["weights"])
    if ref.mood:
        total += W_MOOD
        parts += W_MOOD * (len(set(ref.mood) & set(font["mood"])) / len(ref.mood))
    return parts / total if total else 0.0


def resolve_font(ref: Optional[FontStyle], catalog: Optional[list[dict]] = None) -> FontChoice:
    catalog = catalog if catalog is not None else load_catalog()
    if ref is None or not any([ref.category, ref.geometry, ref.width, ref.weight]):
        return FontChoice(DEFAULT_FONT, DEFAULT_WEIGHT, 0.0, "fallback",
                          "No font attributes were extracted from the reference, so a safe default is used.")
    best = max(catalog, key=lambda f: score_font(ref, f))
    s = round(score_font(ref, best), 3)
    desc = ", ".join(x for x in [ref.category, ref.weight, ref.width, ref.geometry] if x)
    if ref.category == best["category"] and s >= CLOSE_THRESHOLD:
        w = _nearest_weight(ref.weight, best["weights"])
        return FontChoice(best["name"], w, s, "close",
                          f"Reference text reads as [{desc}]. {best['name']} ({best['category']}, {best['geometry']}) "
                          f"is the closest open-license match (score {s:.2f}); this is an attribute match, "
                          f"not an identification of the original font.", best["category"])
    return FontChoice(DEFAULT_FONT, DEFAULT_WEIGHT, s, "fallback",
                      f"Reference text reads as [{desc}] but no catalog font matched well (best candidate "
                      f"{best['name']} scored {s:.2f}); using {DEFAULT_FONT} {DEFAULT_WEIGHT} as a safe default.")
