"""Locate a usable font FILE for a resolved font name, with a transparent fallback chain:

  local (bundled in data/fonts or downloaded earlier)  ->  download from Google Fonts (GitHub)  ->
  a system font of the same category  ->  Pillow's built-in default

Every step returns a LoadedFont describing what was actually used so the caller can log the substitution.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from PIL import ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent / "data" / "fonts"
GITHUB_RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl"
WEIGHT_ORDER = ["light", "regular", "medium", "bold", "black"]
WEIGHT_NAME = {"light": "Light", "regular": "Regular", "medium": "Medium", "bold": "Bold", "black": "Black"}
WEIGHT_NUM = {"light": 300, "regular": 400, "medium": 500, "bold": 700, "black": 900}
MIN_FONT_BYTES = 20_000
DOWNLOAD_TIMEOUT_S = 15


@dataclass
class LoadedFont:
    path: Optional[Path]
    axis_weight: Optional[int]      # numeric weight to set on a variable font, else None
    source: str                     # local | downloaded | system | default
    requested: str                  # e.g. "IBM Plex Mono bold"
    used: str                       # what is really used, human readable
    weight_exact: bool = True

    @property
    def is_fallback(self) -> bool:
        return self.source in ("system", "default") or not self.weight_exact


def _slug(name: str) -> str:
    return name.replace(" ", "")


def _weights_by_distance(weight: str) -> list[str]:
    i = WEIGHT_ORDER.index(weight) if weight in WEIGHT_ORDER else WEIGHT_ORDER.index("bold")
    return sorted(WEIGHT_ORDER, key=lambda w: abs(WEIGHT_ORDER.index(w) - i))


def find_local(name: str, weight: str, font_dir: Optional[Path] = None) -> Optional[LoadedFont]:
    d = Path(font_dir or FONT_DIR)
    slug = _slug(name)
    w = weight if weight in WEIGHT_ORDER else "bold"
    exact = d / f"{slug}-{WEIGHT_NAME[w]}.ttf"
    if exact.exists():
        return LoadedFont(exact, None, "local", f"{name} {w}", exact.name)
    var = d / f"{slug}[wght].ttf"
    if var.exists():
        return LoadedFont(var, WEIGHT_NUM[w], "local", f"{name} {w}", f"{var.name} @ weight {WEIGHT_NUM[w]}")
    for alt in _weights_by_distance(w)[1:]:
        p = d / f"{slug}-{WEIGHT_NAME[alt]}.ttf"
        if p.exists():
            return LoadedFont(p, None, "local", f"{name} {w}", p.name, weight_exact=False)
    return None


def download(name: str, weight: str, font_dir: Optional[Path] = None, timeout: float = DOWNLOAD_TIMEOUT_S
             ) -> Optional[LoadedFont]:
    """Try the Google Fonts GitHub repo (static file for this weight, then the variable font). None on any failure."""
    d = Path(font_dir or FONT_DIR)
    slug, folder = _slug(name), _slug(name).lower()
    w = weight if weight in WEIGHT_ORDER else "bold"
    tries = [(f"{slug}-{WEIGHT_NAME[w]}.ttf", None), (f"{slug}[wght].ttf", WEIGHT_NUM[w])]
    for fname, axis in tries:
        url = f"{GITHUB_RAW}/{folder}/{quote(fname)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "style-transfer-planner"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
        except Exception:
            continue
        if len(data) < MIN_FONT_BYTES:
            continue
        d.mkdir(parents=True, exist_ok=True)
        path = d / fname
        path.write_bytes(data)
        return LoadedFont(path, axis, "downloaded", f"{name} {w}", fname if axis is None else f"{fname} @ weight {axis}")
    return None


def _system_candidates(category: Optional[str], bold: bool) -> list[Path]:
    win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    mac = Path("/System/Library/Fonts/Supplemental")
    lin = Path("/usr/share/fonts/truetype")
    if category == "mono":
        names = [(win, "consolab.ttf" if bold else "consola.ttf"), (win, "courbd.ttf" if bold else "cour.ttf"),
                 (mac, "Courier New Bold.ttf" if bold else "Courier New.ttf"),
                 (lin / "dejavu", "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
                 (lin / "liberation", "LiberationMono-Bold.ttf" if bold else "LiberationMono-Regular.ttf")]
    elif category in ("serif",):
        names = [(win, "georgiab.ttf" if bold else "georgia.ttf"), (win, "timesbd.ttf" if bold else "times.ttf"),
                 (mac, "Georgia Bold.ttf" if bold else "Georgia.ttf"),
                 (lin / "dejavu", "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf")]
    else:
        names = [(win, "arialbd.ttf" if bold else "arial.ttf"), (win, "segoeuib.ttf" if bold else "segoeui.ttf"),
                 (mac, "Arial Bold.ttf" if bold else "Arial.ttf"),
                 (lin / "dejavu", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 (lin / "liberation", "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")]
    return [d / n for d, n in names]


def system_fallback(category: Optional[str], weight: str, requested: str) -> Optional[LoadedFont]:
    bold = weight in ("bold", "black", "medium")
    for p in _system_candidates(category, bold):
        if p.exists():
            return LoadedFont(p, None, "system", requested, f"{p.stem} (system font)", weight_exact=False)
    return None


def load_font(name: str, weight: str = "bold", category: Optional[str] = None, *, allow_download: bool = True,
              font_dir: Optional[Path] = None) -> LoadedFont:
    requested = f"{name} {weight}"
    lf = find_local(name, weight, font_dir)
    if lf and lf.weight_exact:
        return lf
    if allow_download:
        dl = download(name, weight, font_dir)
        if dl:
            return dl
    if lf:
        return lf                       # nearest local weight beats a different family
    sysf = system_fallback(category, weight, requested)
    if sysf:
        return sysf
    return LoadedFont(None, None, "default", requested, "Pillow built-in default font", weight_exact=False)


def pil_font(lf: LoadedFont, size: int):
    size = max(8, int(size))
    if lf.path is None:
        try:
            return ImageFont.load_default(size)
        except TypeError:               # very old Pillow
            return ImageFont.load_default()
    font = ImageFont.truetype(str(lf.path), size)
    if lf.axis_weight is not None:
        try:
            axes = font.get_variation_axes()
            vals = [lf.axis_weight if a["name"] in (b"Weight", "Weight") else a["default"] for a in axes]
            font.set_variation_by_axes(vals)
        except Exception:
            pass
    return font
