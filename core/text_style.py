"""On-screen text: Gemini finds which frames have text, then describes its style (as ATTRIBUTES, never a font name)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import prompts
from .frames import FrameReader, jpeg_bytes, sample_times
from .gemini_client import GeminiClient, Media
from .schema import (VARIETY_VOCAB, CHUNKING_VOCAB, FONT_CATEGORY_VOCAB, FONT_GEOMETRY_VOCAB, FONT_WEIGHT_VOCAB, FONT_WIDTH_VOCAB, FontStyle,
                     OUTLINE_VOCAB, TEXT_ANIMATION_VOCAB, TEXT_CASE_VOCAB, TEXT_POSITION_VOCAB, TextPresenceResponse,
                     TextStyle, TextStyleResponse, normalize_choice, normalize_hex)

PRESENCE_HZ = 1.0
PRESENCE_MAX_FRAMES = 60
SCATTER_STD = 0.10          # std of caption-box centres above this = text is placed all over the frame
BURST_OFFSETS = (-0.6, -0.3, 0.0, 0.3)
EXTRA_FRAMES = 4
CAPTION_SAMPLE_FPS = 8.0
CHANGE_THRESHOLD = 0.4       # fraction of caption-area edge pixels that must change to count as a caption event


def normalize_bbox(b) -> Optional[list[float]]:
    """Return [x0,y0,x1,y1] in 0..1, accepting 0-1000 coordinates (a common Gemini convention); None if unusable."""
    try:
        v = [float(x) for x in b]
    except (TypeError, ValueError):
        return None
    if len(v) != 4:
        return None
    m = max(v)
    if m > 1.5:
        if m > 1000.0:
            return None
        v = [x / 1000.0 for x in v]
    x0, y0, x1, y1 = v
    if not (0 <= x0 < x1 <= 1.0001 and 0 <= y0 < y1 <= 1.0001):
        return None
    return [x0, y0, x1, y1]


@dataclass
class TextDetection:
    flagged_times: list[float] = field(default_factory=list)
    checked_times: list[float] = field(default_factory=list)
    bboxes: list[list[float]] = field(default_factory=list)

    def _usable(self) -> list[list[float]]:
        return [nb for nb in (normalize_bbox(b) for b in self.bboxes) if nb is not None]

    @property
    def scattered(self) -> bool:
        """True if caption boxes are spread all over the frame (kinetic typography): one region is meaningless."""
        ok = self._usable()
        if len(ok) < 3:
            return False
        cx = [(b[0] + b[2]) / 2 for b in ok]
        cy = [(b[1] + b[3]) / 2 for b in ok]
        return float(np.std(cx)) > SCATTER_STD or float(np.std(cy)) > SCATTER_STD

    @property
    def region(self) -> Optional[list[float]]:
        """Median caption box (normalised) if boxes are usable AND consistently placed; else None."""
        ok = self._usable()
        if not ok or self.scattered:
            return None
        return [round(float(np.median([b[i] for b in ok])), 3) for i in range(4)]

    @property
    def present(self) -> bool:
        return bool(self.flagged_times)

    def has_text_between(self, t0: float, t1: float) -> Optional[bool]:
        """True if text was seen in [t0,t1); False only if a sampled frame in that span had none; None if unsampled."""
        if any(t0 <= t < t1 for t in self.flagged_times):
            return True
        if any(t0 <= t < t1 for t in self.checked_times):
            return False
        return None


def detect_text_frames(client: GeminiClient, reader: FrameReader, duration_s: float) -> Optional[TextDetection]:
    times = sample_times(duration_s, hz=PRESENCE_HZ, max_n=PRESENCE_MAX_FRAMES)
    contents: list = [prompts.TEXT_PRESENCE.format(n=len(times))]
    kept = []
    for t in times:
        fr = reader.frame_at(t)
        if fr is None:
            continue
        contents += [f"Frame {len(kept)}", Media(jpeg_bytes(fr, max_side=384), "image/jpeg")]
        kept.append(t)
    if not kept:
        return None
    contents[0] = prompts.TEXT_PRESENCE.format(n=len(kept))
    resp = client.generate_json("text_presence", contents, TextPresenceResponse)
    if resp is None:
        return None
    good = [f for f in resp.frames if f.has_text and 0 <= f.index < len(kept)]
    flagged = sorted({kept[f.index] for f in good})
    return TextDetection(flagged_times=flagged, checked_times=kept, bboxes=[list(f.bbox) for f in good if f.bbox])


def extract_text_style(client: GeminiClient, reader: FrameReader, det: TextDetection, duration_s: float
                       ) -> tuple[Optional[TextStyle], Optional[float]]:
    if not det.present:
        return TextStyle(present=False), None
    t0 = det.flagged_times[0]
    burst_t = [min(max(0.0, t0 + o), max(0.0, duration_s - 0.05)) for o in BURST_OFFSETS]
    rest = [t for t in det.flagged_times if all(abs(t - b) > 0.2 for b in burst_t)]
    if len(rest) > EXTRA_FRAMES:
        step = len(rest) / EXTRA_FRAMES
        rest = [rest[int(i * step)] for i in range(EXTRA_FRAMES)]
    p = prompts.TEXT_STYLE.format(
        cats="/".join(FONT_CATEGORY_VOCAB), weights="/".join(FONT_WEIGHT_VOCAB), widths="/".join(FONT_WIDTH_VOCAB),
        geoms="/".join(FONT_GEOMETRY_VOCAB), cases="/".join(TEXT_CASE_VOCAB), positions="/".join(TEXT_POSITION_VOCAB),
        outlines="/".join(OUTLINE_VOCAB), animations="/".join(TEXT_ANIMATION_VOCAB),
        chunks="/".join(CHUNKING_VOCAB))
    contents: list = [p]
    for i, t in enumerate(burst_t):
        fr = reader.frame_at(t)
        if fr is not None:
            contents += [f"Burst frame {i} (t={t:.2f}s)", Media(jpeg_bytes(fr, max_side=512), "image/jpeg")]
    for i, t in enumerate(rest):
        fr = reader.frame_at(t)
        if fr is not None:
            contents += [f"Other frame {i} (t={t:.2f}s)", Media(jpeg_bytes(fr, max_side=512), "image/jpeg")]
    resp = client.generate_json("text_style", contents, TextStyleResponse)
    if resp is None:
        return None, None
    ts = TextStyle(
        present=True,
        font_style=FontStyle(
            category=normalize_choice(resp.font_category, FONT_CATEGORY_VOCAB),
            weight=normalize_choice(resp.weight, FONT_WEIGHT_VOCAB),
            width=normalize_choice(resp.width, FONT_WIDTH_VOCAB),
            geometry=normalize_choice(resp.geometry, FONT_GEOMETRY_VOCAB),
            mood=[m.strip().lower() for m in resp.mood[:3] if m.strip()]),
        case=normalize_choice(resp.case, TEXT_CASE_VOCAB),
        position=normalize_choice(resp.position, TEXT_POSITION_VOCAB),
        color=normalize_hex(resp.color),
        outline_or_shadow=normalize_choice(resp.outline_or_shadow, OUTLINE_VOCAB),
        animation=normalize_choice(resp.animation, TEXT_ANIMATION_VOCAB),
        chunking=normalize_choice(resp.chunking, CHUNKING_VOCAB),
        variety=normalize_choice(resp.variety, VARIETY_VOCAB),
        description=(resp.description.strip()[:200] or None))
    return ts, resp.confidence


def caption_activity(reader: FrameReader, region: Optional[list[float]], duration_s: float) -> Optional[float]:
    """MEASURED caption change events per second inside the caption region (appear / swap / disappear).

    Edge maps of the cropped region are compared between successive samples (8 fps). Experimental: background
    motion behind the captions can add false events; threshold is uncalibrated."""
    if not region or len(region) != 4 or duration_s <= 0:
        return None
    W, H = reader.width, reader.height
    pad = 0.015
    x0, x1 = int(max(0.0, region[0] - pad) * W), int(min(1.0, region[2] + pad) * W)
    y0, y1 = int(max(0.0, region[1] - pad) * H), int(min(1.0, region[3] + pad) * H)
    if x1 - x0 < 24 or y1 - y0 < 12:
        return None
    prev, changes, n = None, 0, 0
    for _, fr in reader.iter_frames(CAPTION_SAMPLE_FPS):
        g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (160, max(8, int(160 * (y1 - y0) / (x1 - x0)))), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(g, 80, 160) > 0
        if prev is not None:
            union = int(np.logical_or(prev, edges).sum())
            if union >= 20 and np.logical_xor(prev, edges).sum() / union >= CHANGE_THRESHOLD:
                changes += 1
        prev, n = edges, n + 1
    return round(changes / duration_s, 2) if n > 1 else None
