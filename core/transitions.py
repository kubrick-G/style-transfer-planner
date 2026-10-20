"""Cut-point analysis: code measures abruptness; Gemini classifies the transition type from before/after strips."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import prompts
from .frames import FrameReader, jpeg_bytes, montage_rows
from .gemini_client import GeminiClient, Media
from .schema import TRANSITION_VOCAB, TransitionCount, TransitionsResponse, normalize_choice
from .scenes import SceneSpan

BEFORE_OFFSETS = (-6, -3, -1)   # frames relative to the first frame of the new scene
AFTER_OFFSETS = (0, 2, 5)
BATCH = 6
ABRUPT_RATIO = 2.5
MIN_CUT_DIFF = 12.0


@dataclass
class Cut:
    index: int          # cut k sits between scene k and scene k+1
    time_s: float
    abrupt: bool
    ratio: float
    code_label: str


def _gray(f):
    return cv2.cvtColor(cv2.resize(f, (96, 96)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def measure_cut(reader: FrameReader, time_s: float, index: int) -> Cut:
    c = int(round(time_s * reader.fps))
    frames = [reader.frame_index(c + o) for o in range(-3, 3)]   # c-3..c+2
    if any(f is None for f in frames):
        return Cut(index, time_s, True, 0.0, "hard cut")
    g = [_gray(f) for f in frames]
    diffs = [float(np.abs(g[i + 1] - g[i]).mean()) for i in range(len(g) - 1)]   # diffs[2] spans the cut
    cut_diff = diffs[2]
    others = diffs[:2] + diffs[3:]
    ratio = cut_diff / (max(others) + 1e-3)
    abrupt = ratio >= ABRUPT_RATIO and cut_diff >= MIN_CUT_DIFF
    return Cut(index, time_s, bool(abrupt), round(ratio, 2), "hard cut" if abrupt else "other")


def _strip(reader: FrameReader, cut: Cut) -> Optional[np.ndarray]:
    c = int(round(cut.time_s * reader.fps))
    before = [reader.frame_index(c + o) for o in BEFORE_OFFSETS]
    after = [reader.frame_index(c + o) for o in AFTER_OFFSETS]
    if any(f is None for f in before + after):
        return None
    return montage_rows([before, after], ["BEFORE", "AFTER"], cell_w=170)


def classify_transitions(client: Optional[GeminiClient], reader: FrameReader, scenes: list[SceneSpan],
                         notes: Optional[list[str]] = None
                         ) -> tuple[list[Cut], dict[int, str], Optional[float], bool]:
    """Returns (cuts, labels by cut index, gemini self-reported confidence, gemini_ok)."""
    cuts = [measure_cut(reader, sc.start_s, i) for i, sc in enumerate(scenes[1:])]
    labels = {c.index: c.code_label for c in cuts}   # code fallback
    if not cuts or client is None or not client.available:
        return cuts, labels, None, False
    confs, ok_all = [], True
    for b in range(0, len(cuts), BATCH):
        chunk = cuts[b:b + BATCH]
        contents: list = [prompts.TRANSITIONS.format(n=len(chunk), types=", ".join(TRANSITION_VOCAB))]
        used = []
        for local, cut in enumerate(chunk):
            strip = _strip(reader, cut)
            if strip is None:
                continue
            contents += [f"Cut {local} (t={cut.time_s:.2f}s, abrupt={str(cut.abrupt).lower()})",
                         Media(jpeg_bytes(strip, max_side=640), "image/jpeg")]
            used.append((local, cut))
        if not used:
            continue
        resp = client.generate_json(f"transitions[{b}:{b + len(chunk)}]", contents, TransitionsResponse)
        if resp is None:
            ok_all = False
            continue
        confs.append(resp.confidence)
        for item in resp.transitions:
            if 0 <= item.index < len(chunk):
                cut = chunk[item.index]
                label, why = reconcile(normalize_choice(item.type, TRANSITION_VOCAB, "other"), cut.abrupt)
                labels[cut.index] = label
                if why and notes is not None:
                    notes.append(f"Cut at {cut.time_s:.2f}s relabelled '{label}': {why}.")
    return cuts, labels, (float(np.mean(confs)) if confs else None), ok_all and bool(confs)


def reconcile(label: str, abrupt: bool) -> tuple[str, Optional[str]]:
    """A hard cut is by definition a single-frame change. Where Gemini contradicts the measurement,
    the measurement wins. Returns (label, reason_if_changed)."""
    if label == "hard cut" and not abrupt:
        return "other", "Gemini said 'hard cut' but the pixel change was not single-frame"
    if abrupt and label in ("crossfade", "slide/push"):
        return "hard cut", f"Gemini said '{label}' but the change happened within one frame"
    return label, None


def aggregate(labels: dict[int, str]) -> list[TransitionCount]:
    return [TransitionCount(type=t, count=n) for t, n in Counter(labels.values()).most_common()]
