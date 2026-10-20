"""OFFLINE MOCK backend. Returns schema-valid but FAKE labels so the pipeline can be exercised without an API
key. Output is clearly marked (mood/genre start with 'MOCK'); never use it to judge label quality."""
from __future__ import annotations

import re

from .schema import SHOT_TYPE_VOCAB

_N = re.compile(r"N_ITEMS=(\d+)")


def _n_items(contents) -> int:
    for c in contents:
        if isinstance(c, str):
            m = _N.search(c)
            if m:
                return int(m.group(1))
    return 0


def mock_backend(contents, response_model):
    import json
    name, n = response_model.__name__, _n_items(contents)
    if name == "GradeResponse":
        out = {"grade": "natural", "grain": False, "confidence": 0.3}
    elif name == "SceneLabelsResponse":
        out = {"scenes": [{"index": i, "shot_type": SHOT_TYPE_VOCAB[i % 3], "camera_motion": "static"} for i in range(n)],
               "confidence": 0.3}
    elif name == "TransitionsResponse":
        out = {"transitions": [{"index": i, "type": "hard cut"} for i in range(n)], "confidence": 0.3}
    elif name == "TextPresenceResponse":
        out = {"frames": [{"index": i, "has_text": i % 4 == 1, "bbox": [0.1, 0.7, 0.9, 0.85] if i % 4 == 1 else []}
                          for i in range(n)]}
    elif name == "TextStyleResponse":
        out = {"font_category": "sans", "weight": "bold", "width": "normal", "geometry": "geometric",
               "mood": ["mock"], "case": "uppercase", "position": "lower-third", "color": "#ffffff",
               "outline_or_shadow": "shadow", "animation": "fade", "chunking": "single-word", "variety": "consistent", "description": "MOCK text treatment", "confidence": 0.3}
    elif name == "VideoOverviewResponse":
        out = {"mood": "MOCK: placeholder mood", "genre_guess": "MOCK genre", "layout": "full-frame", "structure_ok": True,
               "structure_notes": "MOCK structure note", "confidence": 0.3}
    else:
        raise ValueError(f"mock backend has no response for {name}")
    return json.dumps(out), (0, 0)
