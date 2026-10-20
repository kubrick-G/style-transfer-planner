#!/usr/bin/env python3
"""Draw what the Style Card THINKS onto real frames so you can verify it by eye.

    python scripts/preview_card.py data/examples/opusjake.mp4

Reads outputs/<name>.stylecard.json (run analyze_cli.py first) and writes outputs/<name>.preview.png: one frame per
scene (middle of the scene) with the detected CAPTION REGION as a red box, plus each scene's time, shot type and
transition. If the red box isn't on the captions, the text analysis is wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.frames import FrameReader  # noqa: E402
from core.schema import StyleCard  # noqa: E402

THUMB_H = 360
COLS = 4


def _label(img, text, org, scale=0.5, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build_sheet(video: Path, card: StyleCard) -> np.ndarray:
    region = card.overall.text.region
    tiles = []
    with FrameReader(video) as r:
        for s in card.scenes:
            fr = r.frame_at((s.start_s + s.end_s) / 2)
            if fr is None:
                continue
            h, w = fr.shape[:2]
            t = cv2.resize(fr, (int(w * THUMB_H / h), THUMB_H), interpolation=cv2.INTER_AREA)
            th, tw = t.shape[:2]
            if region and len(region) == 4:
                p1 = (int(region[0] * tw), int(region[1] * th))
                p2 = (int(region[2] * tw), int(region[3] * th))
                cv2.rectangle(t, p1, p2, (0, 0, 255), 2)
            _label(t, f"#{s.index} {s.start_s:.1f}-{s.end_s:.1f}s", (6, 18))
            _label(t, f"{s.shot_type} / cam={s.camera_motion}", (6, 38), 0.45, (200, 255, 200))
            _label(t, f"in: {s.transition_in}  text={s.has_text}", (6, th - 10), 0.45, (200, 220, 255))
            tiles.append(t)
    if not tiles:
        raise SystemExit("No frames could be read from the video.")
    cw = max(t.shape[1] for t in tiles)
    pad = [cv2.copyMakeBorder(t, 0, 4, 0, cw - t.shape[1] + 4, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    while len(pad) % COLS:
        pad.append(np.full_like(pad[0], 30))
    rows = [np.hstack(pad[i:i + COLS]) for i in range(0, len(pad), COLS)]
    sheet = np.vstack(rows)
    o, tx = card.overall, card.overall.text
    head = np.full((96, sheet.shape[1], 3), 20, np.uint8)
    _label(head, f"{video.name}   layout={o.layout}   grade={o.color.grade}   bpm={o.audio.bpm}", (8, 22), 0.55)
    _label(head, f"mood: {o.mood}", (8, 46), 0.5, (200, 255, 200))
    _label(head, f"captions: pos={tx.position} chunking={tx.chunking} anim={tx.animation} changes/s={tx.changes_per_s}",
           (8, 68), 0.5, (120, 160, 255))
    _label(head, f"          font={tx.font_style.category if tx.font_style else None} style={tx.outline_or_shadow} "
                 f"region(red)={tx.region or 'none'}", (8, 88), 0.5, (120, 160, 255))
    return np.vstack([head, sheet])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--card", help="path to the .stylecard.json (default: outputs/<name>.stylecard.json)")
    ap.add_argument("--out-dir", default="outputs")
    a = ap.parse_args()
    video = Path(a.video)
    card_path = Path(a.card) if a.card else Path(a.out_dir) / f"{video.stem}.stylecard.json"
    if not card_path.exists():
        print(f"No style card at {card_path}. Run: python scripts/analyze_cli.py {a.video}")
        return 1
    card = StyleCard.model_validate_json(card_path.read_text())
    out = Path(a.out_dir) / f"{video.stem}.preview.png"
    cv2.imwrite(str(out), build_sheet(video, card))
    print(f"Wrote {out}  (open it; the red box should sit on the captions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
