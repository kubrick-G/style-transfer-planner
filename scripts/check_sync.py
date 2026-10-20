#!/usr/bin/env python3
"""Is the caption transcript EARLY or LATE relative to the audio? Measures it from the audio itself.

    python scripts/check_sync.py data/examples/clip.mp4 outputs/clip.transcript.json
    python scripts/check_sync.py clip.mp4 transcript.json --fix transcript.synced.json   # shifts every word by the global lag

Negative lag = the transcript is LATE (captions appear after the words are spoken): shift captions earlier by that amount.
The estimate needs pauses between phrases to lock on; it reports a confidence and refuses to correct when it is low.
Short runs of late words show up only as 'suspect windows' (diagnostic): fix those with scripts/fix_transcript.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.sync_check import MIN_CONF, analyze_sync, corrected_words, extract_audio  # noqa: E402
from core.transcript import load_words_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("transcript")
    ap.add_argument("--max-lag", type=float, default=1.0, help="search range in seconds (default 1.0; try 3 for big offsets)")
    ap.add_argument("--fix", help="write a corrected transcript here (global shift only)")
    a = ap.parse_args(argv)
    for p in (a.video, a.transcript):
        if not Path(p).exists():
            print(f"ERROR: not found: {p}")
            return 2
    words = load_words_json(a.transcript)
    with tempfile.TemporaryDirectory() as td:
        try:
            rep = analyze_sync(extract_audio(a.video, Path(td) / "a.wav"), words, max_lag_s=a.max_lag)
        except Exception as e:
            print(f"ERROR: could not read the audio: {e}")
            return 2
    print(f"words: {len(words)}   audio: {rep.duration_s}s")
    print(f"GLOBAL  lag = {rep.global_lag_s:+.2f}s   confidence = {rep.global_confidence:.2f}   search range ±{rep.max_lag_s:g}s")
    if rep.trustworthy and abs(rep.global_lag_s) >= 0.06:
        word = "LATE" if rep.global_lag_s < 0 else "EARLY"
        print(f"        TRUSTWORTHY -> the transcript looks {word} by {abs(rep.global_lag_s):.2f}s. Use  --caption-offset {rep.global_lag_s:+.2f}  or --fix.")
    elif rep.trustworthy:
        print("        TRUSTWORTHY -> globally in sync (within 60 ms).")
    else:
        print("        NOT TRUSTWORTHY, do not apply this number:")
        for r in rep.distrust_reasons():
            print(f"          - {r}")
    sus = rep.deviating_windows()
    if sus:
        print("SUSPECT WINDOWS (local lag differs from the global one; diagnostic only):")
        for w in sus:
            print(f"   {w.start_s:6.1f}-{w.end_s:6.1f}s  lag {w.lag_s:+.2f}s  confidence {w.confidence:.2f}  ({w.n_words} words)")
        print("   fix a run with: python scripts/fix_transcript.py <transcript> --shift-time FROM TO BY")
    if a.fix:
        fixed, info = corrected_words(words, rep)
        if not info["applied"]:
            print(f"Not corrected: {info['reason']}")
            return 1
        Path(a.fix).write_text(json.dumps([w.model_dump() for w in fixed], indent=1), encoding="utf-8")
        print(f"wrote {a.fix} (all words shifted by {info['lag_s']:+.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
