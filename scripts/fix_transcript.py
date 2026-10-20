#!/usr/bin/env python3
"""Inspect and locally correct a caption transcript (the <out>.transcript.json saved by apply_cli.py).

Gemini's word timestamps are estimates and can drift for a RUN of words. Fix just that run:

    python scripts/fix_transcript.py outputs/x.transcript.json --list
    python scripts/fix_transcript.py outputs/x.transcript.json --shift-time 11.2 13.8 -0.3     # words starting in 11.2-13.8 s, 0.3 s earlier
    python scripts/fix_transcript.py outputs/x.transcript.json --shift-words 24 29 -0.3         # words #24..#29 (see --list), 0.3 s earlier

Negative = earlier, positive = later. Repeat the shift flags to fix several runs. Writes <name>.fixed.json (or --in-place).
Then render with:  python scripts/apply_cli.py ... --apply text --transcript <name>.fixed.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.transcript import TranscriptWord, clean_words, load_words_json  # noqa: E402
import json  # noqa: E402


def format_list(words: list[TranscriptWord]) -> str:
    L = [" idx    start      end   word"]
    for i, w in enumerate(words):
        L.append(f"{i:4d}  {w.start_s:7.2f}  {w.end_s:7.2f}   {w.text}")
    return "\n".join(L)


def shift_by_time(words: list[TranscriptWord], t0: float, t1: float, delta: float) -> tuple[list[TranscriptWord], int]:
    lo, hi = min(t0, t1), max(t0, t1)
    out, n = [], 0
    for w in words:
        if lo <= w.start_s <= hi:
            out.append(TranscriptWord(text=w.text, start_s=max(0.0, w.start_s + delta), end_s=max(0.06, w.end_s + delta)))
            n += 1
        else:
            out.append(w.model_copy())
    return out, n


def shift_by_index(words: list[TranscriptWord], i0: int, i1: int, delta: float) -> tuple[list[TranscriptWord], int]:
    lo, hi = min(i0, i1), max(i0, i1)
    out, n = [], 0
    for i, w in enumerate(words):
        if lo <= i <= hi:
            out.append(TranscriptWord(text=w.text, start_s=max(0.0, w.start_s + delta), end_s=max(0.06, w.end_s + delta)))
            n += 1
        else:
            out.append(w.model_copy())
    return out, n


def settle(words: list[TranscriptWord]) -> list[TranscriptWord]:
    """After shifting, keep words ordered and stop a word overlapping the next one."""
    duration = max((w.end_s for w in words), default=0.0) + 1.0
    return clean_words(words, duration)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript")
    ap.add_argument("--list", action="store_true", help="print every word with its index and times")
    ap.add_argument("--shift-time", nargs=3, type=float, action="append", metavar=("FROM_S", "TO_S", "BY_S"),
                    help="shift words whose start lies in [FROM_S, TO_S] by BY_S seconds")
    ap.add_argument("--shift-words", nargs=3, action="append", metavar=("FIRST_IDX", "LAST_IDX", "BY_S"),
                    help="shift words FIRST_IDX..LAST_IDX (inclusive, see --list) by BY_S seconds")
    ap.add_argument("--out", help="output file (default <name>.fixed.json)")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input file")
    a = ap.parse_args(argv)
    src = Path(a.transcript)
    if not src.exists():
        print(f"ERROR: transcript not found: {src}")
        return 2
    try:
        words = load_words_json(src)
    except Exception as e:
        print(f"ERROR: could not read {src}: {e}")
        return 2
    if a.list or not (a.shift_time or a.shift_words):
        print(format_list(words))
        if not (a.shift_time or a.shift_words):
            return 0
    changed = 0
    for t0, t1, by in a.shift_time or []:
        words, n = shift_by_time(words, t0, t1, by)
        print(f"shifted {n} word(s) starting in {min(t0, t1):.2f}-{max(t0, t1):.2f}s by {by:+.2f}s")
        changed += n
    for i0, i1, by in a.shift_words or []:
        try:
            words, n = shift_by_index(words, int(i0), int(i1), float(by))
        except ValueError:
            print("ERROR: --shift-words needs integer indices and a number of seconds")
            return 2
        print(f"shifted {n} word(s) #{min(int(i0), int(i1))}..#{max(int(i0), int(i1))} by {float(by):+.2f}s")
        changed += n
    if changed == 0:
        print("No words matched; nothing written.")
        return 1
    words = settle(words)
    dst = src if a.in_place else (Path(a.out) if a.out else src.with_name(src.stem.replace(".fixed", "") + ".fixed.json"))
    dst.write_text(json.dumps([w.model_dump() for w in words], indent=1), encoding="utf-8")
    print(f"wrote {dst}\nrender with:  --apply text --transcript {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
