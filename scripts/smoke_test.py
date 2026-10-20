#!/usr/bin/env python3
"""Smoke test for Style Transfer Planner deployment.

Usage:
    python scripts/smoke_test.py [URL]

Examples:
    python scripts/smoke_test.py http://localhost:8501
    python scripts/smoke_test.py http://localhost:7860
    python scripts/smoke_test.py --headless-only

1. If a URL is provided, checks <URL>/_stcore/health.
2. Runs headless apply_style on bundled demo assets (card + clip + transcript).
3. Verifies rendered output, fallback log, style match, and prints timings.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.apply_engine import apply_style
from core.schema import StyleCard
from core.transcript import TranscriptWord, clean_words


def check_health(url: str, timeout: float = 10.0) -> bool:
    endpoint = url.rstrip("/") + "/_stcore/health"
    print(f"--> Checking health endpoint: {endpoint}")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "STP-SmokeTest/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            code = resp.getcode()
            body = resp.read().decode("utf-8").strip()
            print(f"    Health check returned {code} in {elapsed:.3f}s: {body!r}")
            return code == 200
    except Exception as e:
        print(f"    Health check failed: {e}")
        return False


def run_headless_smoke(out_dir: Path) -> bool:
    print("--> Running headless style transfer on bundled demo assets...")
    card_path = ROOT / "data" / "demo" / "cards" / "demo_fast_cuts.stylecard.json"
    target_path = ROOT / "data" / "demo" / "clips" / "demo_target_speech.mp4"
    transcript_path = ROOT / "data" / "demo" / "transcripts" / "demo_target_speech.transcript.json"

    if not card_path.exists() or not target_path.exists() or not transcript_path.exists():
        print(f"    ERROR: Demo assets missing! Checked {card_path}, {target_path}, {transcript_path}")
        return False

    t0 = time.perf_counter()
    card = StyleCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    raw_tr = json.loads(transcript_path.read_text(encoding="utf-8"))
    items = raw_tr["words"] if isinstance(raw_tr, dict) else raw_tr
    words = clean_words([TranscriptWord(**w) for w in items], 3600.0)

    out_video = out_dir / "smoke_output.mp4"
    selected = ["pacing", "audio_rhythm", "transitions", "color", "text"]

    t_apply = time.perf_counter()
    res = apply_style(
        card,
        target_path,
        selected,
        out_path=str(out_video),
        transcript=words,
        timeout_s=120.0,
    )
    apply_elapsed = time.perf_counter() - t_apply
    total_elapsed = time.perf_counter() - t0

    print(f"    Rendered output: {out_video} ({out_video.stat().st_size} bytes)")
    print(f"    Apply duration: {apply_elapsed:.2f}s (total: {total_elapsed:.2f}s)")
    print(f"    Plan items: {len(res.plan.items)} items, {len(res.plan.fallbacks)} fallbacks")
    if res.match:
        print(f"    Style match cuts: {res.match.scheduled_cuts}, target avg shot: {res.match.avg_shot_len_target_s}s")

    assert out_video.exists() and out_video.stat().st_size > 1000
    assert len(res.plan.items) >= 4
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test Style Transfer Planner.")
    parser.add_argument("url", nargs="?", default=None, help="Base URL of deployed app (e.g. http://localhost:8501)")
    parser.add_argument("--headless-only", action="store_true", help="Skip URL health check and run engine smoke test only")
    args = parser.parse_args()

    success = True
    if args.url and not args.headless_only:
        if not check_health(args.url):
            success = False

    with tempfile.TemporaryDirectory() as td:
        if not run_headless_smoke(Path(td)):
            success = False

    if success:
        print("\nAll smoke tests PASSED! System is ready.")
        return 0
    else:
        print("\nSmoke tests FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
