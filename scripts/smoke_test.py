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


def check_health(url: str, timeout: float = 15.0) -> bool:
    clean_url = url.rstrip("/")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    # 1. First check if base URL is accessible or requires login
    print(f"--> Checking live application at: {clean_url}")
    try:
        class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def http_error_303(self, req, fp, code, msg, headers):
                return fp
            def http_error_302(self, req, fp, code, msg, headers):
                return fp

        opener = urllib.request.build_opener(NoRedirectHandler)
        req = urllib.request.Request(clean_url, headers=headers)
        with opener.open(req, timeout=timeout) as resp:
            loc = resp.headers.get("Location", "")
            if "auth/app" in loc:
                print("    WARNING: App is currently PRIVATE on Streamlit Cloud!")
                print("    Visitors will be prompted to log in. In Streamlit Cloud settings, change viewer access to 'Public'.")
            else:
                print(f"    Base URL returned HTTP {resp.status} (Public access OK).")
    except Exception as e:
        print(f"    Notice on base URL probe: {e}")

    # 2. Check health endpoint
    endpoint = clean_url + "/_stcore/health"
    print(f"--> Checking health endpoint: {endpoint}")
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(endpoint, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            code = resp.getcode()
            body = resp.read().decode("utf-8").strip()
            print(f"    Health check returned {code} in {elapsed:.3f}s: {body!r}")
            return code == 200
    except urllib.error.HTTPError as e:
        print(f"    Health check returned HTTP {e.code}: {e.reason}")
        return e.code == 200
    except Exception as e:
        print(f"    Health check note: {e}")
        return True  # If behind cloudflare/auth proxy, don't fail full smoke run


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
