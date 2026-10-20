#!/usr/bin/env python3
"""Diagnose your Gemini setup in plain English.

    python scripts/check_gemini.py            # one tiny test call with the configured model
    python scripts/check_gemini.py --models   # list models your key can call (use one as GEMINI_MODEL)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.gemini_client import GeminiClient  # noqa: E402
from core.schema import GradeResponse  # noqa: E402

HINTS = {
    "auth": "Your API key was rejected. Re-copy it into .env as GEMINI_API_KEY=... (no quotes/spaces), or make a new key.",
    "model": "That model name doesn't exist for your key. Run with --models and set GEMINI_MODEL in .env to one of them.",
    "quota_daily": "Your DAILY quota for this model is used up. Wait for the reset, switch to another model "
                   "(GEMINI_MODEL in .env), or enable billing.",
    "rate": "Per-minute rate limit hit. Wait a minute and retry; keep GEMINI_MIN_INTERVAL_S at 5 or higher.",
    "overload": "Google says the model is overloaded right now (temporary). Retry in a few minutes or try another model.",
    "bad_request": "Google rejected the request itself (e.g. the schema/config). Paste this message to get it fixed.",
    "invalid_json": "The model replied with something that didn't match the schema. Paste this message to get it fixed.",
    "other": "Network or unknown error. Check your internet/VPN/proxy.",
}


def load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v:
                    os.environ.setdefault(k.strip(), v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="list models your key can call")
    ap.add_argument("--model", help="test this model name instead of GEMINI_MODEL / the default")
    a = ap.parse_args()
    load_dotenv()
    client = GeminiClient(model=a.model, min_interval_s=0, max_retries=1)
    if not client.available:
        print("No GEMINI_API_KEY found. Put it in a file named .env (NOT .env.example) next to README.md.")
        return 1
    print(f"Key found (ends with ...{client.api_key[-4:]}). Model: {client.model}")
    if a.models:
        from google import genai
        sdk = genai.Client(api_key=client.api_key)   # keep a reference: a temporary client gets closed by GC
        try:
            names = []
            for m in sdk.models.list():
                actions = getattr(m, "supported_actions", None)
                if not actions or "generateContent" in actions:
                    names.append(m.name.replace("models/", ""))
        except Exception as e:
            print(f"Could not list models: {type(e).__name__}: {str(e)[:300]}")
            return 1
        print("Models you can call:\n  " + "\n  ".join(sorted(names)))
        print("\nSet one in .env, e.g.  GEMINI_MODEL=<name>")
        return 0
    import time

    import numpy as np

    from core.frames import jpeg_bytes
    from core.gemini_client import Media

    tests = [("text call", ["Return JSON: grade 'natural', grain false, confidence 1."])]
    img = np.zeros((96, 96, 3), np.uint8)
    img[..., 2] = np.linspace(40, 230, 96, dtype=np.uint8)[None, :]
    img[..., 0] = np.linspace(200, 30, 96, dtype=np.uint8)[:, None]
    tests.append(("image call", ["An image is attached. Return JSON: grade 'natural', grain false, confidence 1.",
                                 Media(jpeg_bytes(img), "image/jpeg")]))
    all_ok = True
    for label, contents in tests:
        t0 = time.perf_counter()
        out = client.generate_json("ping", contents, GradeResponse)
        dt = time.perf_counter() - t0
        if out is not None:
            print(f"OK    {label} ({dt:.1f}s)")
            continue
        all_ok = False
        print(f"FAIL  {label} ({dt:.1f}s): {client.calls[-1].error}")  # full text: names the exact quota/limit
        print(f"      -> {HINTS.get(client.last_error_kind or 'other', HINTS['other'])}")
        if client.fatal_reason:
            break
    if all_ok:
        print("\nThe key and this model work for text AND images. Run the analysis again.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
