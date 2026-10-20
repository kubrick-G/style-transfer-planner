"""Gemini wrapper: pacing, error-aware retry/backoff, JSON-schema validation, graceful degradation, usage logging.

`generate_json` NEVER raises for API/validation problems: it returns None (and logs why) so the pipeline can
mark that attribute "unavailable" instead of crashing.

Error handling (learned from real free-tier runs):
- 503/500/504 (overloaded) and 429 per-minute limits -> retry with exponential backoff, honouring any
  "retry in Ns" hint from the API.
- 429 *daily* quota, bad key, model-not-found -> NOT retryable; the whole client short-circuits (later calls are
  skipped instantly) so a doomed run fails fast instead of waiting minutes.
- Calls are spaced by `min_interval_s` to stay under per-minute limits (set GEMINI_MIN_INTERVAL_S=0 on a paid tier).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .prompts import RETRY_REMINDER, SYSTEM_HINT
from .schema import GeminiCallLog

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-flash-latest"   # alias; override with GEMINI_MODEL
DEFAULT_MAX_RETRIES = 3                 # retries after the first attempt (so up to 4 attempts)
DEFAULT_BACKOFF_BASE_S = 4.0
MAX_BACKOFF_S = 45.0
DEFAULT_MIN_INTERVAL_S = 5.0
TIMEOUT_S = 90.0

FATAL_KINDS = {"auth", "model", "quota_daily"}      # no point retrying; skip all later calls
NON_RETRYABLE = FATAL_KINDS | {"bad_request"}


@dataclass
class Media:
    data: bytes
    mime: str


# backend signature: (contents: list[str | Media], response_model) -> (text, (prompt_tokens, output_tokens))
Backend = Callable[[list, Type[BaseModel]], tuple[str, tuple[Optional[int], Optional[int]]]]

_RETRY_HINT = re.compile(r"retry(?:\s+in|Delay['\"]?\s*:\s*['\"]?)\s*([0-9.]+)\s*s", re.IGNORECASE)


def _env_float(name: str, default: float) -> float:
    """Read a float setting; a blank or invalid value is treated as unset (never crash on a typo in .env)."""
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except ValueError:
        return default


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t).strip()


def classify_error(e: Exception) -> tuple[str, Optional[float]]:
    """Return (kind, retry_after_seconds). kinds: auth, model, quota_daily, rate, overload, bad_request, other."""
    msg = str(e)
    low = msg.lower()
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if not isinstance(code, int):
        m = re.search(r"\b([45]\d\d)\b", msg)
        code = int(m.group(1)) if m else None
    hint = _RETRY_HINT.search(msg)
    retry_after = float(hint.group(1)) if hint else None
    if "api key" in low and ("invalid" in low or "not valid" in low or "expired" in low) or code in (401, 403):
        return "auth", None
    if code == 404 or ("model" in low and "not found" in low):
        return "model", None
    if code == 429:
        daily = "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low
        return ("quota_daily" if daily else "rate"), retry_after
    if code in (500, 502, 503, 504) or "unavailable" in low or "overloaded" in low:
        return "overload", retry_after
    if code == 400:
        return "bad_request", None
    return "other", retry_after


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 backend: Optional[Backend] = None, mode: Optional[str] = None,
                 max_retries: Optional[int] = None, timeout_s: float = TIMEOUT_S,
                 backoff_base_s: float = DEFAULT_BACKOFF_BASE_S, min_interval_s: Optional[float] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.max_retries = max_retries if max_retries is not None else _env_int("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        self.min_interval_s = (min_interval_s if min_interval_s is not None
                               else _env_float("GEMINI_MIN_INTERVAL_S", DEFAULT_MIN_INTERVAL_S))
        self.timeout_s, self.backoff_base_s = timeout_s, backoff_base_s
        self._sleep, self._clock = sleep, clock
        self._last_call_t: Optional[float] = None
        self.calls: list[GeminiCallLog] = []
        self.fatal_reason: Optional[str] = None
        self.last_error_kind: Optional[str] = None
        if backend is not None:
            self._backend, self.mode = backend, (mode or "live")
        elif self.api_key:
            self._backend, self.mode = self._live_backend, "live"
        else:
            self._backend, self.mode = None, "unavailable"

    @property
    def available(self) -> bool:
        return self._backend is not None

    # ------------------------------------------------------------ live backend
    def _live_backend(self, contents: list, response_model: Type[BaseModel]):
        from google import genai
        from google.genai import types

        if not hasattr(self, "_sdk"):
            self._sdk = genai.Client(api_key=self.api_key,
                                     http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        parts = [c if isinstance(c, str) else types.Part.from_bytes(data=c.data, mime_type=c.mime) for c in contents]
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_HINT, response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(), temperature=0.0, seed=0)
        resp = self._sdk.models.generate_content(model=self.model, contents=parts, config=cfg)
        um = getattr(resp, "usage_metadata", None)
        return resp.text or "", (getattr(um, "prompt_token_count", None), getattr(um, "candidates_token_count", None))

    # ------------------------------------------------------------------ pacing
    def _pace(self) -> None:
        if self.min_interval_s > 0 and self._last_call_t is not None:
            wait = self.min_interval_s - (self._clock() - self._last_call_t)
            if wait > 0:
                self._sleep(wait)
        self._last_call_t = self._clock()

    # ------------------------------------------------------------------- main
    def generate_json(self, name: str, contents: list, response_model: Type[T]) -> Optional[T]:
        if not self.available:
            self.calls.append(GeminiCallLog(name=name, ok=False, attempts=0,
                                            error="Gemini unavailable (no GEMINI_API_KEY set)"))
            return None
        if self.fatal_reason:
            self.calls.append(GeminiCallLog(name=name, ok=False, attempts=0,
                                            error=f"skipped after earlier fatal error: {self.fatal_reason}"))
            return None
        t_start = time.perf_counter()
        last_err = "unknown error"
        pt = ot = None
        attempts = self.max_retries + 1
        sending = list(contents)
        done = 0
        for attempt in range(1, attempts + 1):
            done = attempt
            self._pace()
            try:
                text, (pt, ot) = self._backend(sending, response_model)
                obj = response_model.model_validate_json(_strip_fences(text))
                self.calls.append(GeminiCallLog(name=name, ok=True, attempts=attempt,
                                                seconds=round(time.perf_counter() - t_start, 3),
                                                prompt_tokens=pt, output_tokens=ot))
                return obj
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                self.last_error_kind = "invalid_json"
                last_err = f"invalid JSON/schema: {str(e)[:160]}"
                if RETRY_REMINDER not in sending:
                    sending = list(contents) + [RETRY_REMINDER]
            except Exception as e:  # network/API/quota errors
                kind, retry_after = classify_error(e)
                self.last_error_kind = kind
                last_err = f"{kind}: {type(e).__name__}: {str(e)[:900]}"
                if kind in FATAL_KINDS:
                    self.fatal_reason = last_err
                if kind in NON_RETRYABLE:
                    break
                if attempt < attempts:
                    backoff = min(MAX_BACKOFF_S, self.backoff_base_s * (2 ** (attempt - 1)))
                    self._sleep(min(MAX_BACKOFF_S, max(backoff, (retry_after or 0) + 1)))
        self.calls.append(GeminiCallLog(name=name, ok=False, attempts=done,
                                        seconds=round(time.perf_counter() - t_start, 3), error=last_err))
        return None
