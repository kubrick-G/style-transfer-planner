"""Usage cap for Gemini API requests: daily limit, UTC rollover, and concurrency-safe tracking."""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from .gemini_client import Backend, DEFAULT_BACKOFF_BASE_S, DEFAULT_MIN_INTERVAL_S, GeminiCallLog, GeminiClient, TIMEOUT_S, _env_int

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DAILY_LIMIT = 40
T = TypeVar("T", bound=BaseModel)


@contextmanager
def _locked(lock_path: Path):
    """File-based lock for cross-process concurrency."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as f:
        if sys.platform == "win32":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


class UsageCap:
    """Tracks daily Gemini requests keyed by UTC date with concurrency safety."""

    def __init__(
        self,
        path: Optional[Path | str] = None,
        daily_limit: Optional[int] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        if path is not None:
            self.path = Path(path)
        else:
            env_path = os.environ.get("STP_USAGE_CAP_FILE")
            self.path = Path(env_path) if env_path else ROOT / "data" / ".usage_cap.json"

        self.lock_path = self.path.with_suffix(".lock")
        if daily_limit is not None:
            self.daily_limit = int(daily_limit)
        else:
            self.daily_limit = _env_int("STP_DAILY_GEMINI_LIMIT", DEFAULT_DAILY_LIMIT)

        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _today_str(self) -> str:
        dt = self._clock()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.date().isoformat()

    def _read_unlocked(self, today: str) -> int:
        if not self.path.exists():
            return 0
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("date") == today:
                return int(data.get("used", 0))
            return 0
        except Exception:
            return 0

    def _write_unlocked(self, today: str, used: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps({"date": today, "used": used}, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get_used(self) -> int:
        today = self._today_str()
        with _locked(self.lock_path):
            return self._read_unlocked(today)

    def remaining(self) -> int:
        return max(0, self.daily_limit - self.get_used())

    def is_exhausted(self) -> bool:
        return self.remaining() <= 0

    def charge(self, n: int = 1) -> bool:
        """Attempt to charge n requests. Returns True if charged, False if limit would be exceeded."""
        if n <= 0:
            return True
        today = self._today_str()
        with _locked(self.lock_path):
            used = self._read_unlocked(today)
            if used + n > self.daily_limit:
                return False
            self._write_unlocked(today, used + n)
            return True


class CappedGeminiClient(GeminiClient):
    """Gemini client that charges a UsageCap for each real request attempt."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[Backend] = None,
        mode: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout_s: float = TIMEOUT_S,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        min_interval_s: Optional[float] = None,
        usage_cap: Optional[UsageCap] = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            backend=backend,
            mode=mode,
            max_retries=max_retries,
            timeout_s=timeout_s,
            backoff_base_s=backoff_base_s,
            min_interval_s=min_interval_s,
            **kwargs,
        )
        self.usage_cap = usage_cap if usage_cap is not None else UsageCap()

    @property
    def available(self) -> bool:
        if not super().available:
            return False
        return self.usage_cap.remaining() > 0

    @property
    def cap_exhausted(self) -> bool:
        return self.usage_cap.remaining() <= 0

    def generate_json(self, name: str, contents: list, response_model: Type[T]) -> Optional[T]:
        if not super().available:
            self.calls.append(
                GeminiCallLog(name=name, ok=False, attempts=0, error="Gemini unavailable (no GEMINI_API_KEY set)")
            )
            return None

        if self.cap_exhausted:
            self.calls.append(
                GeminiCallLog(
                    name=name,
                    ok=False,
                    attempts=0,
                    error=f"Daily Gemini request quota exhausted (0 of {self.usage_cap.daily_limit} left today)",
                )
            )
            return None

        # Intercept the attempt loop to charge per request attempt
        # Wrap original backend with charging logic
        orig_backend = self._backend

        def charging_backend(parts: list, model_cls: Type[BaseModel]):
            if not self.usage_cap.charge(1):
                raise RuntimeError("Daily Gemini request quota exhausted during attempts")
            return orig_backend(parts, model_cls)

        saved_backend = self._backend
        try:
            self._backend = charging_backend
            return super().generate_json(name, contents, response_model)
        finally:
            self._backend = saved_backend
