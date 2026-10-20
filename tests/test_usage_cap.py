from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel
import pytest

from core.gemini_client import GeminiClient
from core.usage_cap import CappedGeminiClient, UsageCap


class DummyResp(BaseModel):
    answer: str


def test_usage_cap_tracks_and_charges(tmp_path):
    cap_file = tmp_path / "cap.json"
    cap = UsageCap(path=cap_file, daily_limit=10)
    assert cap.remaining() == 10
    assert not cap.is_exhausted()

    assert cap.charge(4) is True
    assert cap.remaining() == 6
    assert cap.get_used() == 4

    assert cap.charge(6) is True
    assert cap.remaining() == 0
    assert cap.is_exhausted()

    # Exceeding limit fails
    assert cap.charge(1) is False
    assert cap.remaining() == 0


def test_utc_midnight_rollover(tmp_path):
    cap_file = tmp_path / "cap.json"
    current_dt = datetime(2026, 10, 20, 23, 58, 0, tzinfo=timezone.utc)

    def clock():
        return current_dt

    cap = UsageCap(path=cap_file, daily_limit=10, clock=clock)
    assert cap.charge(8) is True
    assert cap.remaining() == 2
    assert cap.get_used() == 8

    # Roll over to next day UTC
    current_dt = datetime(2026, 10, 21, 0, 2, 0, tzinfo=timezone.utc)
    assert cap.get_used() == 0
    assert cap.remaining() == 10
    assert not cap.is_exhausted()

    assert cap.charge(3) is True
    assert cap.remaining() == 7


def test_env_limits_and_invalid_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("STP_DAILY_GEMINI_LIMIT", "25")
    cap = UsageCap(path=tmp_path / "c1.json")
    assert cap.daily_limit == 25

    monkeypatch.setenv("STP_DAILY_GEMINI_LIMIT", "")
    cap2 = UsageCap(path=tmp_path / "c2.json")
    assert cap2.daily_limit == 40

    monkeypatch.setenv("STP_DAILY_GEMINI_LIMIT", "not_a_number")
    cap3 = UsageCap(path=tmp_path / "c3.json")
    assert cap3.daily_limit == 40


def test_concurrency(tmp_path):
    cap_file = tmp_path / "cap_concurrent.json"
    cap = UsageCap(path=cap_file, daily_limit=100)

    def worker(_):
        c = UsageCap(path=cap_file, daily_limit=100)
        return c.charge(2)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(20)))

    assert all(results)
    assert cap.get_used() == 40
    assert cap.remaining() == 60


def test_capped_gemini_client_exhaustion_never_calls_backend(tmp_path):
    cap_file = tmp_path / "cap.json"
    cap = UsageCap(path=cap_file, daily_limit=2)
    assert cap.charge(2) is True  # exhaust immediately

    backend_calls = 0

    def fake_backend(contents, model_cls):
        nonlocal backend_calls
        backend_calls += 1
        return '{"answer": "ok"}', (10, 10)

    client = CappedGeminiClient(api_key="fake-key", backend=fake_backend, usage_cap=cap)
    assert client.available is False
    assert client.cap_exhausted is True

    res = client.generate_json("test_call", ["prompt"], DummyResp)
    assert res is None
    assert backend_calls == 0
    assert len(client.calls) == 1
    assert "Daily Gemini request quota exhausted" in client.calls[0].error


def test_capped_gemini_client_charges_each_attempt(tmp_path):
    cap_file = tmp_path / "cap.json"
    cap = UsageCap(path=cap_file, daily_limit=10)

    backend_calls = 0

    def retrying_backend(contents, model_cls):
        nonlocal backend_calls
        backend_calls += 1
        if backend_calls < 3:
            raise RuntimeError("503 Service Unavailable")
        return '{"answer": "success"}', (10, 10)

    client = CappedGeminiClient(
        api_key="fake-key",
        backend=retrying_backend,
        usage_cap=cap,
        max_retries=3,
        backoff_base_s=0.01,
        min_interval_s=0,
        sleep=lambda _: None,
    )

    res = client.generate_json("call", ["prompt"], DummyResp)
    assert res is not None
    assert res.answer == "success"
    assert backend_calls == 3
    # Cap should have charged 3 units for 3 attempts
    assert cap.get_used() == 3
    assert cap.remaining() == 7
