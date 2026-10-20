import json

from core.gemini_client import GeminiClient, classify_error
from core.schema import GradeResponse

GOOD = json.dumps({"grade": "natural", "grain": False, "confidence": 0.8})


class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError: has .code and a message."""

    def __init__(self, code, message):
        super().__init__(f"{code} {message}")
        self.code = code


def make(backend, sleeps=None, **kw):
    kw.setdefault("min_interval_s", 0)
    kw.setdefault("backoff_base_s", 1.5)
    return GeminiClient(backend=backend, sleep=(sleeps.append if sleeps is not None else (lambda s: None)), **kw)


# ---------------------------------------------------------------- happy paths
def test_success_first_try_logs_usage():
    c = make(lambda contents, model: (GOOD, (11, 7)))
    out = c.generate_json("g", ["prompt"], GradeResponse)
    assert out.grade == "natural"
    log = c.calls[-1]
    assert log.ok and log.attempts == 1 and (log.prompt_tokens, log.output_tokens) == (11, 7)


def test_markdown_fenced_json_is_accepted():
    c = make(lambda contents, model: ("```json\n" + GOOD + "\n```", (None, None)))
    assert c.generate_json("g", ["p"], GradeResponse).confidence == 0.8


def test_invalid_json_retries_with_reminder_then_succeeds():
    seen = []

    def backend(contents, model):
        seen.append(list(contents))
        return ("not json at all" if len(seen) == 1 else GOOD), (None, None)

    c = make(backend)
    assert c.generate_json("g", ["prompt"], GradeResponse) is not None
    assert c.calls[-1].attempts == 2
    assert "ONLY valid JSON" in seen[1][-1] and len(seen[1]) == len(seen[0]) + 1


def test_schema_violation_is_retried_then_degrades():
    c = make(lambda contents, model: (json.dumps({"wrong": 1}), (None, None)), max_retries=2)
    assert c.generate_json("g", ["p"], GradeResponse) is None
    log = c.calls[-1]
    assert not log.ok and log.attempts == 3 and "invalid" in log.error


# ------------------------------------------------------- transient API errors
def test_overload_503_backs_off_exponentially_then_succeeds():
    state = {"n": 0}

    def backend(contents, model):
        state["n"] += 1
        if state["n"] < 3:
            raise FakeAPIError(503, "UNAVAILABLE. This model is currently experiencing high demand.")
        return GOOD, (None, None)

    sleeps = []
    c = make(backend, sleeps)
    assert c.generate_json("g", ["p"], GradeResponse) is not None
    assert sleeps == [1.5, 3.0] and c.calls[-1].attempts == 3 and not c.fatal_reason


def test_rate_limit_honours_retry_hint():
    state = {"n": 0}

    def backend(contents, model):
        state["n"] += 1
        if state["n"] == 1:
            raise FakeAPIError(429, "RESOURCE_EXHAUSTED. Please retry in 12.5s. quota metric per minute")
        return GOOD, (None, None)

    sleeps = []
    c = make(backend, sleeps)
    assert c.generate_json("g", ["p"], GradeResponse) is not None
    assert sleeps == [13.5]                                  # hint (12.5) + 1s, larger than the 1.5s backoff


def test_persistent_transient_failure_returns_none_never_raises():
    def backend(contents, model):
        raise ConnectionError("down")

    c = make(backend, max_retries=2)
    assert c.generate_json("g", ["p"], GradeResponse) is None
    assert not c.calls[-1].ok and "ConnectionError" in c.calls[-1].error and c.calls[-1].attempts == 3


# ------------------------------------------------- fatal errors fail fast
def test_daily_quota_is_not_retried_and_short_circuits_later_calls():
    n = {"calls": 0}

    def backend(contents, model):
        n["calls"] += 1
        raise FakeAPIError(429, "RESOURCE_EXHAUSTED. Quota exceeded for metric ... PerDay ...")

    sleeps = []
    c = make(backend, sleeps)
    assert c.generate_json("first", ["p"], GradeResponse) is None
    assert c.generate_json("second", ["p"], GradeResponse) is None
    assert n["calls"] == 1 and sleeps == []                  # one attempt, no waiting, second call skipped
    assert c.last_error_kind == "quota_daily" and c.fatal_reason
    assert c.calls[-1].attempts == 0 and "skipped" in c.calls[-1].error


def test_bad_key_and_missing_model_are_fatal():
    for code, msg, kind in [(400, "API key not valid. Please pass a valid API key.", "auth"),
                            (403, "PERMISSION_DENIED", "auth"),
                            (404, "models/nope is not found for API version", "model")]:
        n = {"calls": 0}

        def backend(contents, model, code=code, msg=msg):
            n["calls"] += 1
            raise FakeAPIError(code, msg)

        c = make(backend)
        assert c.generate_json("g", ["p"], GradeResponse) is None
        assert n["calls"] == 1 and c.last_error_kind == kind and c.fatal_reason


def test_other_bad_request_is_not_retried_but_not_fatal():
    c = make(lambda contents, model: (_ for _ in ()).throw(FakeAPIError(400, "Invalid schema field")))
    assert c.generate_json("g", ["p"], GradeResponse) is None
    assert c.last_error_kind == "bad_request" and c.fatal_reason is None and c.calls[-1].attempts == 1


def test_classify_error():
    assert classify_error(FakeAPIError(503, "UNAVAILABLE"))[0] == "overload"
    assert classify_error(FakeAPIError(429, "slow down, retry in 7s"))[0:2] == ("rate", 7.0)
    assert classify_error(FakeAPIError(429, "per day limit"))[0] == "quota_daily"
    assert classify_error(RuntimeError("something odd"))[0] == "other"


# ----------------------------------------------------------- pacing + availability
def test_calls_are_spaced_by_min_interval():
    t = {"now": 100.0}
    sleeps = []

    def fake_sleep(s):
        sleeps.append(round(s, 2))
        t["now"] += s

    c = GeminiClient(backend=lambda contents, model: (GOOD, (None, None)), min_interval_s=5.0,
                     sleep=fake_sleep, clock=lambda: t["now"])
    c.generate_json("a", ["p"], GradeResponse)
    t["now"] += 2.0                                           # only 2s passed since the last call
    c.generate_json("b", ["p"], GradeResponse)
    assert sleeps == [3.0]


def test_unavailable_without_key(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    c = GeminiClient()
    assert not c.available and c.mode == "unavailable"
    assert c.generate_json("g", ["p"], GradeResponse) is None
    assert c.calls[-1].attempts == 0


def test_blank_or_invalid_numeric_env_settings_are_ignored(monkeypatch):
    from core.gemini_client import DEFAULT_MAX_RETRIES, DEFAULT_MIN_INTERVAL_S
    for bad in ("", "   ", "abc", "7s"):
        monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", bad)
        monkeypatch.setenv("GEMINI_MAX_RETRIES", bad)
        c = GeminiClient(backend=lambda contents, model: (GOOD, (None, None)))
        assert c.min_interval_s == DEFAULT_MIN_INTERVAL_S and c.max_retries == DEFAULT_MAX_RETRIES
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "3.5")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "1")
    c = GeminiClient(backend=lambda contents, model: (GOOD, (None, None)))
    assert c.min_interval_s == 3.5 and c.max_retries == 1


def test_dotenv_loader_skips_empty_values(tmp_path, monkeypatch):
    import scripts.analyze_cli as cli
    monkeypatch.delenv("GEMINI_MIN_INTERVAL_S", raising=False)
    monkeypatch.delenv("STP_TEST_KEY", raising=False)
    fake_root = tmp_path / "proj"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / ".env").write_text("GEMINI_MIN_INTERVAL_S=\nSTP_TEST_KEY=\"abc\"\n# comment\n")
    monkeypatch.setattr(cli, "__file__", str(fake_root / "scripts" / "analyze_cli.py"))
    cli._load_dotenv()
    import os
    assert "GEMINI_MIN_INTERVAL_S" not in os.environ and os.environ["STP_TEST_KEY"] == "abc"
    monkeypatch.delenv("STP_TEST_KEY", raising=False)
