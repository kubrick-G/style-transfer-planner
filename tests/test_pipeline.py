import pytest

from core import pipeline
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video
from core.utils import InvalidVideoError, VideoTooLongError


class CountingBackend:
    def __init__(self):
        self.n = 0

    def __call__(self, contents, model):
        self.n += 1
        return mock_backend(contents, model)


@pytest.fixture(scope="module")
def run(vid_beats, tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    backend = CountingBackend()
    client = GeminiClient(backend=backend, sleep=lambda s: None)   # cacheable "live-like" client
    card = analyze_video(vid_beats, client=client, cache_dir=cache)
    return card, backend, client, cache


def test_full_card_against_ground_truth(run):
    card = run[0]
    assert len(card.scenes) == 10 and card.overall.pacing.n_scenes == 10
    assert abs(card.overall.pacing.cuts_per_min - 54) < 3
    assert abs(card.overall.audio.bpm - 120) / 120 < 0.05
    assert card.overall.audio.beat_sync_score >= 0.85          # cuts are exactly on beats
    assert card.coupling.cuts_on_beat_ratio == card.overall.audio.beat_sync_score
    assert card.scenes[0].cut_on_beat is None and card.scenes[1].cut_on_beat is True
    assert card.scenes[0].transition_in is None and card.scenes[1].transition_in == "hard cut"
    assert card.scenes[-1].transition_out is None
    assert len(card.overall.color.palette) >= 3 and card.overall.audio.beat_times and card.overall.audio.energy_curve


def test_gemini_fields_are_validated_and_logged(run):
    card = run[0]
    assert card.overall.text.present is True and card.overall.text.font_style.category == "sans"
    assert card.overall.text.color == "#ffffff"
    assert all(s.shot_type in {"close-up", "medium", "wide"} for s in card.scenes)
    assert not card.meta.degraded
    assert all(c.ok for c in card.meta.gemini_calls) and len(card.meta.gemini_calls) == 8
    assert set(card.meta.stage_seconds) >= {"probe", "scenes", "audio", "color", "motion"}
    assert card.provenance["measured_by_code"] and card.provenance["judged_by_gemini"]
    assert 0 <= min(card.confidence.values()) and max(card.confidence.values()) <= 1


def test_second_run_is_a_cache_hit_with_no_gemini_calls(run, vid_beats):
    card, backend, client, cache = run
    calls = backend.n
    again = analyze_video(vid_beats, client=client, cache_dir=cache)
    assert again.meta.cache_hit is True and backend.n == calls
    assert again.video.sha256 == card.video.sha256 and len(again.scenes) == len(card.scenes)
    forced = analyze_video(vid_beats, client=client, cache_dir=cache, use_cache=False)
    assert forced.meta.cache_hit is False and backend.n > calls
    assert list(cache.glob("*.json"))                          # a forced re-analysis still saves its result


def test_without_gemini_degrades_gracefully_and_is_not_cached(vid_beats, tmp_path, monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    card = analyze_video(vid_beats, client=GeminiClient(), cache_dir=tmp_path)
    assert card.meta.gemini_mode == "unavailable"
    stages = {d.stage for d in card.meta.degraded}
    assert {"color.grade", "scene_labels", "transitions", "text", "overview"} <= stages
    assert card.overall.text.present is None and card.overall.mood is None and card.confidence["text"] == 0
    assert len(card.scenes) == 10 and card.overall.audio.bpm and card.overall.color.grade   # code fields intact
    assert card.overall.transitions[0].type == "hard cut"      # code pre-label
    assert not list(tmp_path.glob("*.json"))                   # degraded results are never cached


def test_one_failing_gemini_call_degrades_only_that_attribute(vid_beats, tmp_path):
    def backend(contents, model):
        if model.__name__ == "TextPresenceResponse":
            raise RuntimeError("boom")
        return mock_backend(contents, model)

    card = analyze_video(vid_beats, client=GeminiClient(backend=backend, sleep=lambda s: None), cache_dir=tmp_path)
    assert [d.stage for d in card.meta.degraded] == ["text"]
    assert card.overall.text.present is None and card.overall.mood is not None


def test_no_audio_video(vid_static, tmp_path):
    card = analyze_video(vid_static, client=GeminiClient(backend=mock_backend, mode="mock"), cache_dir=tmp_path)
    a = card.overall.audio
    assert a.has_audio is False and a.bpm is None and a.beat_sync_score is None and card.confidence["audio"] == 0
    assert len(card.scenes) == 1 and card.overall.transitions == [] and card.coupling.cuts_on_beat_ratio is None
    assert any("No audio track" in n for n in card.meta.notes)


def test_friendly_errors(vid_beats, tmp_path, monkeypatch):
    client = GeminiClient(backend=mock_backend, mode="mock")
    with pytest.raises(InvalidVideoError):
        analyze_video(tmp_path / "missing.mp4", client=client, cache_dir=tmp_path)
    junk = tmp_path / "notes.mp4"
    junk.write_text("this is not a video")
    with pytest.raises(InvalidVideoError):
        analyze_video(junk, client=client, cache_dir=tmp_path)
    monkeypatch.setattr(pipeline, "MAX_DURATION_S", 5.0)
    with pytest.raises(VideoTooLongError) as e:
        analyze_video(vid_beats, client=client, cache_dir=tmp_path)
    assert "limit" in str(e.value)
