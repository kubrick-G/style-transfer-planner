import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from core.apply_engine import apply_style
from core.fonts import resolve_font  # noqa: F401  (import check)
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video
from core.schema import FontStyle
from core.sync_check import MIN_CONF, SyncReport, analyze_sync, corrected_words
from core.transcript import TranscriptWord
from scripts.make_synthetic import make_synthetic

SR = 16000


def speech_words(seed, dur=16.0):
    rng = np.random.default_rng(seed)
    words, t = [], 0.3
    while t < dur - 0.6:
        d = rng.uniform(0.15, 0.42)
        words.append((t, t + d))
        t += d + (rng.uniform(0.6, 0.9) if rng.random() < 0.12 else rng.uniform(0.04, 0.22))
    return words


def speech_audio(words, dur, seed=0):
    rng = np.random.default_rng(seed)
    y = 0.004 * rng.standard_normal(int(dur * SR))
    for a, b in words:
        i0, i1 = int(a * SR), int(b * SR)
        y[i0:i1] += 0.25 * np.hanning(i1 - i0) ** 0.5 * rng.standard_normal(i1 - i0)
    return y.astype(np.float32)


def tw(words, shift=0.0, only=None):
    return [TranscriptWord(text="w", start_s=a + (shift if only is None or only(a) else 0),
                           end_s=b + (shift if only is None or only(a) else 0)) for a, b in words]


@pytest.fixture(scope="module")
def wavs(tmp_path_factory):
    d = tmp_path_factory.mktemp("sync")
    out = {}
    for seed in range(3):
        w = speech_words(seed)
        sf.write(str(d / f"s{seed}.wav"), speech_audio(w, 16.0, seed), SR)
        out[seed] = (d / f"s{seed}.wav", w)
    return out


# ------------------------------------------------------------------- estimator
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("shift", [0.0, 0.3, -0.25, 0.6])
def test_global_lag_is_recovered_exactly(wavs, seed, shift):
    wav, true = wavs[seed]
    r = analyze_sync(wav, tw(true, shift))
    assert abs(r.global_lag_s + shift) < 0.03 and r.trustworthy and r.global_confidence > 0.5


def test_unrelated_audio_is_not_trusted(tmp_path):
    rng = np.random.default_rng(5)
    sf.write(str(tmp_path / "n.wav"), (0.05 * rng.standard_normal(16 * SR)).astype(np.float32), SR)
    r = analyze_sync(tmp_path / "n.wav", tw(speech_words(1)))
    assert r.global_confidence < MIN_CONF and not r.trustworthy


def test_a_long_late_run_shows_up_as_suspect_windows(wavs):
    wav, true = wavs[1]
    r = analyze_sync(wav, tw(true, 0.4, only=lambda a: 6.0 <= a <= 11.5))
    assert abs(r.global_lag_s) < 0.1                                       # most of the transcript is fine
    sus = [w for w in r.deviating_windows() if w.start_s >= 6.0 and w.end_s <= 12.0]
    assert sus and any(abs(w.lag_s + 0.4) < 0.1 for w in sus)


# ------------------------------------------------------------------ correction
def report(lag, conf=0.9):
    return SyncReport(global_lag_s=lag, global_confidence=conf, duration_s=20.0)


def test_global_correction_shifts_every_word():
    words = [TranscriptWord(text="a", start_s=1.0, end_s=1.3), TranscriptWord(text="b", start_s=2.0, end_s=2.4)]
    out, info = corrected_words(words, report(-0.4))
    assert info["applied"] and [round(w.start_s, 2) for w in out] == [0.6, 1.6]


def test_correction_refuses_when_unsure_or_unneeded():
    words = [TranscriptWord(text="a", start_s=1.0, end_s=1.3)]
    out, info = corrected_words(words, report(-0.4, conf=0.02))
    assert not info["applied"] and "not trustworthy" in info["reason"] and "low confidence" in info["reason"] and out[0].start_s == 1.0
    out, info = corrected_words(words, report(0.03))
    assert not info["applied"] and "in sync" in info["reason"]


def test_correction_never_goes_before_zero():
    out, _ = corrected_words([TranscriptWord(text="a", start_s=0.1, end_s=0.4)], report(-0.5))
    assert out[0].start_s == 0.0 and out[0].end_s > out[0].start_s


# ------------------------------------------------------- engine + scripts
@pytest.fixture(scope="module")
def talking(tmp_path_factory):
    """12 s clip whose audio is speech-like bursts at known times."""
    d = tmp_path_factory.mktemp("talk")
    true = speech_words(2, 12.0)
    sf.write(str(d / "a.wav"), speech_audio(true, 12.0, 2), SR)
    silent = make_synthetic(d / "v.mp4", duration=12, fps=30, audio=False, cut_every_beats=1000, speed=0.2, size=(360, 640))
    out = d / "talk.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(d / "a.wav"), "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out, true


@pytest.fixture(scope="module")
def ref_card(vid_beats, tmp_path_factory):
    card = analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path_factory.mktemp("c"))
    t = card.overall.text
    t.font_style = FontStyle(category="mono", weight="bold", width="normal", geometry="slab")
    t.case, t.color, t.outline_or_shadow, t.animation = "uppercase", "#ffffff", "background-box", "none"
    t.chunking, t.changes_per_s, t.region, t.variety = "single-word", None, [0.2, 0.4, 0.8, 0.46], "consistent"
    return card


def test_plan_reports_a_late_transcript_and_suggests_the_fix(ref_card, talking, tmp_path):
    vid, true = talking
    res = apply_style(ref_card, vid, ["text"], str(tmp_path / "o.mp4"), transcript=tw(true, 0.4))
    item = res.plan.items[0]
    sc = item.detail["sync_check"]
    assert abs(sc["estimated_lag_s"] + 0.4) < 0.05 and sc["trustworthy"]
    assert "Timing check" in item.note and "--auto-sync" in item.note and "late" in item.note
    assert res.transcript[0].start_s == pytest.approx(true[0][0] + 0.4)           # untouched unless asked


def test_auto_sync_corrects_the_whole_transcript(ref_card, talking, tmp_path):
    vid, true = talking
    res = apply_style(ref_card, vid, ["text"], str(tmp_path / "o.mp4"), transcript=tw(true, 0.4), auto_sync=True)
    assert res.plan.items[0].detail["sync_check"]["auto_sync"]["applied"] is True
    assert all(abs(w.start_s - a) < 0.05 for w, (a, _) in zip(res.transcript, true))
    assert "Timing check" not in res.plan.items[0].note                           # nothing left to suggest


def test_an_in_sync_transcript_is_left_alone(ref_card, talking, tmp_path):
    vid, true = talking
    res = apply_style(ref_card, vid, ["text"], str(tmp_path / "o.mp4"), transcript=tw(true), auto_sync=True)
    sc = res.plan.items[0].detail["sync_check"]
    assert abs(sc["estimated_lag_s"]) < 0.05 and sc["auto_sync"]["applied"] is False
    assert all(abs(w.start_s - a) < 1e-6 for w, (a, _) in zip(res.transcript, true))


def test_a_failing_sync_check_never_blocks_the_render(ref_card, talking, tmp_path, monkeypatch):
    import core.apply_engine as ae
    monkeypatch.setattr(ae, "analyze_sync", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    vid, true = talking
    res = apply_style(ref_card, vid, ["text"], str(tmp_path / "o.mp4"), transcript=tw(true, 0.4))
    assert res.output_path and "error" in res.plan.items[0].detail["sync_check"]


def test_check_sync_script(talking, tmp_path, capsys):
    from scripts.check_sync import main
    vid, true = talking
    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps([w.model_dump() for w in tw(true, 0.4)]))
    out = tmp_path / "fixed.json"
    assert main([str(vid), str(tpath), "--fix", str(out)]) == 0
    text = capsys.readouterr().out
    assert "LATE" in text and "--caption-offset -0.40" in text.replace("-0.39", "-0.40").replace("-0.41", "-0.40")
    fixed = json.loads(out.read_text())
    assert all(abs(f["start_s"] - a) < 0.05 for f, (a, _) in zip(fixed, true))
    assert main([str(tmp_path / "nope.mp4"), str(tpath)]) == 2


def test_app_passes_the_auto_sync_checkbox_to_the_engine(monkeypatch, tmp_path, vid_static, ref_card):
    import shutil
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import core.apply_engine as ae
    cards, vids, outs = tmp_path / "cards", tmp_path / "videos", tmp_path / "out"
    for d in (cards, vids, outs):
        d.mkdir()
    (cards / "demo.stylecard.json").write_text(ref_card.model_dump_json())
    shutil.copy(vid_static, vids / "target.mp4")
    for k, v in {"STP_CARDS_DIR": cards, "STP_VIDEOS_DIR": vids, "STP_OUT_DIR": outs, "GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}.items():
        monkeypatch.setenv(k, str(v))
    seen = {}
    real = ae.apply_style

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(ae, "apply_style", spy)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=240).run()
    at.selectbox(key="tgt_choice").set_value("target.mp4").run()
    at.toggle(key="sel_text").set_value(False).run()
    box = next(c for c in at.checkbox if "Auto-correct caption timing" in c.label)
    box.set_value(True).run()
    at.button(key="apply_btn").click().run()
    assert not at.exception and seen["auto_sync"] is True


# --------------------------------------------------- honesty about when NOT to trust it
def test_a_transcript_from_different_speech_is_refused_with_reasons(wavs):
    wav, _ = wavs[0]
    r = analyze_sync(wav, tw(speech_words(7)))                 # words of some OTHER utterance
    assert not r.trustworthy and r.distrust_reasons()
    out, info = corrected_words(tw(speech_words(7)), r)
    assert info["applied"] is False and "not trustworthy" in info["reason"]


def test_an_offset_beyond_the_search_range_is_flagged_then_found_with_a_wider_range(wavs):
    wav, true = wavs[1]
    late = tw(true, 1.4)
    narrow = analyze_sync(wav, late, max_lag_s=1.0)
    assert not narrow.trustworthy and any("edge" in r for r in narrow.distrust_reasons())
    wide = analyze_sync(wav, late, max_lag_s=2.5)
    assert abs(wide.global_lag_s + 1.4) < 0.05 and wide.trustworthy


def test_disagreeing_windows_make_an_estimate_untrustworthy():
    from core.sync_check import WindowEstimate
    wins = [WindowEstimate(c, c - 1.5, c + 1.5, lag, 0.3, 8) for c, lag in [(2, 0.8), (3, -0.5), (4, -0.5), (5, 0.2), (6, -0.8)]]
    r = SyncReport(global_lag_s=-0.45, global_confidence=0.5, windows=wins, duration_s=10.0)
    assert not r.trustworthy and any("disagree" in x for x in r.distrust_reasons())
    consistent = SyncReport(global_lag_s=-0.45, global_confidence=0.5, duration_s=10.0,
                            windows=[WindowEstimate(c, c - 1.5, c + 1.5, -0.45 + d, 0.3, 8) for c, d in [(2, 0.0), (3, 0.05), (4, -0.05), (5, 0.1)]])
    assert consistent.trustworthy


def test_check_sync_script_explains_why_it_distrusts(wavs, tmp_path, capsys):
    from scripts.check_sync import main
    wav, _ = wavs[0]
    vid = tmp_path / "v.mp4"
    silent = make_synthetic(tmp_path / "s.mp4", duration=16, fps=30, audio=False, cut_every_beats=1000, speed=0.2, size=(180, 320))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(vid)], check=True)
    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps([w.model_dump() for w in tw(speech_words(9))]))
    assert main([str(vid), str(tpath), "--fix", str(tmp_path / "f.json")]) == 1          # refuses to write a 'fix'
    text = capsys.readouterr().out
    assert "NOT TRUSTWORTHY" in text and "do not apply" in text and not (tmp_path / "f.json").exists()


def test_plot_sync_draws_a_picture(wavs, tmp_path):
    import cv2
    from scripts.plot_sync import main as plot_main
    wav, true = wavs[2]
    vid = tmp_path / "v.mp4"
    silent = make_synthetic(tmp_path / "s.mp4", duration=16, fps=30, audio=False, cut_every_beats=1000, speed=0.2, size=(180, 320))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-shortest", str(vid)], check=True)
    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps([w.model_dump() for w in tw(true)]))
    out = tmp_path / "sync.png"
    assert plot_main([str(vid), str(tpath), "--out", str(out)]) == 0
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] > 1500 and img.shape[0] >= 2 * 230          # 16 s -> two 12 s rows
    assert plot_main([str(tmp_path / "nope.mp4"), str(tpath)]) == 2


def test_the_real_clip_result_that_was_wrongly_trusted_is_now_refused():
    """Numbers from a real run: global -0.97 s at confidence 0.21, five windows disagreeing from +0.79 to -0.76 s."""
    from core.sync_check import WindowEstimate
    wins = [WindowEstimate(c, c - 1.5, c + 1.5, lag, conf, n) for c, lag, conf, n in
            [(4.5, 0.79, 0.25, 10), (5.5, -0.53, 0.31, 10), (6.5, -0.54, 0.21, 11), (7.5, 0.22, 0.25, 9), (8.5, -0.76, 0.33, 12)]]
    r = SyncReport(global_lag_s=-0.97, global_confidence=0.21, windows=wins, duration_s=11.52, max_lag_s=1.0)
    reasons = " | ".join(r.distrust_reasons())
    assert not r.trustworthy
    assert "low confidence" in reasons and "edge" in reasons and "disagree" in reasons
