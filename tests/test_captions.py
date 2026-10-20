import io
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import apply_engine, font_files
from core.apply_engine import apply_style
from core.captions import (Caption, CaptionStyle, _overlay, apply_case, chunk_words, hex_to_rgb, pick_chunk_size,
                           render_captions)
from core.font_files import LoadedFont, download, find_local, load_font, pil_font
from core.gemini_client import GeminiClient
from core.mock_gemini import mock_backend
from core.pipeline import analyze_video
from core.schema import FontStyle
from core.transcript import TranscriptWord, clean_words, load_words_json, transcribe
from core.utils import ffprobe_info


@pytest.fixture(scope="module")
def ref_card(vid_beats, tmp_path_factory):
    card = analyze_video(vid_beats, client=GeminiClient(backend=mock_backend, mode="mock", min_interval_s=0),
                         cache_dir=tmp_path_factory.mktemp("c"))
    t = card.overall.text
    t.font_style = FontStyle(category="mono", weight="bold", width="normal", geometry="slab", mood=["tech"])
    t.case, t.color, t.outline_or_shadow, t.animation = "uppercase", "#ffffff", "background-box", "none"
    t.chunking, t.changes_per_s, t.region, t.variety = "short-chunks", 1.94, [0.241, 0.377, 0.758, 0.421], "consistent"
    return card


def words(n=12, start=0.3, step=0.3):
    return [TranscriptWord(text=f"word{i}", start_s=start + i * step, end_s=start + i * step + 0.25) for i in range(n)]


def streams(path):
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
                          capture_output=True, text=True).stdout.split()


def frame_at(path, t):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * cap.get(cv2.CAP_PROP_FPS)))
    ok, f = cap.read()
    cap.release()
    return f.astype(np.float32)


# ------------------------------------------------------------------ font files
def test_bundled_font_loads_exactly():
    lf = load_font("IBM Plex Mono", "bold", "mono", allow_download=False)
    assert lf.source == "local" and lf.weight_exact and lf.path.exists() and not lf.is_fallback
    assert pil_font(lf, 30).getbbox("REVERSE")[2] > 50


def test_variable_font_gets_its_weight_axis():
    lf = load_font("Montserrat", "bold", allow_download=False)
    assert lf.axis_weight == 700 and "[wght]" in lf.path.name
    bold, light = pil_font(lf, 40), pil_font(load_font("Montserrat", "light", allow_download=False), 40)
    assert bold.getbbox("Hamburgefonts")[2] > light.getbbox("Hamburgefonts")[2]   # heavier weight is wider


def test_nearest_local_weight_is_used_and_flagged():
    lf = load_font("Lato", "light", allow_download=False)
    assert lf.source == "local" and not lf.weight_exact and lf.is_fallback and lf.path.exists()


def test_unknown_family_falls_back_to_system_or_default():
    lf = load_font("Totally Unknown Sans", "bold", "sans", allow_download=False)
    assert lf.source in ("system", "default") and lf.is_fallback
    assert pil_font(lf, 24).getbbox("Test")[2] > 10


def test_download_writes_the_font_and_failure_returns_none(tmp_path, monkeypatch):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    urls = []

    def fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        return Resp(b"x" * 30000)

    monkeypatch.setattr(font_files.urllib.request, "urlopen", fake_urlopen)
    lf = download("Some Font", "bold", font_dir=tmp_path)
    assert lf.source == "downloaded" and lf.path.exists() and urls[0].endswith("/somefont/SomeFont-Bold.ttf")

    def boom(req, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr(font_files.urllib.request, "urlopen", boom)
    assert download("Other Font", "bold", font_dir=tmp_path) is None
    assert find_local("Other Font", "bold", tmp_path) is None


# ------------------------------------------------------------------ transcript
def test_clean_words_repairs_and_drops():
    raw = [TranscriptWord(text=" b ", start_s=1.0, end_s=1.6), TranscriptWord(text="a", start_s=0.2, end_s=1.3),
           TranscriptWord(text="", start_s=2, end_s=3), TranscriptWord(text="bad", start_s=4, end_s=3),
           TranscriptWord(text="late", start_s=99, end_s=100), TranscriptWord(text="long", start_s=5, end_s=30)]
    out = clean_words(raw, 10.0)
    assert [w.text for w in out] == ["a", "b", "long"]
    assert out[0].end_s <= out[1].start_s and out[2].end_s - out[2].start_s <= 3.0


def test_load_words_json_accepts_both_shapes(tmp_path):
    items = [{"text": "hi", "start_s": 0.1, "end_s": 0.4}]
    (tmp_path / "a.json").write_text(json.dumps(items))
    (tmp_path / "b.json").write_text(json.dumps({"words": items}))
    assert load_words_json(tmp_path / "a.json")[0].text == load_words_json(tmp_path / "b.json")[0].text == "hi"


def test_transcribe_with_fake_gemini(vid_beats):
    def backend(contents, model):
        assert model.__name__ == "TranscriptResponse" and any(not isinstance(c, str) and c.mime == "audio/wav" for c in contents)
        return json.dumps({"words": [{"text": "hello", "start_s": 0.5, "end_s": 0.9},
                                     {"text": "world", "start_s": 1.0, "end_s": 1.4}]}), (1, 1)

    out = transcribe(GeminiClient(backend=backend, min_interval_s=0), vid_beats, 10.0)
    assert [w.text for w in out] == ["hello", "world"]


def test_transcribe_returns_none_when_it_cannot(vid_beats, vid_static, monkeypatch):
    assert transcribe(None, vid_beats, 10.0) is None
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert transcribe(GeminiClient(), vid_beats, 10.0) is None

    def boom(contents, model):
        raise RuntimeError("down")

    assert transcribe(GeminiClient(backend=boom, min_interval_s=0, max_retries=0), vid_beats, 10.0) is None
    ok = GeminiClient(backend=lambda c, m: (json.dumps({"words": []}), (1, 1)), min_interval_s=0)
    assert transcribe(ok, vid_beats, 10.0) is None                      # empty transcript -> None
    assert transcribe(ok, vid_static, 6.0) is None                      # no audio track -> None


# -------------------------------------------------------------------- chunking
def test_case_and_color_helpers():
    assert apply_case("hello World", "uppercase") == "HELLO WORLD" and apply_case("Hello", "lowercase") == "hello"
    assert apply_case("hello world", "title") == "Hello World" and apply_case("hello", None) == "hello"
    assert hex_to_rgb("#fffde9") == (255, 253, 233) and hex_to_rgb(None) == (255, 255, 255) and hex_to_rgb("nope") == (255, 255, 255)


def test_chunk_size_prefers_measured_rate_then_label():
    ws = words(30, step=0.4)                                              # 2.5 words/s
    assert pick_chunk_size(ws, 2.5, "single-word") == 1                   # measured rate wins over the label
    assert pick_chunk_size(ws, 1.25, None) == 2
    assert pick_chunk_size(ws, None, "single-word") == 1 and pick_chunk_size(ws, None, "full-lines") == 6
    assert pick_chunk_size(ws, 99.0, "short-chunks") == 2                 # implausible rate ignored
    assert pick_chunk_size(words(2), 2.0, "short-chunks") == 2            # too few words to estimate a rate


def test_chunk_words_groups_times_and_never_overlaps():
    caps = chunk_words(words(7), 3, "uppercase")
    assert [c.text for c in caps] == ["WORD0 WORD1 WORD2", "WORD3 WORD4 WORD5", "WORD6"]
    assert caps[0].start_s == pytest.approx(0.3)
    for a, b in zip(caps, caps[1:]):
        assert a.end_s <= b.start_s + 1e-9 and a.end_s > a.start_s
    assert all(c.end_s - c.start_s >= 0.25 - 1e-9 for c in caps)


# -------------------------------------------------------------------- rendering
def test_overlay_never_exceeds_the_frame():
    lf = load_font("IBM Plex Mono", "bold", allow_download=False)
    style = CaptionStyle(size_rel=0.05, font_px=40, box="background-box", x_center=0.9, y_center=0.98)
    bgr, alpha, x0, y0 = _overlay("A VERY LONG CAPTION " * 6, style, pil_font(lf, 40), 360, 640)
    h, w = alpha.shape
    assert x0 >= 0 and y0 >= 0 and x0 + w <= 360 and y0 + h <= 640 and alpha.max() > 0.5


def test_captions_change_only_the_caption_region_and_window(vid_beats, tmp_path):
    lf = load_font("IBM Plex Mono", "bold", allow_download=False)
    style = CaptionStyle(size_rel=0.06, color=(255, 255, 255), box="background-box", x_center=0.5, y_center=0.4)
    base = render_captions(vid_beats, [], style, lf, tmp_path / "base.mp4")
    out = render_captions(vid_beats, [Caption("HELLO THERE", 2.0, 3.0)], style, lf, tmp_path / "cap.mp4")
    assert "audio" in streams(out) and abs(ffprobe_info(out)["duration_s"] - ffprobe_info(base)["duration_s"]) < 0.3
    inside = np.abs(frame_at(out, 2.5) - frame_at(base, 2.5))
    outside = np.abs(frame_at(out, 5.0) - frame_at(base, 5.0))
    H = inside.shape[0]
    band = inside[int(H * 0.32):int(H * 0.48)]
    assert band.mean() > 5 and inside[: int(H * 0.2)].mean() < 1.0 and inside[int(H * 0.6):].mean() < 1.0
    assert outside.mean() < 1.0                                           # no caption outside its time window


# ------------------------------------------------------------- engine integration
def test_text_is_rendered_when_a_transcript_is_supplied(ref_card, vid_beats, tmp_path):
    out = tmp_path / "o.mp4"
    res = apply_style(ref_card, vid_beats, ["text"], str(out), transcript=words(15))
    item = res.plan.items[0]
    assert out.exists() and item.status == "applied" and "Rendered" in item.note
    d = item.detail
    assert d["captions"] > 0 and d["transcript_source"] == "file" and d["font_file"].startswith("IBMPlexMono")
    assert any(f.attribute == "font" and f.status == "close" for f in res.plan.fallbacks)
    assert "audio" in streams(out)


def test_text_uses_gemini_transcript_when_a_client_is_given(ref_card, vid_beats, tmp_path):
    calls = []

    def backend(contents, model):
        calls.append(model.__name__)
        return json.dumps({"words": [w.model_dump() for w in words(10)]}), (1, 1)

    res = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "o.mp4"),
                      client=GeminiClient(backend=backend, min_interval_s=0))
    assert calls == ["TranscriptResponse"] and res.plan.items[0].detail["transcript_source"] == "gemini"


def test_text_stays_a_plan_when_transcription_fails(ref_card, vid_beats, tmp_path):
    def boom(contents, model):
        raise RuntimeError("down")

    res = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "o.mp4"),
                      client=GeminiClient(backend=boom, min_interval_s=0, max_retries=0))
    assert res.output_path is None and res.plan.items[0].status == "planned" and "not rendered" in res.plan.items[0].note


def test_missing_font_file_is_logged_as_a_fallback(ref_card, vid_beats, tmp_path, monkeypatch):
    sysfont = LoadedFont(None, None, "default", "IBM Plex Mono bold", "Pillow built-in default font", weight_exact=False)
    monkeypatch.setattr(apply_engine, "load_font", lambda *a, **k: sysfont)
    res = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "o.mp4"), transcript=words(8))
    assert res.output_path and any(f.attribute == "font-file" and f.status == "fallback" for f in res.plan.fallbacks)


def test_varied_typography_is_rendered_with_the_dominant_look_and_flagged(ref_card, vid_beats, tmp_path):
    card = ref_card.model_copy(deep=True)
    card.overall.text.variety, card.overall.text.region, card.overall.text.animation = "varied", [], "pop/scale"
    res = apply_style(card, vid_beats, ["text"], str(tmp_path / "o.mp4"), transcript=words(8))
    item = res.plan.items[0]
    kinds = {f.attribute for f in res.plan.fallbacks}
    assert res.output_path and item.status == "fallback" and "dominant" in item.note
    assert {"text", "text-position", "text-animation"} <= kinds


def test_captions_combine_with_cuts_and_color(ref_card, vid_beats, tmp_path):
    out = tmp_path / "o.mp4"
    res = apply_style(ref_card, vid_beats, ["pacing", "color", "text"], str(out), transcript=words(15))
    assert out.exists() and res.plan.cut_times_s and res.match is not None
    assert abs(ffprobe_info(out)["duration_s"] - 10.0) < 0.5


def test_cli_transcript_option(ref_card, vid_static, tmp_path, capsys):
    from scripts.apply_cli import main
    card_path = tmp_path / "r.stylecard.json"
    card_path.write_text(ref_card.model_dump_json())
    tpath = tmp_path / "words.json"
    tpath.write_text(json.dumps([w.model_dump() for w in words(10)]))
    out = tmp_path / "o.mp4"
    rc = main(["--reference", str(card_path), "--target", str(vid_static), "--apply", "text",
               "--transcript", str(tpath), "--out", str(out)])
    assert rc == 0 and out.exists() and "[applied ] text" in capsys.readouterr().out


# --------------------------------------------------------- timing / chunk controls
def test_caption_offset_shifts_every_caption(ref_card, vid_beats, tmp_path):
    base = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "a.mp4"), transcript=words(12))
    early = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "b.mp4"), transcript=words(12),
                        caption_offset_s=-0.2)
    assert base.plan.items[0].detail["caption_offset_s"] == 0.0 and early.plan.items[0].detail["caption_offset_s"] == -0.2
    fa = frame_at(base.output_path, 0.45)          # first word 0.3-0.55 s: captioned without an offset
    fb = frame_at(early.output_path, 0.15)         # with -0.2 s the first caption starts at 0.1 s
    H = fa.shape[0]
    assert np.abs(fa[int(H * 0.32):int(H * 0.48)] - fb[int(H * 0.32):int(H * 0.48)]).mean() < 25   # both show a caption band
    nocap = frame_at(base.output_path, 0.15)
    assert np.abs(nocap[int(H * 0.32):int(H * 0.48)] - fb[int(H * 0.32):int(H * 0.48)]).mean() > 3  # visibly different


def test_words_per_caption_override(ref_card, vid_beats, tmp_path):
    one = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "a.mp4"), transcript=words(12), words_per_caption=1)
    three = apply_style(ref_card, vid_beats, ["text"], str(tmp_path / "b.mp4"), transcript=words(12), words_per_caption=3)
    assert one.plan.items[0].detail["captions"] == 12 and three.plan.items[0].detail["captions"] == 4
    assert three.plan.items[0].detail["words_per_caption_from"] == "override"


def test_gemini_transcript_is_returned_and_saved_by_the_cli(ref_card, vid_beats, tmp_path, monkeypatch, capsys):
    from scripts import apply_cli

    def backend(contents, model):
        return json.dumps({"words": [w.model_dump() for w in words(10)]}), (1, 1)

    monkeypatch.setattr(apply_cli, "GeminiClient", lambda: GeminiClient(backend=backend, min_interval_s=0))
    card_path = tmp_path / "r.stylecard.json"
    card_path.write_text(ref_card.model_dump_json())
    out = tmp_path / "o.mp4"
    rc = apply_cli.main(["--reference", str(card_path), "--target", str(vid_beats), "--apply", "text", "--out", str(out)])
    saved = json.loads(out.with_suffix(".transcript.json").read_text())
    assert rc == 0 and len(saved) == 10 and saved[0]["text"] == "word0" and "transcript " in capsys.readouterr().out
    # the saved file can be fed straight back in, with no Gemini call
    out2 = tmp_path / "o2.mp4"
    assert apply_cli.main(["--reference", str(card_path), "--target", str(vid_beats), "--apply", "text",
                           "--transcript", str(out.with_suffix(".transcript.json")), "--out", str(out2)]) == 0
    assert not out2.with_suffix(".transcript.json").exists()      # user-supplied transcripts are not rewritten


# ------------------------------------------------------------- variable-frame-rate sources
@pytest.fixture(scope="module")
def vid_vfr(tmp_path_factory):
    """12 s source where the second half keeps only every 2nd frame but KEEPS real timestamps (variable frame rate)."""
    from scripts.make_synthetic import make_synthetic
    d = tmp_path_factory.mktemp("vfr")
    base = make_synthetic(d / "base.mp4", duration=12, fps=30, bpm=120, cut_every_beats=1000, speed=0.3, palette="cool", size=(360, 640))
    out = d / "vfr.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(base), "-vf", "select='lt(t,6)+gte(t,6)*not(mod(n,2))'",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-c:a", "copy", str(out)], check=True)
    return out


def rates(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,avg_frame_rate,nb_frames", "-of", "json", str(path)], capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


def test_vfr_detection(vid_vfr, vid_beats):
    from core.captions import is_variable_frame_rate
    assert is_variable_frame_rate(vid_vfr) is True and is_variable_frame_rate(vid_beats) is False


def test_vfr_source_is_not_time_warped_by_the_caption_pass(vid_vfr, tmp_path):
    lf = load_font("IBM Plex Mono", "bold", allow_download=False)
    style = CaptionStyle(size_rel=0.06, box="background-box", y_center=0.4)
    out = render_captions(vid_vfr, [Caption("HELLO", 1.0, 2.0)], style, lf, tmp_path / "o.mp4")
    st = rates(out)
    assert st["r_frame_rate"] == st["avg_frame_rate"]                       # constant frame rate now
    assert abs(int(st["nb_frames"]) - 12 * 30) <= 3                         # real timeline kept (frames duplicated), not 270
    assert abs(ffprobe_info(out)["duration_s"] - 12.0) < 0.3


def test_picture_stays_in_step_with_audio_after_the_caption_pass(vid_vfr, tmp_path):
    """The second half of the source moves; the moving circle must be where it was at the same TIME in the source."""
    lf = load_font("IBM Plex Mono", "bold", allow_download=False)
    out = render_captions(vid_vfr, [], CaptionStyle(size_rel=0.05), lf, tmp_path / "o.mp4")
    for t in (3.0, 8.0, 10.5):
        a, b = frame_at(vid_vfr, t), frame_at(out, t)
        assert np.abs(a - b).mean() < 6.0                                  # same picture at the same time
