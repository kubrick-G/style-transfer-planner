import numpy as np
import soundfile as sf

from core.audio import SR, analyze_audio, beat_sync_score
from core.frames import FrameReader
from core.motion import analyze_motion
from core.scenes import SceneSpan, detect_scenes, pacing_curve, pacing_stats, scene_energy
from core.utils import ffprobe_info
from scripts.make_synthetic import _audio


def test_beat_sync_score_cases():
    beats = [i * 0.5 for i in range(20)]
    assert beat_sync_score([1.0, 2.0, 3.0], beats) == 1.0
    assert beat_sync_score([1.25, 2.25, 3.25], beats) == 0.0
    assert beat_sync_score([1.0, 1.25], beats) == 0.5
    assert beat_sync_score([0.0], beats) is None          # a cut at t=0 is not a cut
    assert beat_sync_score([1.0], []) is None             # no beats -> unmeasurable
    assert beat_sync_score([1.0, 2.0], beats, tol=0.0) == 1.0


def test_analyze_audio_finds_tempo(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), _audio(12, 120, 0), SR)
    res = analyze_audio(wav)
    assert res.bpm is not None and abs(res.bpm - 120) / 120 < 0.05
    assert len(res.beat_times) >= 15 and res.energy_curve and max(res.energy_curve) == 1.0


def test_analyze_audio_silence(tmp_path):
    wav = tmp_path / "s.wav"
    sf.write(str(wav), np.zeros(SR * 3, dtype=np.float32), SR)
    res = analyze_audio(wav)
    assert res.bpm is None and res.beat_times == [] and res.energy == "low"


def test_scene_detection_matches_known_cuts(vid_beats):
    info = ffprobe_info(vid_beats)
    scenes = detect_scenes(vid_beats, info["duration_s"], info["fps"])
    assert len(scenes) == 10                                  # cuts at 1..9 s
    assert scenes[0].role == "hook" and scenes[-1].role == "end" and scenes[4].role == "body"
    for i, s in enumerate(scenes[1:], start=1):
        assert abs(s.start_s - i * 1.0) < 0.1
    stats = pacing_stats(scenes, info["duration_s"])
    assert abs(stats["avg_shot_len_s"] - 1.0) < 0.05
    assert abs(stats["cuts_per_min"] - 54) < 3                # 9 cuts / 10 s


def test_single_scene_when_no_cuts(vid_static):
    info = ffprobe_info(vid_static)
    scenes = detect_scenes(vid_static, info["duration_s"], info["fps"])
    assert len(scenes) == 1 and scenes[0].role == "hook"
    assert pacing_stats(scenes, info["duration_s"])["cuts_per_min"] == 0


def test_static_video_has_no_motion(vid_static):
    with FrameReader(vid_static) as r:
        res = analyze_motion(r, [SceneSpan(0, 0.0, 6.0, "hook")])
    assert res[0].hint == "static" and res[0].magnitude < 0.05


def test_pacing_curve_shapes():
    assert pacing_curve([1, 2, 3, 4, 5, 6, 7, 8, 9], 10) == "steady"
    assert pacing_curve([0.5, 1, 1.5, 2, 2.5, 3, 8], 10) == "fast-start-then-slower"
    assert pacing_curve([7, 7.5, 8, 8.5, 9, 9.5, 1], 10) == "slow-start-then-faster"
    assert pacing_curve([1, 2], 10) == "too-few-cuts"


def test_scene_energy_ordering():
    hi, hl = scene_energy(0.5, 0.9, 0.9)
    lo, ll = scene_energy(6.0, 0.0, 0.1)
    assert hi > lo and hl == "high" and ll == "low"
    assert scene_energy(1.0, 0.5, None)[0] > 0                # works without audio
