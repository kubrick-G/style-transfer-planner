import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.make_synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def vid_beats(tmp_path_factory):
    """10 s, 120 BPM, a cut every 2 beats (1.0 s) -> 9 cuts, 10 scenes, cuts exactly on beats."""
    return make_synthetic(tmp_path_factory.mktemp("v") / "beats.mp4", duration=10, bpm=120, cut_every_beats=2,
                          palette="warm", text="HELLO THERE FRIEND", speed=1.0)


@pytest.fixture(scope="session")
def vid_static(tmp_path_factory):
    """6 s, no audio, no cuts, no motion at all."""
    return make_synthetic(tmp_path_factory.mktemp("v") / "static.mp4", duration=6, audio=False,
                          cut_every_beats=1000, speed=0.0, palette="cool")


@pytest.fixture(scope="session")
def vid_caps(tmp_path_factory):
    """8 s, no audio, no cuts, static background, caption swaps every 0.5 s at the bottom (ground truth ~2 events/s)."""
    return make_synthetic(tmp_path_factory.mktemp("v") / "caps.mp4", duration=8, audio=False,
                          cut_every_beats=1000, speed=0.0, palette="cool", caption_every_s=0.5)
