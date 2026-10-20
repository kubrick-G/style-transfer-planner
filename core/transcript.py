"""Speech transcript with word timestamps, from Gemini (audio) or a user-supplied JSON file."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from . import prompts
from .gemini_client import GeminiClient, Media
from .utils import run, temp_workdir

MAX_WORD_S = 3.0          # a single spoken word longer than this is almost certainly a timestamp error


class TranscriptWord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    start_s: float
    end_s: float


class TranscriptResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    words: list[TranscriptWord]


def clean_words(words: list[TranscriptWord], duration_s: float) -> list[TranscriptWord]:
    """Drop empty/invalid words, clamp to the video, sort, and stop words overlapping the next one."""
    out = []
    for w in words:
        t = w.text.strip()
        if not t or w.end_s <= w.start_s or w.start_s >= duration_s or w.start_s < 0 - 1e-6:
            continue
        end = min(w.end_s, duration_s, w.start_s + MAX_WORD_S)
        out.append(TranscriptWord(text=t, start_s=max(0.0, w.start_s), end_s=end))
    out.sort(key=lambda w: w.start_s)
    for a, b in zip(out, out[1:]):
        if a.end_s > b.start_s:
            a.end_s = max(a.start_s + 0.05, b.start_s)
    return out


def load_words_json(path) -> list[TranscriptWord]:
    """Accepts a bare list or {"words": [...]} of {text,start_s,end_s}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["words"] if isinstance(data, dict) else data
    return [TranscriptWord(**i) for i in items]


def transcribe(client: Optional[GeminiClient], video_path, duration_s: float) -> Optional[list[TranscriptWord]]:
    """Transcribe with word timestamps. Returns None if impossible (no client, no audio, failure, empty)."""
    if client is None or not client.available:
        return None
    with temp_workdir() as tmp:
        wav = tmp / "speech.wav"
        try:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(wav)], timeout=120)
        except Exception:
            return None
        data = wav.read_bytes()
    resp = client.generate_json("transcript", [prompts.TRANSCRIPT.format(duration=round(duration_s, 1)),
                                               Media(data, "audio/wav")], TranscriptResponse)
    if resp is None:
        return None
    words = clean_words(resp.words, duration_s)
    return words or None
