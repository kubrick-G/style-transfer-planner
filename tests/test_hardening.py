from dataclasses import dataclass
from pathlib import Path
import pytest

from core.hardening import (
    JobBusyError,
    PRIVACY_NOTICE,
    allow_analysis,
    apply_timeout_seconds,
    cleanup_old_files,
    heavy_job_semaphore,
    is_public_mode,
    max_clip_seconds,
    max_upload_bytes,
    save_upload_safe,
    validate_clip_duration,
)
from core.utils import VideoTooLongError


@dataclass
class FakeUpload:
    name: str
    size: int
    _buffer: bytes

    def getbuffer(self):
        return self._buffer


def test_public_mode_flags(monkeypatch):
    monkeypatch.delenv("STP_PUBLIC_MODE", raising=False)
    monkeypatch.delenv("STP_ALLOW_ANALYSIS", raising=False)
    assert not is_public_mode()
    assert allow_analysis()

    # In public mode, analysis defaults to enabled so users can upload their own reference
    monkeypatch.setenv("STP_PUBLIC_MODE", "1")
    assert is_public_mode()
    assert allow_analysis()

    # Explicitly disabling analysis
    monkeypatch.setenv("STP_ALLOW_ANALYSIS", "0")
    assert not allow_analysis()

    monkeypatch.setenv("STP_ALLOW_ANALYSIS", "1")
    assert allow_analysis()

    monkeypatch.setenv("STP_MAX_CLIP_S", "20")
    assert max_clip_seconds() == 20.0

    monkeypatch.setenv("STP_MAX_UPLOAD_MB", "10")
    assert max_upload_bytes() == 10 * 1024 * 1024


def test_cleanup_old_files(tmp_path):
    out_dir = tmp_path / "out"
    uploads = out_dir / "uploads"
    uploads.mkdir(parents=True)

    old_file = uploads / "old.mp4"
    new_file = uploads / "new.mp4"
    lock_file = out_dir / ".job.lock"

    old_file.write_text("old")
    new_file.write_text("new")
    lock_file.write_text("lock")

    t_base = 10000.0
    import os
    os.utime(old_file, (t_base - 4000, t_base - 4000))
    os.utime(new_file, (t_base - 1000, t_base - 1000))

    deleted = cleanup_old_files(out_dir, max_age_s=3600.0, clock=lambda: t_base)
    assert old_file in deleted
    assert not old_file.exists()
    assert new_file.exists()
    assert lock_file.exists()  # lock files preserved


def test_heavy_job_semaphore_busy(tmp_path):
    lock_file = tmp_path / ".heavy_job.lock"

    with heavy_job_semaphore(lock_file):
        with pytest.raises(JobBusyError, match="currently busy processing another video"):
            with heavy_job_semaphore(lock_file):
                pass


def test_save_upload_safe_and_path_confinement(tmp_path, monkeypatch):
    monkeypatch.setenv("STP_PUBLIC_MODE", "1")
    monkeypatch.setenv("STP_MAX_UPLOAD_MB", "2")
    out_dir = tmp_path / "out"

    # Too large
    big_up = FakeUpload(name="big.mp4", size=3 * 1024 * 1024, _buffer=b"x" * 100)
    with pytest.raises(ValueError, match="exceeds the public limit"):
        save_upload_safe(big_up, out_dir)

    # Path traversal attempt in name
    traversal_up = FakeUpload(name="../../../evil.mp4", size=100, _buffer=b"clean")
    saved = save_upload_safe(traversal_up, out_dir)
    assert saved.exists()
    assert saved.parent == (out_dir / "uploads").resolve()
    assert str(saved).startswith(str(out_dir.resolve()))


def test_validate_clip_duration_public_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("STP_PUBLIC_MODE", "1")
    monkeypatch.setenv("STP_MAX_CLIP_S", "10")

    clip = tmp_path / "dummy.mp4"
    clip.write_text("")

    # Mock ffprobe_info returning 15s
    monkeypatch.setattr("core.hardening.ffprobe_info", lambda p: {"duration_s": 15.0})
    with pytest.raises(VideoTooLongError, match="exceeds the 10 s limit"):
        validate_clip_duration(clip)

    # When public mode off, allowed
    monkeypatch.setenv("STP_PUBLIC_MODE", "0")
    dur = validate_clip_duration(clip)
    assert dur == 15.0
