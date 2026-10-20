"""Hardening and safety mechanisms for public-mode deployment."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from .utils import AnalysisError, VideoTooLongError, ffprobe_info

ROOT = Path(__file__).resolve().parent.parent
PRIVACY_NOTICE = (
    "🔒 **Privacy Notice**: Uploaded clips and rendered outputs are processed on the server and "
    "automatically deleted within 1 hour. Please do not upload sensitive or private footage."
)


class JobBusyError(AnalysisError):
    """Raised when another heavy render/analysis job is already in progress."""


def is_public_mode() -> bool:
    return os.environ.get("STP_PUBLIC_MODE", "0").strip().lower() in ("1", "true", "yes")


def allow_analysis() -> bool:
    if not is_public_mode():
        return True
    return os.environ.get("STP_ALLOW_ANALYSIS", "0").strip().lower() in ("1", "true", "yes")


def max_clip_seconds() -> float:
    try:
        val = float(os.environ.get("STP_MAX_CLIP_S", "30.0").strip())
        return val if val > 0 else 30.0
    except ValueError:
        return 30.0


def max_upload_bytes() -> int:
    try:
        mb = float(os.environ.get("STP_MAX_UPLOAD_MB", "50").strip())
        return int(mb * 1024 * 1024) if mb > 0 else 50 * 1024 * 1024
    except ValueError:
        return 50 * 1024 * 1024


def apply_timeout_seconds() -> float:
    try:
        val = float(os.environ.get("STP_APPLY_TIMEOUT_S", "120.0").strip())
        return val if val > 0 else 120.0
    except ValueError:
        return 120.0


def cleanup_old_files(out_dir: Path, max_age_s: float = 3600.0, clock: Optional[Callable[[], float]] = None) -> list[Path]:
    """Delete uploads and renders older than max_age_s. Returns list of removed paths."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    now = clock() if clock is not None else time.time()
    deleted: list[Path] = []

    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        # Skip lock files, hidden system files
        if path.name.startswith(".") or path.suffix in (".lock", ".toml"):
            continue
        try:
            mtime = path.stat().st_mtime
            if (now - mtime) > max_age_s:
                path.unlink()
                deleted.append(path)
        except OSError:
            pass

    return deleted


@contextmanager
def heavy_job_semaphore(lock_path: Path) -> Iterator[None]:
    """Ensure only one heavy job (analysis or rendering) runs at a time."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    acquired = False

    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as e:
                raise JobBusyError(
                    "The server is currently busy processing another video. Please wait a minute and try again."
                ) from e
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError as e:
                raise JobBusyError(
                    "The server is currently busy processing another video. Please wait a minute and try again."
                ) from e
        yield
    finally:
        if acquired:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        f.close()


def validate_clip_duration(clip_path: Path, max_s: Optional[float] = None) -> float:
    """Check clip length against public limit before heavy work begins."""
    limit = max_s if max_s is not None else max_clip_seconds()
    info = ffprobe_info(clip_path)
    dur = float(info["duration_s"])
    if is_public_mode() and dur > limit:
        raise VideoTooLongError(
            f"Clip duration ({dur:.1f} s) exceeds the {limit:.0f} s limit for the public demo. "
            f"Please choose or upload a shorter clip."
        )
    return dur


def save_upload_safe(upload, out_dir: Path, subdir: str = "uploads") -> Path:
    """Save upload safely inside out_dir / subdir, enforcing size limits."""
    out_dir = Path(out_dir).resolve()
    max_b = max_upload_bytes()
    if is_public_mode() and upload.size > max_b:
        raise ValueError(
            f"File '{upload.name}' is {upload.size / (1024 * 1024):.1f} MB, which exceeds the public limit "
            f"of {max_b / (1024 * 1024):.0f} MB."
        )

    clean_filename = Path(upload.name).name
    # Strip dangerous directory components
    clean_filename = "".join(c for c in clean_filename if c.isalnum() or c in ".-_")
    if not clean_filename:
        clean_filename = "upload.mp4"

    dest_dir = (out_dir / subdir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = (dest_dir / clean_filename).resolve()

    # Verify destination is strictly under out_dir
    if not str(dest_path).startswith(str(out_dir)):
        raise ValueError("Invalid upload path: attempt to write outside output directory.")

    if not dest_path.exists() or dest_path.stat().st_size != upload.size:
        dest_path.write_bytes(upload.getbuffer())

    return dest_path
