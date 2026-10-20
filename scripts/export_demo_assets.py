"""Export chosen style cards, video clips, and transcripts into data/demo/.

Refuses files > 5 MB unless --force is given.
Validates style cards before copying.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.schema import StyleCard

MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def export_asset(src: Path, dest_dir: Path, force: bool = False, is_card: bool = False) -> Path:
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    size = src.stat().st_size
    if size > MAX_BYTES and not force:
        raise ValueError(
            f"File '{src.name}' is {size / (1024 * 1024):.2f} MB, which exceeds the 5 MB limit. "
            f"Use --force to override if strictly necessary."
        )
    if is_card:
        # Validate that it is a parseable StyleCard
        card = StyleCard.model_validate_json(src.read_text(encoding="utf-8"))
        if card.schema_version is None:
            raise ValueError(f"Invalid style card: {src}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Export demo assets to data/demo/")
    parser.add_argument("--cards", nargs="*", default=[], help="Paths to *.stylecard.json files")
    parser.add_argument("--clips", nargs="*", default=[], help="Paths to video clip files (.mp4, etc.)")
    parser.add_argument("--transcripts", nargs="*", default=[], help="Paths to transcript JSON files")
    parser.add_argument("--force", action="store_true", help="Allow files > 5 MB")
    args = parser.parse_args()

    demo_root = ROOT / "data" / "demo"
    cards_dir = demo_root / "cards"
    clips_dir = demo_root / "clips"
    trans_dir = demo_root / "transcripts"

    for c in args.cards:
        out = export_asset(Path(c), cards_dir, force=args.force, is_card=True)
        print(f"Exported card: {out.name} ({out.stat().st_size} bytes)")

    for v in args.clips:
        out = export_asset(Path(v), clips_dir, force=args.force, is_card=False)
        print(f"Exported clip: {out.name} ({out.stat().st_size} bytes)")

    for t in args.transcripts:
        out = export_asset(Path(t), trans_dir, force=args.force, is_card=False)
        print(f"Exported transcript: {out.name} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
