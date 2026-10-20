import pytest
from pathlib import Path

from scripts.export_demo_assets import export_asset, MAX_BYTES
from core.schema import StyleCard


def test_export_asset_validates_file_size(tmp_path):
    big_file = tmp_path / "big.mp4"
    big_file.write_bytes(b"x" * (MAX_BYTES + 1024))
    dest_dir = tmp_path / "dest"

    with pytest.raises(ValueError, match="exceeds the 5 MB limit"):
        export_asset(big_file, dest_dir, force=False)

    # With force=True it succeeds
    exported = export_asset(big_file, dest_dir, force=True)
    assert exported.exists()
    assert exported.stat().st_size == MAX_BYTES + 1024


def test_export_asset_validates_style_card(tmp_path):
    invalid_card = tmp_path / "invalid.stylecard.json"
    invalid_card.write_text('{"not_a_card": true}')
    dest_dir = tmp_path / "dest"

    with pytest.raises(Exception):
        export_asset(invalid_card, dest_dir, is_card=True)
