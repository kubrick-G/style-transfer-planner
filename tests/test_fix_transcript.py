import json

import pytest

from core.transcript import TranscriptWord
from scripts.fix_transcript import format_list, main, settle, shift_by_index, shift_by_time


def ws():
    return [TranscriptWord(text=f"w{i}", start_s=1.0 + i * 0.5, end_s=1.0 + i * 0.5 + 0.4) for i in range(10)]


def test_shift_by_time_moves_only_the_run():
    out, n = shift_by_time(ws(), 3.0, 4.0, -0.3)               # words starting at 3.0, 3.5, 4.0 (w4, w5, w6)
    assert n == 3
    assert [round(w.start_s, 2) for w in out[4:7]] == [2.7, 3.2, 3.7]
    assert out[3].start_s == 2.5 and out[7].start_s == 4.5     # neighbours untouched


def test_shift_by_index_and_reversed_bounds():
    out, n = shift_by_index(ws(), 6, 4, -0.25)                 # reversed bounds are accepted
    assert n == 3 and out[4].start_s == pytest.approx(2.75) and out[3].start_s == 2.5 and out[7].start_s == 4.5


def test_shift_never_goes_negative_and_keeps_end_after_start():
    out, _ = shift_by_index(ws(), 0, 0, -5.0)
    assert out[0].start_s == 0.0 and out[0].end_s > out[0].start_s


def test_settle_stops_overlaps_after_an_early_shift():
    out, _ = shift_by_index(ws(), 4, 6, -0.45)                 # w4 now starts before w3 ends
    fixed = settle(out)
    for a, b in zip(fixed, fixed[1:]):
        assert a.end_s <= b.start_s + 1e-9 and a.end_s > a.start_s


def test_list_format_shows_index_times_and_word():
    lines = format_list(ws()).splitlines()
    assert "idx" in lines[0] and len(lines) == 11
    row = next(l for l in lines if l.split()[-1] == "w3")
    assert row.split()[:3] == ["3", "2.50", "2.90"]


def test_cli_list_shift_and_output(tmp_path, capsys):
    p = tmp_path / "x.transcript.json"
    p.write_text(json.dumps([w.model_dump() for w in ws()]))
    assert main([str(p), "--list"]) == 0 and "w9" in capsys.readouterr().out
    assert main([str(p), "--shift-time", "3.0", "4.0", "-0.3"]) == 0
    fixed = tmp_path / "x.transcript.fixed.json"
    data = json.loads(fixed.read_text())
    assert fixed.exists() and data[4]["start_s"] == pytest.approx(2.7) and data[3]["start_s"] == 2.5
    assert json.loads(p.read_text())[4]["start_s"] == 3.0       # original untouched by default
    # a second fix builds on the fixed file without piling up '.fixed.fixed'
    assert main([str(fixed), "--shift-words", "8", "9", "0.1"]) == 0
    assert (tmp_path / "x.transcript.fixed.json").exists() and not list(tmp_path.glob("*.fixed.fixed*"))


def test_cli_in_place_and_errors(tmp_path, capsys):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"words": [w.model_dump() for w in ws()]}))        # {"words": [...]} shape also accepted
    assert main([str(p), "--shift-words", "0", "1", "0.2", "--in-place"]) == 0
    assert json.loads(p.read_text())[0]["start_s"] == pytest.approx(1.2)
    assert main([str(tmp_path / "missing.json"), "--list"]) == 2
    assert main([str(p), "--shift-time", "50", "60", "-0.3"]) == 1               # nothing matched -> nothing written
    assert main([str(p), "--shift-words", "a", "b", "0.1"]) == 2
