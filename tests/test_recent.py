"""Tests for the recent-transcripts picker behind `pidcast discord|notes` without a path."""

import os

import pytest

from pidcast import cli
from pidcast.exceptions import PidcastError
from pidcast.recent import list_recent_transcripts, pick_transcript
from pidcast.ui import Chooser


def _touch(path, mtime):
    path.write_text("x")
    os.utime(path, (mtime, mtime))


def test_newest_first_and_limited(tmp_path):
    for i, name in enumerate(["a.md", "b.md", "c.md"]):
        _touch(tmp_path / name, 1000 + i)
    assert [p.name for p in list_recent_transcripts(tmp_path, 2)] == ["c.md", "b.md"]


def test_derived_artifacts_and_sidecars_excluded(tmp_path):
    _touch(tmp_path / "ep.md", 1000)
    for name in (
        "ep.discord-play.md",
        "ep.notes-deepdive.md",
        "ep_analysis_summary.md",
        "2026-10-01_podcast-digest.md",
        "ep.whisper.json",
    ):
        _touch(tmp_path / name, 2000)
    assert [p.name for p in list_recent_transcripts(tmp_path)] == ["ep.md"]


def test_missing_dir_is_empty(tmp_path):
    assert list_recent_transcripts(tmp_path / "nope") == []


def test_pick_without_tty_errors(tmp_path, monkeypatch):
    _touch(tmp_path / "ep.md", 1000)
    monkeypatch.setattr("pidcast.recent.resolve_output_dir", lambda _a: tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(PidcastError, match="needs a TTY"):
        pick_transcript(None)


def test_pick_empty_dir_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("pidcast.recent.resolve_output_dir", lambda _a: tmp_path)
    with pytest.raises(PidcastError, match="No transcripts"):
        pick_transcript(None)


def test_chooser_enter_takes_first_and_digit_jumps():
    assert Chooser(3).press("\r") == "done"
    state = Chooser(3)
    assert state.press("3") == "done"
    assert state.cursor == 2


def test_chooser_wraps_and_quits():
    state = Chooser(2)
    state.press("\x1b[A")
    assert state.cursor == 1
    assert state.press("q") == "quit"


@pytest.mark.parametrize("verb", ["discord", "notes"])
def test_verbs_accept_no_path(verb):
    args = cli.parse_arguments([verb, "-n", "3"])
    assert args.transcript is None
    assert args.recent == 3
