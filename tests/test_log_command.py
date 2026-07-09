"""Unit tests for ``pidcast log`` rendering.

Run dicts in the unified store are not uniformly shaped (legacy/pre-migration
entries can be as sparse as 5 keys), so these tests assert the command
tolerates missing fields instead of crashing on real, already-existing data.
"""

import argparse

import pytest

from pidcast.commands.log import _format_date, _format_status, _row, cmd_log

FULL_ENTRY = {
    "run_uid": "uid-1",
    "run_timestamp": "2026-06-30T10:00:00",
    "video_title": "A Full Episode",
    "video_url": "https://youtube.com/watch?v=abc123",
    "run_duration": 12.0,
    "transcription_duration": 8.0,
    "audio_duration": 3000.0,
    "success": True,
    "transcript_path": "/data/transcripts/2026-06-30_A_Full_Episode.md",
    "transcription_provider": "whisper",
}

SPARSE_LEGACY_ENTRY = {
    "run_uid": "uid-2",
    "run_timestamp": "2025-01-15T09:00:00",
    "video_title": "Legacy Episode",
    "video_url": "https://youtube.com/watch?v=legacy",
    "run_duration": 5.0,
}

FAILED_ENTRY = {
    "run_uid": "uid-3",
    "run_timestamp": "2026-07-01T08:00:00",
    "video_title": "Broken Episode",
    "video_url": "https://youtube.com/watch?v=broken",
    "run_duration": 1.0,
    "success": False,
    "transcript_path": None,
    "transcription_provider": "whisper",
}


def test_row_full_entry_formats_all_columns():
    date, title, provider, duration, status, transcript = _row(FULL_ENTRY)
    assert date == "2026-06-30 10:00"
    assert title == "A Full Episode"
    assert provider == "whisper"
    assert duration == "50m 0s"
    assert status == "✓"
    assert transcript == "2026-06-30_A_Full_Episode.md"


def test_row_sparse_legacy_entry_does_not_raise():
    date, title, provider, duration, status, transcript = _row(SPARSE_LEGACY_ENTRY)
    assert date == "2025-01-15 09:00"
    assert title == "Legacy Episode"
    assert provider == "-"
    assert duration == "-"
    assert status == "?"
    assert transcript == "-"


def test_row_failed_entry_does_not_raise_on_none_transcript_path():
    date, title, provider, duration, status, transcript = _row(FAILED_ENTRY)
    assert status == "✗"
    assert transcript == "-"


def test_format_status_tri_state():
    assert _format_status(True) == "✓"
    assert _format_status(False) == "✗"
    assert _format_status(None) == "?"


def test_format_date_falls_back_on_bad_timestamp():
    assert _format_date("not-a-timestamp") == "not-a-timestamp"
    assert _format_date(None) == "-"


@pytest.fixture
def runs_file(tmp_path):
    import json

    path = tmp_path / "runs.json"
    data = {"by_guid": {}, "runs": [SPARSE_LEGACY_ENTRY, FULL_ENTRY, FAILED_ENTRY]}
    path.write_text(json.dumps(data))
    return path


def test_cmd_log_handles_mixed_shapes_without_raising(runs_file, monkeypatch, capsys):
    monkeypatch.setattr("pidcast.config.RUNS_FILE", runs_file)
    args = argparse.Namespace(limit=10)

    cmd_log(args)

    # Rich wraps long titles across lines, so assert on unwrapped substrings.
    out = capsys.readouterr().out.replace("\n", "")
    assert "Full" in out and "Episode" in out
    assert "Legacy" in out
    assert "Broken" in out


def test_cmd_log_respects_limit_and_orders_newest_first(runs_file, monkeypatch, capsys):
    monkeypatch.setattr("pidcast.config.RUNS_FILE", runs_file)
    args = argparse.Namespace(limit=1)

    cmd_log(args)

    out = capsys.readouterr().out.replace("\n", "")
    assert "Broken" in out
    assert "Full" not in out
    assert "Legacy" not in out


def test_cmd_log_empty_store(tmp_path, monkeypatch, capsys):
    empty_path = tmp_path / "empty_runs.json"
    monkeypatch.setattr("pidcast.config.RUNS_FILE", empty_path)
    args = argparse.Namespace(limit=10)

    cmd_log(args)

    out = capsys.readouterr().out
    assert "No run history yet" in out
