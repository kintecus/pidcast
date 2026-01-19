"""Unit tests for ProcessingHistory."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pidcast.history import HistoryEntry, ProcessingHistory, ProcessingStatus


def test_processing_status_enum():
    """Test ProcessingStatus enum values."""
    assert ProcessingStatus.SUCCESS.value == "success"
    assert ProcessingStatus.FAILED.value == "failed"
    assert ProcessingStatus.IN_PROGRESS.value == "in_progress"


def test_history_entry_to_dict():
    """Test HistoryEntry serialization."""
    now = datetime.now()
    entry = HistoryEntry(
        guid="test-guid",
        show_id=1,
        episode_title="Test Episode",
        processed_at=now,
        status=ProcessingStatus.SUCCESS,
        output_file="/path/to/file.md",
        error_message=None,
    )

    data = entry.to_dict()
    assert data["guid"] == "test-guid"
    assert data["show_id"] == 1
    assert data["episode_title"] == "Test Episode"
    assert data["processed_at"] == now.isoformat()
    assert data["status"] == "success"
    assert data["output_file"] == "/path/to/file.md"
    assert data["error_message"] is None


def test_history_entry_from_dict():
    """Test HistoryEntry deserialization."""
    now = datetime.now()
    data = {
        "guid": "test-guid",
        "show_id": 1,
        "episode_title": "Test Episode",
        "processed_at": now.isoformat(),
        "status": "success",
        "output_file": "/path/to/file.md",
        "error_message": None,
    }

    entry = HistoryEntry.from_dict(data)
    assert entry.guid == "test-guid"
    assert entry.show_id == 1
    assert entry.episode_title == "Test Episode"
    assert entry.processed_at == now
    assert entry.status == ProcessingStatus.SUCCESS
    assert entry.output_file == "/path/to/file.md"
    assert entry.error_message is None


def test_processing_history_init_creates_empty():
    """Test ProcessingHistory initialization with non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        assert history.history_path == history_path
        assert len(history.entries) == 0


def test_processing_history_is_processed():
    """Test is_processed method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        # Not processed yet
        assert not history.is_processed("test-guid")

        # Mark as in progress - should not be considered processed
        history.mark_in_progress("test-guid", 1, "Test Episode")
        assert not history.is_processed("test-guid")

        # Mark as success - should be processed
        history.mark_success("test-guid", "/path/to/file.md")
        assert history.is_processed("test-guid")

        # Mark as failed - should not be considered processed
        history.mark_failed("test-guid", "Some error")
        assert not history.is_processed("test-guid")


def test_processing_history_mark_in_progress():
    """Test mark_in_progress method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        history.mark_in_progress("test-guid", 1, "Test Episode")

        assert "test-guid" in history.entries
        entry = history.entries["test-guid"]
        assert entry.guid == "test-guid"
        assert entry.show_id == 1
        assert entry.episode_title == "Test Episode"
        assert entry.status == ProcessingStatus.IN_PROGRESS
        assert entry.output_file is None
        assert entry.error_message is None

        # Check file was saved
        assert history_path.exists()


def test_processing_history_mark_success():
    """Test mark_success method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        # First mark as in progress
        history.mark_in_progress("test-guid", 1, "Test Episode")

        # Then mark as success
        history.mark_success("test-guid", "/path/to/file.md")

        entry = history.entries["test-guid"]
        assert entry.status == ProcessingStatus.SUCCESS
        assert entry.output_file == "/path/to/file.md"
        assert entry.error_message is None


def test_processing_history_mark_failed():
    """Test mark_failed method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        # First mark as in progress
        history.mark_in_progress("test-guid", 1, "Test Episode")

        # Then mark as failed
        history.mark_failed("test-guid", "Network error")

        entry = history.entries["test-guid"]
        assert entry.status == ProcessingStatus.FAILED
        assert entry.error_message == "Network error"


def test_processing_history_get_failed_episodes():
    """Test get_failed_episodes method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        # Create some entries
        history.mark_in_progress("guid-1", 1, "Episode 1")
        history.mark_success("guid-1", "/path/to/file1.md")

        history.mark_in_progress("guid-2", 1, "Episode 2")
        history.mark_failed("guid-2", "Error 1")

        history.mark_in_progress("guid-3", 1, "Episode 3")
        history.mark_failed("guid-3", "Error 2")

        # Get failed episodes
        failed = history.get_failed_episodes()
        assert len(failed) == 2
        assert all(e.status == ProcessingStatus.FAILED for e in failed)
        assert {e.guid for e in failed} == {"guid-2", "guid-3"}


def test_processing_history_persistence():
    """Test that history persists across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"

        # Create first instance and add entry
        history1 = ProcessingHistory(history_path)
        history1.mark_in_progress("test-guid", 1, "Test Episode")
        history1.mark_success("test-guid", "/path/to/file.md")

        # Create second instance and check entry exists
        history2 = ProcessingHistory(history_path)
        assert "test-guid" in history2.entries
        assert history2.is_processed("test-guid")
        entry = history2.entries["test-guid"]
        assert entry.episode_title == "Test Episode"
        assert entry.output_file == "/path/to/file.md"


def test_processing_history_corrupted_file():
    """Test handling of corrupted history file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"

        # Create corrupted JSON file
        with open(history_path, "w", encoding="utf-8") as f:
            f.write("{invalid json")

        # Should not raise, should create backup and start fresh
        history = ProcessingHistory(history_path)
        assert len(history.entries) == 0

        # Backup should exist
        backup_path = history_path.with_suffix(".json.bak")
        assert backup_path.exists()


def test_processing_history_atomic_save():
    """Test that saves are atomic (using temp file)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history = ProcessingHistory(history_path)

        history.mark_in_progress("test-guid", 1, "Test Episode")

        # Verify file was written atomically
        assert history_path.exists()
        with open(history_path, encoding="utf-8") as f:
            data = json.load(f)
            assert "test-guid" in data
