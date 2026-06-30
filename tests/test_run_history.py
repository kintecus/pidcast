"""Unit tests for the unified RunHistory store.

RunHistory replaces the two legacy stores (history.json keyed by GUID, and
transcription_stats.json - an append-only list keyed by source_id) with one
file shaped ``{"by_guid": {...}, "runs": [...]}``. It must serve BOTH lookup
patterns: guid-keyed sync/digest reads and source_id-keyed dedup/estimate reads.
"""

import json
from datetime import datetime

import pytest

from pidcast.config import TranscriptionStats
from pidcast.history import ProcessingStatus, RunHistory, RunRecord


def _make_stats(**overrides) -> TranscriptionStats:
    """Build a minimal successful TranscriptionStats for the main workflow path."""
    base = {
        "run_uid": "uid-1",
        "run_timestamp": "2026-06-30T10:00:00",
        "video_title": "Some Episode",
        "smart_filename": "2026-06-30_Some_Episode.md",
        "video_url": "https://youtube.com/watch?v=abc123",
        "run_duration": 12.0,
        "transcription_duration": 8.0,
        "audio_duration": 3000.0,
        "success": True,
        "source_id": "yt:abc123",
        "transcript_path": "/data/transcripts/2026-06-30_Some_Episode.md",
    }
    base.update(overrides)
    return TranscriptionStats(**base)


# ---------------------------------------------------------------------------
# On-disk shape
# ---------------------------------------------------------------------------


def test_runs_file_has_two_sections(tmp_path):
    """The store persists a dict with by_guid and runs sections."""
    path = tmp_path / "runs.json"
    history = RunHistory(path)
    history.mark_in_progress("guid-1", 1, "Ep 1")
    history.mark_success("guid-1", "/t/ep1.md")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert set(data.keys()) == {"by_guid", "runs"}
    assert "guid-1" in data["by_guid"]
    assert isinstance(data["runs"], list)


# ---------------------------------------------------------------------------
# Sync / library path - guid-keyed (parity with old ProcessingHistory)
# ---------------------------------------------------------------------------


def test_guid_lifecycle(tmp_path):
    history = RunHistory(tmp_path / "runs.json")
    assert not history.is_processed("g1")

    history.mark_in_progress("g1", 1, "Ep")
    assert not history.is_processed("g1")

    history.mark_success("g1", "/t/ep.md")
    assert history.is_processed("g1")

    history.mark_failed("g1", "boom")
    assert not history.is_processed("g1")


def test_get_failed_episodes(tmp_path):
    history = RunHistory(tmp_path / "runs.json")
    history.mark_in_progress("g1", 1, "E1")
    history.mark_success("g1", "/t/e1.md")
    history.mark_in_progress("g2", 1, "E2")
    history.mark_failed("g2", "err")
    failed = history.get_failed_episodes()
    assert {e.guid for e in failed} == {"g2"}


def test_guid_records_persist_across_instances(tmp_path):
    path = tmp_path / "runs.json"
    h1 = RunHistory(path)
    h1.mark_in_progress("g1", 2, "Ep")
    h1.mark_success("g1", "/t/ep.md")

    h2 = RunHistory(path)
    assert h2.is_processed("g1")
    assert h2.entries["g1"].output_file == "/t/ep.md"
    assert h2.entries["g1"].show_id == 2


# ---------------------------------------------------------------------------
# Main workflow path - source_id-keyed dedup + estimation
# ---------------------------------------------------------------------------


def test_record_run_then_find_by_source(tmp_path):
    history = RunHistory(tmp_path / "runs.json")
    history.record_run(_make_stats())
    rec = history.find_by_source("yt:abc123")
    assert rec is not None
    assert rec.success is True
    assert rec.transcript_path == "/data/transcripts/2026-06-30_Some_Episode.md"
    # A source-only record has no guid.
    assert rec.guid is None


def test_record_run_persists_as_list(tmp_path):
    path = tmp_path / "runs.json"
    history = RunHistory(path)
    history.record_run(_make_stats(run_uid="a", source_id="yt:1"))
    history.record_run(_make_stats(run_uid="b", source_id="yt:2"))

    reloaded = RunHistory(path)
    assert reloaded.find_by_source("yt:1") is not None
    assert reloaded.find_by_source("yt:2") is not None


def test_get_runs_for_estimation_returns_list_of_dicts(tmp_path):
    """estimate_transcription_time consumes a list of raw stat dicts."""
    history = RunHistory(tmp_path / "runs.json")
    history.record_run(_make_stats())
    runs = history.get_runs_for_estimation()
    assert isinstance(runs, list)
    assert runs and isinstance(runs[0], dict)
    # The dict carries the fields the estimator reads.
    assert runs[0]["success"] is True
    assert runs[0]["audio_duration"] == 3000.0
    assert runs[0]["transcription_duration"] == 8.0


def test_dual_index_source_only_record_not_in_by_guid(tmp_path):
    path = tmp_path / "runs.json"
    history = RunHistory(path)
    history.record_run(_make_stats())
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["by_guid"] == {}
    assert len(data["runs"]) == 1


# ---------------------------------------------------------------------------
# Phantom pruning (dedup hygiene) - moved from utils._is_phantom
# ---------------------------------------------------------------------------


def test_phantom_records_flags_missing_transcript(tmp_path):
    history = RunHistory(tmp_path / "runs.json")
    history.record_run(_make_stats(transcript_path=str(tmp_path / "gone.md")))
    phantoms = history.phantom_records()
    assert len(phantoms) == 1


def test_prune_phantoms_removes_only_missing(tmp_path):
    real = tmp_path / "real.md"
    real.write_text("x")
    history = RunHistory(tmp_path / "runs.json")
    history.record_run(_make_stats(run_uid="r", source_id="yt:real", transcript_path=str(real)))
    history.record_run(
        _make_stats(run_uid="g", source_id="yt:gone", transcript_path=str(tmp_path / "gone.md"))
    )
    removed = history.prune_phantoms()
    assert removed == 1
    assert history.find_by_source("yt:real") is not None
    assert history.find_by_source("yt:gone") is None


# ---------------------------------------------------------------------------
# Corruption handling - backs up and starts empty (loses both sections)
# ---------------------------------------------------------------------------


def test_corrupted_store_backed_up_and_empty(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text("{not valid json")
    history = RunHistory(path)
    assert history.entries == {}
    assert history.get_runs_for_estimation() == []
    assert path.with_suffix(".json.bak").exists()


def test_atomic_save_writes_unified_shape(tmp_path):
    path = tmp_path / "runs.json"
    history = RunHistory(path)
    history.mark_in_progress("g1", 1, "Ep")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # The guid lives under by_guid, NOT at the top level.
    assert "g1" not in data
    assert "g1" in data["by_guid"]


# ---------------------------------------------------------------------------
# RunRecord serialization
# ---------------------------------------------------------------------------


def test_run_record_roundtrip():
    rec = RunRecord(
        guid="g1",
        show_id=1,
        episode_title="Ep",
        processed_at=datetime(2026, 6, 30, 10, 0, 0),
        status=ProcessingStatus.SUCCESS,
        output_file="/t/ep.md",
        one_liner="a liner",
        source_id="yt:1",
        transcript_path="/t/ep.md",
        audio_path="/a/ep.wav",
    )
    restored = RunRecord.from_dict(rec.to_dict())
    assert restored == rec


def test_run_record_tolerates_missing_guid_fields():
    """A source-only record (no guid/show_id) round-trips with None defaults."""
    rec = RunRecord(
        guid=None,
        show_id=None,
        episode_title="Ep",
        processed_at=datetime(2026, 6, 30, 10, 0, 0),
        status=ProcessingStatus.SUCCESS,
        source_id="yt:1",
        transcript_path="/t/ep.md",
    )
    restored = RunRecord.from_dict(rec.to_dict())
    assert restored.guid is None
    assert restored.show_id is None
    assert restored.source_id == "yt:1"


@pytest.mark.parametrize("missing_key", ["by_guid", "runs"])
def test_load_tolerates_partial_legacy_shape(tmp_path, missing_key):
    """A store written with only one section still loads (forward/back compat)."""
    path = tmp_path / "runs.json"
    payload = {"by_guid": {}, "runs": []}
    del payload[missing_key]
    path.write_text(json.dumps(payload))
    history = RunHistory(path)  # must not raise
    assert history.entries == {}
    assert history.get_runs_for_estimation() == []
