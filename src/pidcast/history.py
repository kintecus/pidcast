"""Unified run history for pidcast.

A single store (``runs.json``) replaces the two legacy stores:

- ``history.json`` - a dict keyed by episode GUID, written by the ``lib sync``
  path via :class:`ProcessingHistory`.
- ``transcription_stats.json`` - an append-only list keyed by ``source_id``,
  written by the main transcription workflow via ``save_statistics``.

They were disjoint write paths (sync records have a guid; ad-hoc URL/file runs
have a source_id) so the unified store keeps two sections::

    {"by_guid": {"<guid>": {RunRecord...}}, "runs": [{RunRecord...}, ...]}

``by_guid`` backs the guid-keyed sync/digest reads; ``runs`` is the append-only
list the dedup and estimation code scans by ``source_id``/``transcript_path``.

``ProcessingHistory`` / ``HistoryEntry`` / ``ProcessingStatus`` are kept as thin
back-compat shims so digest internals (``.entries`` / ``._save()``) and existing
callers keep working against the new backing store.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import TranscriptionStats

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Episode processing status."""

    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


@dataclass
class RunRecord:
    """A single processing run.

    Superset of the legacy ``HistoryEntry`` (sync/library fields) and the dedup
    fields from ``TranscriptionStats``. Sync runs populate ``guid``/``show_id``;
    ad-hoc workflow runs populate ``source_id``/``transcript_path`` and leave the
    guid fields ``None``. Every field tolerates the other path's absence.
    """

    # Sync/library identity (None for ad-hoc URL/file runs)
    guid: str | None = None
    show_id: int | None = None
    episode_title: str = ""
    processed_at: datetime | None = None
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    output_file: str | None = None
    error_message: str | None = None
    one_liner: str | None = None  # Cached one-liner summary for digest

    # Dedup / estimation fields (None for sync-only records)
    source_id: str | None = None
    transcript_path: str | None = None
    analysis_file: str | None = None
    audio_path: str | None = None
    run_timestamp: str | None = None
    success: bool | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "guid": self.guid,
            "show_id": self.show_id,
            "episode_title": self.episode_title,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "status": self.status.value,
            "output_file": self.output_file,
            "error_message": self.error_message,
            "one_liner": self.one_liner,
            "source_id": self.source_id,
            "transcript_path": self.transcript_path,
            "analysis_file": self.analysis_file,
            "audio_path": self.audio_path,
            "run_timestamp": self.run_timestamp,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        """Create from dictionary."""
        processed_at = data.get("processed_at")
        return cls(
            guid=data.get("guid"),
            show_id=data.get("show_id"),
            episode_title=data.get("episode_title", ""),
            processed_at=datetime.fromisoformat(processed_at) if processed_at else None,
            status=ProcessingStatus(data.get("status", "success")),
            output_file=data.get("output_file"),
            error_message=data.get("error_message"),
            one_liner=data.get("one_liner"),
            source_id=data.get("source_id"),
            transcript_path=data.get("transcript_path"),
            analysis_file=data.get("analysis_file"),
            audio_path=data.get("audio_path"),
            run_timestamp=data.get("run_timestamp"),
            success=data.get("success"),
        )


class RunHistory:
    """Unified run-history store backed by a two-section JSON file.

    - ``by_guid``: dict[guid -> RunRecord] for the sync/library path.
    - ``runs``: append-only list of raw stats dicts (full TranscriptionStats
      output) for the dedup/estimation path. Stored as plain dicts so every
      field the legacy consumers read (smart_filename, video_url, durations,
      provider, whisper_model, diarization_performed, ...) is preserved without
      enumerating each as a RunRecord attribute.
    """

    def __init__(self, path: Path):
        """Initialize the store.

        Args:
            path: Path to the unified runs JSON file.
        """
        self.path = Path(path)
        self._by_guid: dict[str, RunRecord] = {}
        self._runs: list[dict] = []
        self._load()

    # -- guid-keyed sync/library API ----------------------------------------

    @property
    def entries(self) -> dict[str, RunRecord]:
        """Live view of guid-keyed records (back-compat with ProcessingHistory)."""
        return self._by_guid

    def is_processed(self, guid: str) -> bool:
        """Check if an episode was already processed successfully."""
        entry = self._by_guid.get(guid)
        return entry is not None and entry.status == ProcessingStatus.SUCCESS

    def mark_in_progress(self, guid: str, show_id: int, title: str) -> None:
        """Mark an episode as currently processing."""
        self._by_guid[guid] = RunRecord(
            guid=guid,
            show_id=show_id,
            episode_title=title,
            processed_at=datetime.now(),
            status=ProcessingStatus.IN_PROGRESS,
        )
        self.save()

    def mark_success(self, guid: str, output_file: str) -> None:
        """Mark an episode as successfully processed."""
        if guid in self._by_guid:
            entry = self._by_guid[guid]
            entry.status = ProcessingStatus.SUCCESS
            entry.output_file = output_file
            entry.success = True
            entry.processed_at = datetime.now()
            self.save()

    def mark_failed(self, guid: str, error: str) -> None:
        """Mark an episode as failed."""
        if guid in self._by_guid:
            entry = self._by_guid[guid]
            entry.status = ProcessingStatus.FAILED
            entry.error_message = error
            entry.success = False
            entry.processed_at = datetime.now()
            self.save()

    def get_failed_episodes(self) -> list[RunRecord]:
        """Get all failed episodes for retry."""
        return [e for e in self._by_guid.values() if e.status == ProcessingStatus.FAILED]

    # -- source_id-keyed dedup / estimation API -----------------------------

    def record_run(self, stats: "TranscriptionStats") -> None:
        """Append a completed workflow run (from a TranscriptionStats object)."""
        self._runs.append(stats.to_dict())
        self.save()

    def find_by_source(self, source_id: str) -> RunRecord | None:
        """Most recent successful run for a source id, else None."""
        for entry in reversed(self._runs):
            if entry.get("success") and entry.get("source_id") == source_id:
                return RunRecord.from_dict(entry)
        return None

    def find_by_transcript(self, transcript_path: str | Path) -> RunRecord | None:
        """Most recent run whose transcript_path matches, else None."""
        target = str(Path(transcript_path).resolve())
        for entry in reversed(self._runs):
            stored = entry.get("transcript_path")
            if stored and str(Path(stored).resolve()) == target:
                return RunRecord.from_dict(entry)
        return None

    def get_runs_for_estimation(self) -> list[dict]:
        """Return raw run dicts (list shape) for estimate_transcription_time."""
        return list(self._runs)

    def phantom_records(self) -> list[dict]:
        """Successful runs whose recorded transcript file no longer exists."""
        return [r for r in self._runs if _is_phantom_entry(r)]

    def prune_phantoms(self) -> int:
        """Drop runs whose transcript file is gone. Returns count removed."""
        kept = [r for r in self._runs if not _is_phantom_entry(r)]
        removed = len(self._runs) - len(kept)
        if removed:
            self._runs = kept
            self.save()
        return removed

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # Legacy transcription_stats.json: a bare list of run dicts.
                # Read it as the runs section so pre-migration files and existing
                # fixtures keep working.
                by_guid = {}
                runs = data
            else:
                by_guid = data.get("by_guid", {})
                runs = data.get("runs", [])
            self._by_guid = {guid: RunRecord.from_dict(d) for guid, d in by_guid.items()}
            self._runs = list(runs)
        except Exception as e:
            logger.error(f"Failed to load run history from {self.path}: {e}")
            if self.path.exists():
                backup_path = self.path.with_suffix(".json.bak")
                try:
                    self.path.rename(backup_path)
                    logger.warning(f"Corrupted run history backed up to: {backup_path}")
                except Exception as backup_error:
                    logger.error(f"Failed to backup corrupted run history: {backup_error}")
            self._by_guid = {}
            self._runs = []

    def save(self) -> None:
        """Persist the store atomically (temp file + rename)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "by_guid": {guid: e.to_dict() for guid, e in self._by_guid.items()},
                "runs": list(self._runs),
            }
            temp_path = self.path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.path)
        except Exception as e:
            logger.error(f"Failed to save run history to {self.path}: {e}")
            raise


def _is_phantom_entry(entry: dict) -> bool:
    """A successful stats run whose recorded transcript path no longer exists.

    Mirrors the legacy ``utils._is_phantom``: entries with only a basename
    (no full ``transcript_path``) can't be located, so they are NOT phantoms.
    """
    if not (entry.get("success") and entry.get("smart_filename")):
        return False
    path = entry.get("transcript_path")
    return bool(path) and not Path(path).exists()


# ============================================================================
# Back-compat shims
# ============================================================================

# digest.py and existing tests construct HistoryEntry(...) with guid fields and
# read .entries values; RunRecord is a superset, so HistoryEntry is an alias.
HistoryEntry = RunRecord


class ProcessingHistory(RunHistory):
    """Legacy alias for RunHistory.

    Preserves the ``history_path`` attribute, the ``.entries`` dict, and the
    ``._save()`` method that digest.py and older callers depend on.
    """

    def __init__(self, history_path: Path):
        self.history_path = Path(history_path)
        super().__init__(history_path)

    def _save(self) -> None:  # noqa: D401 - back-compat name
        """Deprecated alias for :meth:`save`."""
        self.save()
