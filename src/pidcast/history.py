"""Processing history management for podcast sync."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Episode processing status."""

    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"


@dataclass
class HistoryEntry:
    """Represents a processed episode."""

    guid: str
    show_id: int
    episode_title: str
    processed_at: datetime
    status: ProcessingStatus
    output_file: str | None = None
    error_message: str | None = None
    one_liner: str | None = None  # Cached one-liner summary for digest

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "guid": self.guid,
            "show_id": self.show_id,
            "episode_title": self.episode_title,
            "processed_at": self.processed_at.isoformat(),
            "status": self.status.value,
            "output_file": self.output_file,
            "error_message": self.error_message,
            "one_liner": self.one_liner,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryEntry":
        """Create from dictionary."""
        return cls(
            guid=data["guid"],
            show_id=data["show_id"],
            episode_title=data["episode_title"],
            processed_at=datetime.fromisoformat(data["processed_at"]),
            status=ProcessingStatus(data["status"]),
            output_file=data.get("output_file"),
            error_message=data.get("error_message"),
            one_liner=data.get("one_liner"),
        )


class ProcessingHistory:
    """Manage episode processing history."""

    def __init__(self, history_path: Path):
        """Initialize ProcessingHistory.

        Args:
            history_path: Path to history JSON file
        """
        self.history_path = history_path
        self.entries: dict[str, HistoryEntry] = {}
        self._load()

    def is_processed(self, guid: str) -> bool:
        """Check if episode already processed successfully.

        Args:
            guid: Episode GUID

        Returns:
            True if episode was successfully processed
        """
        entry = self.entries.get(guid)
        return entry is not None and entry.status == ProcessingStatus.SUCCESS

    def mark_in_progress(self, guid: str, show_id: int, title: str) -> None:
        """Mark episode as currently processing.

        Args:
            guid: Episode GUID
            show_id: Show ID
            title: Episode title
        """
        self.entries[guid] = HistoryEntry(
            guid=guid,
            show_id=show_id,
            episode_title=title,
            processed_at=datetime.now(),
            status=ProcessingStatus.IN_PROGRESS,
            output_file=None,
        )
        self._save()

    def mark_success(self, guid: str, output_file: str) -> None:
        """Mark episode as successfully processed.

        Args:
            guid: Episode GUID
            output_file: Path to output markdown file
        """
        if guid in self.entries:
            self.entries[guid].status = ProcessingStatus.SUCCESS
            self.entries[guid].output_file = output_file
            self.entries[guid].processed_at = datetime.now()
            self._save()

    def mark_failed(self, guid: str, error: str) -> None:
        """Mark episode as failed.

        Args:
            guid: Episode GUID
            error: Error message
        """
        if guid in self.entries:
            self.entries[guid].status = ProcessingStatus.FAILED
            self.entries[guid].error_message = error
            self.entries[guid].processed_at = datetime.now()
            self._save()

    def get_failed_episodes(self) -> list[HistoryEntry]:
        """Get all failed episodes for retry.

        Returns:
            List of failed history entries
        """
        return [e for e in self.entries.values() if e.status == ProcessingStatus.FAILED]

    def _load(self) -> None:
        """Load history from JSON file."""
        if not self.history_path.exists():
            return

        try:
            with open(self.history_path, encoding="utf-8") as f:
                data = json.load(f)
                for guid, entry_dict in data.items():
                    self.entries[guid] = HistoryEntry.from_dict(entry_dict)
        except Exception as e:
            logger.error(f"Failed to load history from {self.history_path}: {e}")
            # Backup corrupted file
            if self.history_path.exists():
                backup_path = self.history_path.with_suffix(".json.bak")
                try:
                    self.history_path.rename(backup_path)
                    logger.warning(f"Corrupted history backed up to: {backup_path}")
                except Exception as backup_error:
                    logger.error(f"Failed to backup corrupted history: {backup_error}")
            # Start with empty history
            self.entries = {}

    def _save(self) -> None:
        """Save history to JSON file (incremental)."""
        try:
            # Ensure directory exists
            self.history_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            data = {guid: e.to_dict() for guid, e in self.entries.items()}

            # Write atomically (write to temp file, then rename)
            temp_path = self.history_path.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_path.replace(self.history_path)

        except Exception as e:
            logger.error(f"Failed to save history to {self.history_path}: {e}")
            raise
