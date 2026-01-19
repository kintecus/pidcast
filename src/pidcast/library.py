"""Library management for podcast shows."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .config import CONFIG_DIR, LIBRARY_FILE
from .exceptions import DuplicateShowError, ShowNotFoundError
from .rss import Episode, RSSParser

logger = logging.getLogger(__name__)


@dataclass
class Show:
    """Represents a podcast show in the library."""

    id: int
    title: str
    feed_url: str
    description: str = ""
    author: str = ""
    artwork_url: str = ""
    added_at: datetime = field(default_factory=datetime.now)
    last_checked: datetime | None = None
    backfill_count: int = 5  # Uses global default

    def to_dict(self) -> dict[str, Any]:
        """Convert Show to dictionary for YAML serialization.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "title": self.title,
            "feed_url": self.feed_url,
            "description": self.description,
            "author": self.author,
            "artwork_url": self.artwork_url,
            "added_at": self.added_at.isoformat() if self.added_at else None,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "backfill_count": self.backfill_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Show":
        """Create Show from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Show instance
        """
        # Parse datetime fields
        added_at = data.get("added_at")
        if isinstance(added_at, str):
            added_at = datetime.fromisoformat(added_at)
        elif not added_at:
            added_at = datetime.now()

        last_checked = data.get("last_checked")
        if isinstance(last_checked, str):
            last_checked = datetime.fromisoformat(last_checked)

        return cls(
            id=data["id"],
            title=data["title"],
            feed_url=data["feed_url"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            artwork_url=data.get("artwork_url", ""),
            added_at=added_at,
            last_checked=last_checked,
            backfill_count=data.get("backfill_count", 5),
        )


class LibraryManager:
    """Manages podcast library operations."""

    def __init__(self, library_path: Path | None = None):
        """Initialize LibraryManager.

        Args:
            library_path: Path to library file (defaults to LIBRARY_FILE)
        """
        self.library_path = library_path or LIBRARY_FILE
        self.shows: list[Show] = []
        self.next_id: int = 1
        self._load()

    def add_show(
        self, feed_url: str, backfill_count: int | None = None, verbose: bool = False
    ) -> Show:
        """Add show to library from RSS feed URL.

        Args:
            feed_url: RSS feed URL
            backfill_count: Number of recent episodes to backfill (uses global default if None)
            verbose: Enable verbose logging

        Returns:
            Created Show instance

        Raises:
            DuplicateShowError: If show already exists in library
            FeedFetchError: If feed can't be fetched
            FeedParseError: If feed format is invalid
        """
        # Check for duplicate feed URL
        existing = self.find_show_by_url(feed_url)
        if existing:
            raise DuplicateShowError(
                f"Show already in library (ID: {existing.id}, Title: {existing.title})"
            )

        if verbose:
            logger.info(f"Fetching feed metadata for: {feed_url}")

        # Parse feed to extract metadata
        show_meta, episodes = RSSParser.parse_feed(feed_url, verbose=verbose)

        # Create show
        show = Show(
            id=self.next_id,
            title=show_meta["title"],
            feed_url=feed_url,
            description=show_meta["description"],
            author=show_meta["author"],
            artwork_url=show_meta["artwork_url"],
            added_at=datetime.now(),
            last_checked=None,
            backfill_count=backfill_count if backfill_count is not None else 5,
        )

        # Add to library
        self.shows.append(show)
        self.next_id += 1
        self._save()

        if verbose:
            logger.info(f"Added show: {show.title} (ID: {show.id})")

        return show

    def get_show(self, show_id: int) -> Show | None:
        """Get show by ID.

        Args:
            show_id: Show ID

        Returns:
            Show instance or None if not found
        """
        for show in self.shows:
            if show.id == show_id:
                return show
        return None

    def find_show_by_url(self, feed_url: str) -> Show | None:
        """Find show by feed URL.

        Args:
            feed_url: RSS feed URL

        Returns:
            Show instance or None if not found
        """
        # Normalize URLs for comparison (strip trailing slashes, etc.)
        normalized_url = feed_url.rstrip("/").lower()
        for show in self.shows:
            if show.feed_url.rstrip("/").lower() == normalized_url:
                return show
        return None

    def list_shows(self) -> list[Show]:
        """Get all shows in library.

        Returns:
            List of Show instances
        """
        # Return sorted by ID for consistent ordering
        return sorted(self.shows, key=lambda s: s.id)

    def remove_show(self, show_id: int) -> bool:
        """Remove show from library.

        Args:
            show_id: Show ID

        Returns:
            True if removed, False if not found
        """
        show = self.get_show(show_id)
        if not show:
            return False

        self.shows.remove(show)
        self._save()
        return True

    def get_episodes(
        self, show_id: int, limit: int | None = None, verbose: bool = False
    ) -> list[Episode]:
        """Get episodes for a show by fetching its RSS feed.

        Args:
            show_id: Show ID
            limit: Maximum number of episodes to return (None = all)
            verbose: Enable verbose logging

        Returns:
            List of Episode instances

        Raises:
            ShowNotFoundError: If show not found
            FeedFetchError: If feed can't be fetched
            FeedParseError: If feed format is invalid
        """
        show = self.get_show(show_id)
        if not show:
            raise ShowNotFoundError(f"Show ID {show_id} not found")

        if verbose:
            logger.info(f"Fetching episodes for: {show.title}")

        # Parse feed
        _, episodes = RSSParser.parse_feed(show.feed_url, verbose=verbose)

        # Update last_checked timestamp
        show.last_checked = datetime.now()
        self._save()

        # Apply limit if specified
        if limit is not None:
            episodes = episodes[:limit]

        return episodes

    def update_last_checked(self, show_id: int) -> bool:
        """Update last_checked timestamp for a show.

        Args:
            show_id: Show ID

        Returns:
            True if updated, False if show not found
        """
        show = self.get_show(show_id)
        if not show:
            return False

        show.last_checked = datetime.now()
        self._save()
        return True

    def _load(self) -> None:
        """Load library from YAML file."""
        if not self.library_path.exists():
            # Initialize with empty library
            self.shows = []
            self.next_id = 1
            return

        try:
            yaml = YAML()
            with open(self.library_path, encoding="utf-8") as f:
                data = yaml.load(f)

            if not data:
                self.shows = []
                self.next_id = 1
                return

            # Load shows
            shows_data = data.get("shows", [])
            self.shows = [Show.from_dict(show_data) for show_data in shows_data]

            # Load next_id
            self.next_id = data.get("next_id", 1)

            # Ensure next_id is greater than all existing IDs
            if self.shows:
                max_id = max(show.id for show in self.shows)
                if self.next_id <= max_id:
                    self.next_id = max_id + 1

        except Exception as e:
            logger.error(f"Failed to load library from {self.library_path}: {e}")
            # Initialize with empty library on error
            self.shows = []
            self.next_id = 1

    def _save(self) -> None:
        """Save library to YAML file."""
        try:
            # Ensure config directory exists
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)

            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 4096  # Prevent line wrapping

            # Prepare data
            data = {
                "shows": [show.to_dict() for show in self.shows],
                "next_id": self.next_id,
            }

            # Write to file with comments
            with open(self.library_path, "w", encoding="utf-8") as f:
                f.write("# Pidcast Library Configuration\n")
                f.write("# This file is auto-managed but safe to edit manually\n\n")
                yaml.dump(data, f)

        except Exception as e:
            logger.error(f"Failed to save library to {self.library_path}: {e}")
            raise
