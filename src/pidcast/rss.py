"""RSS feed parsing for podcast feeds."""

import logging
from dataclasses import dataclass
from datetime import datetime

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import FeedFetchError, FeedParseError

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a podcast episode from RSS feed."""

    guid: str  # Unique episode identifier from RSS
    title: str
    description: str
    pub_date: datetime
    duration: int | None  # Seconds
    audio_url: str  # Enclosure URL

    def __str__(self) -> str:
        """Return human-readable episode string."""
        date_str = self.pub_date.strftime("%Y-%m-%d")
        duration_str = f"{self.duration // 60}m" if self.duration else "Unknown"
        return f"{date_str} - {self.title} ({duration_str})"


class RSSParser:
    """Parse podcast RSS feeds."""

    @staticmethod
    def parse_feed(
        feed_url: str, max_retries: int = 3, verbose: bool = False
    ) -> tuple[dict[str, str], list[Episode]]:
        """Parse RSS feed and return (show_metadata, episodes).

        Args:
            feed_url: URL of the RSS feed
            max_retries: Maximum number of retry attempts
            verbose: Enable verbose logging

        Returns:
            Tuple of (show_metadata, episodes)

        Raises:
            FeedFetchError: If feed can't be fetched
            FeedParseError: If feed format is invalid
        """
        if verbose:
            logger.info(f"Fetching RSS feed: {feed_url}")

        # Handle file:// URLs for testing
        if feed_url.startswith("file://"):
            feed = feedparser.parse(feed_url)
        else:
            # Configure requests session with retry logic
            session = requests.Session()
            retry_strategy = Retry(
                total=max_retries,
                backoff_factor=1,  # 1s, 2s, 4s delays
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Fetch feed with timeout
            try:
                response = session.get(feed_url, timeout=10)
                response.raise_for_status()
            except requests.exceptions.Timeout as e:
                raise FeedFetchError(f"Request timeout while fetching feed: {feed_url}") from e
            except requests.exceptions.ConnectionError as e:
                raise FeedFetchError(f"Connection error while fetching feed: {e}") from e
            except requests.exceptions.HTTPError as e:
                raise FeedFetchError(
                    f"HTTP error {response.status_code} while fetching feed: {e}"
                ) from e
            except requests.exceptions.RequestException as e:
                raise FeedFetchError(f"Failed to fetch feed: {e}") from e

            # Parse feed
            feed = feedparser.parse(response.content)

        # Check for parsing errors
        if feed.bozo and hasattr(feed, "bozo_exception"):
            # Some feeds have minor issues but are still usable
            error_msg = str(feed.bozo_exception)
            # Only raise on critical errors
            if "not well-formed" in error_msg.lower() or "no element found" in error_msg.lower():
                raise FeedParseError(f"Invalid feed format: {error_msg}")
            elif verbose:
                logger.warning(f"Feed has minor issues but continuing: {error_msg}")

        # Validate feed has required fields
        if not hasattr(feed, "feed") or not feed.feed:
            raise FeedParseError("Feed has no channel information")

        # Extract show metadata
        show_meta = {
            "title": RSSParser._get_feed_field(feed.feed, "title", "Unknown Podcast"),
            "description": RSSParser._get_feed_field(feed.feed, "description", ""),
            "author": RSSParser._extract_author(feed.feed),
            "artwork_url": RSSParser._extract_artwork(feed.feed),
        }

        if verbose:
            logger.info(f"Found show: {show_meta['title']}")
            logger.info(f"Author: {show_meta['author']}")

        # Extract episodes
        episodes = []
        for entry in feed.entries:
            try:
                episode = RSSParser._parse_episode(entry)
                episodes.append(episode)
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to parse episode '{entry.get('title', 'Unknown')}': {e}")
                continue

        if not episodes:
            raise FeedParseError("Feed has no valid episodes")

        if verbose:
            logger.info(f"Found {len(episodes)} episodes")

        return show_meta, episodes

    @staticmethod
    def _parse_episode(entry: feedparser.FeedParserDict) -> Episode:
        """Parse single RSS entry into Episode.

        Args:
            entry: Feed entry from feedparser

        Returns:
            Episode object

        Raises:
            ValueError: If episode is missing required fields
        """
        # Extract title
        title = entry.get("title", "").strip()
        if not title:
            raise ValueError("Episode missing title")

        # Extract GUID (use link as fallback)
        guid = entry.get("id") or entry.get("guid") or entry.get("link", "")
        if not guid:
            raise ValueError("Episode missing GUID")

        # Extract description
        description = (
            entry.get("summary", "") or entry.get("description", "") or entry.get("subtitle", "")
        )

        # Extract publication date
        pub_date = RSSParser._parse_date(entry)

        # Extract duration (iTunes extension or generic)
        duration = RSSParser._parse_duration(entry)

        # Extract audio URL from enclosure
        audio_url = RSSParser._extract_audio_url(entry)
        if not audio_url:
            raise ValueError("Episode missing audio enclosure")

        return Episode(
            guid=guid,
            title=title,
            description=description,
            pub_date=pub_date,
            duration=duration,
            audio_url=audio_url,
        )

    @staticmethod
    def _get_feed_field(feed_dict: feedparser.FeedParserDict, field: str, default: str = "") -> str:
        """Safely extract a field from feed dictionary.

        Args:
            feed_dict: Feed dictionary
            field: Field name
            default: Default value if field not found

        Returns:
            Field value or default
        """
        value = feed_dict.get(field, default)
        return value.strip() if isinstance(value, str) else default

    @staticmethod
    def _extract_author(feed_dict: feedparser.FeedParserDict) -> str:
        """Extract author/creator from feed.

        Args:
            feed_dict: Feed dictionary

        Returns:
            Author name or empty string
        """
        # Try various author fields
        author = (
            feed_dict.get("author")
            or feed_dict.get("itunes_author")
            or feed_dict.get("creator")
            or feed_dict.get("publisher")
            or ""
        )
        return author.strip() if isinstance(author, str) else ""

    @staticmethod
    def _extract_artwork(feed_dict: feedparser.FeedParserDict) -> str:
        """Extract artwork URL from feed.

        Args:
            feed_dict: Feed dictionary

        Returns:
            Artwork URL or empty string
        """
        # Try iTunes image first (standard for podcasts)
        if hasattr(feed_dict, "image") and isinstance(feed_dict.image, dict):
            itunes_image = feed_dict.image.get("href")
            if itunes_image:
                return itunes_image

        # Try standard RSS image
        if hasattr(feed_dict, "image") and isinstance(feed_dict.image, dict):
            standard_image = feed_dict.image.get("url")
            if standard_image:
                return standard_image

        # Try any image link in the feed
        links = feed_dict.get("links", [])
        for link in links:
            if isinstance(link, dict) and link.get("rel") == "image":
                return link.get("href", "")

        return ""

    @staticmethod
    def _parse_date(entry: feedparser.FeedParserDict) -> datetime:
        """Parse publication date from entry.

        Args:
            entry: Feed entry

        Returns:
            Publication date (defaults to current time if not found)
        """
        # Try published_parsed first
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except Exception:
                pass

        # Try updated_parsed
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            try:
                return datetime(*entry.updated_parsed[:6])
            except Exception:
                pass

        # Fallback to current time
        logger.warning(f"Could not parse date for episode '{entry.get('title', 'Unknown')}'")
        return datetime.now()

    @staticmethod
    def _parse_duration(entry: feedparser.FeedParserDict) -> int | None:
        """Parse duration from entry.

        Args:
            entry: Feed entry

        Returns:
            Duration in seconds, or None if not found
        """
        # Try iTunes duration (format: HH:MM:SS or MM:SS or seconds)
        itunes_duration = entry.get("itunes_duration")
        if itunes_duration:
            try:
                # Handle seconds as integer
                if isinstance(itunes_duration, int):
                    return itunes_duration

                # Handle time format strings
                if isinstance(itunes_duration, str):
                    parts = itunes_duration.split(":")
                    if len(parts) == 3:  # HH:MM:SS
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    elif len(parts) == 2:  # MM:SS
                        return int(parts[0]) * 60 + int(parts[1])
                    else:  # Just seconds
                        return int(itunes_duration)
            except Exception:
                pass

        # Try generic duration field
        duration = entry.get("duration")
        if duration:
            try:
                return int(duration)
            except Exception:
                pass

        return None

    @staticmethod
    def _extract_audio_url(entry: feedparser.FeedParserDict) -> str:
        """Extract audio URL from enclosures.

        Args:
            entry: Feed entry

        Returns:
            Audio URL or empty string
        """
        # Check for enclosures
        enclosures = entry.get("enclosures", [])
        for enclosure in enclosures:
            # Look for audio MIME types
            mime_type = enclosure.get("type", "")
            if mime_type.startswith("audio/") or mime_type.startswith("video/"):
                return enclosure.get("href", enclosure.get("url", ""))

        # Fallback: return first enclosure if any
        if enclosures and len(enclosures) > 0:
            return enclosures[0].get("href", enclosures[0].get("url", ""))

        # Last resort: check for media:content
        if hasattr(entry, "media_content"):
            media = entry.media_content
            if isinstance(media, list) and len(media) > 0:
                return media[0].get("url", "")

        return ""
