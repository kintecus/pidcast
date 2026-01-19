"""Tests for RSS feed parsing."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from pidcast.exceptions import FeedFetchError, FeedParseError
from pidcast.rss import Episode, RSSParser


@pytest.fixture
def sample_feed_path():
    """Path to sample RSS feed fixture."""
    return Path(__file__).parent / "fixtures" / "sample_feed.xml"


@pytest.fixture
def sample_feed_content(sample_feed_path):
    """Content of sample RSS feed."""
    with open(sample_feed_path, encoding="utf-8") as f:
        return f.read()


def test_parse_feed_from_file(sample_feed_path):
    """Test parsing RSS feed from local file."""
    # Convert file path to file:// URL for testing
    feed_url = f"file://{sample_feed_path.absolute()}"

    show_meta, episodes = RSSParser.parse_feed(feed_url)

    # Verify show metadata
    assert show_meta["title"] == "Test Podcast"
    assert show_meta["description"] == "A sample podcast feed for testing"
    assert show_meta["author"] == "Test Author"
    assert show_meta["artwork_url"] == "https://example.com/artwork.jpg"

    # Verify episodes
    assert len(episodes) == 3
    assert all(isinstance(ep, Episode) for ep in episodes)


def test_parse_episode_fields(sample_feed_path):
    """Test that episode fields are correctly parsed."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    _, episodes = RSSParser.parse_feed(feed_url)

    # Check first episode
    ep1 = episodes[0]
    assert ep1.title == "Episode 1: Introduction"
    assert ep1.description == "This is the first episode"
    assert "episode-1" in ep1.guid  # GUID may be resolved relative to base URL
    assert ep1.duration == 3600  # 1 hour in seconds
    assert ep1.audio_url == "https://example.com/episodes/episode-1.mp3"
    assert isinstance(ep1.pub_date, datetime)


def test_parse_duration_formats(sample_feed_path):
    """Test parsing different duration formats."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    _, episodes = RSSParser.parse_feed(feed_url)

    # Episode 1: integer seconds
    assert episodes[0].duration == 3600

    # Episode 2: MM:SS format
    assert episodes[1].duration == 2730  # 45:30 = 45*60 + 30

    # Episode 3: integer seconds
    assert episodes[2].duration == 1800


def test_episode_string_representation(sample_feed_path):
    """Test Episode __str__ method."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    _, episodes = RSSParser.parse_feed(feed_url)

    ep_str = str(episodes[0])
    assert "2026-01-15" in ep_str
    assert "Episode 1: Introduction" in ep_str
    assert "60m" in ep_str  # 3600 seconds = 60 minutes


def test_parse_feed_invalid_url():
    """Test parsing with invalid URL."""
    with pytest.raises(FeedFetchError):
        RSSParser.parse_feed("https://nonexistent.example.com/feed.xml")


def test_parse_feed_with_missing_fields():
    """Test parsing feed with missing optional fields."""
    # Create minimal valid RSS feed
    minimal_feed = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Minimal Podcast</title>
        <item>
          <title>Episode 1</title>
          <guid>ep1</guid>
          <pubDate>Mon, 15 Jan 2026 10:00:00 GMT</pubDate>
          <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg"/>
        </item>
      </channel>
    </rss>
    """

    # Write to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(minimal_feed)
        temp_path = f.name

    try:
        feed_url = f"file://{temp_path}"
        show_meta, episodes = RSSParser.parse_feed(feed_url)

        # Should handle missing fields gracefully
        assert show_meta["title"] == "Minimal Podcast"
        assert show_meta["description"] == ""  # Missing
        assert show_meta["author"] == ""  # Missing
        assert show_meta["artwork_url"] == ""  # Missing

        assert len(episodes) == 1
        assert episodes[0].title == "Episode 1"
        assert episodes[0].description == ""  # Missing
        assert episodes[0].duration is None  # Missing
    finally:
        os.unlink(temp_path)


def test_get_feed_field():
    """Test _get_feed_field helper method."""
    # Mock feed dictionary
    feed_dict = {"title": "Test", "description": "  Test Description  "}

    assert RSSParser._get_feed_field(feed_dict, "title") == "Test"
    assert RSSParser._get_feed_field(feed_dict, "description") == "Test Description"
    assert RSSParser._get_feed_field(feed_dict, "missing", "default") == "default"


def test_parse_duration_edge_cases():
    """Test duration parsing with various edge cases."""
    import feedparser

    # Mock entry
    class MockEntry:
        def get(self, key, default=None):
            return self.data.get(key, default)

        def __init__(self, data):
            self.data = data

    # Test integer duration
    entry = MockEntry({"itunes_duration": 1800})
    assert RSSParser._parse_duration(entry) == 1800

    # Test HH:MM:SS format
    entry = MockEntry({"itunes_duration": "01:30:00"})
    assert RSSParser._parse_duration(entry) == 5400

    # Test MM:SS format
    entry = MockEntry({"itunes_duration": "30:00"})
    assert RSSParser._parse_duration(entry) == 1800

    # Test seconds as string
    entry = MockEntry({"itunes_duration": "900"})
    assert RSSParser._parse_duration(entry) == 900

    # Test missing duration
    entry = MockEntry({})
    assert RSSParser._parse_duration(entry) is None
