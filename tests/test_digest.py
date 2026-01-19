"""Unit tests for digest generation."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from pidcast.digest import DigestData, DigestFormatter, DigestGenerator, ShowRollup, TopicCluster
from pidcast.history import HistoryEntry, ProcessingHistory, ProcessingStatus
from pidcast.library import LibraryManager, Show


@pytest.fixture
def mock_library():
    """Create a mock library manager."""
    library = Mock(spec=LibraryManager)
    show1 = Show(
        id=1,
        title="Test Podcast 1",
        feed_url="https://example.com/feed1.xml",
        description="Test podcast 1",
        author="Author 1",
    )
    show2 = Show(
        id=2,
        title="Test Podcast 2",
        feed_url="https://example.com/feed2.xml",
        description="Test podcast 2",
        author="Author 2",
    )
    library.get_show = lambda show_id: show1 if show_id == 1 else show2
    return library


@pytest.fixture
def mock_history(tmp_path):
    """Create a mock processing history."""
    history_file = tmp_path / "history.json"
    history = ProcessingHistory(history_file)

    now = datetime.now()

    # Add some test episodes
    entries = [
        HistoryEntry(
            guid="ep1-guid",
            show_id=1,
            episode_title="Episode 1",
            processed_at=now,
            status=ProcessingStatus.SUCCESS,
            output_file=str(tmp_path / "ep1.md"),
            one_liner="A great episode about technology",
        ),
        HistoryEntry(
            guid="ep2-guid",
            show_id=1,
            episode_title="Episode 2",
            processed_at=now,
            status=ProcessingStatus.SUCCESS,
            output_file=str(tmp_path / "ep2.md"),
            one_liner="Another tech episode",
        ),
        HistoryEntry(
            guid="ep3-guid",
            show_id=2,
            episode_title="Episode 3",
            processed_at=now,
            status=ProcessingStatus.SUCCESS,
            output_file=str(tmp_path / "ep3.md"),
            one_liner="Discussion about science",
        ),
    ]

    for entry in entries:
        history.entries[entry.guid] = entry

    return history


@pytest.fixture
def mock_summarizer():
    """Create a mock summarizer."""
    summarizer = Mock()
    summarizer.generate_one_liners = Mock(
        return_value={
            "ep1-guid": "A great episode about technology",
            "ep2-guid": "Another tech episode",
            "ep3-guid": "Discussion about science",
        }
    )
    summarizer.generate_show_rollup = Mock(return_value="2 episodes covering technology topics")
    summarizer.generate_topic_clusters = Mock(
        return_value=[
            {
                "topic": "Technology",
                "description": "Episodes discussing tech trends",
                "episode_guids": ["ep1-guid", "ep2-guid"],
            }
        ]
    )
    return summarizer


def test_digest_generator_initialization(mock_library, mock_history, mock_summarizer):
    """Test DigestGenerator initialization."""
    generator = DigestGenerator(mock_library, mock_history, mock_summarizer)
    assert generator.library == mock_library
    assert generator.history == mock_history
    assert generator.summarizer == mock_summarizer


def test_filter_episodes_by_date(mock_library, mock_history, mock_summarizer):
    """Test filtering episodes by specific date."""
    generator = DigestGenerator(mock_library, mock_history, mock_summarizer)

    # Filter by today
    episodes = generator._filter_episodes(date_filter=datetime.now(), date_range=None)
    assert len(episodes) == 3

    # Filter by yesterday (should be empty)
    yesterday = datetime.now() - timedelta(days=1)
    episodes = generator._filter_episodes(date_filter=yesterday, date_range=None)
    assert len(episodes) == 0


def test_filter_episodes_by_range(mock_library, mock_history, mock_summarizer):
    """Test filtering episodes by date range."""
    generator = DigestGenerator(mock_library, mock_history, mock_summarizer)

    # Filter by last 7 days
    episodes = generator._filter_episodes(date_filter=None, date_range=timedelta(days=7))
    assert len(episodes) == 3

    # Filter by a future date range (should still include today's episodes)
    episodes = generator._filter_episodes(date_filter=None, date_range=timedelta(days=1))
    assert len(episodes) == 3


def test_get_one_liners_from_cache(mock_library, mock_history, mock_summarizer):
    """Test getting one-liners from cache."""
    generator = DigestGenerator(mock_library, mock_history, mock_summarizer)

    episodes = list(mock_history.entries.values())
    one_liners = generator._get_one_liners(episodes)

    # All episodes have cached one-liners
    assert len(one_liners) == 3
    assert one_liners["ep1-guid"] == "A great episode about technology"
    assert one_liners["ep2-guid"] == "Another tech episode"
    assert one_liners["ep3-guid"] == "Discussion about science"

    # Summarizer should not have been called since all are cached
    mock_summarizer.generate_one_liners.assert_not_called()


def test_get_one_liners_generate_missing(mock_library, tmp_path):
    """Test generating one-liners for episodes without cache."""
    # Create a fresh mock summarizer for this test
    summarizer = Mock()
    summarizer.generate_one_liners = Mock(
        return_value={"ep4-guid": "Generated one-liner for episode 4"}
    )

    # Create history without cached one-liners
    history_file = tmp_path / "history.json"
    history = ProcessingHistory(history_file)

    now = datetime.now()
    entry = HistoryEntry(
        guid="ep4-guid",
        show_id=1,
        episode_title="Episode 4",
        processed_at=now,
        status=ProcessingStatus.SUCCESS,
        output_file=str(tmp_path / "ep4.md"),
        one_liner=None,  # No cached one-liner
    )
    history.entries[entry.guid] = entry

    generator = DigestGenerator(mock_library, history, summarizer)

    episodes = [entry]
    one_liners = generator._get_one_liners(episodes)

    # Summarizer should have been called
    summarizer.generate_one_liners.assert_called_once_with([entry])

    # One-liner should be from mock
    assert "ep4-guid" in one_liners
    assert one_liners["ep4-guid"] == "Generated one-liner for episode 4"


def test_generate_show_rollups(mock_library, mock_history, mock_summarizer):
    """Test generating show-level rollups."""
    generator = DigestGenerator(mock_library, mock_history, mock_summarizer)

    episodes = list(mock_history.entries.values())
    one_liners = {
        "ep1-guid": "A great episode about technology",
        "ep2-guid": "Another tech episode",
        "ep3-guid": "Discussion about science",
    }

    rollups = generator._generate_show_rollups(episodes, one_liners)

    # Should have rollups for 2 shows
    assert len(rollups) == 2
    assert 1 in rollups
    assert 2 in rollups

    # Check show 1 rollup
    rollup1 = rollups[1]
    assert rollup1.episode_count == 2
    assert rollup1.show.title == "Test Podcast 1"


def test_generate_digest(mock_library, mock_history, mock_summarizer):
    """Test full digest generation."""
    generator = DigestGenerator(mock_library, mock_history, mock_summarizer)

    digest = generator.generate_digest(date_filter=datetime.now())

    assert len(digest.episodes) == 3
    assert len(digest.one_liners) == 3
    assert len(digest.show_rollups) == 2
    # Topic clustering should be attempted for 3 episodes
    assert isinstance(digest.topic_clusters, list)


def test_generate_digest_empty(mock_library, tmp_path, mock_summarizer):
    """Test digest generation with no episodes."""
    # Create empty history
    history_file = tmp_path / "history.json"
    history = ProcessingHistory(history_file)

    generator = DigestGenerator(mock_library, history, mock_summarizer)

    digest = generator.generate_digest(date_filter=datetime.now())

    assert len(digest.episodes) == 0
    assert len(digest.one_liners) == 0
    assert len(digest.show_rollups) == 0
    assert len(digest.topic_clusters) == 0


def test_digest_formatter_markdown(mock_library, mock_history):
    """Test markdown formatting."""
    show_rollup = ShowRollup(
        show=Show(
            id=1,
            title="Test Podcast",
            feed_url="https://example.com/feed.xml",
            description="Test",
            author="Author",
        ),
        episode_count=2,
        episodes=list(mock_history.entries.values())[:2],
        summary="2 episodes about tech",
    )

    digest = DigestData(
        episodes=list(mock_history.entries.values()),
        one_liners={
            "ep1-guid": "Episode 1 summary",
            "ep2-guid": "Episode 2 summary",
        },
        show_rollups={1: show_rollup},
        topic_clusters=[],
    )

    markdown = DigestFormatter.format_markdown(digest)

    # Check markdown structure
    assert "---" in markdown  # YAML front matter
    assert "title: Podcast Digest" in markdown
    assert "episode_count: 3" in markdown
    assert "## Shows" in markdown
    assert "Test Podcast" in markdown


def test_topic_clustering_minimum_episodes(mock_library, tmp_path, mock_summarizer):
    """Test that topic clustering is skipped with too few episodes."""
    # Create history with only 2 episodes
    history_file = tmp_path / "history.json"
    history = ProcessingHistory(history_file)

    now = datetime.now()
    for i in range(2):
        entry = HistoryEntry(
            guid=f"ep{i}-guid",
            show_id=1,
            episode_title=f"Episode {i}",
            processed_at=now,
            status=ProcessingStatus.SUCCESS,
            output_file=str(tmp_path / f"ep{i}.md"),
        )
        history.entries[entry.guid] = entry

    generator = DigestGenerator(mock_library, history, mock_summarizer)

    episodes = list(history.entries.values())
    one_liners = {ep.guid: f"Summary {ep.guid}" for ep in episodes}

    clusters = generator._generate_topic_clusters(episodes, one_liners)

    # Should return empty list for < 3 episodes
    assert len(clusters) == 0
    # Summarizer should not be called
    mock_summarizer.generate_topic_clusters.assert_not_called()
