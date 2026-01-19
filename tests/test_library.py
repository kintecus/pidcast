"""Tests for library management."""

import tempfile
from pathlib import Path

import pytest

from pidcast.exceptions import DuplicateShowError, ShowNotFoundError
from pidcast.library import LibraryManager, Show


@pytest.fixture
def temp_library_path():
    """Create a temporary library file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def library_manager(temp_library_path):
    """Create a LibraryManager with temporary storage."""
    return LibraryManager(library_path=temp_library_path)


@pytest.fixture
def sample_feed_path():
    """Path to sample RSS feed fixture."""
    return Path(__file__).parent / "fixtures" / "sample_feed.xml"


def test_library_manager_init(library_manager):
    """Test LibraryManager initialization."""
    assert isinstance(library_manager, LibraryManager)
    assert library_manager.shows == []
    assert library_manager.next_id == 1


def test_add_show(library_manager, sample_feed_path):
    """Test adding a show to library."""
    feed_url = f"file://{sample_feed_path.absolute()}"

    show = library_manager.add_show(feed_url)

    assert show.id == 1
    assert show.title == "Test Podcast"
    assert show.author == "Test Author"
    assert show.feed_url == feed_url
    assert len(library_manager.shows) == 1


def test_add_show_increments_id(library_manager, sample_feed_path):
    """Test that adding shows increments IDs."""
    feed_url = f"file://{sample_feed_path.absolute()}"

    show1 = library_manager.add_show(feed_url)
    assert show1.id == 1

    # Remove and add again (should get new ID)
    library_manager.remove_show(show1.id)

    # Modify URL slightly to avoid duplicate detection
    feed_url2 = feed_url + "#modified"
    show2 = library_manager.add_show(feed_url2)
    assert show2.id == 2


def test_add_duplicate_show(library_manager, sample_feed_path):
    """Test that adding duplicate show raises error."""
    feed_url = f"file://{sample_feed_path.absolute()}"

    library_manager.add_show(feed_url)

    with pytest.raises(DuplicateShowError):
        library_manager.add_show(feed_url)


def test_add_duplicate_show_normalized_url(library_manager, sample_feed_path):
    """Test that URL normalization prevents duplicates."""
    feed_url = f"file://{sample_feed_path.absolute()}"

    library_manager.add_show(feed_url)

    # Try adding with trailing slash (should be detected as duplicate)
    with pytest.raises(DuplicateShowError):
        library_manager.add_show(feed_url + "/")


def test_get_show(library_manager, sample_feed_path):
    """Test getting a show by ID."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    added_show = library_manager.add_show(feed_url)

    show = library_manager.get_show(added_show.id)

    assert show is not None
    assert show.id == added_show.id
    assert show.title == added_show.title


def test_get_show_not_found(library_manager):
    """Test getting non-existent show."""
    show = library_manager.get_show(999)
    assert show is None


def test_find_show_by_url(library_manager, sample_feed_path):
    """Test finding show by feed URL."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    added_show = library_manager.add_show(feed_url)

    show = library_manager.find_show_by_url(feed_url)

    assert show is not None
    assert show.id == added_show.id
    assert show.feed_url == feed_url


def test_find_show_by_url_not_found(library_manager):
    """Test finding non-existent show by URL."""
    show = library_manager.find_show_by_url("https://nonexistent.example.com/feed.xml")
    assert show is None


def test_list_shows(library_manager, sample_feed_path):
    """Test listing all shows."""
    assert library_manager.list_shows() == []

    feed_url = f"file://{sample_feed_path.absolute()}"
    show1 = library_manager.add_show(feed_url)

    shows = library_manager.list_shows()
    assert len(shows) == 1
    assert shows[0].id == show1.id


def test_list_shows_sorted(library_manager, sample_feed_path):
    """Test that list_shows returns sorted by ID."""
    feed_url1 = f"file://{sample_feed_path.absolute()}"
    feed_url2 = f"file://{sample_feed_path.absolute()}#2"

    show1 = library_manager.add_show(feed_url1)
    show2 = library_manager.add_show(feed_url2)

    shows = library_manager.list_shows()
    assert len(shows) == 2
    assert shows[0].id == show1.id
    assert shows[1].id == show2.id


def test_remove_show(library_manager, sample_feed_path):
    """Test removing a show."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    show = library_manager.add_show(feed_url)

    assert len(library_manager.shows) == 1

    result = library_manager.remove_show(show.id)

    assert result is True
    assert len(library_manager.shows) == 0


def test_remove_show_not_found(library_manager):
    """Test removing non-existent show."""
    result = library_manager.remove_show(999)
    assert result is False


def test_get_episodes(library_manager, sample_feed_path):
    """Test getting episodes for a show."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    show = library_manager.add_show(feed_url)

    episodes = library_manager.get_episodes(show.id)

    assert len(episodes) == 3
    assert episodes[0].title == "Episode 1: Introduction"
    # Verify last_checked was updated
    assert show.last_checked is not None


def test_get_episodes_with_limit(library_manager, sample_feed_path):
    """Test getting limited number of episodes."""
    feed_url = f"file://{sample_feed_path.absolute()}"
    show = library_manager.add_show(feed_url)

    episodes = library_manager.get_episodes(show.id, limit=2)

    assert len(episodes) == 2


def test_get_episodes_show_not_found(library_manager):
    """Test getting episodes for non-existent show."""
    with pytest.raises(ShowNotFoundError):
        library_manager.get_episodes(999)


def test_library_persistence(temp_library_path, sample_feed_path):
    """Test that library persists across instances."""
    feed_url = f"file://{sample_feed_path.absolute()}"

    # Create first instance and add show
    lib1 = LibraryManager(library_path=temp_library_path)
    show1 = lib1.add_show(feed_url)

    # Create second instance and verify show exists
    lib2 = LibraryManager(library_path=temp_library_path)
    shows = lib2.list_shows()

    assert len(shows) == 1
    assert shows[0].id == show1.id
    assert shows[0].title == show1.title
    assert shows[0].feed_url == show1.feed_url


def test_show_to_dict():
    """Test Show.to_dict() method."""
    from datetime import datetime

    show = Show(
        id=1,
        title="Test Show",
        feed_url="https://example.com/feed.xml",
        description="Test description",
        author="Test Author",
        artwork_url="https://example.com/art.jpg",
    )

    data = show.to_dict()

    assert data["id"] == 1
    assert data["title"] == "Test Show"
    assert data["feed_url"] == "https://example.com/feed.xml"
    assert data["description"] == "Test description"
    assert data["author"] == "Test Author"
    assert data["artwork_url"] == "https://example.com/art.jpg"
    assert isinstance(data["added_at"], str)
    assert data["backfill_count"] == 5


def test_show_from_dict():
    """Test Show.from_dict() method."""
    data = {
        "id": 1,
        "title": "Test Show",
        "feed_url": "https://example.com/feed.xml",
        "description": "Test description",
        "author": "Test Author",
        "artwork_url": "https://example.com/art.jpg",
        "added_at": "2026-01-19T10:00:00",
        "last_checked": None,
        "backfill_count": 10,
    }

    show = Show.from_dict(data)

    assert show.id == 1
    assert show.title == "Test Show"
    assert show.feed_url == "https://example.com/feed.xml"
    assert show.description == "Test description"
    assert show.author == "Test Author"
    assert show.backfill_count == 10
