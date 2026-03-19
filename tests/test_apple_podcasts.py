"""Tests for Apple Podcasts URL resolution."""

from datetime import datetime
from unittest.mock import patch

import pytest

from pidcast.apple_podcasts import (
    find_episode_in_feed,
    is_apple_podcasts_url,
    parse_apple_podcasts_url,
    resolve_apple_podcasts_url,
)
from pidcast.exceptions import ApplePodcastsResolutionError
from pidcast.rss import Episode

# ---------------------------------------------------------------------------
# is_apple_podcasts_url
# ---------------------------------------------------------------------------


class TestIsApplePodcastsUrl:
    def test_standard_url(self):
        assert is_apple_podcasts_url(
            "https://podcasts.apple.com/us/podcast/some-show/id123456?i=789"
        )

    def test_no_country(self):
        assert is_apple_podcasts_url(
            "https://podcasts.apple.com/podcast/id123456?i=789"
        )

    def test_itunes_domain(self):
        assert is_apple_podcasts_url(
            "https://itunes.apple.com/us/podcast/id123456?i=789"
        )

    def test_youtube_url(self):
        assert not is_apple_podcasts_url("https://www.youtube.com/watch?v=abc")

    def test_random_url(self):
        assert not is_apple_podcasts_url("https://example.com/podcast")

    def test_empty_string(self):
        assert not is_apple_podcasts_url("")

    def test_not_a_url(self):
        assert not is_apple_podcasts_url("/some/local/path.mp3")


# ---------------------------------------------------------------------------
# parse_apple_podcasts_url
# ---------------------------------------------------------------------------


class TestParseApplePodcastsUrl:
    def test_full_url_with_episode(self):
        url = "https://podcasts.apple.com/us/podcast/ep-title/id1234567890?i=1000600000001"
        cid, tid = parse_apple_podcasts_url(url)
        assert cid == "1234567890"
        assert tid == "1000600000001"

    def test_show_only_url(self):
        url = "https://podcasts.apple.com/us/podcast/show-name/id1234567890"
        cid, tid = parse_apple_podcasts_url(url)
        assert cid == "1234567890"
        assert tid is None

    def test_url_with_extra_query_params(self):
        url = "https://podcasts.apple.com/us/podcast/show/id111?i=222&mt=2"
        cid, tid = parse_apple_podcasts_url(url)
        assert cid == "111"
        assert tid == "222"

    def test_no_id_in_path(self):
        with pytest.raises(ApplePodcastsResolutionError, match="Could not extract"):
            parse_apple_podcasts_url("https://podcasts.apple.com/us/podcast/show-name")


# ---------------------------------------------------------------------------
# find_episode_in_feed
# ---------------------------------------------------------------------------


def _make_episode(title: str, pub_date: datetime, audio_url: str = "https://cdn.example.com/ep.mp3") -> Episode:
    return Episode(
        guid=f"guid-{title}",
        title=title,
        description="Test episode",
        pub_date=pub_date,
        duration=3600,
        audio_url=audio_url,
    )


class TestFindEpisodeInFeed:
    @patch("pidcast.apple_podcasts.RSSParser.parse_feed")
    def test_match_by_title(self, mock_parse):
        episodes = [
            _make_episode("Episode One", datetime(2024, 1, 1)),
            _make_episode("Episode Two", datetime(2024, 1, 8)),
        ]
        mock_parse.return_value = ({}, episodes)

        result = find_episode_in_feed(
            "https://feed.example.com/rss",
            {"trackName": "Episode Two"},
        )
        assert result.title == "Episode Two"

    @patch("pidcast.apple_podcasts.RSSParser.parse_feed")
    def test_match_by_title_normalized(self, mock_parse):
        episodes = [
            _make_episode("The Best Episode! (Part 1)", datetime(2024, 1, 1)),
        ]
        mock_parse.return_value = ({}, episodes)

        result = find_episode_in_feed(
            "https://feed.example.com/rss",
            {"trackName": "The Best Episode (Part 1)"},
        )
        assert result.title == "The Best Episode! (Part 1)"

    @patch("pidcast.apple_podcasts.RSSParser.parse_feed")
    def test_match_by_date_fallback(self, mock_parse):
        episodes = [
            _make_episode("Ep A", datetime(2024, 3, 10)),
            _make_episode("Ep B", datetime(2024, 3, 15)),
        ]
        mock_parse.return_value = ({}, episodes)

        result = find_episode_in_feed(
            "https://feed.example.com/rss",
            {"trackName": "Completely Different Title", "releaseDate": "2024-03-15T00:00:00Z"},
        )
        assert result.title == "Ep B"

    @patch("pidcast.apple_podcasts.RSSParser.parse_feed")
    def test_no_match_raises(self, mock_parse):
        episodes = [_make_episode("Only Episode", datetime(2024, 1, 1))]
        mock_parse.return_value = ({}, episodes)

        with pytest.raises(ApplePodcastsResolutionError, match="Could not find"):
            find_episode_in_feed(
                "https://feed.example.com/rss",
                {"trackName": "Nonexistent", "releaseDate": "2024-06-01T00:00:00Z"},
            )


# ---------------------------------------------------------------------------
# resolve_apple_podcasts_url (integration with mocks)
# ---------------------------------------------------------------------------


class TestResolveApplePodcastsUrl:
    @patch("pidcast.apple_podcasts.RSSParser.parse_feed")
    @patch("pidcast.apple_podcasts._itunes_fetch")
    def test_full_resolution(self, mock_fetch, mock_parse):
        # First call: collection lookup returns show with feedUrl
        collection_result = [
            {
                "wrapperType": "track",
                "kind": "podcast",
                "collectionName": "Test Podcast",
                "artistName": "Test Author",
                "feedUrl": "https://feed.example.com/rss",
            },
        ]
        # Second call: episode lookup returns collection + matching episode
        episode_results = [
            collection_result[0],
            {
                "wrapperType": "track",
                "kind": "podcast-episode",
                "trackId": 222,
                "trackName": "Great Episode",
                "releaseDate": "2024-05-01T00:00:00Z",
            },
        ]
        mock_fetch.side_effect = [collection_result, episode_results]

        # Mock RSS feed
        episodes = [
            _make_episode("Great Episode", datetime(2024, 5, 1), "https://cdn.example.com/great.mp3"),
        ]
        mock_parse.return_value = ({"title": "Test Podcast"}, episodes)

        url = "https://podcasts.apple.com/us/podcast/test/id111?i=222"
        audio_url, video_info = resolve_apple_podcasts_url(url)

        assert audio_url == "https://cdn.example.com/great.mp3"
        assert video_info.title == "Great Episode"
        assert video_info.channel == "Test Podcast"
        assert video_info.uploader == "Test Author"
        assert video_info.webpage_url == url

    def test_show_level_url_raises(self):
        with pytest.raises(ApplePodcastsResolutionError, match="show-level URL"):
            resolve_apple_podcasts_url(
                "https://podcasts.apple.com/us/podcast/show-name/id123456"
            )
