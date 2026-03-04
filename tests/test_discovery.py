"""Tests for pidcast.discovery module."""

import sqlite3
from unittest.mock import MagicMock, patch

from pidcast.discovery import (
    discover_podcast,
    prompt_user_selection,
    search_apple_podcasts_local,
    search_itunes_api,
)

# ============================================================================
# search_apple_podcasts_local
# ============================================================================


class TestSearchApplePodcastsLocal:
    def test_returns_empty_when_db_missing(self):
        with patch("pidcast.discovery.APPLE_PODCASTS_DB") as mock_path:
            mock_path.exists.return_value = False
            result = search_apple_podcasts_local("changelog")
        assert result == []

    def test_queries_db_and_returns_results(self, tmp_path):
        db_path = tmp_path / "MTLibrary.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE ZMTPODCAST "
            "(ZTITLE TEXT, ZFEEDURL TEXT, ZAUTHOR TEXT, ZITEMDESCRIPTION TEXT)"
        )
        conn.execute(
            "INSERT INTO ZMTPODCAST VALUES (?, ?, ?, ?)",
            ("The Changelog", "https://feeds.changelog.com/podcast.xml", "Changelog", "Dev podcast"),
        )
        conn.commit()
        conn.close()

        with patch("pidcast.discovery.APPLE_PODCASTS_DB", db_path):
            result = search_apple_podcasts_local("changelog")

        assert len(result) == 1
        assert result[0]["title"] == "The Changelog"
        assert result[0]["feed_url"] == "https://feeds.changelog.com/podcast.xml"
        assert result[0]["author"] == "Changelog"
        assert result[0]["source"] == "local"

    def test_skips_rows_with_empty_feed_url(self, tmp_path):
        db_path = tmp_path / "MTLibrary.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE ZMTPODCAST "
            "(ZTITLE TEXT, ZFEEDURL TEXT, ZAUTHOR TEXT, ZITEMDESCRIPTION TEXT)"
        )
        conn.execute("INSERT INTO ZMTPODCAST VALUES (?, ?, ?, ?)", ("No Feed", "", "Author", "Desc"))
        conn.execute("INSERT INTO ZMTPODCAST VALUES (?, ?, ?, ?)", ("No Feed2", None, "Author", "Desc"))
        conn.commit()
        conn.close()

        with patch("pidcast.discovery.APPLE_PODCASTS_DB", db_path):
            result = search_apple_podcasts_local("No Feed")

        assert result == []

    def test_returns_empty_on_sqlite_error(self, tmp_path):
        db_path = tmp_path / "broken.sqlite"
        db_path.write_bytes(b"not a sqlite database")

        with patch("pidcast.discovery.APPLE_PODCASTS_DB", db_path):
            result = search_apple_podcasts_local("anything")

        assert result == []

    def test_truncates_long_description(self, tmp_path):
        db_path = tmp_path / "MTLibrary.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE ZMTPODCAST "
            "(ZTITLE TEXT, ZFEEDURL TEXT, ZAUTHOR TEXT, ZITEMDESCRIPTION TEXT)"
        )
        long_desc = "x" * 500
        conn.execute(
            "INSERT INTO ZMTPODCAST VALUES (?, ?, ?, ?)",
            ("Show", "https://feed.example.com/rss", "Author", long_desc),
        )
        conn.commit()
        conn.close()

        with patch("pidcast.discovery.APPLE_PODCASTS_DB", db_path):
            result = search_apple_podcasts_local("Show")

        assert len(result[0]["description"]) == 200


# ============================================================================
# search_itunes_api
# ============================================================================


class TestSearchItunesApi:
    def _make_response(self, results):
        import json

        body = json.dumps({"results": results}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_matching_podcasts(self):
        results = [
            {
                "collectionName": "The Changelog",
                "feedUrl": "https://feeds.changelog.com/podcast.xml",
                "artistName": "Changelog Media",
                "description": "Dev podcast",
            }
        ]
        with patch("urllib.request.urlopen", return_value=self._make_response(results)):
            found = search_itunes_api("changelog")

        assert len(found) == 1
        assert found[0]["title"] == "The Changelog"
        assert found[0]["feed_url"] == "https://feeds.changelog.com/podcast.xml"
        assert found[0]["source"] == "itunes"

    def test_skips_items_without_feed_url(self):
        results = [{"collectionName": "No Feed", "feedUrl": "", "artistName": "x"}]
        with patch("urllib.request.urlopen", return_value=self._make_response(results)):
            found = search_itunes_api("no feed")

        assert found == []

    def test_returns_empty_on_network_error(self):
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            found = search_itunes_api("anything")

        assert found == []

    def test_returns_empty_on_timeout(self):

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            found = search_itunes_api("anything")

        assert found == []


# ============================================================================
# discover_podcast
# ============================================================================


class TestDiscoverPodcast:
    def test_merges_and_deduplicates_by_feed_url(self):
        local = [{"title": "Show A", "feed_url": "https://feed.example.com/rss", "author": "A", "description": "", "source": "local"}]
        remote = [
            {"title": "Show A", "feed_url": "https://feed.example.com/rss/", "author": "A", "description": "", "source": "itunes"},
            {"title": "Show B", "feed_url": "https://other.example.com/rss", "author": "B", "description": "", "source": "itunes"},
        ]
        with (
            patch("pidcast.discovery.search_apple_podcasts_local", return_value=local),
            patch("pidcast.discovery.search_itunes_api", return_value=remote),
        ):
            result = discover_podcast("show")

        assert len(result) == 2
        titles = [r["title"] for r in result]
        assert "Show A" in titles
        assert "Show B" in titles

    def test_local_results_take_precedence_in_dedup(self):
        local = [{"title": "Local Show", "feed_url": "https://feed.example.com/rss", "author": "L", "description": "", "source": "local"}]
        remote = [{"title": "Remote Show", "feed_url": "https://feed.example.com/rss", "author": "R", "description": "", "source": "itunes"}]
        with (
            patch("pidcast.discovery.search_apple_podcasts_local", return_value=local),
            patch("pidcast.discovery.search_itunes_api", return_value=remote),
        ):
            result = discover_podcast("show")

        assert len(result) == 1
        assert result[0]["source"] == "local"

    def test_skips_itunes_when_local_has_10_results(self):
        local = [
            {"title": f"Show {i}", "feed_url": f"https://feed{i}.example.com/rss", "author": "", "description": "", "source": "local"}
            for i in range(10)
        ]
        with (
            patch("pidcast.discovery.search_apple_podcasts_local", return_value=local),
            patch("pidcast.discovery.search_itunes_api") as mock_itunes,
        ):
            result = discover_podcast("show")

        mock_itunes.assert_not_called()
        assert len(result) == 10

    def test_calls_itunes_when_local_has_fewer_than_10(self):
        local = [{"title": "Show 1", "feed_url": "https://feed1.example.com/rss", "author": "", "description": "", "source": "local"}]
        remote = [{"title": "Show 2", "feed_url": "https://feed2.example.com/rss", "author": "", "description": "", "source": "itunes"}]
        with (
            patch("pidcast.discovery.search_apple_podcasts_local", return_value=local),
            patch("pidcast.discovery.search_itunes_api", return_value=remote),
        ):
            result = discover_podcast("show")

        assert len(result) == 2


# ============================================================================
# prompt_user_selection
# ============================================================================


class TestPromptUserSelection:
    RESULTS = [
        {"title": "Show A", "feed_url": "https://a.com/rss", "author": "Auth A", "source": "local"},
        {"title": "Show B", "feed_url": "https://b.com/rss", "author": "Auth B", "source": "itunes"},
    ]

    def test_returns_selected_item(self):
        with patch("builtins.input", return_value="1"):
            result = prompt_user_selection(self.RESULTS)
        assert result == self.RESULTS[0]

    def test_returns_none_on_quit(self):
        with patch("builtins.input", return_value="q"):
            result = prompt_user_selection(self.RESULTS)
        assert result is None

    def test_returns_none_on_empty_input(self):
        with patch("builtins.input", return_value=""):
            result = prompt_user_selection(self.RESULTS)
        assert result is None

    def test_reprompts_on_invalid_input_then_selects(self):
        with patch("builtins.input", side_effect=["99", "abc", "2"]):
            result = prompt_user_selection(self.RESULTS)
        assert result == self.RESULTS[1]
