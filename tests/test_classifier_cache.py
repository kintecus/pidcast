"""Tests for the on-disk cache around the Haiku story/theme pass."""

import json
from unittest.mock import patch

import pytest

from pidcast import classifier_cache
from pidcast.discord_recap import find_stories
from pidcast.providers.claude_provider import ClaudeRun

RAW = json.dumps(
    {"stories": [{"kind": "story", "headline": "A", "gist": "g", "start": 0, "end": 1}]}
)
SEGMENTS = ["one", "two"]


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier_cache, "CACHE_DIR", tmp_path)


def _run(text=RAW):
    return ClaudeRun(text=text, model="haiku", cost_usd=0.002)


def test_second_call_replays_cache_for_free():
    with patch("pidcast.classifier_cache.run_claude", return_value=_run()) as mock:
        first, cost1 = find_stories(SEGMENTS, "Ep", "haiku")
        second, cost2 = find_stories(SEGMENTS, "Ep", "haiku")
    assert mock.call_count == 1
    assert first == second
    assert (cost1, cost2) == (0.002, 0.0)


def test_refresh_bypasses_and_overwrites():
    with patch("pidcast.classifier_cache.run_claude", return_value=_run()) as mock:
        find_stories(SEGMENTS, "Ep", "haiku")
        find_stories(SEGMENTS, "Ep", "haiku", refresh=True)
    assert mock.call_count == 2


@pytest.mark.parametrize(
    ("segments", "title", "model"),
    [(["one", "changed"], "Ep", "haiku"), (SEGMENTS, "Other", "haiku"), (SEGMENTS, "Ep", "sonnet")],
)
def test_changed_input_misses(segments, title, model):
    with patch("pidcast.classifier_cache.run_claude", return_value=_run()) as mock:
        find_stories(SEGMENTS, "Ep", "haiku")
        find_stories(segments, title, model)
    assert mock.call_count == 2


def test_unparseable_reply_is_not_cached():
    with patch("pidcast.classifier_cache.run_claude", return_value=_run("no json")) as mock:
        for _ in range(2):
            with pytest.raises(Exception, match="no JSON"):
                find_stories(SEGMENTS, "Ep", "haiku")
    assert mock.call_count == 2


def test_corrupt_entry_falls_back_to_model(tmp_path):
    with patch("pidcast.classifier_cache.run_claude", return_value=_run()) as mock:
        find_stories(SEGMENTS, "Ep", "haiku")
        for f in tmp_path.glob("*.json"):
            f.write_text("{broken")
        find_stories(SEGMENTS, "Ep", "haiku")
    assert mock.call_count == 2
