"""Tests for pidcast.discord_recap (story pass, picking, prompt building, length guard)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pidcast.discord_recap import (
    Story,
    build_recap_prompt,
    list_styles,
    load_preset,
    parse_pick,
    parse_stories,
    segment,
    skipped_summary,
    source_line,
    write_recap,
)
from pidcast.exceptions import AnalysisError
from pidcast.providers.claude_provider import ClaudeRun


def _stories() -> list[Story]:
    return [
        Story(None, "intro", "Welcome", "Host says hi.", 0, 0),
        Story(1, "story", "Nvidia B400", "3x faster.", 1, 2),
        Story(None, "ad", "VPN promo", "Code BREW.", 3, 3),
        Story(2, "story", "Meta layoffs", "1,200 cut.", 4, 5),
    ]


def _run(text: str, cost: float | None = 0.01) -> ClaudeRun:
    return ClaudeRun(text=text, model="claude-opus-5-5", cost_usd=cost)


class TestSegment:
    def test_groups_sentences_to_target_and_drops_blanks(self):
        body = "One. Two is longer.\n\nThree!   Four?"
        segments = segment(body, target=10)
        assert segments == ["One. Two is longer.", "Three! Four?"]
        assert all(s.strip() for s in segments)

    def test_keeps_trailing_remainder(self):
        assert segment("Short.", target=150) == ["Short."]


class TestParseStories:
    def test_renumbers_and_clamps_ranges(self):
        raw = json.dumps(
            {
                "stories": [
                    {"id": 7, "kind": "story", "headline": "A", "gist": "a", "start": -3, "end": 2},
                    {"id": 7, "kind": "weird", "headline": "B", "gist": "b", "start": 3, "end": 99},
                ]
            }
        )
        stories = parse_stories(f"```json\n{raw}\n```", n_segments=5)
        assert [(s.id, s.kind, s.start, s.end) for s in stories] == [
            (1, "story", 0, 2),
            (2, "story", 3, 4),
        ]

    def test_numbers_only_pickable_stories(self):
        kinds = ["ad", "intro", "story", "ad", "story", "outro"]
        raw = json.dumps(
            {"stories": [{"kind": k, "start": i, "end": i} for i, k in enumerate(kinds)]}
        )
        stories = parse_stories(raw, n_segments=6)
        assert [s.id for s in stories] == [None, None, 1, None, 2, None]

    def test_raises_on_missing_json(self):
        with pytest.raises(AnalysisError, match="no JSON"):
            parse_stories("I could not find any stories.", n_segments=5)

    def test_raises_on_empty_list(self):
        with pytest.raises(AnalysisError, match="no stories"):
            parse_stories('{"stories": []}', n_segments=5)


class TestParsePick:
    def test_keeps_given_order_and_dedupes(self):
        assert parse_pick("2, 1 2", _stories()) == [2, 1]

    def test_empty_means_all_pickable(self):
        assert parse_pick("  ", _stories()) == [1, 2]

    @pytest.mark.parametrize("raw", ["0", "3", "9", "x"])
    def test_rejects_ads_intros_and_unknown_ids(self, raw):
        with pytest.raises(ValueError, match="not a story number"):
            parse_pick(raw, _stories())


class TestBuildRecapPrompt:
    def test_includes_only_picked_excerpts_in_pick_order(self):
        segments = [
            "hello",
            "B400 is out.",
            "Ships March.",
            "Use code BREW.",
            "Meta cuts.",
            "1,200.",
        ]
        preset = {
            "system_prompt": "SYS",
            "user_prompt": "Source line: {source_line}\n{n_stories} stories:\n{stories_block}",
        }
        system, user = build_recap_prompt(_stories(), [2, 1], segments, "Show | 2026-10-16", preset)
        assert system == "SYS"
        assert user.index("Meta cuts. 1,200.") < user.index("B400 is out. Ships March.")
        assert "Story 1: Meta layoffs" in user
        assert "BREW" not in user
        assert "hello" not in user
        assert "2 stories" in user

    def test_shipped_presets_render_without_episode_title(self):
        for style in list_styles():
            preset = load_preset(style)
            assert "{title}" not in preset["user_prompt"]
            _, user = build_recap_prompt(_stories(), [1], ["a", "b", "c"], "Show", preset)
            assert "Show" in user


class TestWriteRecap:
    def test_within_budget_makes_one_call(self):
        with patch("pidcast.discord_recap.run_claude", return_value=_run("x" * 100)) as run:
            recap = write_recap("SYS", "USER", "opus", budget=1900)
        assert run.call_count == 1
        assert recap.fits and not recap.shortened
        assert run.call_args.kwargs["system_prompt"] == "SYS"

    def test_over_budget_retries_once_with_draft_and_limit(self):
        long, short = "y" * 2100, "z" * 1500
        with patch(
            "pidcast.discord_recap.run_claude", side_effect=[_run(long), _run(short)]
        ) as run:
            recap = write_recap("SYS", "USER", "opus", budget=1900)
        assert run.call_count == 2
        retry_prompt = run.call_args.args[0]
        assert long in retry_prompt and "1850" in retry_prompt and "2100" in retry_prompt
        assert recap.text == short and recap.shortened and recap.fits
        assert recap.cost_usd == pytest.approx(0.02)

    def test_still_over_after_retry_is_flagged_not_retried_again(self):
        with patch(
            "pidcast.discord_recap.run_claude", side_effect=[_run("y" * 2100), _run("y" * 1950)]
        ) as run:
            recap = write_recap("SYS", "USER", "opus", budget=1900)
        assert run.call_count == 2
        assert not recap.fits


class TestPresetsAndSourceLine:
    def test_play_and_digest_ship(self):
        assert {"play", "digest"} <= set(list_styles())
        assert load_preset("play")["budget_chars"] == 1900

    def test_unknown_style_lists_available(self):
        with pytest.raises(ValueError, match="Available: .*play"):
            load_preset("sonnet")

    def test_source_line_uses_channel_and_filename_date(self):
        path = Path("/x/2026-10-16_Anthropics_Growth.md")
        assert source_line("Tech Brew Ride Home", path) == "Tech Brew Ride Home | 2026-10-16"
        assert source_line("Unknown", path) == "2026-10-16"
        assert source_line(None, Path("/x/episode.md")) == "episode"

    def test_skipped_summary_counts_kinds_in_order(self):
        stories = _stories() + [
            Story(None, "ad", "Promo", "", 6, 6),
            Story(None, "outro", "Bye", "", 7, 7),
        ]
        assert skipped_summary(stories) == "intro, 2 ads, outro"
        assert skipped_summary([s for s in stories if s.pickable]) == ""
