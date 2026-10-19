"""Tests for pidcast.knowledge_notes (theme pass, focus prompt, document rendering)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pidcast.config import VideoInfo
from pidcast.discord_recap import Story
from pidcast.knowledge_notes import (
    WRITER_TIMEOUT,
    Notes,
    build_notes_prompt,
    clean_output,
    find_topics,
    list_styles,
    load_preset,
    render_document,
    write_notes,
)
from pidcast.providers.claude_provider import ClaudeRun


def _topics() -> list[Story]:
    return [
        Story(None, "intro", "Welcome", "Guest joined in 2015.", 0, 0),
        Story(1, "story", "Michelangelo ML platform", "Built in 2016.", 1, 3),
        Story(None, "ad", "WorkOS Airlock", "Agent auth.", 4, 4),
        Story(2, "story", "MCP Gateway", "Central registry for tools.", 5, 7),
        Story(3, "story", "Code review agents", "Catch 30% of bugs.", 8, 9),
    ]


def _info() -> VideoInfo:
    return VideoInfo(
        title="Uber's AI infra",
        webpage_url="https://example.com/ep",
        channel="The Pragmatic Engineer",
        duration_string="1:33:18",
    )


class TestPresets:
    def test_deepdive_is_listed_and_has_prompts(self):
        assert "deepdive" in list_styles()
        preset = load_preset("deepdive")
        assert "## Executive summary" in preset["system_prompt"]
        for var in ("{title}", "{topics_block}", "{steer_block}", "{transcript}"):
            assert var in preset["user_prompt"]

    def test_unknown_style_lists_available(self):
        with pytest.raises(ValueError, match="deepdive"):
            load_preset("nope")


class TestFindTopics:
    def test_topic_kind_becomes_pickable(self):
        raw = json.dumps(
            {
                "stories": [
                    {"kind": "intro", "headline": "Hi", "gist": "g", "start": 0, "end": 0},
                    {"kind": "topic", "headline": "A", "gist": "g", "start": 1, "end": 2},
                    {"kind": "ad", "headline": "Ad", "gist": "g", "start": 3, "end": 3},
                    {"kind": "topic", "headline": "B", "gist": "g", "start": 4, "end": 5},
                ]
            }
        )
        run = ClaudeRun(text=raw, model="haiku", cost_usd=0.002)
        with patch("pidcast.knowledge_notes.run_claude", return_value=run) as mock:
            topics, cost = find_topics([f"s{i}" for i in range(6)], "Ep", "haiku")
        assert [t.id for t in topics] == [None, 1, None, 2]
        assert [t.pickable for t in topics] == [False, True, False, True]
        assert cost == 0.002
        assert "[5] s5" in mock.call_args.args[0]


class TestBuildNotesPrompt:
    def test_tags_focus_and_brief_with_focus_first_in_pick_order(self):
        _, user = build_notes_prompt(
            "BODY", _info(), _topics(), [3, 1], None, load_preset("deepdive")
        )
        lines = [ln for ln in user.splitlines() if ln.startswith("- [")]
        assert lines == [
            "- [FOCUS] Code review agents: Catch 30% of bugs.",
            "- [FOCUS] Michelangelo ML platform: Built in 2016.",
            "- [BRIEF] MCP Gateway: Central registry for tools.",
        ]
        assert "WorkOS" not in user.split("Transcript:")[0]

    def test_sends_full_transcript_and_metadata(self):
        _, user = build_notes_prompt(
            "FULL TRANSCRIPT", _info(), _topics(), [1], None, load_preset("deepdive")
        )
        assert user.rstrip().endswith("Write the notes now.")
        assert "FULL TRANSCRIPT" in user
        assert "The Pragmatic Engineer" in user and "1:33:18" in user

    def test_steer_only_when_given(self):
        preset = load_preset("deepdive")
        _, without = build_notes_prompt("B", _info(), _topics(), [1], None, preset)
        _, with_steer = build_notes_prompt("B", _info(), _topics(), [1], "  eval infra ", preset)
        assert "Reader's focus" not in without
        assert "Reader's focus (weight the notes toward this): eval infra\n" in with_steer

    def test_unknown_channel_and_duration(self):
        info = VideoInfo(title="T", channel="Unknown", duration_string="")
        _, user = build_notes_prompt("B", info, _topics(), [1], None, load_preset("deepdive"))
        assert "Channel: unknown" in user and "Duration: unknown" in user


class TestCleanOutput:
    @pytest.mark.parametrize(
        "raw", ["```markdown\n## A\nbody\n```", "```\n## A\nbody\n```", "  ## A\nbody\n"]
    )
    def test_strips_wrapping_fence(self, raw):
        assert clean_output(raw) == "## A\nbody"

    def test_keeps_inner_fences(self):
        raw = "## A\n```py\nx = 1\n```\nmore"
        assert clean_output(raw) == raw


def test_write_notes_uses_long_timeout():
    run = ClaudeRun(text="```md\n## Executive summary\nX\n```", model="opus", cost_usd=0.4)
    with patch("pidcast.knowledge_notes.run_claude", return_value=run) as mock:
        notes = write_notes("SYS", "USER", "opus")
    assert notes == Notes(text="## Executive summary\nX", model="opus", cost_usd=0.4)
    assert mock.call_args.kwargs == {"system_prompt": "SYS", "timeout": WRITER_TIMEOUT}


class TestRenderDocument:
    def test_front_matter_h1_and_body(self):
        notes = Notes(text="## Executive summary\nX", model="claude-opus-5-5", cost_usd=None)
        path = Path("/data/2026-10-19_Ubers_AI_infra.md")
        doc = render_document(notes, _info(), path, [_topics()[3]], "deepdive", ["uber"])
        front, body = doc.split("---\n", 2)[1:]
        assert "date: 2026-10-19" in front
        assert 'url: "https://example.com/ep"' in front
        assert "transcript: /data/2026-10-19_Ubers_AI_infra.md" in front
        assert "focus: ['MCP Gateway']" in front
        assert "tags: ['podcast-notes', 'notes-deepdive', 'uber']" in front
        assert body.startswith("\n# Uber's AI infra\n\n## Executive summary\nX")

    def test_omits_unknown_metadata(self):
        notes = Notes(text="X", model="m", cost_usd=None)
        doc = render_document(
            notes, VideoInfo(title="T", channel="Unknown"), Path("t.md"), [], "deepdive"
        )
        assert "channel" not in doc and "date:" not in doc and "url:" not in doc
