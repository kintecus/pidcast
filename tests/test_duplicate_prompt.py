"""Tests for the redesigned duplicate-detection prompt.

Detection logic lives in test_duplicate_detection.py; this file covers the new
single-keypress selector (ui.select_key) and the panel/action mapping in
duplicate.prompt_duplicate_detected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pidcast import ui
from pidcast.duplicate import DuplicateAction, prompt_duplicate_detected

CHOICES = [
    ("r", "re-transcribe"),
    ("a", "analyze"),
    ("f", "force"),
    ("c", "cancel"),
]


# ---------------------------------------------------------------------------
# select_key — TTY path (monkeypatch the inner getch)
# ---------------------------------------------------------------------------


def _feed_keys(monkeypatch, keys):
    """Make stdin a TTY and feed select_key one key per _read_raw_key() call."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    seq = iter(keys)
    monkeypatch.setattr(ui, "_read_raw_key", lambda: next(seq))


def test_tty_keypress_matches_key(monkeypatch):
    _feed_keys(monkeypatch, ["r"])
    assert ui.select_key("", CHOICES, default="c") == "r"


def test_tty_keypress_case_insensitive(monkeypatch):
    _feed_keys(monkeypatch, ["R"])
    assert ui.select_key("", CHOICES, default="c") == "r"


def test_tty_enter_returns_default(monkeypatch):
    _feed_keys(monkeypatch, ["\r"])
    assert ui.select_key("", CHOICES, default="c") == "c"

    _feed_keys(monkeypatch, ["\n"])
    assert ui.select_key("", CHOICES, default="c") == "c"


def test_tty_esc_returns_cancel(monkeypatch):
    _feed_keys(monkeypatch, ["\x1b"])
    assert ui.select_key("", CHOICES, default="r") == "c"


def test_tty_ctrl_c_returns_cancel(monkeypatch):
    _feed_keys(monkeypatch, ["\x03"])
    assert ui.select_key("", CHOICES, default="r") == "c"


def test_tty_eof_returns_cancel(monkeypatch):
    _feed_keys(monkeypatch, [""])
    assert ui.select_key("", CHOICES, default="r") == "c"


def test_tty_invalid_key_rereads_until_valid(monkeypatch):
    # 'x' and 'z' are not choices -> re-read; 'f' is -> returned.
    _feed_keys(monkeypatch, ["x", "z", "f"])
    assert ui.select_key("", CHOICES, default="c") == "f"


def test_cancel_falls_back_to_default_when_no_cancel_choice(monkeypatch):
    # No 'c' choice -> Esc resolves to the default key, not a missing cancel.
    no_cancel = [("r", "re-transcribe"), ("f", "force")]
    _feed_keys(monkeypatch, ["\x1b"])
    assert ui.select_key("", no_cancel, default="r") == "r"


# ---------------------------------------------------------------------------
# select_key — non-TTY fallback path (input())
# ---------------------------------------------------------------------------


def test_non_tty_matches_first_char(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "a")
    assert ui.select_key("hint", CHOICES, default="c") == "a"


def test_non_tty_empty_line_returns_default(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    assert ui.select_key("hint", CHOICES, default="c") == "c"


def test_non_tty_eof_returns_cancel(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    def _raise(*a, **k):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert ui.select_key("hint", CHOICES, default="r") == "c"


def test_non_tty_invalid_then_valid(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    lines = iter(["nope", "f"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(lines))
    assert ui.select_key("hint", CHOICES, default="c") == "f"


# ---------------------------------------------------------------------------
# prompt_duplicate_detected — key -> DuplicateAction mapping
# ---------------------------------------------------------------------------


class _FakePrev:
    """Minimal stand-in for PreviousTranscription."""

    def __init__(self, transcript_path: Path, *, analysis_performed=False, analysis_type=None):
        self.video_title = "Some Episode Title"
        self.formatted_date = "June 30, 2026 at 10:00"
        self.analysis_performed = analysis_performed
        self.analysis_type = analysis_type
        self._transcript_path = transcript_path

    @property
    def transcript_path(self) -> Path:
        return self._transcript_path


@pytest.fixture
def existing_transcript(tmp_path) -> Path:
    p = tmp_path / "2026-06-30_Some_Episode.md"
    p.write_text("# transcript\n")
    return p


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("r", DuplicateAction.RE_TRANSCRIBE),
        ("a", DuplicateAction.ANALYZE_EXISTING),
        ("f", DuplicateAction.FORCE_CONTINUE),
        ("c", DuplicateAction.CANCEL),
    ],
)
def test_key_maps_to_action(monkeypatch, existing_transcript, key, expected):
    prev = _FakePrev(existing_transcript)
    monkeypatch.setattr(ui, "select_key", lambda *a, **k: key)
    assert prompt_duplicate_detected(prev) is expected


def test_analyze_choice_omitted_when_transcript_missing(monkeypatch, tmp_path):
    prev = _FakePrev(tmp_path / "gone.md")  # never created -> does not exist

    captured = {}

    def _fake_select(prompt, choices, *, default):
        captured["choices"] = choices
        captured["default"] = default
        return "c"

    monkeypatch.setattr(ui, "select_key", _fake_select)

    action = prompt_duplicate_detected(prev)
    assert action is DuplicateAction.CANCEL

    keys = [k for k, _ in captured["choices"]]
    assert keys == ["r", "f", "c"]  # 'a' omitted
    assert "a" not in keys
    assert captured["default"] == "c"


def test_analyze_choice_present_when_transcript_exists(monkeypatch, existing_transcript):
    prev = _FakePrev(existing_transcript)

    captured = {}

    def _fake_select(prompt, choices, *, default):
        captured["choices"] = choices
        return "c"

    monkeypatch.setattr(ui, "select_key", _fake_select)

    prompt_duplicate_detected(prev)
    keys = [k for k, _ in captured["choices"]]
    assert keys == ["r", "a", "f", "c"]
