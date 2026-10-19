"""Tests for the IO-free checklist state behind ui.select_many."""

import pytest

from pidcast.ui import Checklist, _split_keys


def _press(state: Checklist, *keys: str) -> str | None:
    result = None
    for key in keys:
        result = state.press(key)
    return result


def test_starts_all_selected_and_enter_confirms():
    state = Checklist(selected=[True, True, True])
    assert state.press("\r") == "done"
    assert state.picked == [0, 1, 2]


def test_arrows_and_space_toggle_under_cursor():
    state = Checklist(selected=[True, True, True])
    _press(state, "\x1b[B", " ", "j", " ")
    assert state.picked == [0]


def test_cursor_wraps():
    state = Checklist(selected=[True, True])
    _press(state, "\x1b[A")
    assert state.cursor == 1


def test_digit_toggles_that_item_and_moves_cursor():
    state = Checklist(selected=[True, True, True])
    _press(state, "2")
    assert state.picked == [0, 2]
    assert state.cursor == 1


def test_out_of_range_digit_is_ignored():
    state = Checklist(selected=[True, True])
    _press(state, "9", "0")
    assert state.picked == [0, 1]


def test_a_toggles_all_then_none():
    state = Checklist(selected=[True, False])
    _press(state, "a")
    assert state.picked == [0, 1]
    _press(state, "a")
    assert state.picked == []


def test_enter_with_nothing_selected_hints_instead_of_finishing():
    state = Checklist(selected=[False, False])
    assert state.press("\r") is None
    assert state.hint
    state.press("1")
    assert state.hint == ""


@pytest.mark.parametrize("key", ["\x1b", "\x03", "q", "Q"])
def test_quit_keys(key):
    assert Checklist(selected=[True]).press(key) == "quit"


def test_split_keys_keeps_escape_sequences_whole():
    assert _split_keys("\x1b[A") == ["\x1b[A"]
    assert _split_keys("13") == ["1", "3"]
    assert _split_keys("\x1b") == ["\x1b"]
