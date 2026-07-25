"""Tests for revealing the finished file in the OS file manager.

Covers the low-level ``reveal_in_file_manager`` platform dispatch and the
``_maybe_reveal`` gating in the workflow (flag off, non-interactive, once-per-run).
"""

import argparse
import subprocess
from unittest.mock import patch

import pytest

from pidcast import workflow
from pidcast.utils import reveal_in_file_manager


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "2026-07-25_Episode.md"
    path.write_text("# transcript", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def reset_latch():
    workflow._revealed_once = False
    yield
    workflow._revealed_once = False


def _args(**overrides):
    defaults = {"reveal": True}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- reveal_in_file_manager ------------------------------------------------


def test_macos_selects_the_file(transcript):
    with patch("sys.platform", "darwin"), patch("subprocess.run") as run:
        assert reveal_in_file_manager(transcript) is True
    assert run.call_args[0][0] == ["open", "-R", str(transcript.resolve())]


def test_linux_opens_the_parent_directory(transcript):
    with patch("sys.platform", "linux"), patch("subprocess.run") as run:
        assert reveal_in_file_manager(transcript) is True
    assert run.call_args[0][0] == ["xdg-open", str(transcript.resolve().parent)]


def test_windows_ignores_explorer_exit_status(transcript):
    # explorer.exe exits non-zero even on success, so check must be disabled.
    with patch("sys.platform", "win32"), patch("subprocess.run") as run:
        assert reveal_in_file_manager(transcript) is True
    assert run.call_args.kwargs["check"] is False


def test_missing_path_is_a_no_op(tmp_path):
    with patch("subprocess.run") as run:
        assert reveal_in_file_manager(tmp_path / "gone.md") is False
    run.assert_not_called()


def test_file_manager_failure_is_swallowed(transcript):
    with patch("subprocess.run", side_effect=FileNotFoundError("no open(1)")):
        assert reveal_in_file_manager(transcript) is False


def test_nonzero_exit_is_swallowed(transcript):
    error = subprocess.CalledProcessError(1, "open")
    with patch("sys.platform", "darwin"), patch("subprocess.run", side_effect=error):
        assert reveal_in_file_manager(transcript) is False


# --- _maybe_reveal gating --------------------------------------------------


def test_reveals_first_existing_candidate(tmp_path, transcript):
    analysis = tmp_path / "analysis.md"
    analysis.write_text("# analysis", encoding="utf-8")
    with (
        patch("pidcast.utils.is_interactive", return_value=True),
        patch("pidcast.utils.reveal_in_file_manager", return_value=True) as reveal,
    ):
        workflow._maybe_reveal(_args(), analysis, transcript)
    reveal.assert_called_once_with(analysis)


def test_falls_back_when_earlier_candidate_is_missing(tmp_path, transcript):
    with (
        patch("pidcast.utils.is_interactive", return_value=True),
        patch("pidcast.utils.reveal_in_file_manager", return_value=True) as reveal,
    ):
        workflow._maybe_reveal(_args(), None, tmp_path / "gone.md", transcript)
    reveal.assert_called_once_with(transcript)


def test_no_reveal_when_flag_is_off(transcript):
    with patch("pidcast.utils.reveal_in_file_manager") as reveal:
        workflow._maybe_reveal(_args(reveal=False), transcript)
    reveal.assert_not_called()


def test_no_reveal_when_flag_is_absent(transcript):
    with patch("pidcast.utils.reveal_in_file_manager") as reveal:
        workflow._maybe_reveal(argparse.Namespace(), transcript)
    reveal.assert_not_called()


def test_no_reveal_in_non_interactive_runs(transcript):
    # A scheduled `lib sync` must never pop a file-manager window.
    with (
        patch("pidcast.utils.is_interactive", return_value=False),
        patch("pidcast.utils.reveal_in_file_manager") as reveal,
    ):
        workflow._maybe_reveal(_args(), transcript)
    reveal.assert_not_called()


def test_reveals_at_most_once_per_process(transcript):
    with (
        patch("pidcast.utils.is_interactive", return_value=True),
        patch("pidcast.utils.reveal_in_file_manager", return_value=True) as reveal,
    ):
        workflow._maybe_reveal(_args(), transcript)
        workflow._maybe_reveal(_args(), transcript)
    assert reveal.call_count == 1


def _resolved_reveal(argv):
    """Run ``main()`` with a stubbed handler and return the resolved ``reveal``."""
    captured = {}

    def _capture(args):
        captured["reveal"] = args.reveal

    with (
        patch("sys.argv", ["pidcast", *argv]),
        patch("pidcast.commands.transcribe.cmd_transcribe", _capture),
        patch("pidcast.commands.lib.handle_sync", _capture),
        patch("pidcast.cli.setup_logging"),
    ):
        from pidcast.cli import main

        main()
    return captured["reveal"]


def _config(**values):
    return patch("pidcast.config_manager.ConfigManager.load_config", return_value=values)


def test_config_default_applies_to_transcribe():
    with _config(reveal_on_success=True):
        assert _resolved_reveal(["transcribe", "audio.mp3"]) is True


def test_explicit_no_reveal_beats_config():
    with _config(reveal_on_success=True):
        assert _resolved_reveal(["transcribe", "audio.mp3", "--no-reveal"]) is False


def test_reveal_defaults_off_without_config_key():
    with _config(backfill_limit=5):
        assert _resolved_reveal(["transcribe", "audio.mp3"]) is False


def test_lib_sync_ignores_the_config_default():
    # Batch verbs opt out: a config-wide default must not open a window per episode.
    with _config(reveal_on_success=True):
        assert _resolved_reveal(["lib", "sync"]) is False


def test_lib_sync_honors_an_explicit_flag():
    with _config(reveal_on_success=False):
        assert _resolved_reveal(["lib", "sync", "--reveal"]) is True


def test_a_failed_reveal_does_not_latch(transcript):
    with (
        patch("pidcast.utils.is_interactive", return_value=True),
        patch("pidcast.utils.reveal_in_file_manager", return_value=False) as reveal,
    ):
        workflow._maybe_reveal(_args(), transcript)
        workflow._maybe_reveal(_args(), transcript)
    assert reveal.call_count == 2
