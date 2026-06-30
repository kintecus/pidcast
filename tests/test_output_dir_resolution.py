"""Tests for resolve_output_dir precedence: flag -> config.yaml -> XDG default.

This pins the cwd-footgun fix: with no flag and no configured value, the default
must be the canonical TRANSCRIPTS_DIR, never the current working directory.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

from pidcast.config import TRANSCRIPTS_DIR, resolve_output_dir


def _args(output_dir=None):
    return argparse.Namespace(output_dir=output_dir)


def test_flag_wins():
    result = resolve_output_dir(_args(output_dir="/explicit/from/flag"))
    assert result == Path("/explicit/from/flag")


def test_flag_beats_config():
    with patch(
        "pidcast.config_manager.ConfigManager.load_config",
        return_value={"output_dir": "/from/config"},
    ):
        result = resolve_output_dir(_args(output_dir="/from/flag"))
    assert result == Path("/from/flag")


def test_config_value_used_when_no_flag():
    with patch(
        "pidcast.config_manager.ConfigManager.load_config",
        return_value={"output_dir": "/from/config"},
    ):
        result = resolve_output_dir(_args(output_dir=None))
    assert result == Path("/from/config")


def test_falls_back_to_transcripts_dir_when_config_missing_key():
    # config dict present but WITHOUT output_dir -> XDG transcripts dir, not cwd.
    with patch(
        "pidcast.config_manager.ConfigManager.load_config",
        return_value={"backfill_limit": 5},
    ):
        result = resolve_output_dir(_args(output_dir=None))
    assert result == TRANSCRIPTS_DIR


def test_never_returns_cwd():
    with patch(
        "pidcast.config_manager.ConfigManager.load_config",
        return_value={},
    ):
        result = resolve_output_dir(_args(output_dir=None))
    assert result == TRANSCRIPTS_DIR
    assert result != Path.cwd()


def test_legacy_repo_path_in_config_is_ignored():
    # An older config.yaml pinned the in-repo data/transcripts path before the
    # storage move; that stale value must fall through to the XDG dir, not send
    # transcripts back into the source tree.
    from pidcast.config import PROJECT_ROOT

    legacy = str(PROJECT_ROOT / "data" / "transcripts")
    with patch(
        "pidcast.config_manager.ConfigManager.load_config",
        return_value={"output_dir": legacy},
    ):
        result = resolve_output_dir(_args(output_dir=None))
    assert result == TRANSCRIPTS_DIR


def test_legacy_repo_path_via_flag_is_still_honored():
    # An explicit --output-dir always wins, even if it's the legacy path - the
    # user asked for it directly this run.
    from pidcast.config import PROJECT_ROOT

    legacy = str(PROJECT_ROOT / "data" / "transcripts")
    result = resolve_output_dir(_args(output_dir=legacy))
    assert result == Path(legacy)
