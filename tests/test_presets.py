"""Tests for CLI preset loading."""

import argparse
import logging
from unittest.mock import patch

import pytest

from pidcast.cli import apply_preset
from pidcast.config_manager import ConfigManager


class TestLoadPreset:
    def test_load_existing_preset(self):
        config = {
            "backfill_limit": 5,
            "presets": {
                "daily": {
                    "whisper_model": "large-v3",
                    "language": "uk",
                    "diarize": True,
                    "no_analyze": True,
                }
            },
        }
        with patch.object(ConfigManager, "load_config", return_value=config):
            result = ConfigManager.load_preset("daily")
        assert result == {
            "whisper_model": "large-v3",
            "language": "uk",
            "diarize": True,
            "no_analyze": True,
        }

    def test_load_nonexistent_preset_raises(self):
        config = {"presets": {"daily": {"diarize": True}}}
        with (
            patch.object(ConfigManager, "load_config", return_value=config),
            pytest.raises(ValueError, match="Unknown preset 'nope'"),
        ):
            ConfigManager.load_preset("nope")

    def test_load_preset_no_presets_section(self):
        config = {"backfill_limit": 5}
        with (
            patch.object(ConfigManager, "load_config", return_value=config),
            pytest.raises(ValueError, match="No presets defined"),
        ):
            ConfigManager.load_preset("daily")

    def test_list_presets_returns_names_and_flags(self):
        config = {
            "presets": {
                "daily": {"whisper_model": "large-v3", "diarize": True},
                "meeting": {"whisper_model": "small"},
            }
        }
        with patch.object(ConfigManager, "load_config", return_value=config):
            result = ConfigManager.list_presets()
        assert result == {
            "daily": {"whisper_model": "large-v3", "diarize": True},
            "meeting": {"whisper_model": "small"},
        }

    def test_list_presets_empty(self):
        config = {"backfill_limit": 5}
        with patch.object(ConfigManager, "load_config", return_value=config):
            result = ConfigManager.list_presets()
        assert result == {}


class TestApplyPreset:
    def test_preset_sets_unset_args(self):
        args = argparse.Namespace(
            preset="daily",
            whisper_model=None,
            language=None,
            diarize=False,
            no_analyze=False,
            verbose=False,
        )
        preset_values = {
            "whisper_model": "large-v3",
            "language": "uk",
            "diarize": True,
            "no_analyze": True,
        }
        with patch.object(ConfigManager, "load_preset", return_value=preset_values):
            apply_preset(args)

        assert args.whisper_model == "large-v3"
        assert args.language == "uk"
        assert args.diarize is True
        assert args.no_analyze is True

    def test_explicit_cli_flag_overrides_preset(self):
        args = argparse.Namespace(
            preset="daily",
            whisper_model="medium",
            language=None,
            diarize=False,
            no_analyze=False,
            verbose=False,
        )
        preset_values = {
            "whisper_model": "large-v3",
            "language": "uk",
        }
        explicitly_set = {"whisper_model"}
        with patch.object(ConfigManager, "load_preset", return_value=preset_values):
            apply_preset(args, explicitly_set=explicitly_set)

        assert args.whisper_model == "medium"  # NOT overridden
        assert args.language == "uk"  # set from preset

    def test_unknown_preset_key_warns(self, caplog):
        args = argparse.Namespace(
            preset="daily",
            verbose=False,
        )
        preset_values = {"bogus_flag": "value"}
        with (
            patch.object(ConfigManager, "load_preset", return_value=preset_values),
            caplog.at_level(logging.WARNING),
        ):
            apply_preset(args)
        assert "Unknown preset key 'bogus_flag'" in caplog.text
