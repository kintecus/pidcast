"""Tests for CLI preset loading."""

from unittest.mock import patch

import pytest

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
