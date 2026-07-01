"""Tests for CLI preset loading."""

import argparse
import logging
from unittest.mock import patch

import pytest

from pidcast.cli import _explicitly_set_dests, apply_preset
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

    def test_hyphenated_preset_key_resolves_to_dest(self):
        # Keys written with hyphens (matching CLI flags) should map to the
        # underscore argparse dest. Regression: --transcription-provider was
        # silently skipped when written as 'transcription-provider' in a preset.
        args = argparse.Namespace(
            preset="elabs",
            transcription_provider="whisper",
            keep_audio=False,
            verbose=False,
        )
        preset_values = {
            "transcription-provider": "elevenlabs",
            "keep-audio": True,
        }
        with patch.object(ConfigManager, "load_preset", return_value=preset_values):
            apply_preset(args)

        assert args.transcription_provider == "elevenlabs"
        assert args.keep_audio is True

    def test_explicit_flag_overrides_hyphenated_preset_key(self):
        args = argparse.Namespace(
            preset="elabs",
            transcription_provider="whisper",
            verbose=False,
        )
        preset_values = {"transcription-provider": "elevenlabs"}
        explicitly_set = {"transcription_provider"}
        with patch.object(ConfigManager, "load_preset", return_value=preset_values):
            apply_preset(args, explicitly_set=explicitly_set)

        assert args.transcription_provider == "whisper"  # CLI flag wins

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


class TestExplicitlySetDetection:
    """Detecting which dests the user passed, regardless of flag spelling.

    Replaces a brittle sys.argv substring scan that missed short aliases and
    BooleanOptionalAction, letting a preset wrongly clobber a flag the user set.
    """

    def test_long_flag_detected(self):
        dests = _explicitly_set_dests(["transcribe", "x.mp3", "--groq-model", "llama33"])
        assert "groq_model" in dests

    def test_short_alias_detected(self):
        # The old scan missed `-m` (only matched --groq-model).
        dests = _explicitly_set_dests(["transcribe", "x.mp3", "-m", "llama33"])
        assert "groq_model" in dests

    def test_boolean_optional_action_detected(self):
        # The old scan missed `--no-suppress-nst` (dest is suppress_nst).
        dests = _explicitly_set_dests(["transcribe", "x.mp3", "--no-suppress-nst"])
        assert "suppress_nst" in dests

    def test_unset_flag_not_detected(self):
        dests = _explicitly_set_dests(["transcribe", "x.mp3"])
        assert "groq_model" not in dests
        assert "language" not in dests

    def test_preset_flag_itself_not_reported_as_clobberable(self):
        dests = _explicitly_set_dests(["transcribe", "x.mp3", "-p", "daily"])
        # `input` is positional and present, but a preset can't target it anyway.
        assert "groq_model" not in dests
