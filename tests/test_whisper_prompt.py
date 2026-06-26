"""Tests for whisper --prompt glossary biasing (CLI -> resolver -> argv)."""

import argparse
from unittest.mock import patch

import pytest

from pidcast.config import load_glossaries
from pidcast.exceptions import ConfigurationError
from pidcast.transcription import run_whisper_transcription
from pidcast.workflow import _resolve_whisper_prompt

# ============================================================================
# argv construction: the real contract is what lands in the whisper command
# ============================================================================


def _capture_command(**kwargs):
    """Run run_whisper_transcription with the streaming layer mocked and return
    the command list it built."""
    with patch("pidcast.transcription._run_whisper_streaming") as mock_stream:
        run_whisper_transcription(
            "audio.wav",
            "ggml-small.en.bin",
            "txt",
            "out",
            show_progress=False,
            **kwargs,
        )
        # command is the first positional arg to _run_whisper_streaming
        return mock_stream.call_args[0][0]


class TestWhisperPromptArgv:
    def test_prompt_emits_flag_and_carry(self):
        cmd = _capture_command(prompt="Names: Sabre, Xenoss.")
        assert "--prompt" in cmd
        # text follows the flag immediately
        assert cmd[cmd.index("--prompt") + 1] == "Names: Sabre, Xenoss."
        # carry-initial-prompt keeps the bias across all windows
        assert "--carry-initial-prompt" in cmd

    def test_no_prompt_omits_flag(self):
        cmd = _capture_command(prompt=None)
        assert "--prompt" not in cmd
        assert "--carry-initial-prompt" not in cmd

    def test_empty_prompt_omits_flag(self):
        cmd = _capture_command(prompt="")
        assert "--prompt" not in cmd

    def test_never_uses_dash_p(self):
        # -p is whisper's --processors, NOT prompt - regression guard.
        cmd = _capture_command(prompt="x")
        assert "-p" not in cmd


# ============================================================================
# resolver precedence
# ============================================================================


def _ns(**kw):
    defaults = {"whisper_prompt": None, "glossary": None, "language": None, "whisper_model": ""}
    defaults.update(kw)
    return argparse.Namespace(**defaults)


class TestResolveWhisperPrompt:
    def test_neither_returns_none(self):
        assert _resolve_whisper_prompt(_ns()) is None

    def test_raw_prompt_passthrough(self):
        assert _resolve_whisper_prompt(_ns(whisper_prompt="raw text")) == "raw text"

    def test_whisper_prompt_overrides_glossary(self):
        with patch("pidcast.workflow.logger") as mock_log:
            out = _resolve_whisper_prompt(_ns(whisper_prompt="raw", glossary="adtech-ai"))
        assert out == "raw"
        mock_log.warning.assert_called_once()

    def test_known_glossary_resolves(self):
        out = _resolve_whisper_prompt(_ns(glossary="adtech-ai"))
        assert out and "Sabre" in out

    def test_unknown_glossary_raises_with_available(self):
        with pytest.raises(ConfigurationError) as exc:
            _resolve_whisper_prompt(_ns(glossary="does-not-exist"))
        assert "adtech-ai" in str(exc.value)

    def test_en_model_non_english_warns(self):
        with patch("pidcast.workflow.logger") as mock_log:
            _resolve_whisper_prompt(
                _ns(glossary="adtech-ai", language="uk", whisper_model="ggml-small.en.bin")
            )
        mock_log.warning.assert_called_once()

    def test_en_model_english_no_warn(self):
        with patch("pidcast.workflow.logger") as mock_log:
            _resolve_whisper_prompt(
                _ns(glossary="adtech-ai", language="en", whisper_model="ggml-small.en.bin")
            )
        mock_log.warning.assert_not_called()


# ============================================================================
# glossary loader
# ============================================================================


class TestLoadGlossaries:
    def test_seeded_glossary_present(self):
        g = load_glossaries()
        assert "adtech-ai" in g
        assert "Xenoss" in g["adtech-ai"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_glossaries(tmp_path / "nope.yaml") == {}

    def test_empty_file_returns_empty(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text("")
        assert load_glossaries(f) == {}

    def test_invalid_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text("glossaries: [unclosed")
        assert load_glossaries(f) == {}

    def test_parses_custom_glossary(self, tmp_path):
        f = tmp_path / "g.yaml"
        f.write_text("glossaries:\n  foo: Bar baz qux.\n")
        g = load_glossaries(f)
        assert g == {"foo": "Bar baz qux."}
