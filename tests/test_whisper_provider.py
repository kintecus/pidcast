"""Tests for Whisper transcription provider."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pidcast.exceptions import DiarizationError
from pidcast.providers import TranscriptionProvider, TranscriptionResult

# ============================================================================
# Protocol conformance
# ============================================================================


class TestWhisperProviderProtocol:
    def test_satisfies_protocol(self):
        from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

        provider = WhisperTranscriptionProvider(
            whisper_model="base.en",
            output_format="txt",
            output_dir="/tmp",
        )
        assert isinstance(provider, TranscriptionProvider)


# ============================================================================
# Basic transcription
# ============================================================================


class TestWhisperBasicTranscription:
    @patch("pidcast.providers.whisper_provider.run_whisper_transcription")
    def test_basic_transcription_returns_result(self, mock_whisper, tmp_path):
        from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

        def fake_whisper(audio_file, model, fmt, output_file, verbose, **kwargs):
            Path(f"{output_file}.txt").write_text("Hello world", encoding="utf-8")
            return True

        mock_whisper.side_effect = fake_whisper

        provider = WhisperTranscriptionProvider(
            whisper_model="base.en",
            output_format="txt",
            output_dir=str(tmp_path),
        )
        result = provider.transcribe(audio_file="/tmp/test.wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.provider == "whisper"
        assert result.diarized is False
        assert result.speaker_count is None

    @patch("pidcast.providers.whisper_provider.run_whisper_transcription")
    def test_language_passed_to_whisper(self, mock_whisper, tmp_path):
        from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

        def fake_whisper(audio_file, model, fmt, output_file, verbose, **kwargs):
            Path(f"{output_file}.txt").write_text("Привіт", encoding="utf-8")
            return True

        mock_whisper.side_effect = fake_whisper

        provider = WhisperTranscriptionProvider(
            whisper_model="base",
            output_format="txt",
            output_dir=str(tmp_path),
        )
        provider.transcribe(audio_file="/tmp/test.wav", language="uk")

        _, call_kwargs = mock_whisper.call_args
        assert call_kwargs.get("language") == "uk"

    @patch("pidcast.providers.whisper_provider.run_whisper_transcription")
    def test_temp_files_cleaned_up(self, mock_whisper, tmp_path):
        from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

        created_files = []

        def fake_whisper(audio_file, model, fmt, output_file, verbose, **kwargs):
            txt_path = Path(f"{output_file}.txt")
            txt_path.write_text("Hello", encoding="utf-8")
            created_files.append(txt_path)
            return True

        mock_whisper.side_effect = fake_whisper

        provider = WhisperTranscriptionProvider(
            whisper_model="base.en",
            output_format="txt",
            output_dir=str(tmp_path),
        )
        provider.transcribe(audio_file="/tmp/test.wav")

        # Temp files should be cleaned up
        for f in created_files:
            assert not f.exists()


# ============================================================================
# Diarization
# ============================================================================


class TestWhisperDiarization:
    @patch("pidcast.providers.whisper_provider.run_diarization")
    @patch("pidcast.providers.whisper_provider.merge_whisper_with_diarization")
    @patch("pidcast.providers.whisper_provider.run_whisper_transcription")
    def test_diarization_merges_output(self, mock_whisper, mock_merge, mock_diarize, tmp_path):
        from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

        def fake_whisper(audio_file, model, fmt, output_file, verbose, **kwargs):
            json_path = Path(f"{output_file}.json")
            json_path.write_text(
                json.dumps(
                    {"transcription": [{"text": "Hello", "offsets": {"from": 0, "to": 500}}]}
                ),
                encoding="utf-8",
            )
            return True

        mock_whisper.side_effect = fake_whisper
        mock_diarize.return_value = [MagicMock()]
        mock_merge.return_value = ("**Speaker 1**\nHello", 1)

        provider = WhisperTranscriptionProvider(
            whisper_model="base.en",
            output_format="txt",
            output_dir=str(tmp_path),
        )

        with patch("pidcast.providers.whisper_provider.HUGGINGFACE_TOKEN", "fake-token"):
            result = provider.transcribe(audio_file="/tmp/test.wav", diarize=True)

        assert result.diarized is True
        assert result.speaker_count == 1
        assert "**Speaker 1**" in result.text

    def test_diarize_without_hf_token_raises(self, tmp_path):
        from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

        provider = WhisperTranscriptionProvider(
            whisper_model="base.en",
            output_format="txt",
            output_dir=str(tmp_path),
        )

        with (
            patch("pidcast.providers.whisper_provider.HUGGINGFACE_TOKEN", None),
            pytest.raises(DiarizationError, match="HUGGINGFACE_TOKEN"),
        ):
            provider.transcribe(audio_file="/tmp/test.wav", diarize=True)
