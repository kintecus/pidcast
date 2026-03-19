"""Tests for ElevenLabs transcription provider."""

from unittest.mock import MagicMock, patch

import pytest

from pidcast.exceptions import ConfigurationError, TranscriptionError
from pidcast.providers import TranscriptionProvider, TranscriptionResult

# ============================================================================
# Helpers
# ============================================================================


def _make_word(text, speaker_id=None, word_type="word", start=0.0, end=0.5):
    """Create a mock word object matching ElevenLabs API response."""
    word = MagicMock()
    word.text = text
    word.speaker_id = speaker_id
    word.type = word_type
    word.start = start
    word.end = end
    return word


def _make_response(text="Hello world", words=None, language_code="eng", language_probability=0.98):
    """Create a mock ElevenLabs STT response."""
    response = MagicMock()
    response.text = text
    response.words = words or []
    response.language_code = language_code
    response.language_probability = language_probability
    return response


# ============================================================================
# Construction and validation
# ============================================================================


class TestElevenLabsProviderInit:
    def test_missing_api_key_raises_configuration_error(self):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        with pytest.raises(ConfigurationError, match="ELEVENLABS_API_KEY"):
            ElevenLabsTranscriptionProvider(api_key=None)

    def test_empty_api_key_raises_configuration_error(self):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        with pytest.raises(ConfigurationError, match="ELEVENLABS_API_KEY"):
            ElevenLabsTranscriptionProvider(api_key="")

    def test_satisfies_protocol(self):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        assert isinstance(ElevenLabsTranscriptionProvider(api_key="test"), TranscriptionProvider)


# ============================================================================
# Missing package
# ============================================================================


class TestElevenLabsMissingPackage:
    def test_missing_elevenlabs_package_raises_transcription_error(self):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        provider = ElevenLabsTranscriptionProvider(api_key="test")
        # _create_client() imports elevenlabs at call time, so patch it there
        with (
            patch.dict("sys.modules", {"elevenlabs": None}),
            pytest.raises(TranscriptionError, match="elevenlabs.*not installed"),
        ):
            provider.transcribe(audio_file="/tmp/test.wav")


# ============================================================================
# Basic transcription (no diarization)
# ============================================================================


class TestElevenLabsBasicTranscription:
    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_basic_transcription_returns_result(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        response = _make_response(text="Hello world", language_code="eng")
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.return_value = response
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        result = provider.transcribe(audio_file=str(audio_file))

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.provider == "elevenlabs"
        assert result.diarized is False
        assert result.speaker_count is None
        assert result.language == "eng"

    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_language_passed_to_api(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        response = _make_response()
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.return_value = response
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        provider.transcribe(audio_file=str(audio_file), language="uk")

        call_kwargs = mock_client.speech_to_text.convert.call_args[1]
        assert call_kwargs["language_code"] == "uk"

    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_api_error_wrapped_in_transcription_error(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        mock_client = MagicMock()
        mock_client.speech_to_text.convert.side_effect = Exception("API rate limit")
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        with pytest.raises(TranscriptionError, match="API rate limit"):
            provider.transcribe(audio_file=str(audio_file))


# ============================================================================
# Diarization
# ============================================================================


class TestElevenLabsDiarization:
    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_diarization_produces_speaker_labels(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        words = [
            _make_word("Hello", speaker_id="speaker_0", start=0.0, end=0.5),
            _make_word(" ", speaker_id="speaker_0", word_type="spacing"),
            _make_word("there.", speaker_id="speaker_0", start=0.5, end=1.0),
            _make_word(" ", speaker_id="speaker_1", word_type="spacing"),
            _make_word("Hi!", speaker_id="speaker_1", start=1.0, end=1.5),
        ]
        response = _make_response(text="Hello there. Hi!", words=words)
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.return_value = response
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        result = provider.transcribe(audio_file=str(audio_file), diarize=True)

        assert "**Speaker 1**" in result.text
        assert "**Speaker 2**" in result.text
        assert result.speaker_count == 2
        assert result.diarized is True

    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_diarize_flag_passed_to_api(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        response = _make_response()
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.return_value = response
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        provider.transcribe(audio_file=str(audio_file), diarize=True)

        call_kwargs = mock_client.speech_to_text.convert.call_args[1]
        assert call_kwargs["diarize"] is True

    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_consecutive_same_speaker_grouped(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        words = [
            _make_word("Hello", speaker_id="speaker_0", start=0.0, end=0.3),
            _make_word(" ", speaker_id="speaker_0", word_type="spacing"),
            _make_word("world", speaker_id="speaker_0", start=0.3, end=0.6),
            _make_word(" ", speaker_id="speaker_0", word_type="spacing"),
            _make_word("foo", speaker_id="speaker_0", start=0.6, end=0.9),
        ]
        response = _make_response(text="Hello world foo", words=words)
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.return_value = response
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        result = provider.transcribe(audio_file=str(audio_file), diarize=True)

        # Only one speaker label, not repeated
        assert result.text.count("**Speaker 1**") == 1

    @patch("pidcast.providers.elevenlabs_provider._create_client")
    def test_audio_events_preserved_inline(self, mock_create_client, tmp_path):
        from pidcast.providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake wav data")

        words = [
            _make_word("Hello", speaker_id="speaker_0", start=0.0, end=0.5),
            _make_word(" ", speaker_id="speaker_0", word_type="spacing"),
            _make_word("(laughter)", speaker_id="speaker_0", word_type="audio_event"),
            _make_word(" ", speaker_id="speaker_0", word_type="spacing"),
            _make_word("world", speaker_id="speaker_0", start=0.5, end=1.0),
        ]
        response = _make_response(text="Hello (laughter) world", words=words)
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.return_value = response
        mock_create_client.return_value = mock_client

        provider = ElevenLabsTranscriptionProvider(api_key="test-key")
        result = provider.transcribe(audio_file=str(audio_file), diarize=True)

        assert "(laughter)" in result.text
        # Audio event should not break speaker grouping - still one speaker label
        assert result.text.count("**Speaker 1**") == 1
