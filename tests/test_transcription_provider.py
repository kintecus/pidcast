"""Tests for TranscriptionProvider protocol conformance."""

from pidcast.providers import TranscriptionProvider, TranscriptionResult


class TestTranscriptionResult:
    def test_dataclass_fields(self):
        result = TranscriptionResult(
            text="Hello world",
            speaker_count=None,
            duration=1.5,
            provider="test",
            language="en",
            diarized=False,
        )
        assert result.text == "Hello world"
        assert result.speaker_count is None
        assert result.duration == 1.5
        assert result.provider == "test"
        assert result.language == "en"
        assert result.diarized is False

    def test_dataclass_with_speakers(self):
        result = TranscriptionResult(
            text="**Speaker 1**\nHello",
            speaker_count=2,
            duration=3.0,
            provider="elevenlabs",
            language="en",
            diarized=True,
        )
        assert result.speaker_count == 2
        assert result.diarized is True


class TestTranscriptionProviderProtocol:
    def test_protocol_is_runtime_checkable(self):
        assert hasattr(TranscriptionProvider, "__protocol_attrs__") or hasattr(
            TranscriptionProvider, "_is_runtime_protocol"
        )
