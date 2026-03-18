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


class TestSpeakerLabelConsistency:
    """Both providers must produce the same **Speaker N** label format."""

    def test_elevenlabs_speaker_format(self):
        from pidcast.providers.elevenlabs_provider import _format_elevenlabs_speaker

        assert _format_elevenlabs_speaker("speaker_0") == "Speaker 1"
        assert _format_elevenlabs_speaker("speaker_1") == "Speaker 2"
        assert _format_elevenlabs_speaker(None) == "Unknown Speaker"

    def test_pyannote_speaker_format(self):
        from pidcast.diarization import _format_speaker_label

        assert _format_speaker_label("SPEAKER_00") == "Speaker 1"
        assert _format_speaker_label("SPEAKER_01") == "Speaker 2"
        assert _format_speaker_label(None) == "Unknown Speaker"

    def test_both_produce_same_format(self):
        from pidcast.diarization import _format_speaker_label
        from pidcast.providers.elevenlabs_provider import _format_elevenlabs_speaker

        for i in range(5):
            el_label = _format_elevenlabs_speaker(f"speaker_{i}")
            py_label = _format_speaker_label(f"SPEAKER_{i:02d}")
            assert el_label == py_label, f"Mismatch at index {i}: {el_label} != {py_label}"
