"""ElevenLabs Scribe v2 transcription provider."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..exceptions import ConfigurationError, TranscriptionError
from . import TranscriptionResult

logger = logging.getLogger(__name__)


def _create_client(api_key: str):
    """Create ElevenLabs client. Separated for testability."""
    try:
        from elevenlabs import ElevenLabs
    except ImportError as e:
        raise TranscriptionError(
            "elevenlabs package is not installed. "
            "Install with: uv pip install 'pidcast[elevenlabs]'"
        ) from e

    return ElevenLabs(api_key=api_key)


def _format_elevenlabs_speaker(speaker_id: str | None) -> str:
    """Convert ElevenLabs speaker ID to human-readable label.

    'speaker_0' -> 'Speaker 1', 'speaker_1' -> 'Speaker 2', etc.
    """
    if speaker_id is None:
        return "Unknown Speaker"
    try:
        idx = int(speaker_id.split("_")[-1])
        return f"Speaker {idx + 1}"
    except (ValueError, IndexError):
        return speaker_id


def _build_diarized_text(words) -> tuple[str, int]:
    """Build diarized transcript text from ElevenLabs word-level response.

    Groups consecutive words by speaker_id and inserts **Speaker N** labels.

    Returns:
        Tuple of (labeled_text, speaker_count).
    """
    if not words:
        return "", 0

    lines: list[str] = []
    current_speaker = None
    current_words: list[str] = []
    speakers_seen: set[str] = set()

    for word in words:
        if word.type == "spacing":
            continue

        speaker = getattr(word, "speaker_id", None)

        if speaker != current_speaker and word.type == "word":
            # Flush current words
            if current_words:
                lines.append(" ".join(current_words))
                current_words = []

            current_speaker = speaker
            if speaker is not None:
                speakers_seen.add(speaker)
                label = _format_elevenlabs_speaker(speaker)
                lines.append(f"\n**{label}**")

        if word.type in ("word", "audio_event"):
            current_words.append(word.text)

    # Flush remaining words
    if current_words:
        lines.append(" ".join(current_words))

    text = "\n".join(lines).strip()
    return text, len(speakers_seen)


class ElevenLabsTranscriptionProvider:
    """Transcription provider using ElevenLabs Scribe v2 API."""

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise ConfigurationError(
                "ELEVENLABS_API_KEY environment variable not set. "
                "Required for ElevenLabs transcription."
            )
        self._api_key = api_key

    def transcribe(
        self,
        audio_file: str | Path,
        language: str | None = None,
        diarize: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using ElevenLabs Scribe v2."""
        client = _create_client(self._api_key)

        kwargs = {
            "model_id": "scribe_v2",
            "tag_audio_events": True,
            "timestamps_granularity": "word",
            "diarize": diarize,
        }

        if language:
            kwargs["language_code"] = language

        start_time = time.time()

        try:
            response = self._call_api(client, kwargs, audio_file)
        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"ElevenLabs transcription failed: {e}") from e

        duration = time.time() - start_time

        # Build output text
        if diarize and response.words:
            text, speaker_count = _build_diarized_text(response.words)
        else:
            text = response.text
            speaker_count = None

        language_code = getattr(response, "language_code", None)

        return TranscriptionResult(
            text=text,
            speaker_count=speaker_count,
            duration=duration,
            provider="elevenlabs",
            language=language_code,
            diarized=diarize and speaker_count is not None and speaker_count > 0,
        )

    @staticmethod
    def _call_api(client, kwargs: dict, audio_file: str | Path):
        """Call ElevenLabs API with optional Rich progress spinner."""
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Transcribing with ElevenLabs..."),
                TimeElapsedColumn(),
            ) as progress:
                progress.add_task("transcribe", total=None)
                with open(audio_file, "rb") as f:
                    kwargs["file"] = f
                    return client.speech_to_text.convert(**kwargs)
        except ImportError:
            logger.info("Transcribing with ElevenLabs...")
            with open(audio_file, "rb") as f:
                kwargs["file"] = f
                return client.speech_to_text.convert(**kwargs)
