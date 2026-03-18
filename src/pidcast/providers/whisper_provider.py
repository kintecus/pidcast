"""Whisper.cpp transcription provider."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from ..config import HUGGINGFACE_TOKEN
from ..diarization import merge_whisper_with_diarization, run_diarization
from ..exceptions import DiarizationError, TranscriptionError
from ..transcription import run_whisper_transcription
from . import TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperTranscriptionProvider:
    """Transcription provider using local whisper.cpp."""

    def __init__(
        self,
        whisper_model: str,
        output_format: str,
        output_dir: str | Path,
        estimated_duration: float | None = None,
    ) -> None:
        self._whisper_model = whisper_model
        self._output_format = output_format
        self._output_dir = Path(output_dir)
        self._estimated_duration = estimated_duration

    def transcribe(
        self,
        audio_file: str | Path,
        language: str | None = None,
        diarize: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using whisper.cpp."""
        temp_output = self._output_dir / f"temp_transcript_{uuid.uuid4().hex[:8]}"
        output_format = "json" if diarize else self._output_format

        if diarize and not HUGGINGFACE_TOKEN:
            raise DiarizationError(
                "HUGGINGFACE_TOKEN environment variable not set. "
                "Required for speaker diarization with Whisper."
            )

        start_time = time.time()

        try:
            run_whisper_transcription(
                audio_file,
                self._whisper_model,
                output_format,
                str(temp_output),
                verbose,
                estimated_duration=self._estimated_duration,
                language=language,
            )
        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}") from e

        duration = time.time() - start_time

        # Handle diarization
        speaker_count = None
        if diarize:
            text, speaker_count = self._run_diarization(audio_file, temp_output, verbose)
            diarized = speaker_count is not None and speaker_count > 0
        else:
            txt_file = Path(f"{temp_output}.txt")
            try:
                text = txt_file.read_text(encoding="utf-8")
            except FileNotFoundError as e:
                raise TranscriptionError(f"Whisper output file not found: {txt_file}") from e
            diarized = False

        # Clean up temp files
        self._cleanup_temp_files(temp_output)

        return TranscriptionResult(
            text=text,
            speaker_count=speaker_count,
            duration=duration,
            provider="whisper",
            language=language,
            diarized=diarized,
        )

    def _run_diarization(
        self,
        audio_file: str | Path,
        temp_output: Path,
        verbose: bool,
    ) -> tuple[str, int]:
        """Run pyannote diarization and merge with whisper output."""
        logger.info("Running speaker diarization...")
        diarization_segments = run_diarization(audio_file, HUGGINGFACE_TOKEN, verbose)

        # Read whisper JSON output
        json_file = Path(f"{temp_output}.json")
        try:
            with open(json_file, encoding="utf-8") as f:
                whisper_data = json.load(f)
        except FileNotFoundError as e:
            raise TranscriptionError(f"Whisper JSON output not found: {json_file}") from e

        whisper_segments = whisper_data.get("transcription", [])
        text, speaker_count = merge_whisper_with_diarization(whisper_segments, diarization_segments)

        return text, speaker_count

    def _cleanup_temp_files(self, temp_output: Path) -> None:
        """Remove temporary whisper output files."""
        for ext in (".txt", ".json", ".vtt", ".srt"):
            temp_file = Path(f"{temp_output}{ext}")
            if temp_file.exists():
                temp_file.unlink()
