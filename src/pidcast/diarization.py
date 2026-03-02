"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .exceptions import DiarizationError

logger = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    """A speaker segment with start/end times in seconds."""

    start: float
    end: float
    speaker: str


def _load_pipeline(hf_token: str):
    """Load pyannote speaker diarization pipeline.

    Raises:
        DiarizationError: If pyannote.audio is not installed or token is invalid.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationError(
            "pyannote.audio is not installed. "
            "Install with: uv pip install 'pidcast[diarize]'"
        ) from e

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        raise DiarizationError(f"Failed to load pyannote pipeline: {e}") from e

    return pipeline


def run_diarization(
    audio_file: str | Path,
    hf_token: str,
    verbose: bool = False,
) -> list[DiarizationSegment]:
    """Run pyannote speaker diarization on a WAV file.

    Args:
        audio_file: Path to 16kHz mono WAV file.
        hf_token: HuggingFace API token.
        verbose: Enable verbose logging.

    Returns:
        List of DiarizationSegment sorted by start time.

    Raises:
        DiarizationError: On any failure.
    """
    pipeline = _load_pipeline(hf_token)

    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Diarizing speakers..."),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("diarize", total=None)
            diarization = pipeline(str(audio_file))
    except ImportError:
        logger.info("Diarizing speakers...")
        diarization = pipeline(str(audio_file))
    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            )
        )

    segments.sort(key=lambda s: s.start)
    return segments


def merge_whisper_with_diarization(
    whisper_segments: list[dict],
    diarization_segments: list[DiarizationSegment],
) -> tuple[str, int]:
    """Align whisper transcript segments with diarization speaker labels.

    Uses maximum overlap to assign each whisper segment to a speaker.

    Args:
        whisper_segments: List of dicts from whisper.cpp JSON output, each with
            'offsets' (containing 'from'/'to' in milliseconds) and 'text' keys.
        diarization_segments: Speaker segments from pyannote.

    Returns:
        Tuple of (labeled_transcript_text, speaker_count).
    """
    if not diarization_segments:
        text = "\n".join(seg["text"].strip() for seg in whisper_segments if seg["text"].strip())
        return text, 0

    lines: list[str] = []
    current_speaker = None

    for wseg in whisper_segments:
        w_start = wseg["offsets"]["from"] / 1000.0
        w_end = wseg["offsets"]["to"] / 1000.0
        text = wseg["text"].strip()

        if not text:
            continue

        best_speaker = _find_speaker(w_start, w_end, diarization_segments)

        if best_speaker != current_speaker:
            current_speaker = best_speaker
            label = _format_speaker_label(best_speaker)
            lines.append(f"\n**{label}**")

        lines.append(text)

    speaker_count = len({s.speaker for s in diarization_segments})
    return "\n".join(lines).strip(), speaker_count


def _find_speaker(
    w_start: float,
    w_end: float,
    diarization_segments: list[DiarizationSegment],
) -> str | None:
    """Find the diarization speaker with maximum overlap for a whisper segment."""
    best_speaker = None
    best_overlap = 0.0

    for dseg in diarization_segments:
        overlap_start = max(w_start, dseg.start)
        overlap_end = min(w_end, dseg.end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = dseg.speaker

    return best_speaker


def _format_speaker_label(speaker_id: str | None) -> str:
    """Convert pyannote speaker ID to human-readable label.

    'SPEAKER_00' -> 'Speaker 1', 'SPEAKER_01' -> 'Speaker 2', etc.
    """
    if speaker_id is None:
        return "Unknown Speaker"
    try:
        idx = int(speaker_id.split("_")[-1])
        return f"Speaker {idx + 1}"
    except (ValueError, IndexError):
        return speaker_id
