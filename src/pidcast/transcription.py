"""Whisper transcription functionality."""

import logging
import subprocess
import threading
import time
from pathlib import Path

from .config import (
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_SAMPLE_RATE,
    FFMPEG_PATH,
    WHISPER_CPP_PATH,
)
from .exceptions import TranscriptionError
from .utils import format_duration, load_json_file

logger = logging.getLogger(__name__)


# ============================================================================
# PROGRESS DISPLAY
# ============================================================================


def show_transcription_progress(start_time: float, estimated_duration: float | None) -> None:
    """Show progress indicator for transcription.

    Args:
        start_time: When transcription started
        estimated_duration: Estimated total duration in seconds (or None)
    """
    elapsed = time.time() - start_time

    if estimated_duration and estimated_duration > 0:
        # Show progress with percentage
        progress = min(elapsed / estimated_duration, 0.99)  # Cap at 99% until done
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed_str = format_duration(elapsed)
        remaining = max(0, estimated_duration - elapsed)
        remaining_str = format_duration(remaining)

        print(
            f"\r  Progress: [{bar}] {progress * 100:.0f}% | "
            f"Elapsed: {elapsed_str} | Remaining: ~{remaining_str}",
            end="",
            flush=True,
        )
    else:
        # Just show elapsed time with spinner
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        elapsed_str = format_duration(elapsed)
        spinner_char = spinner[int(elapsed * 2) % len(spinner)]
        print(
            f"\r  {spinner_char} Transcribing... Elapsed: {elapsed_str}",
            end="",
            flush=True,
        )


# ============================================================================
# AUDIO PROCESSING
# ============================================================================


def build_ffmpeg_audio_conversion_command(
    input_file: str, output_file: str, overwrite: bool = False
) -> list[str]:
    """Build FFmpeg command for audio conversion to 16kHz mono WAV.

    Args:
        input_file: Path to input audio file
        output_file: Path to output WAV file
        overwrite: Whether to overwrite existing output file

    Returns:
        FFmpeg command as list of arguments
    """
    command = [FFMPEG_PATH]
    if overwrite:
        command.append("-y")
    command.extend(
        [
            "-i",
            input_file,
            "-ar",
            AUDIO_SAMPLE_RATE,
            "-ac",
            AUDIO_CHANNELS,
            "-c:a",
            AUDIO_CODEC,
            output_file,
        ]
    )
    return command


# ============================================================================
# TIME ESTIMATION
# ============================================================================


def estimate_transcription_time(
    stats_file: str | Path, audio_duration: float, max_records: int = 100
) -> float | None:
    """Estimate transcription time based on historical data.

    Args:
        stats_file: Path to statistics JSON file
        audio_duration: Duration of audio in seconds
        max_records: Maximum historical records to consider

    Returns:
        Estimated time in seconds, or None if not enough data
    """
    existing_stats = load_json_file(stats_file, default=[])

    if not existing_stats:
        return None

    successful_runs = [
        s
        for s in existing_stats
        if s.get("success") and "transcription_duration" in s and "audio_duration" in s
    ]

    if not successful_runs:
        return None

    recent_runs = successful_runs[-max_records:]

    ratios = []
    for run in recent_runs:
        trans_duration = run["transcription_duration"]
        audio_dur = run["audio_duration"]
        if audio_dur > 0:
            ratios.append(trans_duration / audio_dur)

    if not ratios:
        return None

    avg_ratio = sum(ratios) / len(ratios)
    return audio_duration * avg_ratio


# ============================================================================
# TRANSCRIPTION
# ============================================================================


def run_whisper_transcription(
    audio_file: str | Path,
    whisper_model: str,
    output_format: str,
    output_file: str | Path,
    verbose: bool = False,
    estimated_duration: float | None = None,
    show_progress: bool = True,
) -> bool:
    """Run whisper transcription.

    Args:
        audio_file: Path to audio file
        whisper_model: Path to Whisper model
        output_format: Output format (txt, vtt, srt, json)
        output_file: Output file path (without extension)
        verbose: Enable verbose output
        estimated_duration: Estimated transcription duration for progress bar
        show_progress: Whether to show progress indicator

    Returns:
        True if successful

    Raises:
        TranscriptionError: If transcription fails
    """
    # Build base command
    command = [
        WHISPER_CPP_PATH,
        "-f",
        str(audio_file),
        "-m",
        whisper_model,
        "-t",
        "8",
    ]

    # Add output format flag
    format_flags = {
        "txt": "--output-txt",
        "vtt": "--output-vtt",
        "srt": "--output-srt",
        "json": "--output-json",
    }
    if output_format in format_flags:
        command.append(format_flags[output_format])

    # Add output file path
    if output_file:
        command.extend(["-of", str(output_file)])

    if verbose:
        logger.info(f"Running Whisper command: {' '.join(command)}")

    # Start progress tracking in a separate thread
    progress_thread = None
    stop_progress = None

    if show_progress and not verbose:
        stop_progress = threading.Event()
        start_time = time.time()

        def update_progress():
            while not stop_progress.is_set():
                show_transcription_progress(start_time, estimated_duration)
                time.sleep(0.1)

        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

    try:
        # Run transcription
        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # Stop progress indicator
        if stop_progress:
            stop_progress.set()
            if progress_thread:
                progress_thread.join(timeout=0.5)
            print()  # New line after progress bar

        if verbose:
            logger.info("✓ Transcription completed successfully.")

        return True

    except subprocess.CalledProcessError as e:
        if stop_progress:
            stop_progress.set()
        error_msg = f"Whisper transcription failed with exit code {e.returncode}"
        if hasattr(e, "stderr") and e.stderr:
            error_msg += f": {e.stderr.decode()}"
        raise TranscriptionError(error_msg) from e

    except FileNotFoundError as e:
        if stop_progress:
            stop_progress.set()
        raise TranscriptionError(
            f"Whisper binary not found at: {WHISPER_CPP_PATH}. "
            "Please check the WHISPER_CPP_PATH configuration."
        ) from e


def process_local_file(
    input_file: str | Path,
    output_dir: str | Path,
    verbose: bool = False,
) -> tuple[str, dict]:
    """Process a local audio file for transcription.

    Args:
        input_file: Path to local audio file
        output_dir: Directory to store processed files
        verbose: Enable verbose output

    Returns:
        Tuple of (audio_file_path, info_dict)

    Raises:
        FileProcessingError: If file processing fails
    """
    import datetime

    from .config import VideoInfo
    from .exceptions import FileProcessingError
    from .utils import format_duration as fmt_dur

    input_file = Path(input_file)
    output_dir = Path(output_dir)

    if not input_file.exists():
        raise FileProcessingError(f"File not found: {input_file}")

    name = input_file.stem

    # Create info dict similar to yt-dlp output
    info = VideoInfo(
        title=name.replace("_", " "),
        webpage_url=f"file://{input_file.absolute()}",
        channel="Local File",
        uploader="User",
        duration=0,
        duration_string="Unknown",
        view_count=0,
        upload_date=datetime.datetime.fromtimestamp(input_file.stat().st_mtime).strftime("%Y%m%d"),
    )

    # Try to get duration using ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            info.duration = duration
            info.duration_string = fmt_dur(duration)
    except Exception as e:
        if verbose:
            logger.warning(f"Could not determine duration: {e}")

    # Convert to 16kHz mono WAV if needed
    output_wav = output_dir / f"{name}_16k.wav"

    if verbose:
        logger.info(f"Converting {input_file.name} to 16kHz mono WAV...")

    try:
        command = build_ffmpeg_audio_conversion_command(
            str(input_file), str(output_wav), overwrite=True
        )

        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        return str(output_wav), info

    except subprocess.CalledProcessError as e:
        raise FileProcessingError(f"Error converting audio: {e}") from e
