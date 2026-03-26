"""Whisper transcription functionality."""

import logging
import subprocess
import time
from pathlib import Path

from .config import (
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_SAMPLE_RATE,
    FFMPEG_PATH,
    WHISPER_CPP_PATH,
    WHISPER_MODEL,
    WHISPER_MODELS_DIR,
)
from .exceptions import TranscriptionError
from .utils import load_json_file

logger = logging.getLogger(__name__)


# ============================================================================
# WHISPER MODEL RESOLUTION
# ============================================================================


def _get_models_dir() -> Path | None:
    """Derive the whisper models directory from config."""
    if WHISPER_MODELS_DIR:
        p = Path(WHISPER_MODELS_DIR)
        return p if p.is_dir() else None
    if WHISPER_MODEL:
        p = Path(WHISPER_MODEL).parent
        return p if p.is_dir() else None
    return None


def list_whisper_models() -> list[dict[str, str]]:
    """List available whisper models in the models directory.

    Returns:
        List of dicts with 'name', 'path', and 'size' keys.
    """
    models_dir = _get_models_dir()
    if not models_dir:
        return []

    models = []
    for f in sorted(models_dir.glob("ggml-*.bin")):
        if f.name.startswith("for-tests-"):
            continue
        # Extract name: ggml-base.en.bin -> base.en
        name = f.name.removeprefix("ggml-").removesuffix(".bin")
        size_mb = f.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"
        models.append({"name": name, "path": str(f), "size": size_str})
    return models


def resolve_whisper_model(model_arg: str) -> str:
    """Resolve a whisper model argument to a file path.

    Accepts either a full file path or a model name (e.g., 'medium', 'large-v3-turbo').

    Args:
        model_arg: Model name or path.

    Returns:
        Absolute path to the model file.

    Raises:
        TranscriptionError: If model cannot be resolved.
    """
    # If it's an existing file, use it directly
    if Path(model_arg).is_file():
        return model_arg

    # Treat as model name - look in models dir
    models_dir = _get_models_dir()
    if not models_dir:
        raise TranscriptionError(
            f"Cannot resolve model name '{model_arg}': no models directory found. "
            "Set WHISPER_MODELS_DIR or WHISPER_MODEL environment variable."
        )

    candidate = models_dir / f"ggml-{model_arg}.bin"
    if candidate.is_file():
        return str(candidate)

    # List available models for error message
    available = list_whisper_models()
    names = [m["name"] for m in available]
    raise TranscriptionError(
        f"Whisper model '{model_arg}' not found in {models_dir}.\n"
        f"Available models: {', '.join(names) if names else 'none'}\n"
        f"Download models with: {models_dir}/download-ggml-model.sh {model_arg}"
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
    stats_file: str | Path,
    audio_duration: float,
    provider: str = "whisper",
    whisper_model: str | None = None,
    diarize: bool = False,
    max_records: int = 100,
) -> float | None:
    """Estimate transcription time based on historical data, filtered by provider.

    Uses tiered filtering with fallback to provide the most relevant estimate:
    1. Provider + whisper_model + diarization (if whisper with model specified)
    2. Provider + diarization
    3. Provider only
    4. All records (cold-start fallback)

    Each tier requires >= 3 records to be used.

    Args:
        stats_file: Path to statistics JSON file
        audio_duration: Duration of audio in seconds
        provider: Transcription provider ("whisper" or "elevenlabs")
        whisper_model: Whisper model name (e.g. "large-v3") for model-specific estimates
        diarize: Whether diarization is enabled (affects whisper speed significantly)
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

    min_records = 3

    # Tier 1: provider + whisper_model + diarization (most specific)
    if provider == "whisper" and whisper_model:
        tier1 = [
            r
            for r in recent_runs
            if (r.get("transcription_provider") or "whisper") == provider
            and r.get("whisper_model") == whisper_model
            and r.get("diarization_performed", False) == diarize
        ]
        if len(tier1) >= min_records:
            return _avg_ratio(tier1, audio_duration)

    # Tier 2: provider + diarization
    tier2 = [
        r
        for r in recent_runs
        if (r.get("transcription_provider") or "whisper") == provider
        and r.get("diarization_performed", False) == diarize
    ]
    if len(tier2) >= min_records:
        return _avg_ratio(tier2, audio_duration)

    # Tier 3: provider only
    tier3 = [
        r
        for r in recent_runs
        if (r.get("transcription_provider") or "whisper") == provider
    ]
    if len(tier3) >= min_records:
        return _avg_ratio(tier3, audio_duration)

    # Tier 4: all records (cold-start fallback)
    return _avg_ratio(recent_runs, audio_duration)


def _avg_ratio(runs: list[dict], audio_duration: float) -> float | None:
    """Compute estimated transcription time from average ratio of historical runs."""
    ratios = []
    for run in runs:
        audio_dur = run["audio_duration"]
        if audio_dur > 0:
            ratios.append(run["transcription_duration"] / audio_dur)
    if not ratios:
        return None
    return audio_duration * (sum(ratios) / len(ratios))


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
    language: str | None = None,
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
        language: Language code for transcription (e.g., 'uk', 'en', 'de')

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

    # Add language if specified
    if language:
        command.extend(["-l", language])

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

    try:
        # Run transcription with Rich progress bar
        if show_progress and not verbose:
            try:
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[cyan]Transcribing..."),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                ) as progress:
                    # Add task with total based on estimated duration
                    if estimated_duration and estimated_duration > 0:
                        # Use estimated duration in deciseconds for granular updates
                        total = int(estimated_duration * 10)
                        task = progress.add_task("transcribe", total=total)

                        # Run transcription in subprocess
                        proc = subprocess.Popen(
                            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                        )

                        # Update progress based on elapsed time
                        start_time = time.time()
                        while proc.poll() is None:
                            elapsed = time.time() - start_time
                            completed = min(int(elapsed * 10), total - 1)  # Cap at 99%
                            progress.update(task, completed=completed)
                            time.sleep(0.1)

                        # Mark as complete
                        progress.update(task, completed=total)

                        # Check return code
                        if proc.returncode != 0:
                            stderr = proc.stderr.read().decode() if proc.stderr else ""
                            raise subprocess.CalledProcessError(
                                proc.returncode, command, stderr=stderr
                            )

                    else:
                        # No estimate - just show spinner
                        task = progress.add_task("transcribe", total=None)
                        subprocess.run(
                            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                        )
                        progress.update(task, completed=1)

            except ImportError:
                # Fallback if rich is not installed
                logger.info("Transcribing (install 'rich' for progress display)...")
                subprocess.run(
                    command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
        else:
            # Verbose mode or no progress
            if verbose:
                subprocess.run(command, check=True)
            else:
                subprocess.run(
                    command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )

        if verbose:
            logger.info("✓ Transcription completed successfully.")

        return True

    except subprocess.CalledProcessError as e:
        error_msg = f"Whisper transcription failed with exit code {e.returncode}"
        if hasattr(e, "stderr") and e.stderr:
            stderr_text = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
            error_msg += f": {stderr_text}"
        raise TranscriptionError(error_msg) from e

    except FileNotFoundError as e:
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
