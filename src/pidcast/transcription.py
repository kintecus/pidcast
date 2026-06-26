"""Whisper transcription functionality."""

import contextlib
import logging
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .config import (
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_SAMPLE_RATE,
    FFMPEG_PATH,
    WHISPER_CPP_PATH,
    WHISPER_MODEL,
    WHISPER_MODELS_DIR,
    WHISPER_VAD_MODEL,
)
from .exceptions import TranscriptionError, TranscriptionPaused
from .utils import load_json_file

# whisper.cpp stdout segment line: "[HH:MM:SS.mmm --> HH:MM:SS.mmm]   text".
# Tolerant of '.' or ',' as the ms separator (varies by build) and inner spacing.
_SEGMENT_LINE_RE = re.compile(
    r"^\s*\[(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\]\s*(.*)$"
)


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def parse_whisper_segment_line(line: str) -> dict | None:
    """Parse one whisper stdout segment line into ``{from_ms, to_ms, text}``.

    Returns None for lines that are not segment lines (progress, banners, blank).
    A line that *looks* like a segment (starts with ``[``) but fails to parse is
    logged at WARNING so a format drift is caught rather than silently dropped.
    """
    match = _SEGMENT_LINE_RE.match(line)
    if not match:
        if line.lstrip().startswith("[") and "-->" in line:
            logger.warning("Unparseable whisper segment line (format drift?): %r", line.strip())
        return None
    g = match.groups()
    return {
        "from_ms": _ts_to_ms(g[0], g[1], g[2], g[3]),
        "to_ms": _ts_to_ms(g[4], g[5], g[6], g[7]),
        "text": g[8].strip(),
    }


def ms_to_timestamp(ms: int) -> str:
    """Render absolute milliseconds as a whisper-style ``HH:MM:SS,mmm`` string."""
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def merge_segments_to_whisper_json(
    segments: list[dict],
    base_template: dict | None = None,
) -> dict:
    """Assemble a whisper-shaped JSON dict from streamed ``{from_ms,to_ms,text}`` records.

    Each ``transcription[]`` entry carries the ``offsets`` sub-dict (ms ints) that
    :func:`pidcast.diarization.merge_whisper_with_diarization` and the plain-text
    fallback both read, plus regenerated ``timestamps`` strings so the saved file
    is internally consistent. Segments are de-overlapped (a re-decoded segment from
    a VAD-backed-off resume whose start precedes the previous segment's end is
    dropped) and then run through :func:`dedupe_whisper_segments`.
    """
    deoverlapped: list[dict] = []
    last_to = -1
    for seg in sorted(segments, key=lambda s: int(s.get("from_ms", 0))):
        from_ms = int(seg.get("from_ms", 0))
        to_ms = int(seg.get("to_ms", from_ms))
        # Drop a segment fully inside what we already have (resume re-decode overlap).
        if to_ms <= last_to:
            continue
        deoverlapped.append(
            {
                "text": seg.get("text", ""),
                "offsets": {"from": from_ms, "to": to_ms},
                "timestamps": {"from": ms_to_timestamp(from_ms), "to": ms_to_timestamp(to_ms)},
            }
        )
        last_to = to_ms

    transcription = dedupe_whisper_segments(deoverlapped)
    result = dict(base_template) if base_template else {}
    result["transcription"] = transcription
    return result


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


def resolve_vad_model(model_arg: str | None = None, verbose: bool = False) -> str | None:
    """Resolve a Silero VAD model path for whisper.cpp's --vad.

    Resolution precedence:
        1. Explicit ``model_arg`` (CLI/preset value), if it is an existing file.
        2. ``WHISPER_VAD_MODEL`` environment variable, if it is an existing file.
        3. Auto-detect ``ggml-silero-*.bin`` in the models directory, skipping the
           ``for-tests-*`` fixtures bundled with whisper.cpp.

    Args:
        model_arg: Explicit VAD model name or path (overrides env/auto-detect).
        verbose: Log the resolved path when found.

    Returns:
        Absolute path to a VAD model, or None if none can be resolved. Never raises -
        the caller decides whether to warn and continue without VAD.
    """
    if model_arg and Path(model_arg).is_file():
        if verbose:
            logger.info(f"Using VAD model: {model_arg}")
        return str(model_arg)

    if WHISPER_VAD_MODEL and Path(WHISPER_VAD_MODEL).is_file():
        if verbose:
            logger.info(f"Using VAD model from WHISPER_VAD_MODEL: {WHISPER_VAD_MODEL}")
        return WHISPER_VAD_MODEL

    models_dir = _get_models_dir()
    if models_dir:
        for f in sorted(models_dir.glob("ggml-silero-*.bin")):
            if f.name.startswith("for-tests-"):
                continue
            if verbose:
                logger.info(f"Auto-detected VAD model: {f}")
            return str(f)

    return None


# ============================================================================
# TRANSCRIPT POST-PROCESSING
# ============================================================================


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for repeat detection: strip, casefold, drop trailing punctuation."""
    return text.strip().casefold().rstrip(".!?…").strip()


def dedupe_whisper_segments(
    segments: list[dict],
    max_repeats: int = 2,
) -> list[dict]:
    """Collapse runs of identical consecutive whisper segments.

    Whisper hallucinates repeated boilerplate (e.g. "Дякую за перегляд!") on
    low-speech / silence stretches. This collapses any run of more than
    ``max_repeats`` consecutive segments with identical normalized text down to a
    single segment, keeping the first occurrence. A run of ``max_repeats`` or fewer
    is left intact (a legitimately doubled line survives).

    Args:
        segments: Whisper JSON ``transcription`` list (each dict has a ``text`` key).
        max_repeats: Maximum consecutive identical lines to keep (default 2 -> a run
            of 3 or more collapses to one).

    Returns:
        A new list with over-long identical runs collapsed. Non-text/edge cases pass
        through unchanged.
    """
    if not segments:
        return segments

    result: list[dict] = []
    run_key: str | None = None
    run_len = 0

    for seg in segments:
        key = _normalize_for_dedup(seg.get("text", ""))
        if key and key == run_key:
            run_len += 1
            # Beyond the allowed repeats, drop the segment entirely.
            if run_len > max_repeats:
                continue
        else:
            run_key = key
            run_len = 1
        result.append(seg)

    return result


def collapse_repeated_lines(text: str, max_repeats: int = 2) -> str:
    """Collapse runs of identical consecutive lines in a plain-text transcript.

    Line-level counterpart to :func:`dedupe_whisper_segments` for the plain ``.txt``
    output path (no JSON segments available). Preserves blank-line structure.

    Args:
        text: Transcript text.
        max_repeats: Maximum consecutive identical lines to keep.

    Returns:
        Text with over-long identical runs collapsed.
    """
    lines = text.splitlines()
    result: list[str] = []
    run_key: str | None = None
    run_len = 0

    for line in lines:
        key = _normalize_for_dedup(line)
        if key and key == run_key:
            run_len += 1
            if run_len > max_repeats:
                continue
        else:
            run_key = key
            run_len = 1
        result.append(line)

    return "\n".join(result)


# ============================================================================
# AUDIO PROCESSING
# ============================================================================


def build_ffmpeg_audio_conversion_command(
    input_file: str,
    output_file: str,
    overwrite: bool = False,
    start_offset: float | None = None,
    max_duration: float | None = None,
) -> list[str]:
    """Build FFmpeg command for audio conversion to 16kHz mono WAV.

    Args:
        input_file: Path to input audio file
        output_file: Path to output WAV file
        overwrite: Whether to overwrite existing output file
        start_offset: Start time in seconds (seek before decoding)
        max_duration: Maximum duration in seconds to extract

    Returns:
        FFmpeg command as list of arguments
    """
    command = [FFMPEG_PATH]
    if overwrite:
        command.append("-y")
    command.extend(["-i", input_file])
    if start_offset is not None and start_offset > 0:
        command.extend(["-ss", str(start_offset)])
    if max_duration is not None:
        command.extend(["-t", str(max_duration)])
    command.extend(
        [
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
    tier3 = [r for r in recent_runs if (r.get("transcription_provider") or "whisper") == provider]
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
# AUDIO SEGMENT EXTRACTION
# ============================================================================


def extract_audio_segment(
    audio_file: str | Path,
    output_file: str | Path,
    start_offset: float = 0,
    max_duration: float = 120,
    verbose: bool = False,
) -> str:
    """Extract a segment from an audio file using ffmpeg.

    Args:
        audio_file: Path to input audio file
        output_file: Path to output WAV file
        start_offset: Start time in seconds
        max_duration: Duration in seconds to extract
        verbose: Enable verbose output

    Returns:
        Path to the extracted segment file
    """
    command = build_ffmpeg_audio_conversion_command(
        str(audio_file),
        str(output_file),
        overwrite=True,
        start_offset=start_offset,
        max_duration=max_duration,
    )

    if verbose:
        logger.info(f"Extracting segment: offset={start_offset}s, duration={max_duration}s")
        subprocess.run(command, check=True)
    else:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    return str(output_file)


# ============================================================================
# TRANSCRIPTION
# ============================================================================


def _run_whisper_streaming(
    command: list[str],
    *,
    verbose: bool,
    show_progress: bool,
    estimated_duration: float | None,
    audio_duration: float | None,
    offset_ms: int | None,
    segment_callback: Callable[[dict], None] | None,
    pause_check: Callable[[], bool] | None,
) -> None:
    """Run whisper as a single Popen, reading stdout line-by-line.

    Unifies every non-verbose path (estimate / no-estimate / no-rich) onto one
    streaming loop so the segment callback and pause check work regardless of
    whether a duration estimate is available. whisper.cpp prints each completed
    segment to stdout and flushes, so we parse those lines live.

    In verbose mode stdout is inherited (whisper prints directly) so we cannot
    parse segments - pause still works via process polling, but per-segment
    checkpointing does not. Callers needing checkpointing run non-verbose.

    Raises:
        TranscriptionPaused: if ``pause_check`` returns True mid-run.
        subprocess.CalledProcessError: on non-zero exit.
    """
    offset_base = int(offset_ms or 0)

    # Progress is driven by AUDIO POSITION (segment end-time vs total audio
    # duration), not wall-clock, so it stays correct when resuming from an offset:
    # the bar starts at offset/duration and Rich derives the ETA from throughput.
    # Fall back to the wall-clock estimate only when the true duration is unknown.
    total = None
    if audio_duration and audio_duration > 0:
        total = int(audio_duration * 1000)  # ms scale (audio timeline)
    elif estimated_duration and estimated_duration > 0:
        total = int(estimated_duration * 1000)

    label = "Resuming transcription" if offset_base > 0 else "Transcribing"
    progress_ctx = None
    progress = None
    task = None
    if show_progress and not verbose:
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{label}..."),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[dim]elapsed[/dim]"),
                TimeElapsedColumn(),
                TextColumn("[dim]eta[/dim]"),
                TimeRemainingColumn(),
            )
            progress_ctx = progress
        except ImportError:
            logger.info(f"{label} (install 'rich' for progress display)...")

    # Verbose: inherit stdout so whisper prints directly; only poll for pause.
    stdout_target = None if verbose else subprocess.PIPE
    proc = subprocess.Popen(
        command,
        stdout=stdout_target,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _terminate_for_pause() -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    last_to_ms = offset_base
    try:
        if progress_ctx is not None:
            progress_ctx.start()
            task = progress.add_task("transcribe", total=total)
            # Seed the bar at the resume offset so % and ETA are right from the
            # first new segment (a resumed run starts partway through the audio).
            if total and offset_base > 0:
                progress.update(task, completed=min(offset_base, total - 1))

        if proc.stdout is not None:
            for line in proc.stdout:
                if pause_check is not None and pause_check():
                    _terminate_for_pause()
                    raise TranscriptionPaused(last_offset_ms=last_to_ms)

                seg = parse_whisper_segment_line(line)
                if seg is None:
                    continue
                last_to_ms = seg["to_ms"]
                if segment_callback is not None:
                    segment_callback(seg)
                if progress is not None and task is not None and total:
                    progress.update(task, completed=min(seg["to_ms"], total - 1))
        else:
            # Verbose path: no stdout to read; poll for pause.
            while proc.poll() is None:
                if pause_check is not None and pause_check():
                    _terminate_for_pause()
                    raise TranscriptionPaused(last_offset_ms=last_to_ms)
                time.sleep(0.2)

        proc.wait()
        if progress is not None and task is not None and total:
            progress.update(task, completed=total)

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise subprocess.CalledProcessError(proc.returncode, command, stderr=stderr)
    finally:
        if progress_ctx is not None:
            progress_ctx.stop()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None and not proc.stderr.closed:
            with contextlib.suppress(Exception):
                proc.stderr.close()


def run_whisper_transcription(
    audio_file: str | Path,
    whisper_model: str,
    output_format: str,
    output_file: str | Path,
    verbose: bool = False,
    estimated_duration: float | None = None,
    show_progress: bool = True,
    language: str | None = None,
    *,
    threads: int = 8,
    vad_model: str | None = None,
    vad_threshold: float | None = None,
    no_speech_thold: float | None = None,
    temperature: float | None = None,
    no_fallback: bool = False,
    suppress_nst: bool = False,
    prompt: str | None = None,
    audio_duration: float | None = None,
    offset_ms: int | None = None,
    segment_callback: "Callable[[dict], None] | None" = None,
    pause_check: "Callable[[], bool] | None" = None,
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
        threads: Number of decoding threads (whisper -t)
        vad_model: Resolved path to a Silero VAD model. When set, enables VAD
            (--vad) to strip silence before decoding, the primary defense against
            silence hallucinations. None disables VAD.
        vad_threshold: VAD speech-probability threshold (whisper -vt). None uses
            the whisper default (0.50).
        no_speech_thold: No-speech threshold (whisper -nth). None uses the whisper
            default (0.60).
        temperature: Sampling temperature (whisper -tp). None uses the whisper
            default (0.00).
        no_fallback: Disable temperature fallback while decoding (whisper -nf).
        suppress_nst: Suppress non-speech tokens (whisper -sns). Reduces boilerplate
            hallucinations on low-speech audio.
        prompt: Initial-prompt text passed to whisper --prompt to bias decoding
            toward domain terms (proper nouns, jargon). None omits the flag. When
            set, --carry-initial-prompt is also passed so the bias persists across
            all decode windows, not just the first (whisper applies the initial
            prompt only to the first window otherwise). Limited to roughly half the
            model's text context (~200+ tokens); longer prompts are truncated.
        offset_ms: Resume offset in milliseconds (whisper -ot). Decoding starts here;
            emitted timestamps remain absolute (relative to file start).
        segment_callback: Invoked once per completed segment as it streams off
            whisper's stdout, with ``{"from_ms", "to_ms", "text"}`` (absolute ms).
            Enables live checkpointing for pause/resume.
        pause_check: Polled between stdout lines; if it returns True the whisper
            subprocess is terminated cleanly and the function raises
            :class:`TranscriptionPaused` (already-streamed segments are preserved).

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
        str(threads),
    ]

    # Add language if specified
    if language:
        command.extend(["-l", language])

    # Anti-hallucination / decoding quality flags
    if no_speech_thold is not None:
        command.extend(["-nth", str(no_speech_thold)])
    if temperature is not None:
        command.extend(["-tp", str(temperature)])
    if no_fallback:
        command.append("-nf")
    if suppress_nst:
        command.append("-sns")
    if vad_model:
        command.extend(["--vad", "-vm", str(vad_model)])
        if vad_threshold is not None:
            command.extend(["-vt", str(vad_threshold)])

    # Glossary/initial-prompt biasing. --carry-initial-prompt makes whisper apply
    # the prompt to every decode window (not just the first), so the bias holds
    # across a long file. NOTE: -p is --processors, NOT prompt - do not use it.
    if prompt:
        command.extend(["--prompt", prompt, "--carry-initial-prompt"])

    # Resume: start decoding at a time offset. whisper.cpp emits ABSOLUTE
    # timestamps under -ot (verified), so the streamed segments need no
    # offset-correction - they align directly onto the pre-pause segments.
    if offset_ms and offset_ms > 0:
        command.extend(["-ot", str(int(offset_ms))])

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
        _run_whisper_streaming(
            command,
            verbose=verbose,
            show_progress=show_progress,
            estimated_duration=estimated_duration,
            audio_duration=audio_duration,
            offset_ms=offset_ms,
            segment_callback=segment_callback,
            pause_check=pause_check,
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
            f"Whisper binary not found at: {WHISPER_CPP_PATH}\n"
            "  Fix: Set WHISPER_CPP_PATH in .env to your whisper-cli binary path.\n"
            "  Or use cloud transcription: pidcast <input> --transcription-provider elevenlabs\n"
            "  Run 'pidcast doctor' to check your setup."
        ) from e


def process_local_file(
    input_file: str | Path,
    output_dir: str | Path,
    verbose: bool = False,
    start_offset: float | None = None,
    max_duration: float | None = None,
) -> tuple[str, dict]:
    """Process a local audio file for transcription.

    Args:
        input_file: Path to local audio file
        output_dir: Directory to store processed files
        verbose: Enable verbose output
        start_offset: Start time in seconds for segment extraction
        max_duration: Maximum duration in seconds for segment extraction

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

    # Clamp duration for segment extraction
    if max_duration is not None and info.duration > 0:
        offset = start_offset or 0
        info.duration = min(info.duration - offset, max_duration)
        info.duration_string = fmt_dur(info.duration)

    # Convert to 16kHz mono WAV if needed
    output_wav = output_dir / f"{name}_16k.wav"

    if verbose:
        logger.info(f"Converting {input_file.name} to 16kHz mono WAV...")

    try:
        command = build_ffmpeg_audio_conversion_command(
            str(input_file),
            str(output_wav),
            overwrite=True,
            start_offset=start_offset,
            max_duration=max_duration,
        )

        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        return str(output_wav), info

    except subprocess.CalledProcessError as e:
        raise FileProcessingError(f"Error converting audio: {e}") from e
