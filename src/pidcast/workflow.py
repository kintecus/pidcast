"""Core workflow for media processing."""

import argparse
import json
import logging
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .analysis import (
    load_analysis_prompts,
    render_analysis_to_terminal,
)
from .apple_podcasts import is_apple_podcasts_url, resolve_apple_podcasts_url
from .config import (
    TranscriptionStats,
    VideoInfo,
)
from .download import download_audio
from .exceptions import (
    AnalysisError,
    ApplePodcastsResolutionError,
    ConfigurationError,
    DiarizationError,
    DownloadError,
    FileProcessingError,
    TranscriptionError,
)
from .markdown import create_analysis_markdown_file, create_markdown_file
from .transcription import (
    estimate_transcription_time,
    extract_audio_segment,
    process_local_file,
)
from .utils import (
    cleanup_temp_files,
    create_smart_filename,
    format_duration,
    get_unique_filename,
    log_error,
    log_section,
    log_success,
    save_statistics,
    validate_input_source,
)

logger = logging.getLogger(__name__)


def _extract_whisper_model_name(model_path: str | None) -> str | None:
    """Extract clean model name from whisper model path.

    Example: '/path/to/ggml-large-v3.bin' -> 'large-v3'
    """
    if not model_path:
        return None
    return Path(model_path).stem.removeprefix("ggml-")


def parse_transcript_file(filepath: str, verbose: bool = False) -> tuple[str, VideoInfo]:
    """Parse a transcript file and extract transcript text and metadata.

    Args:
        filepath: Path to transcript file (.md or .txt)
        verbose: Enable verbose output

    Returns:
        Tuple of (transcript_text, video_info)

    Raises:
        FileProcessingError: If file doesn't exist, is empty, or can't be parsed
    """
    file_path = Path(filepath)

    # Validate file exists
    if not file_path.exists():
        raise FileProcessingError(f"File not found: {filepath}")

    # Validate extension
    if file_path.suffix not in [".md", ".txt"]:
        raise FileProcessingError(
            f"Unsupported file type: {file_path.suffix}. Only .md and .txt files are supported."
        )

    # Read file content
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise FileProcessingError(f"Failed to read file: {e}") from e

    # Validate not empty
    if not content.strip():
        raise FileProcessingError("File is empty")

    if verbose:
        logger.info(f"Parsing {file_path.suffix} file: {filepath}")

    # Parse based on file type
    if file_path.suffix == ".md":
        # Try to parse YAML front matter
        parts = content.split("---", 2)

        if len(parts) >= 3:
            # Has YAML front matter
            try:
                yaml_content = parts[1].strip()
                transcript_text = parts[2].strip()

                # Parse YAML manually (simple key: value parsing)
                metadata: dict[str, Any] = {}
                for line in yaml_content.split("\n"):
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        metadata[key] = value

                # Create VideoInfo from metadata
                video_info = VideoInfo(
                    title=metadata.get("title", file_path.stem),
                    webpage_url=metadata.get("url", ""),
                    channel=metadata.get("channel", "Unknown"),
                    duration_string=metadata.get("duration", "Unknown"),
                    description=metadata.get("description", ""),
                )

                if verbose:
                    logger.info(f"Extracted metadata: title='{video_info.title}'")

            except Exception as e:
                # Fallback: treat as plain text
                if verbose:
                    logger.warning(
                        f"Failed to parse YAML front matter, treating as plain text: {e}"
                    )
                transcript_text = content
                video_info = VideoInfo(title=file_path.stem)
        else:
            # No YAML front matter, treat as plain text
            transcript_text = content
            video_info = VideoInfo(title=file_path.stem)

    else:  # .txt file
        transcript_text = content
        video_info = VideoInfo(title=file_path.stem)

        if verbose:
            logger.info(f"Plain text file - using filename as title: '{video_info.title}'")

    # Validate transcript is not empty
    if not transcript_text.strip():
        raise FileProcessingError("No transcript content found in file")

    return transcript_text, video_info


def render_analysis_to_terminal_direct(
    result: Any,
    video_info: VideoInfo,
    verbose: bool = False,
) -> None:
    """Render analysis result directly to terminal without saving to file.

    Args:
        result: AnalysisResult from LLM analysis
        video_info: Video metadata
        verbose: Enable verbose output
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        # Fallback to plain text
        print(f"Title: {video_info.title}")
        print(f"Analysis Type: {result.analysis_name}")
        print(f"Model: {result.model}")
        print("-" * 60)
        print(result.analysis_text)
        return

    console = Console()

    # Build metadata table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan bold", width=20)
    table.add_column(style="white")

    table.add_row("Title", video_info.title)
    table.add_row("Analysis Type", result.analysis_name)
    table.add_row("Model", result.model)
    table.add_row("Tokens", str(result.tokens_total))

    if result.estimated_cost:
        table.add_row("Cost", f"${result.estimated_cost:.4f}")

    if verbose:
        table.add_row("", "")  # Blank row
        table.add_row("Channel", video_info.channel or "Unknown")
        table.add_row("Duration", video_info.duration_string or "Unknown")
        table.add_row(
            "Tokens (In/Out)",
            f"{result.tokens_input}/{result.tokens_output}",
        )
        table.add_row("Truncated", "Yes" if result.truncated else "No")

    # Render metadata panel
    console.print(
        Panel(table, title="[bold cyan]Analysis Metadata[/bold cyan]", border_style="cyan")
    )
    console.print()

    # Render markdown content
    md = Markdown(result.analysis_text)
    console.print(md)


def run_analysis(
    markdown_file: Path | None,
    video_info: VideoInfo,
    output_dir: Path,
    args: argparse.Namespace,
    transcript_text: str | None = None,
    save_to_file: bool = True,
    is_local_file: bool = False,
    custom_tags: list[str] | None = None,
) -> tuple[Path | None, float, dict[str, Any]]:
    """Run LLM analysis on transcript.

    Args:
        markdown_file: Path to transcript markdown file (None if using transcript_text)
        video_info: Video metadata
        output_dir: Output directory
        args: Parsed arguments
        transcript_text: Optional pre-loaded transcript text
        save_to_file: Whether to save the analysis to a file
        is_local_file: Whether the source is a local file
        custom_tags: User-provided tags (overrides inferred source tags)

    Returns:
        Tuple of (analysis_file, duration, metadata)
    """
    log_section("Starting LLM Analysis")

    prompts_file = args.prompts_file
    prompts_config = load_analysis_prompts(prompts_file, args.verbose)

    # Get transcript text
    if transcript_text is None:
        if markdown_file is None:
            raise AnalysisError("Either markdown_file or transcript_text must be provided")

        # Read transcript from markdown file
        with open(markdown_file, encoding="utf-8") as f:
            content = f.read()

        # Extract transcript (everything after front matter)
        parts = content.split("---", 2)
        transcript_text = parts[2].strip() if len(parts) >= 3 else content

    # Build provider
    provider_name = getattr(args, "provider", "groq")

    if provider_name == "claude":
        from .providers.claude_provider import ClaudeProvider

        analysis_provider = ClaudeProvider(model=getattr(args, "claude_model", None))
    else:
        from .providers.groq_provider import GroqProvider

        groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise AnalysisError(
                "Groq API key not found. Set GROQ_API_KEY environment variable or use --groq_api_key"
            )
        analysis_provider = GroqProvider(api_key=groq_api_key, model=args.groq_model)

    # Analyze
    analysis_start = time.time()
    result = analysis_provider.analyze(
        transcript_text,
        video_info,
        args.analysis_type,
        prompts_config,
        args.verbose,
    )

    analysis_duration = time.time() - analysis_start
    metadata = {
        "analysis_type": result.analysis_type,
        "analysis_name": result.analysis_name,
        "model": result.model,
        "tokens_total": result.tokens_total,
        "estimated_cost": result.estimated_cost,
        "truncated": result.truncated,
    }

    analysis_file: Path | None = None

    # Create analysis markdown file only if saving
    if save_to_file:
        # If markdown_file is None, create a placeholder filename from video title
        source_file_for_naming = markdown_file
        if source_file_for_naming is None:
            # Create a smart filename for the analysis based on video title
            smart_filename = create_smart_filename(
                video_info.title, max_length=60, include_date=True
            )
            source_file_for_naming = output_dir / f"{smart_filename}.md"

        analysis_file = create_analysis_markdown_file(
            {
                "analysis_text": result.analysis_text,
                "analysis_type": result.analysis_type,
                "analysis_name": result.analysis_name,
                "model": result.model,
                "provider": result.provider,
                "tokens_input": result.tokens_input,
                "tokens_output": result.tokens_output,
                "tokens_total": result.tokens_total,
                "estimated_cost": result.estimated_cost,
                "duration": result.duration,
                "truncated": result.truncated,
                "contextual_tags": result.contextual_tags,
            },
            source_file_for_naming,
            video_info,
            output_dir,
            args.verbose,
            is_local_file=is_local_file,
            custom_tags=custom_tags,
        )

        if not analysis_file:
            raise AnalysisError("Failed to create analysis file")

        logger.info(f"\n✓ Analysis completed in {format_duration(analysis_duration)}")
        logger.info(f"✓ Analysis file: file://{analysis_file.absolute()}")

        # Render analysis to terminal from file
        print()
        render_analysis_to_terminal(analysis_file, verbose=args.verbose)
    else:
        # Terminal-only output - render directly without saving
        logger.info(f"\n✓ Analysis completed in {format_duration(analysis_duration)}")
        print()
        render_analysis_to_terminal_direct(result, video_info, args.verbose)

    if result.estimated_cost:
        logger.info(f"\n✓ Cost: ${result.estimated_cost:.4f}")

    logger.info(f"✓ Generated with: {result.model}")

    return analysis_file, analysis_duration, metadata


def run_analyze_existing_mode(
    transcript_path: str | Path,
    args: argparse.Namespace,
    output_dir: Path,
    analysis_output_dir: Path,
    stats_file: Path,
    run_uid: str,
    run_timestamp: str,
    start_time: float,
) -> bool:
    """Run analysis on an existing transcript file.

    Args:
        transcript_path: Path to the existing transcript file
        args: Parsed command-line arguments
        output_dir: Output directory for transcript files
        analysis_output_dir: Output directory for analysis files (may be Obsidian vault)
        stats_file: Path to statistics file
        run_uid: Unique run identifier
        run_timestamp: ISO timestamp for this run
        start_time: Start time for duration calculation

    Returns:
        True if analysis completed successfully, False otherwise
    """
    transcript_path = Path(transcript_path)

    if args.verbose:
        log_section("Analyze Existing Transcript")
        logger.info(f"Input file: {transcript_path}")
        logger.info(f"Run ID: {run_uid}")
        logger.info(f"Timestamp: {run_timestamp}")

    try:
        # Ensure output directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_output_dir.mkdir(parents=True, exist_ok=True)

        # Parse transcript file
        transcript_text, video_info = parse_transcript_file(str(transcript_path), args.verbose)

        logger.info(f"Title: {video_info.title}")
        if video_info.channel != "Unknown":
            logger.info(f"Channel: {video_info.channel}")

        # Determine if we should save to file
        should_save = args.save or args.save_to_obsidian

        # Run analysis
        analysis_file, analysis_duration, analysis_metadata = run_analysis(
            None,  # No source markdown file in analyze-existing mode
            video_info,
            analysis_output_dir,
            args,
            transcript_text=transcript_text,
            save_to_file=should_save,
        )

        # Save statistics
        end_time = time.time()
        duration = end_time - start_time

        # Store the original transcript filename
        transcript_filename = Path(transcript_path).name

        stats = TranscriptionStats(
            run_uid=run_uid,
            run_timestamp=run_timestamp,
            video_title=video_info.title,
            smart_filename=transcript_filename,
            video_url=str(transcript_path),
            run_duration=duration,
            transcription_duration=0,
            audio_duration=0,
            success=True,
            saved_to_obsidian=args.save_to_obsidian,
            is_local_file=True,
            analysis_only=True,
            analysis_performed=True,
            analysis_type=analysis_metadata.get("analysis_type"),
            analysis_name=analysis_metadata.get("analysis_name"),
            analysis_duration=analysis_duration,
            analysis_model=analysis_metadata.get("model"),
            analysis_tokens=analysis_metadata.get("tokens_total", 0),
            analysis_cost=analysis_metadata.get("estimated_cost", 0) or 0,
            analysis_truncated=analysis_metadata.get("truncated", False),
            analysis_file=analysis_file.name if analysis_file else None,
        )

        save_statistics(stats_file, stats, args.verbose)

        log_section("✓ Analysis completed successfully!")
        logger.info(f"Total duration: {format_duration(duration)}")

        return True

    except (FileProcessingError, AnalysisError) as e:
        log_error(str(e))
        if args.verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        log_error(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


def run_diarize_existing_mode(
    transcript_path: str | Path,
    audio_override: str | None = None,
    verbose: bool = False,
) -> bool:
    """Run diarization on an existing transcript using saved whisper JSON.

    Args:
        transcript_path: Path to the existing transcript .md file
        audio_override: Optional explicit path to audio file
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    from .config import HUGGINGFACE_TOKEN
    from .diarization import merge_whisper_with_diarization, run_diarization
    from .markdown import format_yaml_front_matter

    transcript_path = Path(transcript_path)

    if verbose:
        log_section("Diarize Existing Transcript")
        logger.info(f"Input file: {transcript_path}")

    try:
        # Parse transcript to get metadata and text
        _, video_info = parse_transcript_file(str(transcript_path), verbose)
        logger.info(f"Title: {video_info.title}")

        # Locate whisper JSON
        stem = transcript_path.stem
        parent = transcript_path.parent
        whisper_json = parent / f"{stem}.whisper.json"

        if not whisper_json.exists():
            log_error(
                f"Whisper JSON not found: {whisper_json}\n"
                "This file is auto-saved during transcription. "
                "Re-run transcription to generate it."
            )
            return False

        # Locate audio file
        audio_file = None
        if audio_override:
            audio_file = Path(audio_override)
            if not audio_file.exists():
                log_error(f"Audio file not found: {audio_file}")
                return False
        else:
            # Look for .wav next to the transcript
            audio_wav = parent / f"{stem}.wav"
            if audio_wav.exists():
                audio_file = audio_wav
            else:
                log_error(
                    f"Audio file not found: {audio_wav}\n"
                    "Provide an audio file with --audio /path/to/file.wav"
                )
                return False

        logger.info(f"Audio: {audio_file}")
        logger.info(f"Whisper JSON: {whisper_json}")

        # Validate HuggingFace token
        if not HUGGINGFACE_TOKEN:
            from .exceptions import DiarizationError

            raise DiarizationError(
                "HUGGINGFACE_TOKEN environment variable not set. Required for speaker diarization."
            )

        # Run diarization
        logger.info("Running speaker diarization...")
        import time

        start_time = time.time()
        diarization_segments = run_diarization(str(audio_file), HUGGINGFACE_TOKEN, verbose)
        diarization_duration = time.time() - start_time

        # Read whisper JSON and merge
        with open(whisper_json, encoding="utf-8") as f:
            whisper_data = json.load(f)

        whisper_segments = whisper_data.get("transcription", [])
        diarized_text, speaker_count = merge_whisper_with_diarization(
            whisper_segments, diarization_segments
        )

        if speaker_count and speaker_count > 0:
            logger.info(f"Speakers detected: {speaker_count}")
        else:
            logger.warning("No speaker segments detected")

        # Re-read the original file to preserve front matter structure
        content = transcript_path.read_text(encoding="utf-8")
        parts = content.split("---", 2)

        if len(parts) >= 3:
            # Parse and update front matter
            import yaml

            try:
                metadata = yaml.safe_load(parts[1]) or {}
            except Exception:
                metadata = {}
            metadata["diarized"] = True
            metadata["speaker_count"] = speaker_count

            front_matter_str = format_yaml_front_matter(metadata)
            updated_content = f"{front_matter_str}\n\n{diarized_text}"
        else:
            updated_content = diarized_text

        transcript_path.write_text(updated_content, encoding="utf-8")

        log_section("Diarization completed successfully!")
        logger.info(f"Updated: file://{transcript_path.absolute()}")
        logger.info(f"Diarization time: {format_duration(diarization_duration)}")
        logger.info(f"Speakers: {speaker_count}")

        return True

    except DiarizationError as e:
        log_error(str(e))
        if verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        log_error(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def process_input_source(
    input_source: str,
    args: argparse.Namespace,
    output_dir: Path,
    analysis_output_dir: Path,
    stats_file: Path,
    run_uid: str,
    run_timestamp: str,
    start_time: float,
    video_info_override: VideoInfo | None = None,
) -> bool:
    """Execute the main transcription workflow.

    Args:
        input_source: URL or path to file
        args: Parsed command-line arguments
        output_dir: Output directory for transcript files
        analysis_output_dir: Output directory for analysis files (may be Obsidian vault)
        stats_file: Path to statistics file
        run_uid: Unique run identifier
        run_timestamp: ISO timestamp for this run
        start_time: Start time for duration calculation
        video_info_override: Optional VideoInfo to override detected metadata

    Returns:
        True if successful, False otherwise
    """
    audio_file: str | None = None
    is_local_file = False
    video_info: VideoInfo | None = None
    smart_filename: str | None = None
    transcription_result = None
    success = False

    try:
        # Validate input
        source, is_local_file = validate_input_source(input_source)

        if args.verbose:
            if is_local_file:
                source_type = "Local File"
            elif is_apple_podcasts_url(source):
                source_type = "Apple Podcasts URL"
            else:
                source_type = "YouTube URL"
            logger.info(f"Type: {source_type}")

        # Ensure output directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_output_dir.mkdir(parents=True, exist_ok=True)

        # Compute segment parameters
        test_segment = getattr(args, "test_segment", None)
        start_at = getattr(args, "start_at", None)
        segment_offset = start_at * 60 if start_at else None
        segment_duration = test_segment * 60 if test_segment is not None else None

        if test_segment is not None:
            offset_str = f" from {start_at:.0f}:00" if start_at else ""
            log_section(f"Test Segment: {test_segment:.0f} min{offset_str}")

        # Process input source
        if is_local_file:
            logger.info("Processing local file...")
            audio_file, video_info = process_local_file(
                source,
                output_dir,
                args.verbose,
                start_offset=segment_offset,
                max_duration=segment_duration,
            )
            logger.info("\n✓ Local file processed successfully!")
        elif is_apple_podcasts_url(source):
            logger.info("Resolving Apple Podcasts URL...")
            audio_url, video_info = resolve_apple_podcasts_url(source, args.verbose)
            logger.info(f"Found: {video_info.title}")
            logger.info("Downloading podcast audio...")
            audio_file, _ = download_audio(audio_url, "temp_audio.%(ext)s", args.verbose)
            logger.info("\n✓ Audio downloaded successfully!")
        else:
            logger.info("Downloading audio from YouTube...")
            audio_file, video_info = download_audio(
                source,
                "temp_audio.%(ext)s",
                args.verbose,
                args.po_token,
                cookies_from_browser=getattr(args, "cookies_from_browser", None),
                cookies=getattr(args, "cookies", None),
                chrome_profile=getattr(args, "chrome_profile", None),
            )
            logger.info("\n✓ Audio downloaded successfully!")

        # Apply override if provided
        if video_info_override and video_info:
            video_info.title = video_info_override.title or video_info.title
            video_info.description = video_info_override.description or video_info.description
            if video_info_override.channel:
                video_info.channel = video_info_override.channel
            if video_info_override.upload_date:
                video_info.upload_date = video_info_override.upload_date

        logger.info(f"Title: {video_info.title}")

        # Create smart filename
        smart_filename = create_smart_filename(video_info.title, max_length=60, include_date=True)
        if args.verbose:
            logger.info(f"Smart filename: {smart_filename}")

        # Verify audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Extract segment for URL sources (local files already handled in process_local_file)
        if segment_duration is not None and not is_local_file:
            trimmed_file = str(output_dir / f"test_segment_{uuid.uuid4().hex[:8]}.wav")
            extract_audio_segment(
                audio_file,
                trimmed_file,
                start_offset=segment_offset or 0,
                max_duration=segment_duration,
                verbose=args.verbose,
            )
            cleanup_temp_files(audio_file, args.verbose)
            audio_file = trimmed_file
            # Clamp video_info duration
            if video_info.duration > 0:
                offset = segment_offset or 0
                video_info.duration = min(video_info.duration - offset, segment_duration)
                video_info.duration_string = format_duration(video_info.duration)

        # Resolve provider and diarization settings
        from .config import ELEVENLABS_API_KEY, WHISPER_CPP_PATH

        transcription_provider_name = getattr(args, "transcription_provider", "whisper")
        diarize = getattr(args, "diarize", False)
        whisper_model_path = getattr(args, "whisper_model", None)
        whisper_model_name = _extract_whisper_model_name(whisper_model_path)

        # Auto-select provider if whisper isn't configured
        if transcription_provider_name == "whisper" and not WHISPER_CPP_PATH:
            if ELEVENLABS_API_KEY:
                logger.info("Whisper not configured, using ElevenLabs provider")
                transcription_provider_name = "elevenlabs"
            else:
                raise ConfigurationError(
                    "No transcription provider configured.\n"
                    "  Run 'pidcast setup' to get started, or configure:\n"
                    "  - WHISPER_CPP_PATH in .env (for local transcription)\n"
                    "  - ELEVENLABS_API_KEY in .env (for cloud transcription)"
                )

        # Estimate transcription time (filtered by provider/model/diarization)
        audio_duration = video_info.duration
        estimated_time = estimate_transcription_time(
            stats_file,
            audio_duration,
            provider=transcription_provider_name,
            whisper_model=whisper_model_name if transcription_provider_name == "whisper" else None,
            diarize=diarize,
        )

        # Run transcription
        if estimated_time:
            logger.info(
                f"Estimated transcription time: ~{format_duration(estimated_time)} "
                "(based on historical data)"
            )
        elif audio_duration > 0:
            logger.info(f"Audio duration: {format_duration(audio_duration)}")

        if transcription_provider_name == "elevenlabs":
            from .config import ELEVENLABS_API_KEY
            from .providers.elevenlabs_provider import ElevenLabsTranscriptionProvider

            logger.info("\nTranscribing audio with ElevenLabs Scribe v2...")
            transcription_provider = ElevenLabsTranscriptionProvider(
                api_key=ELEVENLABS_API_KEY,
            )
        else:
            from .providers.whisper_provider import WhisperTranscriptionProvider

            # Save whisper JSON for diarization retry (always, since it's small)
            whisper_json_dest = output_dir / f"{smart_filename}.whisper.json"

            logger.info("\nTranscribing audio with Whisper...")
            transcription_provider = WhisperTranscriptionProvider(
                whisper_model=args.whisper_model,
                output_format=args.output_format.replace("o", ""),
                output_dir=output_dir,
                estimated_duration=estimated_time,
                save_whisper_json_to=whisper_json_dest,
            )

        transcription_result = transcription_provider.transcribe(
            audio_file=audio_file,
            language=getattr(args, "language", None),
            diarize=diarize,
            verbose=args.verbose,
        )

        transcription_duration = transcription_result.duration
        speaker_count = transcription_result.speaker_count

        # Auto-save audio when diarizing (for diarization retry)
        saved_audio_path = None
        if diarize and audio_file and Path(audio_file).exists():
            saved_audio_path = output_dir / f"{smart_filename}.wav"
            if not saved_audio_path.exists():
                shutil.copy2(audio_file, saved_audio_path)
                if args.verbose:
                    logger.info(f"Saved audio for diarization retry: {saved_audio_path}")

        if diarize and speaker_count and speaker_count > 0:
            logger.info(f"\n✓ Diarization complete - {speaker_count} speaker(s) detected")
        elif diarize:
            logger.warning("Diarization returned no speaker segments, using plain transcript")

        # Test segment: show preview and exit early
        if test_segment is not None:
            log_section("Test Segment Results")
            preview = transcription_result.text[:3000]
            if len(transcription_result.text) > 3000:
                preview += "\n\n[... truncated ...]"
            print(preview)
            print()
            if transcription_result.diarized and speaker_count:
                logger.info(f"Speakers detected: {speaker_count}")
            logger.info(f"Transcription time: {format_duration(transcription_duration)}")
            logger.info(
                f"Segment: {test_segment:.0f} min"
                + (f" from {start_at:.0f}:00" if start_at else "")
            )
            logger.info("\nResults look good? Run without --test-segment for full transcription.")
            success = True
            return True

        # Write transcript to temp file for markdown creation
        transcript_file = str(output_dir / f"temp_transcript_{uuid.uuid4().hex[:8]}.txt")
        Path(transcript_file).write_text(transcription_result.text, encoding="utf-8")

        # Create Markdown file (transcript)

        logger.info("\nCreating Markdown file...")
        markdown_file = get_unique_filename(output_dir, smart_filename, ".md")

        try:
            front_matter = json.loads(args.front_matter)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in front_matter, using empty dict: {e}")
            front_matter = {}

        if transcription_result.diarized and transcription_result.speaker_count:
            front_matter["diarized"] = True
            front_matter["speaker_count"] = transcription_result.speaker_count

        # Parse custom tags from CLI
        custom_tags = None
        raw_tags = getattr(args, "tags", None)
        if raw_tags:
            custom_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

        if not create_markdown_file(
            markdown_file,
            transcript_file,
            video_info,
            front_matter,
            args.verbose,
            is_local_file=is_local_file,
            custom_tags=custom_tags,
        ):
            raise FileProcessingError("Failed to create Markdown file")

        # LLM Analysis (default: enabled unless --no-analyze)
        analysis_file: Path | None = None
        analysis_duration = 0.0
        analysis_performed = False
        analysis_metadata: dict[str, Any] = {}

        should_analyze = not args.no_analyze
        # Auto-skip analysis if no API key configured
        if should_analyze:
            groq_key = os.environ.get("GROQ_API_KEY") or getattr(args, "groq_api_key", None)
            provider_name = getattr(args, "provider", "groq")
            if provider_name == "groq" and not groq_key:
                logger.info(
                    "Skipping analysis (GROQ_API_KEY not set). "
                    "Get a free key at: https://console.groq.com/"
                )
                should_analyze = False
        should_save_analysis = args.save or args.save_to_obsidian

        if should_analyze:
            try:
                analysis_file, analysis_duration, analysis_metadata = run_analysis(
                    markdown_file,
                    video_info,
                    analysis_output_dir,
                    args,
                    transcript_text=None,
                    save_to_file=should_save_analysis,
                    is_local_file=is_local_file,
                    custom_tags=custom_tags,
                )
                analysis_performed = True
            except AnalysisError as e:
                if args.skip_analysis_on_error:
                    log_error(f"{e} (continuing)")
                else:
                    raise

        # Clean up transcript file
        transcript_path = Path(transcript_file)
        if not args.keep_transcript and transcript_path.exists():
            transcript_path.unlink()
            if args.verbose:
                logger.debug(f"Cleaned up temporary transcript file: {transcript_file}")
        elif args.keep_transcript and transcript_path.exists():
            if args.verbose:
                log_success(f"Kept transcript file: {transcript_file}")

        # Store statistics
        end_time = time.time()
        duration = end_time - start_time
        filename_for_stats = markdown_file.name if markdown_file else ""

        stats = TranscriptionStats(
            run_uid=run_uid,
            run_timestamp=run_timestamp,
            video_title=video_info.title,
            smart_filename=filename_for_stats,
            video_url=input_source,
            run_duration=duration,
            transcription_duration=transcription_duration,
            audio_duration=audio_duration,
            success=True,
            saved_to_obsidian=args.save_to_obsidian,
            is_local_file=is_local_file,
            analysis_performed=analysis_performed,
            analysis_type=analysis_metadata.get("analysis_type"),
            analysis_name=analysis_metadata.get("analysis_name"),
            analysis_duration=analysis_duration,
            analysis_model=analysis_metadata.get("model"),
            analysis_tokens=analysis_metadata.get("tokens_total", 0),
            analysis_cost=analysis_metadata.get("estimated_cost", 0) or 0,
            analysis_truncated=analysis_metadata.get("truncated", False),
            analysis_file=analysis_file.name if analysis_file else None,
            diarization_performed=diarize,
            transcription_provider=transcription_provider_name,
            whisper_model=whisper_model_name if transcription_provider_name == "whisper" else None,
            speaker_count=speaker_count,
        )

        save_statistics(stats_file, stats, args.verbose)

        success = True
        log_section("✓ Transcription completed successfully!")

        # Log transcript location
        if markdown_file:
            logger.info(f"Transcript file: file://{markdown_file.absolute()}")

        # Log analysis location if performed
        if analysis_performed and analysis_file:
            if args.save_to_obsidian:
                logger.info(f"Analysis saved to Obsidian vault: file://{analysis_file.absolute()}")
            else:
                logger.info(f"Analysis file: file://{analysis_file.absolute()}")

        logger.info(f"Total duration: {format_duration(duration)}")
        logger.info(f"Transcription duration: {format_duration(transcription_duration)}")

        if estimated_time:
            diff = transcription_duration - estimated_time
            diff_abs = abs(diff)
            percentage_diff = (diff_abs / estimated_time) * 100
            direction = "slower" if diff > 0 else "faster"
            logger.info(
                f"Estimation accuracy: {percentage_diff:.1f}% "
                f"({format_duration(diff_abs)}) {direction} than estimated"
            )

        # Save audio file if requested (skip if already auto-saved for diarization)
        if getattr(args, "keep_audio", False) and audio_file and Path(audio_file).exists():
            saved_audio = output_dir / f"{smart_filename}.wav"
            if not saved_audio.exists():
                shutil.copy2(audio_file, saved_audio)
            logger.info(f"Audio saved to: file://{saved_audio.absolute()}")

        return True

    except KeyboardInterrupt:
        logger.info("\n\n✗ Process interrupted by user.")
        return False

    except DiarizationError as e:
        log_error(str(e))
        # Check if whisper JSON was saved - if so, hint about retry
        whisper_json = output_dir / f"{smart_filename}.whisper.json" if smart_filename else None
        if whisper_json and whisper_json.exists():
            # Save a non-diarized markdown from whisper JSON
            fallback_md = output_dir / f"{smart_filename}.md"
            if not fallback_md.exists() and video_info:
                try:
                    with open(whisper_json, encoding="utf-8") as f:
                        whisper_data = json.load(f)
                    plain_text = "\n".join(
                        seg["text"].strip()
                        for seg in whisper_data.get("transcription", [])
                        if seg["text"].strip()
                    )
                    from .markdown import create_markdown_file as _create_md

                    temp_txt = output_dir / f"_fallback_{uuid.uuid4().hex[:8]}.txt"
                    temp_txt.write_text(plain_text, encoding="utf-8")
                    _create_md(fallback_md, temp_txt, video_info, {}, args.verbose)
                    temp_txt.unlink(missing_ok=True)
                    logger.info(f"\nTranscript saved: file://{fallback_md.absolute()}")
                except Exception:
                    pass  # Best-effort fallback

            logger.info("Retry diarization without re-transcribing:")
            logger.info(f"  pidcast --diarize-existing {fallback_md}")
        if args.verbose:
            traceback.print_exc()
        return False

    except (
        ConfigurationError,
        DownloadError,
        TranscriptionError,
        FileProcessingError,
        AnalysisError,
        ApplePodcastsResolutionError,
    ) as e:
        log_error(str(e))
        if args.verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        log_error(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return False

    finally:
        # Clean up temporary audio files
        if audio_file:
            if is_local_file:
                audio_path = Path(audio_file)
                if audio_path.exists() and "_16k.wav" in audio_file:
                    try:
                        audio_path.unlink()
                        if args.verbose:
                            logger.debug(f"Cleaned up temporary file: {audio_file}")
                    except Exception as e:
                        if args.verbose:
                            logger.warning(f"Could not remove {audio_file}: {e}")
            else:
                cleanup_temp_files(audio_file, args.verbose)

        # Save failure statistics if needed
        if not success and video_info:
            end_time = time.time()
            duration = end_time - start_time
            stats = TranscriptionStats(
                run_uid=run_uid,
                run_timestamp=run_timestamp,
                video_title=video_info.title,
                smart_filename="",
                video_url=input_source,
                run_duration=duration,
                transcription_duration=0,
                audio_duration=0,
                success=False,
                is_local_file=is_local_file,
            )
            save_statistics(stats_file, stats, False)
