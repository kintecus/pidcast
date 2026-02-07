"""Core workflow for media processing."""

import argparse
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .analysis import (
    analyze_transcript_with_llm,
    load_analysis_prompts,
    render_analysis_to_terminal,
)
from .config import (
    TranscriptionStats,
    VideoInfo,
)
from .download import download_audio
from .exceptions import AnalysisError, DownloadError, FileProcessingError, TranscriptionError
from .markdown import create_analysis_markdown_file, create_markdown_file
from .transcription import (
    estimate_transcription_time,
    process_local_file,
    run_whisper_transcription,
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
) -> tuple[Path | None, float, dict[str, Any]]:
    """Run LLM analysis on transcript.

    Args:
        markdown_file: Path to transcript markdown file (None if using transcript_text)
        video_info: Video metadata
        output_dir: Output directory
        args: Parsed arguments
        transcript_text: Optional pre-loaded transcript text
        save_to_file: Whether to save the analysis to a file

    Returns:
        Tuple of (analysis_file, duration, metadata)
    """
    log_section("Starting LLM Analysis")

    # Get API key
    groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise AnalysisError(
            "Groq API key not found. Set GROQ_API_KEY environment variable or use --groq_api_key"
        )

    # Load prompts (use new flag or legacy flag for backward compatibility)
    prompts_file = args.prompts_file or args.prompts_file_legacy
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

    # Analyze
    analysis_start = time.time()
    result = analyze_transcript_with_llm(
        transcript_text,
        video_info,
        args.analysis_type,
        prompts_config,
        groq_api_key,
        args.groq_model,
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
    success = False

    try:
        # Validate input
        source, is_local_file = validate_input_source(input_source)

        if args.verbose:
            logger.info(f"Type: {'Local File' if is_local_file else 'YouTube URL'}")

        # Ensure output directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_output_dir.mkdir(parents=True, exist_ok=True)

        # Process input source
        if is_local_file:
            logger.info("Processing local file...")
            audio_file, video_info = process_local_file(source, output_dir, args.verbose)
            logger.info("\n✓ Local file processed successfully!")
        else:
            logger.info("Downloading audio from YouTube...")
            audio_file, video_info = download_audio(
                source, "temp_audio.%(ext)s", args.verbose, args.po_token
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

        # Estimate transcription time
        audio_duration = video_info.duration
        estimated_time = estimate_transcription_time(stats_file, audio_duration)

        # Run Whisper transcription
        logger.info("\nTranscribing audio with Whisper...")
        if estimated_time:
            logger.info(
                f"Estimated transcription time: ~{format_duration(estimated_time)} (based on historical data)"
            )
        elif audio_duration > 0:
            logger.info(f"Audio duration: {format_duration(audio_duration)}")

        transcription_start = time.time()
        temp_whisper_output = output_dir / f"temp_transcript_{uuid.uuid4().hex[:8]}"
        output_format = args.output_format.replace("o", "")

        run_whisper_transcription(
            audio_file,
            args.whisper_model,
            output_format,
            str(temp_whisper_output),
            args.verbose,
            estimated_duration=estimated_time,
        )

        transcription_duration = time.time() - transcription_start

        # Create Markdown file (transcript)
        transcript_file = f"{temp_whisper_output}.txt"

        logger.info("\nCreating Markdown file...")
        markdown_file = get_unique_filename(output_dir, smart_filename, ".md")

        try:
            front_matter = json.loads(args.front_matter)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in front_matter, using empty dict: {e}")
            front_matter = {}

        if not create_markdown_file(
            markdown_file,
            transcript_file,
            video_info,
            front_matter,
            args.verbose,
        ):
            raise FileProcessingError("Failed to create Markdown file")

        # LLM Analysis (default: enabled unless --no-analyze)
        analysis_file: Path | None = None
        analysis_duration = 0.0
        analysis_performed = False
        analysis_metadata: dict[str, Any] = {}

        should_analyze = not args.no_analyze
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

        return True

    except KeyboardInterrupt:
        logger.info("\n\n✗ Process interrupted by user.")
        return False

    except (DownloadError, TranscriptionError, FileProcessingError, AnalysisError) as e:
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
