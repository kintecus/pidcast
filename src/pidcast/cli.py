"""Command-line interface for pidcast."""

import argparse
import datetime
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
    DEFAULT_ANALYSIS_PROMPTS_FILE,
    DEFAULT_GROQ_MODEL,
    DEFAULT_STATS_FILE,
    DEFAULT_TRANSCRIPTS_DIR,
    OBSIDIAN_PATH,
    WHISPER_MODEL,
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
    save_json_file,
    setup_logging,
    validate_input_source,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STATISTICS
# ============================================================================


def save_statistics(stats_file: Path, stats: TranscriptionStats, verbose: bool = False) -> bool:
    """Save transcription statistics to a JSON file.

    Args:
        stats_file: Path to stats file
        stats: Statistics to save
        verbose: Enable verbose output

    Returns:
        True if successful
    """
    from .utils import load_json_file

    existing_stats = load_json_file(stats_file, default=[])
    existing_stats.append(stats.to_dict())
    return save_json_file(stats_file, existing_stats, verbose=verbose)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Automate audio transcription with Whisper (YouTube URL or local file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"
  %(prog)s "/path/to/audio/file.mp3"
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir ./transcripts --verbose
  %(prog)s "VIDEO_URL" --analyze --analysis_type key_points
        """,
    )

    parser.add_argument("input_source", help="YouTube video URL or path to local audio file")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Output directory for Markdown files (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--save_to_obsidian",
        action="store_true",
        help=f"Save to Obsidian vault at: {OBSIDIAN_PATH}",
    )
    parser.add_argument("--whisper_model", default=WHISPER_MODEL, help="Path to Whisper model file")
    parser.add_argument(
        "--output_format",
        default="otxt",
        help="Whisper output format (txt, vtt, srt, json). Prefix with 'o' for original filename.",
    )
    parser.add_argument(
        "--front_matter", default="{}", help="JSON string for Markdown front matter"
    )
    parser.add_argument(
        "--stats_file",
        default=None,
        help=f"File to store statistics (default: {DEFAULT_STATS_FILE})",
    )
    parser.add_argument(
        "--keep_transcript",
        action="store_true",
        help="Keep the .txt transcript file alongside the .md file",
    )
    parser.add_argument(
        "--po_token",
        default=None,
        help="PO Token for bypassing YouTube restrictions (format: 'client.type+TOKEN')",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # LLM Analysis arguments
    analysis_group = parser.add_argument_group("LLM Analysis Options")
    analysis_group.add_argument(
        "--analyze", action="store_true", help="Enable LLM-based transcript analysis"
    )
    analysis_group.add_argument(
        "--analysis_type",
        default="summary",
        help="Analysis type/prompt template to use (default: summary)",
    )
    analysis_group.add_argument(
        "--analysis_prompts_file",
        default=None,
        help=f"Path to analysis prompts JSON file (default: {DEFAULT_ANALYSIS_PROMPTS_FILE})",
    )
    analysis_group.add_argument(
        "--groq_api_key",
        default=None,
        help="Groq API key (default: GROQ_API_KEY environment variable)",
    )
    analysis_group.add_argument(
        "--groq_model",
        default=DEFAULT_GROQ_MODEL,
        help=f"Groq model to use for analysis (default: {DEFAULT_GROQ_MODEL})",
    )
    analysis_group.add_argument(
        "--skip_analysis_on_error",
        action="store_true",
        help="Continue if analysis fails instead of aborting",
    )

    return parser.parse_args()


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_analysis(
    markdown_file: Path,
    video_info: VideoInfo,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path | None, float, dict[str, Any]]:
    """Run LLM analysis on transcript.

    Args:
        markdown_file: Path to transcript markdown file
        video_info: Video metadata
        output_dir: Output directory
        args: Parsed arguments

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

    # Load prompts
    prompts_file = Path(args.analysis_prompts_file or DEFAULT_ANALYSIS_PROMPTS_FILE)
    prompts_config = load_analysis_prompts(prompts_file, args.verbose)
    if not prompts_config:
        raise AnalysisError(f"Failed to load analysis prompts from: {prompts_file}")

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

    # Create analysis markdown
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
        },
        markdown_file,
        video_info,
        output_dir,
        args.verbose,
    )

    if not analysis_file:
        raise AnalysisError("Failed to create analysis file")

    analysis_duration = time.time() - analysis_start
    metadata = {
        "analysis_type": result.analysis_type,
        "analysis_name": result.analysis_name,
        "model": result.model,
        "tokens_total": result.tokens_total,
        "estimated_cost": result.estimated_cost,
        "truncated": result.truncated,
    }

    logger.info(f"\n✓ Analysis completed in {format_duration(analysis_duration)}")
    logger.info(f"✓ Analysis file: file://{analysis_file.absolute()}")

    # Render analysis to terminal
    print()  # Blank line
    render_analysis_to_terminal(analysis_file, verbose=args.verbose)

    if result.estimated_cost:
        logger.info(f"\n✓ Cost: ${result.estimated_cost:.4f}")

    return analysis_file, analysis_duration, metadata


def main() -> None:
    """Main entry point for pidcast CLI."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.verbose)

    # Set defaults for paths
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TRANSCRIPTS_DIR
    stats_file = Path(args.stats_file) if args.stats_file else DEFAULT_STATS_FILE

    if args.save_to_obsidian:
        output_dir = Path(OBSIDIAN_PATH)
        if args.verbose:
            logger.info(f"Saving to Obsidian vault: {output_dir}")

    # Initialize tracking variables
    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()
    audio_file: str | None = None
    success = False
    is_local_file = False
    video_info: VideoInfo | None = None

    if args.verbose:
        log_section("Transcription Tool")
        logger.info(f"Input: {args.input_source}")
        logger.info(f"Run ID: {run_uid}")
        logger.info(f"Timestamp: {run_timestamp}")

    try:
        # Validate input
        source, is_local_file = validate_input_source(args.input_source)

        if args.verbose:
            logger.info(f"Type: {'Local File' if is_local_file else 'YouTube URL'}")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

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

        # Create Markdown file
        logger.info("\nCreating Markdown file...")
        markdown_file = get_unique_filename(output_dir, smart_filename, ".md")
        transcript_file = f"{temp_whisper_output}.txt"

        try:
            front_matter = json.loads(args.front_matter)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in front_matter, using empty dict: {e}")
            front_matter = {}

        if not create_markdown_file(
            markdown_file, transcript_file, video_info, front_matter, args.verbose
        ):
            raise FileProcessingError("Failed to create Markdown file")

        # LLM Analysis
        analysis_file: Path | None = None
        analysis_duration = 0.0
        analysis_performed = False
        analysis_metadata: dict[str, Any] = {}

        if args.analyze:
            try:
                analysis_file, analysis_duration, analysis_metadata = run_analysis(
                    markdown_file, video_info, output_dir, args
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

        stats = TranscriptionStats(
            run_uid=run_uid,
            run_timestamp=run_timestamp,
            video_title=video_info.title,
            smart_filename=markdown_file.name,
            video_url=args.input_source,
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
        logger.info(f"Markdown file: file://{markdown_file.absolute()}")
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

        if args.save_to_obsidian:
            log_success("Saved to Obsidian vault")

    except KeyboardInterrupt:
        logger.info("\n\n✗ Process interrupted by user.")

    except (DownloadError, TranscriptionError, FileProcessingError, AnalysisError) as e:
        log_error(str(e))
        if args.verbose:
            traceback.print_exc()

    except Exception as e:
        log_error(f"An unexpected error occurred: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()

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
                video_title=video_info.title if video_info else "Unknown",
                smart_filename="",
                video_url=args.input_source,
                run_duration=duration,
                transcription_duration=0,
                audio_duration=0,
                success=False,
                is_local_file=is_local_file,
            )
            save_statistics(stats_file, stats, False)


if __name__ == "__main__":
    main()
