"""Command-line interface for pidcast."""

import argparse
import datetime
import json
import logging
import os
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from .analysis import (
    analyze_transcript_with_llm,
    load_analysis_prompts,
    render_analysis_to_terminal,
)
from .config import (
    DEFAULT_PROMPTS_FILE,
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
    extract_youtube_video_id,
    find_existing_transcription,
    format_duration,
    get_unique_filename,
    is_interactive,
    log_error,
    log_section,
    log_success,
    save_json_file,
    setup_logging,
    validate_input_source,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DUPLICATE DETECTION
# ============================================================================


class DuplicateAction(Enum):
    """User's choice when duplicate transcription is detected."""

    RE_TRANSCRIBE = "retranscribe"
    ANALYZE_EXISTING = "analyze"
    FORCE_CONTINUE = "force"
    CANCEL = "cancel"


def prompt_duplicate_detected(
    prev: "PreviousTranscription",
    verbose: bool = False,
) -> DuplicateAction:
    """Display duplicate detection UI and get user's choice.

    Args:
        prev: Information about the previous transcription
        verbose: Enable verbose output

    Returns:
        User's selected action
    """
    from .config import PreviousTranscription

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        return _prompt_duplicate_basic(prev)

    console = Console()

    # Build info panel
    console.print()
    console.print(
        Panel(
            "[yellow bold]Duplicate Detected![/yellow bold]\n\n"
            "This video was previously transcribed.",
            border_style="yellow",
        )
    )

    # Show previous transcription details
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", width=20)
    table.add_column(style="white")

    table.add_row("Title", prev.video_title)
    table.add_row("Transcribed", prev.formatted_date)
    table.add_row("File", prev.smart_filename)

    # Check if transcript file still exists
    transcript_exists = prev.transcript_path.exists()
    if transcript_exists:
        table.add_row("Status", "[green]Transcript file exists[/green]")
    else:
        table.add_row("Status", "[red]Transcript file not found[/red]")

    if prev.analysis_performed:
        table.add_row("Previous Analysis", prev.analysis_type or "Yes")

    console.print(table)
    console.print()

    # Build options
    console.print("[bold]What would you like to do?[/bold]")
    console.print()
    console.print("  [cyan]1[/cyan] - Re-transcribe the video")

    if transcript_exists:
        console.print("  [cyan]2[/cyan] - Analyze existing transcript (skip re-transcription)")
    else:
        console.print("  [dim]2 - Analyze existing (unavailable - file not found)[/dim]")

    console.print("  [cyan]3[/cyan] - Continue anyway (force re-transcription)")
    console.print("  [cyan]4[/cyan] - Cancel")
    console.print()

    # Get choice
    valid_choices = ["1", "2", "3", "4"] if transcript_exists else ["1", "3", "4"]
    while True:
        choice = Prompt.ask(
            "Enter choice",
            choices=valid_choices,
            default="4",
        )

        if choice == "1":
            return DuplicateAction.RE_TRANSCRIBE
        elif choice == "2" and transcript_exists:
            return DuplicateAction.ANALYZE_EXISTING
        elif choice == "3":
            return DuplicateAction.FORCE_CONTINUE
        elif choice == "4":
            return DuplicateAction.CANCEL


def _prompt_duplicate_basic(prev: "PreviousTranscription") -> DuplicateAction:
    """Basic fallback prompt without rich."""
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTED!")
    print("=" * 60)
    print(f"Title: {prev.video_title}")
    print(f"Previously transcribed: {prev.formatted_date}")
    print(f"File: {prev.smart_filename}")
    print()
    print("Options:")
    print("  1 - Re-transcribe the video")
    print("  2 - Analyze existing transcript")
    print("  3 - Continue anyway")
    print("  4 - Cancel")
    print()

    transcript_exists = prev.transcript_path.exists()
    valid_choices = {"1", "3", "4"}
    if transcript_exists:
        valid_choices.add("2")
    else:
        print("  (Option 2 unavailable - transcript file not found)")

    while True:
        choice = input("Enter choice [4]: ").strip() or "4"
        if choice in valid_choices:
            break
        print(f"Invalid choice. Please enter: {', '.join(sorted(valid_choices))}")

    mapping = {
        "1": DuplicateAction.RE_TRANSCRIBE,
        "2": DuplicateAction.ANALYZE_EXISTING,
        "3": DuplicateAction.FORCE_CONTINUE,
        "4": DuplicateAction.CANCEL,
    }
    return mapping[choice]


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
  %(prog)s --analyze_existing transcript.md --analysis_type summary
        """,
    )

    # Input source group (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "input_source", nargs="?", help="YouTube video URL or path to local audio file"
    )
    input_group.add_argument(
        "--analyze_existing",
        help="Path to existing transcript file (.md or .txt) to analyze without re-transcribing",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Output directory for Markdown files (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--save_to_obsidian",
        action="store_true",
        help=f"Save analysis files to Obsidian vault (transcripts still saved to output_dir) at: {OBSIDIAN_PATH}",
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
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip duplicate detection and force transcription",
    )

    # LLM Analysis arguments
    analysis_group = parser.add_argument_group("LLM Analysis Options")
    analysis_group.add_argument(
        "--no-analyze",
        action="store_true",
        dest="no_analyze",
        help="Skip LLM analysis (default: analyze is enabled)",
    )
    analysis_group.add_argument(
        "--analyze",
        action="store_true",
        help="(Deprecated) Enable LLM analysis - this is now the default behavior",
    )
    analysis_group.add_argument(
        "--analysis_type",
        default="executive_summary",
        help="Analysis type/prompt template to use (default: executive_summary)",
    )
    analysis_group.add_argument(
        "--prompts_file",
        default=None,
        help=f"Path to prompts YAML file (default: {DEFAULT_PROMPTS_FILE})",
    )
    # Keep old flag name for backward compatibility
    analysis_group.add_argument(
        "--analysis_prompts_file",
        default=None,
        dest="prompts_file_legacy",
        help="(Deprecated) Use --prompts_file instead",
    )
    analysis_group.add_argument(
        "--groq_api_key",
        default=None,
        help="Groq API key (default: GROQ_API_KEY environment variable)",
    )
    analysis_group.add_argument(
        "--groq_model",
        default=None,
        help="Groq model to use for analysis (default: from config/models.yaml)",
    )
    analysis_group.add_argument(
        "--skip_analysis_on_error",
        action="store_true",
        help="Continue if analysis fails instead of aborting",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save",
        action="store_true",
        help="Save analysis output to file (default: terminal only)",
    )

    return parser.parse_args()


# ============================================================================
# MAIN PIPELINE
# ============================================================================


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


def _render_analysis_to_terminal_direct(
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
        print()  # Blank line
        render_analysis_to_terminal(analysis_file, verbose=args.verbose)
    else:
        # Terminal-only output - render directly without saving
        logger.info(f"\n✓ Analysis completed in {format_duration(analysis_duration)}")
        print()  # Blank line
        _render_analysis_to_terminal_direct(result, video_info, args.verbose)

    if result.estimated_cost:
        logger.info(f"\n✓ Cost: ${result.estimated_cost:.4f}")

    logger.info(f"✓ Generated with: {result.model}")

    return analysis_file, analysis_duration, metadata


def _run_analyze_existing_mode(
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

    This is extracted from the analyze_existing mode in main() to allow reuse
    when the user selects "Analyze existing" from the duplicate detection prompt.

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

        # Store the original transcript filename, not the analysis filename
        # This ensures duplicate detection can find the raw transcript on subsequent runs
        transcript_filename = markdown_file.name if markdown_file else ""

        stats = TranscriptionStats(
            run_uid=run_uid,
            run_timestamp=run_timestamp,
            video_title=video_info.title,
            smart_filename=transcript_filename,
            video_url=video_info.webpage_url or str(transcript_path),
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


def main() -> None:
    """Main entry point for pidcast CLI."""
    args = parse_arguments()

    # Set up logging
    setup_logging(args.verbose)

    # Set defaults for paths
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TRANSCRIPTS_DIR
    stats_file = Path(args.stats_file) if args.stats_file else DEFAULT_STATS_FILE

    # Determine where analysis files should go
    # Transcripts always go to output_dir, but analysis can go to Obsidian
    analysis_output_dir = Path(OBSIDIAN_PATH) if args.save_to_obsidian else output_dir

    if args.save_to_obsidian and args.verbose:
        logger.info(f"Analysis will be saved to Obsidian vault: {analysis_output_dir}")
        logger.info(f"Transcripts will be saved to: {output_dir}")

    # Initialize tracking variables
    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()
    audio_file: str | None = None
    success = False
    is_local_file = False
    video_info: VideoInfo | None = None

    # Handle analyze-existing mode
    if args.analyze_existing:
        # --no-analyze doesn't make sense with --analyze_existing
        if args.no_analyze:
            log_error(
                "--no-analyze cannot be used with --analyze_existing. "
                "The purpose of --analyze_existing is to analyze a transcript."
            )
            return

        _run_analyze_existing_mode(
            args.analyze_existing, args, output_dir, analysis_output_dir, stats_file, run_uid, run_timestamp, start_time
        )
        return

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

        # Check for duplicate transcription (unless --force is used)
        if not args.force:
            prev_transcription = find_existing_transcription(
                stats_file, args.input_source, output_dir
            )

            if prev_transcription:
                # Handle non-interactive mode
                if not is_interactive():
                    log_error(
                        f"Duplicate detected: '{prev_transcription.video_title}' "
                        f"was already transcribed on {prev_transcription.formatted_date}. "
                        "Use --force to proceed anyway."
                    )
                    return

                # Interactive mode: prompt user for action
                action = prompt_duplicate_detected(prev_transcription, args.verbose)

                if action == DuplicateAction.CANCEL:
                    logger.info("Operation cancelled.")
                    return

                elif action == DuplicateAction.ANALYZE_EXISTING:
                    # Redirect to analyze-existing mode
                    _run_analyze_existing_mode(
                        prev_transcription.transcript_path,
                        args,
                        output_dir,
                        analysis_output_dir,
                        stats_file,
                        run_uid,
                        run_timestamp,
                        start_time,
                    )
                    return

                elif action == DuplicateAction.RE_TRANSCRIBE:
                    # Continue with normal transcription
                    logger.info("Re-transcribing video...")

                elif action == DuplicateAction.FORCE_CONTINUE:
                    # Continue with normal transcription
                    logger.info("Continuing with transcription...")

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
        # Always create transcript file in output_dir
        transcript_file = f"{temp_whisper_output}.txt"

        logger.info("\nCreating Markdown file...")
        markdown_file = get_unique_filename(output_dir, smart_filename, ".md")

        try:
            front_matter = json.loads(args.front_matter)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in front_matter, using empty dict: {e}")
            front_matter = {}

        if not create_markdown_file(
            markdown_file, transcript_file, video_info, front_matter, args.verbose
        ):
            raise FileProcessingError("Failed to create Markdown file")

        # LLM Analysis (default: enabled unless --no-analyze)
        analysis_file: Path | None = None
        analysis_duration = 0.0
        analysis_performed = False
        analysis_metadata: dict[str, Any] = {}

        # Analyze by default unless --no-analyze is set
        should_analyze = not args.no_analyze
        # Save to file if --save or --save_to_obsidian is set
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

        # Use transcript filename for stats (always created now)
        filename_for_stats = markdown_file.name if markdown_file else ""

        stats = TranscriptionStats(
            run_uid=run_uid,
            run_timestamp=run_timestamp,
            video_title=video_info.title,
            smart_filename=filename_for_stats,
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
