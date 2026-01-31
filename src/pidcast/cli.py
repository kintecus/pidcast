"Command-line interface for pidcast."

import argparse
import datetime
import logging
import os
import time
import traceback
import uuid
from enum import Enum
from pathlib import Path

from .config import (
    DEFAULT_PROMPTS_FILE,
    DEFAULT_STATS_FILE,
    DEFAULT_TRANSCRIPTS_DIR,
    OBSIDIAN_PATH,
    WHISPER_MODEL,
)
from .exceptions import DuplicateShowError, FeedFetchError, FeedParseError, ShowNotFoundError
from .utils import (
    find_existing_transcription,
    is_interactive,
    log_error,
    log_section,
    setup_logging,
)
from .workflow import (
    process_input_source,
    run_analyze_existing_mode,
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
    prev,
    verbose: bool = False,
) -> DuplicateAction:
    """Display duplicate detection UI and get user's choice.

    Args:
        prev: Information about the previous transcription (PreviousTranscription)
        verbose: Enable verbose output

    Returns:
        User's selected action
    """

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
            "[yellow bold]Duplicate Detected![/yellow bold]\n\n" \
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


def _prompt_duplicate_basic(prev) -> DuplicateAction:
    """Basic fallback prompt without rich.

    Args:
        prev: Information about the previous transcription (PreviousTranscription)
    """
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
# ARGUMENT PARSING
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    import sys

    # Check if first argument is 'lib' to determine which parser to use
    is_lib_command = len(sys.argv) > 1 and sys.argv[1] == "lib"

    parser = argparse.ArgumentParser(
        description="Automate audio transcription with Whisper (YouTube URL or local file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common Workflows:
  # Quick transcription with defaults
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"

  # Save to Obsidian vault
  %(prog)s "VIDEO_URL" -o

  # Custom analysis type (supports fuzzy matching)
  %(prog)s "VIDEO_URL" -o -a exec          # Matches 'executive_summary'
  %(prog)s "VIDEO_URL" -o -a detailed -v   # Verbose output

  # Choose specific model
  %(prog)s "VIDEO_URL" -m llama33           # Matches 'llama-3.3-70b-versatile'

  # Analyze existing transcript
  %(prog)s --analyze_existing transcript.md -a summary

Discovery:
  # List available analysis types
  %(prog)s -L

  # List available models
  %(prog)s -M

Library Management:
  # Add podcast to library
  %(prog)s lib add "https://feeds.example.com/podcast.xml"

  # Process latest episode from library
  %(prog)s lib process "Lex Fridman" --latest

  # List all shows
  %(prog)s lib list

  # Sync library and process new episodes
  %(prog)s lib sync

Short Flags:
  -o  --save_to_obsidian    Save to Obsidian vault
  -a  --analysis_type       Analysis type (fuzzy matching enabled)
  -m  --groq_model          Model name (fuzzy matching enabled)
  -f  --force               Force re-transcription
  -v  --verbose             Verbose output
  -L  --list-analyses       List available analysis types
  -M  --list-models         List available models
        """,
    )

    # Only create subparsers if 'lib' command is used
    if is_lib_command:
        subparsers = parser.add_subparsers(dest="mode", help="Command mode", required=False)

        # Library subcommand with nested subparsers
        lib_parser = subparsers.add_parser('lib', help='Podcast library management')
        lib_subparsers = lib_parser.add_subparsers(dest='lib_command', help="Library management commands", required=True)

        # Add command
        add_parser = lib_subparsers.add_parser("add", help="Add podcast to library")
        add_parser.add_argument("feed_url", help="RSS feed URL")
        add_parser.add_argument(
            "--preview", action="store_true", help="Preview episodes before adding"
        )
        add_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

        # Process command (NEW)
        process_parser = lib_subparsers.add_parser("process", help="Process an episode from a library show")
        process_parser.add_argument("show_query", help="Show ID or partial name")
        process_parser.add_argument("--latest", action="store_true", help="Process the latest episode")
        process_parser.add_argument("--match", help="Process episode matching this title string")
        # Reuse common flags for processing
        process_parser.add_argument("--output_dir", help="Output directory")
        process_parser.add_argument("--save_to_obsidian", action="store_true", help="Save to Obsidian")
        process_parser.add_argument("--whisper_model", help="Whisper model path")
        process_parser.add_argument("--groq_api_key", help="Groq API key")
        process_parser.add_argument("--analysis_type", default="executive_summary", help="Analysis type")
        process_parser.add_argument("--prompts_file", help="Prompts file path")
        process_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        process_parser.add_argument("--force", action="store_true", help="Force reprocessing")
        process_parser.add_argument("--output_format", default="otxt", help="Output format")
        process_parser.add_argument("--keep_transcript", action="store_true", help="Keep transcript")
        process_parser.add_argument("--po_token", help="PO Token")
        process_parser.add_argument("--front_matter", default="{}", help="Front matter JSON")
        process_parser.add_argument("--save", action="store_true", help="Save output")
        process_parser.add_argument("--no-analyze", action="store_true", help="Skip analysis")
        process_parser.add_argument("--skip_analysis_on_error", action="store_true", help="Skip analysis on error")
        process_parser.add_argument("--stats_file", help="Stats file path")
        process_parser.add_argument("--groq_model", help="Groq model")

        # List command
        list_parser = lib_subparsers.add_parser("list", help="List all shows in library")
        list_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

        # Show command
        show_parser = lib_subparsers.add_parser("show", help="Show details for a podcast")
        show_parser.add_argument("show_id", type=int, help="Show ID")
        show_parser.add_argument(
            "--episodes", type=int, default=5, help="Number of recent episodes to show (default: 5)"
        )
        show_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

        # Remove command
        remove_parser = lib_subparsers.add_parser("remove", help="Remove podcast from library")
        remove_parser.add_argument("show_id", type=int, help="Show ID")
        remove_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

        # Sync command
        sync_parser = lib_subparsers.add_parser("sync", help="Sync library shows and process new episodes")
        sync_parser.add_argument("--show", type=int, metavar="ID", help="Sync only specific show by ID")
        sync_parser.add_argument("--dry-run", action="store_true", help="Preview only")
        sync_parser.add_argument("--force", action="store_true", help="Reprocess episodes")
        sync_parser.add_argument("--backfill", type=int, metavar="N", help="Override backfill limit")
        sync_parser.add_argument("--output_dir", help="Output directory")
        sync_parser.add_argument("--whisper_model", help="Whisper model path")
        sync_parser.add_argument("--groq_api_key", help="Groq API key")
        sync_parser.add_argument("--analysis_type", default="executive_summary", help="Analysis type")
        sync_parser.add_argument("--prompts_file", help="Prompts file path")
        sync_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        sync_parser.add_argument("--no-digest", action="store_true", help="Skip digest generation")

        # Digest command
        digest_parser = lib_subparsers.add_parser("digest", help="Generate podcast digest")
        digest_parser.add_argument("--date", help="Specific date (YYYY-MM-DD)")
        digest_parser.add_argument("--range", help="Date range (e.g., 7d)")
        digest_parser.add_argument("--output_dir", help="Output directory")
        digest_parser.add_argument("--groq_api_key", help="Groq API key")
        digest_parser.add_argument("--prompts_file", help="Prompts file path")
        digest_parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Input source group (mutually exclusive) - for original workflow
    input_group = parser.add_mutually_exclusive_group(required=False)
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
        "-o",
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
        "-a",
        "--analysis_type",
        default="executive_summary",
        help="Analysis type/prompt template to use (default: executive_summary). Use -L to list available types.",
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
        "-m",
        "--groq_model",
        default=None,
        help="Groq model to use for analysis (default: from config/models.yaml). Use -M to list available models.",
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

    # Discoverability options
    discovery_group = parser.add_argument_group("Discovery Options")
    discovery_group.add_argument(
        "-L",
        "--list-analyses",
        action="store_true",
        dest="list_analyses",
        help="List available analysis types and exit",
    )
    discovery_group.add_argument(
        "-M",
        "--list-models",
        action="store_true",
        dest="list_models",
        help="List available Groq models and exit",
    )

    return parser.parse_args()


# ============================================================================
# LIBRARY COMMAND HANDLERS
# ============================================================================


def cmd_process(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib process' command."""
    import uuid

    from .config import DEFAULT_STATS_FILE, DEFAULT_TRANSCRIPTS_DIR, OBSIDIAN_PATH, VideoInfo
    from .library import LibraryManager, Show

    library = LibraryManager()

    # Find show
    show_query = args.show_query
    show: Show | None = None

    # Try as ID
    try:
        show_id = int(show_query)
        show = library.get_show(show_id)
    except ValueError:
        pass

    # Try as name (case insensitive partial match)
    if not show:
        matches = [s for s in library.list_shows() if show_query.lower() in s.title.lower()]
        if len(matches) == 1:
            show = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous show name '{show_query}'. Matches:")
            for s in matches:
                print(f"  {s.id}: {s.title}")
            return

    if not show:
        log_error(f"Show '{show_query}' not found.")
        return

    # Fetch episodes
    episodes = library.get_episodes(show.id, limit=50 if args.match else 20, verbose=args.verbose)

    selected_episode = None

    if args.latest:
        selected_episode = episodes[0] if episodes else None
    elif args.match:
        # fuzzy match title
        query = args.match.lower()
        for ep in episodes:
            if query in ep.title.lower():
                selected_episode = ep
                break
        if not selected_episode:
            log_error(f"No episode found matching '{args.match}'")
            return
    else:
        # Interactive selection
        try:
            from rich.console import Console
            from rich.prompt import Prompt
            from rich.table import Table
            console = Console()

            table = Table(show_header=True)
            table.add_column("#", style="cyan", width=4)
            table.add_column("Date", style="yellow", width=12)
            table.add_column("Title", style="white")

            # Show top 10
            display_episodes = episodes[:10]
            for i, ep in enumerate(display_episodes, 1):
                table.add_row(str(i), ep.pub_date.strftime("%Y-%m-%d"), ep.title)

            console.print(table)

            # Ask for selection
            choices = [str(i) for i in range(1, len(display_episodes) + 1)]
            choice = Prompt.ask("Select episode", choices=choices)
            selected_episode = display_episodes[int(choice)-1]
        except ImportError:
            # Fallback
            print("Rich not installed, and no selection flags used. Defaulting to latest.")
            selected_episode = episodes[0] if episodes else None

    if not selected_episode:
        log_error("No episodes found.")
        return

    logger.info(f"\nProcessing: {selected_episode.title} ({show.title})")

    # Construct override
    video_info = VideoInfo(
        title=selected_episode.title,
        webpage_url=selected_episode.audio_url,
        channel=show.title,
        upload_date=selected_episode.pub_date.strftime("%Y%m%d"),
        description=selected_episode.description,
        duration=selected_episode.duration or 0
    )

    # Run workflow
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TRANSCRIPTS_DIR
    stats_file = Path(args.stats_file) if args.stats_file else DEFAULT_STATS_FILE
    analysis_output_dir = Path(OBSIDIAN_PATH) if args.save_to_obsidian else output_dir

    process_input_source(
        selected_episode.audio_url,
        args,
        output_dir,
        analysis_output_dir,
        stats_file,
        str(uuid.uuid4()),
        datetime.datetime.now().isoformat(),
        time.time(),
        video_info_override=video_info
    )


def cmd_add(args: argparse.Namespace) -> None:
    """Handle 'pidcast add' command."""
    from .config_manager import ConfigManager
    from .library import LibraryManager
    from .rss import RSSParser

    try:
        from rich.console import Console
        from rich.prompt import Confirm
        from rich.table import Table

        has_rich = True
    except ImportError:
        has_rich = False

    # Initialize config and library
    ConfigManager.init_default_config()
    library = LibraryManager()

    try:
        # Preview mode: show episodes before adding
        if args.preview:
            if args.verbose:
                logger.info(f"Fetching feed preview: {args.feed_url}")

            show_meta, episodes = RSSParser.parse_feed(args.feed_url, verbose=args.verbose)

            if has_rich:
                console = Console()
                console.print(f"\n[bold cyan]Show:[/bold cyan] {show_meta['title']}")
                console.print(f"[bold cyan]Author:[/bold cyan] {show_meta['author']}")
                console.print("\n[bold]Recent Episodes:[/bold]")

                table = Table(show_header=True)
                table.add_column("#", style="cyan", width=4)
                table.add_column("Date", style="yellow", width=12)
                table.add_column("Title", style="white")

                for i, episode in enumerate(episodes[:5], 1):
                    date_str = episode.pub_date.strftime("%Y-%m-%d")
                    table.add_row(str(i), date_str, episode.title)

                console.print(table)
                console.print(f"\nTotal episodes: {len(episodes)}")

                if not Confirm.ask("\nAdd this show to library?", default=True):
                    logger.info("Cancelled.")
                    return
            else:
                print(f"\nShow: {show_meta['title']}")
                print(f"Author: {show_meta['author']}")
                print("\nRecent Episodes:")
                for i, episode in enumerate(episodes[:5], 1):
                    print(f"  {i}. {episode}")
                print(f"\nTotal episodes: {len(episodes)}")

                confirm = input("\nAdd this show to library? [Y/n]: ").strip().lower()
                if confirm and confirm not in ["y", "yes"]:
                    logger.info("Cancelled.")
                    return

        # Add show to library
        show = library.add_show(args.feed_url, verbose=args.verbose)

        if has_rich:
            console = Console()
            console.print(f"\n[green]✓[/green] Added: [bold]{show.title}[/bold] (ID: {show.id})")
        else:
            print(f"\n✓ Added: {show.title} (ID: {show.id})")

    except DuplicateShowError as e:
        log_error(str(e))
    except (FeedFetchError, FeedParseError) as e:
        log_error(f"Failed to add show: {e}")
        if args.verbose:
            traceback.print_exc()
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
        if args.verbose:
            traceback.print_exc()


def cmd_list(args: argparse.Namespace) -> None:
    """Handle 'pidcast list' command."""
    from .library import LibraryManager

    try:
        from rich.console import Console
        from rich.table import Table

        has_rich = True
    except ImportError:
        has_rich = False

    # Initialize library
    library = LibraryManager()
    shows = library.list_shows()

    if not shows:
        logger.info("No shows in library. Use 'pidcast add <feed-url>' to add shows.")
        return

    if has_rich:
        console = Console()
        table = Table(show_header=True, title=f"Podcast Library ({len(shows)} shows)")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Title", style="white")
        table.add_column("Author", style="yellow")
        table.add_column("Added", style="dim", width=12)

        for show in shows:
            added_str = show.added_at.strftime("%Y-%m-%d")
            table.add_row(str(show.id), show.title, show.author or "Unknown", added_str)

        console.print(table)
    else:
        print(f"\nPodcast Library ({len(shows)} shows):")
        print("-" * 80)
        for show in shows:
            added_str = show.added_at.strftime("%Y-%m-%d")
            print(f"ID: {show.id}")
            print(f"  Title: {show.title}")
            print(f"  Author: {show.author or 'Unknown'}")
            print(f"  Added: {added_str}")
            print()


def cmd_show(args: argparse.Namespace) -> None:
    """Handle 'pidcast show' command."""
    from .library import LibraryManager

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        has_rich = True
    except ImportError:
        has_rich = False

    # Initialize library
    library = LibraryManager()

    try:
        show = library.get_show(args.show_id)
        if not show:
            log_error(f"Show ID {args.show_id} not found. Run 'pidcast list' to see available shows.")
            return

        # Fetch episodes
        episodes = library.get_episodes(args.show_id, limit=args.episodes, verbose=args.verbose)

        if has_rich:
            console = Console()

            # Show metadata
            info_table = Table(show_header=False, box=None, padding=(0, 2))
            info_table.add_column(style="cyan bold", width=15)
            info_table.add_column(style="white")

            info_table.add_row("ID", str(show.id))
            info_table.add_row("Title", show.title)
            info_table.add_row("Author", show.author or "Unknown")
            info_table.add_row("Feed URL", show.feed_url)
            info_table.add_row("Added", show.added_at.strftime("%Y-%m-%d %H:%M"))
            if show.last_checked:
                info_table.add_row("Last Checked", show.last_checked.strftime("%Y-%m-%d %H:%M"))

            console.print(Panel(info_table, title=f"[bold]{show.title}[/bold]", border_style="cyan"))

            # Episodes
            console.print(f"\n[bold]Recent Episodes (showing {len(episodes)}):[/bold]")
            ep_table = Table(show_header=True)
            ep_table.add_column("#", style="cyan", width=4)
            ep_table.add_column("Date", style="yellow", width=12)
            ep_table.add_column("Duration", style="dim", width=10)
            ep_table.add_column("Title", style="white")

            for i, episode in enumerate(episodes, 1):
                date_str = episode.pub_date.strftime("%Y-%m-%d")
                duration_str = f"{episode.duration // 60}m" if episode.duration else "Unknown"
                ep_table.add_row(str(i), date_str, duration_str, episode.title)

            console.print(ep_table)
        else:
            print(f"\n{'=' * 80}")
            print(f"Show Details (ID: {show.id})")
            print(f"{ '=' * 80}")
            print(f"Title: {show.title}")
            print(f"Author: {show.author or 'Unknown'}")
            print(f"Feed URL: {show.feed_url}")
            print(f"Added: {show.added_at.strftime('%Y-%m-%d %H:%M')}")
            if show.last_checked:
                print(f"Last Checked: {show.last_checked.strftime('%Y-%m-%d %H:%M')}")

            print(f"\nRecent Episodes (showing {len(episodes)}):")
            print("-" * 80)
            for i, episode in enumerate(episodes, 1):
                print(f"{i}. {episode}")
            print()

    except ShowNotFoundError as e:
        log_error(str(e))
    except (FeedFetchError, FeedParseError) as e:
        log_error(f"Failed to fetch episodes: {e}")
        if args.verbose:
            traceback.print_exc()
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
        if args.verbose:
            traceback.print_exc()


def cmd_remove(args: argparse.Namespace) -> None:
    """Handle 'pidcast remove' command."""
    from .library import LibraryManager

    try:
        from rich.console import Console

        has_rich = True
    except ImportError:
        has_rich = False

    # Initialize library
    library = LibraryManager()

    # Get show details before removing
    show = library.get_show(args.show_id)
    if not show:
        log_error(f"Show ID {args.show_id} not found. Run 'pidcast list' to see available shows.")
        return

    # Remove show
    if library.remove_show(args.show_id):
        if has_rich:
            console = Console()
            console.print(f"\n[green]✓[/green] Removed: [bold]{show.title}[/bold] (ID: {show.id})")
        else:
            print(f"\n✓ Removed: {show.title} (ID: {show.id})")
    else:
        log_error(f"Failed to remove show ID {args.show_id}")


def cmd_digest(args: argparse.Namespace) -> None:
    """Handle 'pidcast digest' command."""
    from datetime import datetime, timedelta

    from .config import DEFAULT_PROMPTS_FILE, HISTORY_FILE, get_digest_output_path
    from .digest import DigestFormatter, DigestGenerator
    from .history import ProcessingHistory
    from .library import LibraryManager
    from .summarization import Summarizer

    # Get Groq API key
    groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        log_error(
            "Groq API key not found. Set GROQ_API_KEY environment variable or use --groq_api_key"
        )
        return

    # Initialize components
    library = LibraryManager()
    history = ProcessingHistory(HISTORY_FILE)
    prompts_file = Path(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS_FILE
    summarizer = Summarizer(prompts_file, groq_api_key)

    generator = DigestGenerator(library, history, summarizer)

    # Parse date filters
    date_filter = None
    date_range = None

    if args.date:
        try:
            date_filter = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            log_error(f"Invalid date format: {args.date}. Use YYYY-MM-DD format.")
            return
    elif args.range:
        # Parse range like "7d", "30d"
        try:
            if args.range.endswith("d"):
                days = int(args.range[:-1])
                date_range = timedelta(days=days)
            else:
                log_error(f"Invalid range format: {args.range}. Use format like '7d' or '30d'.")
                return
        except ValueError:
            log_error(f"Invalid range format: {args.range}. Use format like '7d' or '30d'.")
            return
    else:
        # Default: today's episodes
        date_filter = datetime.now()

    # Generate digest
    try:
        logger.info("Generating digest...")
        digest = generator.generate_digest(date_filter, date_range)

        if not digest.episodes:
            logger.info("No episodes found for specified date range.")
            return

        # Display in terminal
        DigestFormatter.format_terminal(digest)

        # Save to file
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TRANSCRIPTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = get_digest_output_path(date_filter or datetime.now())

        markdown = DigestFormatter.format_markdown(digest)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        logger.info(f"\nDigest saved to: {output_path}")

    except Exception as e:
        log_error(f"Failed to generate digest: {e}")
        if args.verbose:
            traceback.print_exc()


def cmd_sync(args: argparse.Namespace) -> None:
    """Handle 'pidcast sync' command."""
    from .config import DEFAULT_BACKFILL_LIMIT, HISTORY_FILE, WHISPER_MODEL
    from .history import ProcessingHistory
    from .library import LibraryManager
    from .sync import SyncEngine

    # Initialize library and history
    library = LibraryManager()
    history = ProcessingHistory(HISTORY_FILE)

    # Get output directory
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TRANSCRIPTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get Whisper model
    whisper_model = args.whisper_model or WHISPER_MODEL
    if not whisper_model:
        log_error(
            "Whisper model not specified. Set WHISPER_MODEL environment variable "
            "or use --whisper_model flag"
        )
        return

    # Get Groq API key (optional)
    groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")

    # Build config
    config = {
        "backfill_limit": DEFAULT_BACKFILL_LIMIT,
    }

    # Create sync engine
    engine = SyncEngine(
        library=library,
        history=history,
        config=config,
        output_dir=output_dir,
        whisper_model=whisper_model,
        groq_api_key=groq_api_key,
        analysis_type=args.analysis_type,
        prompts_file=Path(args.prompts_file) if args.prompts_file else None,
        verbose=args.verbose,
    )

    # Run sync
    try:
        stats = engine.sync(
            show_id=args.show,
            dry_run=args.dry_run,
            force=args.force,
            backfill=args.backfill,
        )

        # Print summary
        logger.info(f"\n{'=' * 60}")
        logger.info("Sync Complete!")
        logger.info(f"{ '=' * 60}")
        logger.info(f"Processed: {stats['processed']} episodes")
        logger.info(f"Succeeded: {stats['succeeded']}")
        logger.info(f"Failed: {stats['failed']}")
        if stats["skipped"] > 0:
            logger.info(f"Skipped: {stats['skipped']} (already processed)")
        logger.info(f"{ '=' * 60}\n")

        # Generate digest unless --no-digest flag is set
        if not args.no_digest and stats["succeeded"] > 0 and groq_api_key:
            from datetime import datetime

            from .config import get_digest_output_path
            from .digest import DigestFormatter, DigestGenerator
            from .summarization import Summarizer

            try:
                logger.info("Generating digest...")
                prompts_file = Path(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS_FILE
                summarizer = Summarizer(prompts_file, groq_api_key)
                digest_generator = DigestGenerator(library, history, summarizer)

                # Generate digest for today's processed episodes
                digest = digest_generator.generate_digest(date_filter=datetime.now())

                if digest.episodes:
                    DigestFormatter.format_terminal(digest)

                    # Save digest
                    output_path = get_digest_output_path(datetime.now())
                    markdown = DigestFormatter.format_markdown(digest)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(markdown)

                    logger.info(f"\n✓ Digest saved to: {output_path}")

            except Exception as e:
                logger.warning(f"Failed to generate digest: {e}")
                if args.verbose:
                    traceback.print_exc()

    except Exception as e:
        log_error(f"Sync failed: {e}")
        if args.verbose:
            traceback.print_exc()


def main() -> None:
    """Main entry point for pidcast CLI."""
    args = parse_arguments()

    # Set up logging
    setup_logging(getattr(args, "verbose", False))

    # Handle discovery/list commands first (they exit immediately)
    if getattr(args, "list_analyses", False):
        from .utils import list_available_analyses
        list_available_analyses()
        return

    if getattr(args, "list_models", False):
        from .utils import list_available_models
        list_available_models()
        return

    # Route to library commands if specified
    if getattr(args, "mode", None) == "lib":
        lib_commands = {
            "add": cmd_add,
            "list": cmd_list,
            "show": cmd_show,
            "remove": cmd_remove,
            "sync": cmd_sync,
            "digest": cmd_digest,
            "process": cmd_process,
        }
        handler = lib_commands.get(args.lib_command)
        if handler:
            handler(args)
        return

    # Validate that we have either input_source or analyze_existing for transcription workflow
    if not args.input_source and not args.analyze_existing:
        log_error("Error: Either provide a URL/file path or use a library command")
        log_error("Run 'pidcast --help' for usage information")
        return

    # Resolve analysis type with fuzzy matching
    if args.analysis_type and args.analysis_type != "executive_summary":
        from .utils import resolve_analysis_type
        try:
            resolved_type = resolve_analysis_type(args.analysis_type, args.prompts_file)
            if resolved_type != args.analysis_type and args.verbose:
                logger.info(f"Matched '{args.analysis_type}' → '{resolved_type}'")
            args.analysis_type = resolved_type
        except ValueError as e:
            log_error(str(e))
            return

    # Resolve model name with fuzzy matching
    if args.groq_model:
        from .utils import resolve_model_name
        try:
            resolved_model = resolve_model_name(args.groq_model)
            if resolved_model != args.groq_model and args.verbose:
                logger.info(f"Matched '{args.groq_model}' → '{resolved_model}'")
            args.groq_model = resolved_model
        except ValueError as e:
            log_error(str(e))
            return

    # Set defaults for paths
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_TRANSCRIPTS_DIR
    stats_file = Path(args.stats_file) if args.stats_file else DEFAULT_STATS_FILE

    # Determine where analysis files should go
    analysis_output_dir = Path(OBSIDIAN_PATH) if args.save_to_obsidian else output_dir

    if args.save_to_obsidian and args.verbose:
        logger.info(f"Analysis will be saved to Obsidian vault: {analysis_output_dir}")
        logger.info(f"Transcripts will be saved to: {output_dir}")

    # Initialize tracking variables
    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()

    # Handle analyze-existing mode
    if args.analyze_existing:
        if args.no_analyze:
            log_error(
                "--no-analyze cannot be used with --analyze_existing. "
                "The purpose of --analyze_existing is to analyze a transcript."
            )
            return

        run_analyze_existing_mode(
            args.analyze_existing, args, output_dir, analysis_output_dir,
            stats_file, run_uid, run_timestamp, start_time
        )
        return

    if args.verbose:
        log_section("Transcription Tool")
        logger.info(f"Input: {args.input_source}")
        logger.info(f"Run ID: {run_uid}")
        logger.info(f"Timestamp: {run_timestamp}")

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
                run_analyze_existing_mode(
                    prev_transcription.transcript_path, args, output_dir,
                    analysis_output_dir, stats_file, run_uid, run_timestamp, start_time
                )
                return

            elif action == DuplicateAction.RE_TRANSCRIBE:
                logger.info("Re-transcribing video...")

            elif action == DuplicateAction.FORCE_CONTINUE:
                logger.info("Continuing with transcription...")

    # Run the main transcription workflow
    process_input_source(
        args.input_source, args, output_dir, analysis_output_dir, stats_file,
        run_uid, run_timestamp, start_time
    )


if __name__ == "__main__":
    main()
