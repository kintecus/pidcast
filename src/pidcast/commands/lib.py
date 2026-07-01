"""Handlers for the ``pidcast lib`` subcommands.

Bodies lifted verbatim from the old monolithic ``cli.py`` (cmd_* -> handle_*),
with ``resolve_output_dir`` re-homed to ``config``. Kept out of ``library.py``
(the data layer) so the show store stays free of workflow coupling.
"""

import argparse
import datetime
import logging
import os
import time
import traceback
import uuid
from pathlib import Path

from ..config import RUNS_FILE, resolve_output_dir
from ..exceptions import DuplicateShowError, FeedFetchError, FeedParseError, ShowNotFoundError
from ..utils import log_error
from ..workflow import process_input_source

logger = logging.getLogger(__name__)


def handle_process(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib process' command."""
    from ..config import OBSIDIAN_PATH, VideoInfo
    from ..library import LibraryManager, Show

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
            selected_episode = display_episodes[int(choice) - 1]
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
        duration=selected_episode.duration or 0,
    )

    # Resolve whisper model
    if args.whisper_model:
        from ..transcription import resolve_whisper_model

        try:
            args.whisper_model = resolve_whisper_model(args.whisper_model)
        except Exception as e:
            log_error(str(e))
            return

    # Run workflow
    output_dir = resolve_output_dir(args)
    stats_file = Path(args.stats_file) if args.stats_file else RUNS_FILE
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
        video_info_override=video_info,
    )


def handle_add(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib add' command."""
    from ..config_manager import ConfigManager
    from ..library import LibraryManager
    from ..rss import RSSParser

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

    # Resolve feed URL from name if user didn't pass a URL
    feed_url = args.feed_url
    if not feed_url.startswith(("http://", "https://", "feed://")):
        from ..discovery import discover_podcast, prompt_user_selection

        query = feed_url
        if has_rich:
            Console().print(f"\nSearching for podcasts matching [bold]{query!r}[/bold]...")
        else:
            print(f"\nSearching for podcasts matching {query!r}...")

        results = discover_podcast(query)
        if not results:
            log_error(f"No podcasts found for {query!r}. Try using the RSS feed URL directly.")
            return

        chosen = prompt_user_selection(results)
        if chosen is None:
            logger.info("Cancelled.")
            return

        feed_url = chosen["feed_url"]
        args.feed_url = feed_url

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


def handle_list(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib list' command."""
    from ..library import LibraryManager

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
        logger.info("No shows in library. Use 'pidcast lib add <feed-url>' to add shows.")
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


def handle_show(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib show' command."""
    from ..library import LibraryManager

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
            log_error(
                f"Show ID {args.show_id} not found. Run 'pidcast lib list' to see available shows."
            )
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

            console.print(
                Panel(info_table, title=f"[bold]{show.title}[/bold]", border_style="cyan")
            )

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
            print(f"{'=' * 80}")
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


def handle_remove(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib remove' command."""
    from ..library import LibraryManager

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
        log_error(
            f"Show ID {args.show_id} not found. Run 'pidcast lib list' to see available shows."
        )
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


def handle_digest(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib digest' command."""
    from datetime import datetime, timedelta

    from ..config import DEFAULT_PROMPTS_FILE, HISTORY_FILE, get_digest_output_path
    from ..digest import DigestFormatter, DigestGenerator
    from ..history import ProcessingHistory
    from ..library import LibraryManager
    from ..summarization import Summarizer

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

        # Save to file (digest path comes from DIGESTS_DIR via get_digest_output_path)
        output_dir = resolve_output_dir(args)
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


def handle_sync(args: argparse.Namespace) -> None:
    """Handle 'pidcast lib sync' command."""
    from ..config import DEFAULT_BACKFILL_LIMIT, HISTORY_FILE, WHISPER_MODEL
    from ..history import ProcessingHistory
    from ..library import LibraryManager
    from ..sync import SyncEngine

    # Initialize library and history
    library = LibraryManager()
    history = ProcessingHistory(HISTORY_FILE)

    # Get output directory
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get Whisper model
    whisper_model = args.whisper_model or WHISPER_MODEL
    if not whisper_model:
        log_error(
            "Whisper model not specified. Set WHISPER_MODEL environment variable "
            "or use --whisper_model flag"
        )
        return

    # Resolve model name to path
    from ..transcription import resolve_whisper_model

    try:
        whisper_model = resolve_whisper_model(whisper_model)
    except Exception as e:
        log_error(str(e))
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
        logger.info(f"{'=' * 60}")
        logger.info(f"Processed: {stats['processed']} episodes")
        logger.info(f"Succeeded: {stats['succeeded']}")
        logger.info(f"Failed: {stats['failed']}")
        if stats["skipped"] > 0:
            logger.info(f"Skipped: {stats['skipped']} (already processed)")
        logger.info(f"{'=' * 60}\n")

        # Generate digest unless --no-digest flag is set
        if not args.no_digest and stats["succeeded"] > 0 and groq_api_key:
            from datetime import datetime

            from ..config import DEFAULT_PROMPTS_FILE, get_digest_output_path
            from ..digest import DigestFormatter, DigestGenerator
            from ..summarization import Summarizer

            try:
                logger.info("Generating digest...")
                prompts_file = (
                    Path(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS_FILE
                )
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
