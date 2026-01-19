"""Sync engine for automated podcast processing."""

import logging
import time
import uuid
from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .analysis import analyze_transcript_with_llm, load_analysis_prompts
from .config import DEFAULT_PROMPTS_FILE, VideoInfo
from .download import download_audio
from .exceptions import AnalysisError, DownloadError, TranscriptionError
from .history import ProcessingHistory
from .library import LibraryManager, Show
from .markdown import create_markdown_file
from .rss import Episode, RSSParser
from .transcription import run_whisper_transcription
from .utils import create_smart_filename, format_duration, get_unique_filename

logger = logging.getLogger(__name__)


class SyncEngine:
    """Orchestrate podcast sync workflow."""

    def __init__(
        self,
        library: LibraryManager,
        history: ProcessingHistory,
        config: dict,
        output_dir: Path,
        whisper_model: str,
        groq_api_key: str | None = None,
        analysis_type: str = "executive_summary",
        prompts_file: Path | None = None,
        verbose: bool = False,
    ):
        """Initialize SyncEngine.

        Args:
            library: Library manager instance
            history: Processing history manager
            config: Configuration dict
            output_dir: Output directory for transcripts
            whisper_model: Path to Whisper model
            groq_api_key: Groq API key for analysis
            analysis_type: Type of analysis to perform
            prompts_file: Path to prompts YAML file
            verbose: Enable verbose logging
        """
        self.library = library
        self.history = history
        self.config = config
        self.output_dir = output_dir
        self.whisper_model = whisper_model
        self.groq_api_key = groq_api_key
        self.analysis_type = analysis_type
        self.prompts_file = prompts_file or DEFAULT_PROMPTS_FILE
        self.verbose = verbose
        self.stats = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    def sync(
        self,
        show_id: int | None = None,
        dry_run: bool = False,
        force: bool = False,
        backfill: int | None = None,
    ) -> dict:
        """Run sync for library shows.

        Args:
            show_id: Sync only specific show (None = all shows)
            dry_run: Preview mode, no actual processing
            force: Reprocess episodes even if already successful
            backfill: Override global backfill limit

        Returns:
            Stats dict with processed, succeeded, failed, skipped counts
        """
        # Get shows to sync
        shows = [self.library.get_show(show_id)] if show_id else self.library.list_shows()

        if not shows:
            logger.info("No shows to sync")
            return self.stats

        if show_id and shows[0] is None:
            logger.error(f"Show ID {show_id} not found")
            return self.stats

        # Filter out None values
        shows = [s for s in shows if s is not None]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Syncing {len(shows)} show(s)")
        if dry_run:
            logger.info("[DRY RUN MODE - No files will be created]")
        logger.info(f"{'=' * 60}\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:

            for show in shows:
                task = progress.add_task(f"[cyan]{show.title}", total=None)

                try:
                    # Fetch new episodes
                    episodes = self._fetch_new_episodes(show, backfill, force)
                    progress.update(task, total=len(episodes))

                    if not episodes:
                        logger.info(f"  No new episodes for: {show.title}")
                        progress.update(task, completed=len(episodes))
                        continue

                    logger.info(
                        f"  Found {len(episodes)} new episode(s) for: {show.title}"
                    )

                    for episode in episodes:
                        if dry_run:
                            logger.info(
                                f"    [DRY RUN] Would process: {episode.title}"
                            )
                            self.stats["processed"] += 1
                            progress.advance(task)
                            continue

                        # Process episode
                        success = self._process_episode(show, episode)
                        if success:
                            self.stats["succeeded"] += 1
                        else:
                            self.stats["failed"] += 1

                        self.stats["processed"] += 1
                        progress.advance(task)

                    # Update last checked timestamp
                    if not dry_run:
                        self.library.update_last_checked(show.id)

                except Exception as e:
                    logger.error(f"  Error syncing {show.title}: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
                    continue

        return self.stats

    def _fetch_new_episodes(
        self,
        show: Show,
        backfill: int | None,
        force: bool,
    ) -> list[Episode]:
        """Fetch new episodes from show's RSS feed.

        Args:
            show: Show to fetch episodes for
            backfill: Override backfill limit
            force: Reprocess all episodes

        Returns:
            List of episodes to process
        """
        # Parse RSS feed
        _, episodes = RSSParser.parse_feed(show.feed_url, verbose=self.verbose)

        # Handle backfill for newly added shows
        if show.last_checked is None:
            # New show: use backfill limit
            limit = backfill if backfill is not None else show.backfill_count
            episodes = episodes[:limit]
            if self.verbose:
                logger.info(f"    Backfilling {limit} episodes for new show")
        else:
            # Existing show: filter by pub_date
            episodes = [
                ep for ep in episodes
                if ep.pub_date > show.last_checked
            ]
            if self.verbose:
                logger.info(
                    f"    Found {len(episodes)} episodes since last check"
                )

        # Filter out already processed (unless force)
        if not force:
            unprocessed = []
            for ep in episodes:
                if not self.history.is_processed(ep.guid):
                    unprocessed.append(ep)
                else:
                    self.stats["skipped"] += 1

            if self.verbose and len(episodes) != len(unprocessed):
                logger.info(
                    f"    Skipping {len(episodes) - len(unprocessed)} "
                    "already processed episodes"
                )

            episodes = unprocessed

        return episodes

    def _process_episode(self, show: Show, episode: Episode) -> bool:
        """Process single episode through pipeline.

        Args:
            show: Show the episode belongs to
            episode: Episode to process

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n  Processing: {episode.title}")

        # Mark as in progress
        self.history.mark_in_progress(episode.guid, show.id, episode.title)

        audio_file = None
        transcript_file = None

        try:
            # Create video info for this episode
            video_info = VideoInfo(
                title=episode.title,
                webpage_url=episode.audio_url,
                channel=show.author,
                uploader=show.author,
                duration=episode.duration or 0,
                duration_string=f"{episode.duration // 60}m" if episode.duration else "Unknown",
                upload_date=episode.pub_date.strftime("%Y%m%d"),
            )

            # Download audio
            if self.verbose:
                logger.info("    Downloading audio...")

            audio_file, _ = download_audio(
                episode.audio_url,
                "temp_audio.%(ext)s",
                verbose=self.verbose,
            )

            # Transcribe
            if self.verbose:
                logger.info("    Transcribing with Whisper...")

            temp_output = self.output_dir / f"temp_transcript_{uuid.uuid4().hex[:8]}"
            run_whisper_transcription(
                audio_file,
                self.whisper_model,
                "txt",
                str(temp_output),
                self.verbose,
            )

            transcript_file = f"{temp_output}.txt"

            # Create markdown file
            smart_filename = create_smart_filename(
                episode.title, max_length=60, include_date=True
            )
            markdown_file = get_unique_filename(
                self.output_dir, smart_filename, ".md"
            )

            # Build front matter with show metadata
            front_matter = {
                "show": show.title,
                "show_id": show.id,
                "episode_guid": episode.guid,
                "pub_date": episode.pub_date.isoformat(),
                "audio_url": episode.audio_url,
            }

            if not create_markdown_file(
                markdown_file,
                transcript_file,
                video_info,
                front_matter,
                self.verbose,
            ):
                raise TranscriptionError("Failed to create markdown file")

            # Analyze with LLM (if API key provided)
            if self.groq_api_key:
                if self.verbose:
                    logger.info("    Analyzing transcript with LLM...")

                # Read transcript
                with open(transcript_file, encoding="utf-8") as f:
                    transcript_text = f.read()

                # Load prompts
                prompts_config = load_analysis_prompts(
                    self.prompts_file, self.verbose
                )

                # Run analysis
                analysis_start = time.time()
                _ = analyze_transcript_with_llm(
                    transcript_text,
                    video_info,
                    self.analysis_type,
                    prompts_config,
                    self.groq_api_key,
                    None,  # groq_model
                    self.verbose,
                )
                analysis_duration = time.time() - analysis_start

                if self.verbose:
                    logger.info(
                        f"    Analysis completed in {format_duration(analysis_duration)}"
                    )

            # Clean up temp files
            if transcript_file and Path(transcript_file).exists():
                Path(transcript_file).unlink()

            # Mark success
            self.history.mark_success(episode.guid, str(markdown_file))

            logger.info(f"    ✓ Saved to: {markdown_file.name}")
            return True

        except (DownloadError, TranscriptionError, AnalysisError) as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"    ✗ Failed: {error_msg}")
            self.history.mark_failed(episode.guid, error_msg)
            return False

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"    ✗ {error_msg}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            self.history.mark_failed(episode.guid, error_msg)
            return False

        finally:
            # Clean up temp audio files
            if audio_file and Path(audio_file).exists():
                try:
                    Path(audio_file).unlink()
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"    Could not remove {audio_file}: {e}")

            # Clean up temp transcript if still exists
            if transcript_file and Path(transcript_file).exists():
                Path(transcript_file).unlink(missing_ok=True)
