# Implementation Spec: Pidcast Sync - Phase 2

**PRD**: ./prd-phase-2.md
**Estimated Effort**: L (Large - 5-7 days)

## Technical Approach

Phase 2 builds the sync orchestration layer that ties together library management (Phase 1) with existing transcription pipeline. The core challenge is coordinating multiple async operations (feed fetching, downloads, transcriptions) while maintaining state and handling partial failures gracefully.

We'll implement a `SyncEngine` class that orchestrates the workflow: fetch feeds → identify new episodes → process each through existing pipeline → update history. Processing history uses a simple JSON file with episode GUIDs as keys for O(1) lookups. Each episode's processing result (success/failure) is persisted incrementally to ensure Ctrl+C interruptions don't lose progress.

The existing transcription modules (`download.py`, `transcription.py`, `analysis.py`, `markdown.py`) will be reused with minimal changes. The sync engine acts as a coordinator, calling these modules for each episode. Progress tracking uses the `rich` library for clean terminal output with progress bars.

Key architectural decisions:
- Sequential processing: Process episodes one at a time for MVP (simpler error handling, no resource contention)
- Incremental history saves: Persist after each episode to survive interruptions
- Fail-fast on critical errors (can't load library) but continue on per-episode errors (one bad feed doesn't stop sync)
- Dry-run mode: Full simulation without side effects (no downloads, no transcriptions)

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `pidcast/sync.py` | Sync engine orchestration and episode processing pipeline |
| `pidcast/history.py` | Processing history manager with GUID-based tracking |
| `tests/test_sync.py` | Unit tests for sync orchestration logic |
| `tests/test_history.py` | Unit tests for history tracking and persistence |
| `tests/fixtures/mock_episodes.json` | Mock episode data for testing |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `pidcast/cli.py` | Add `sync` subcommand with flags (--show, --dry-run, --force, --backfill) |
| `pidcast/library.py` | Add `update_last_checked()` method to update show timestamps |
| `pidcast/config.py` | Add sync configuration constants (history path, concurrency limits) |
| `pidcast/markdown.py` | Add show metadata to YAML front matter for synced episodes |
| `requirements.txt` | Add `rich` for progress bars and terminal output |

## Implementation Details

### ProcessingHistory Class

**Pattern to follow**: Similar to `LibraryManager` with JSON persistence

**Overview**: Track which episodes have been processed to prevent duplicate transcriptions.

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from pathlib import Path
import json

class ProcessingStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

@dataclass
class HistoryEntry:
    """Represents a processed episode."""
    guid: str
    show_id: int
    episode_title: str
    processed_at: datetime
    status: ProcessingStatus
    output_file: Optional[str]  # Path to markdown file
    error_message: Optional[str] = None

class ProcessingHistory:
    """Manage episode processing history."""

    def __init__(self, history_path: Path):
        self.history_path = history_path
        self.entries: Dict[str, HistoryEntry] = {}  # guid -> entry
        self._load()

    def is_processed(self, guid: str) -> bool:
        """Check if episode already processed successfully."""
        entry = self.entries.get(guid)
        return entry is not None and entry.status == ProcessingStatus.SUCCESS

    def mark_in_progress(self, guid: str, show_id: int, title: str):
        """Mark episode as currently processing."""
        self.entries[guid] = HistoryEntry(
            guid=guid,
            show_id=show_id,
            episode_title=title,
            processed_at=datetime.now(),
            status=ProcessingStatus.IN_PROGRESS,
            output_file=None
        )
        self._save()

    def mark_success(self, guid: str, output_file: str):
        """Mark episode as successfully processed."""
        if guid in self.entries:
            self.entries[guid].status = ProcessingStatus.SUCCESS
            self.entries[guid].output_file = output_file
            self.entries[guid].processed_at = datetime.now()
            self._save()

    def mark_failed(self, guid: str, error: str):
        """Mark episode as failed."""
        if guid in self.entries:
            self.entries[guid].status = ProcessingStatus.FAILED
            self.entries[guid].error_message = error
            self.entries[guid].processed_at = datetime.now()
            self._save()

    def get_failed_episodes(self) -> list[HistoryEntry]:
        """Get all failed episodes for retry."""
        return [e for e in self.entries.values() if e.status == ProcessingStatus.FAILED]

    def _load(self):
        """Load history from JSON file."""
        if not self.history_path.exists():
            return

        with open(self.history_path) as f:
            data = json.load(f)
            for guid, entry_dict in data.items():
                self.entries[guid] = HistoryEntry(**entry_dict)

    def _save(self):
        """Save history to JSON file (incremental)."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'w') as f:
            data = {
                guid: {
                    'guid': e.guid,
                    'show_id': e.show_id,
                    'episode_title': e.episode_title,
                    'processed_at': e.processed_at.isoformat(),
                    'status': e.status.value,
                    'output_file': e.output_file,
                    'error_message': e.error_message,
                }
                for guid, e in self.entries.items()
            }
            json.dump(data, f, indent=2)
```

**Key decisions**:
- GUID as key: RSS standard, guaranteed unique, O(1) lookup
- Incremental saves: Persist after each episode status change to survive interruptions
- Failed episodes tracked: Can be retried on next sync without --force
- In-progress status: Detect crashed/interrupted syncs

**Implementation steps**:
1. Create `ProcessingStatus` enum and `HistoryEntry` dataclass
2. Implement `ProcessingHistory` with load/save using JSON
3. Add `is_processed()` for quick success checks
4. Add `mark_in_progress()`, `mark_success()`, `mark_failed()` with incremental saves
5. Add `get_failed_episodes()` for automatic retry logic
6. Handle corrupted history.json gracefully (backup and recreate)

### SyncEngine Class

**Pattern to follow**: Similar to existing CLI flow but multi-episode orchestration

**Overview**: Orchestrate sync workflow across all library shows.

```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from typing import List, Tuple

class SyncEngine:
    """Orchestrate podcast sync workflow."""

    def __init__(
        self,
        library: LibraryManager,
        history: ProcessingHistory,
        config: dict
    ):
        self.library = library
        self.history = history
        self.config = config
        self.stats = {'processed': 0, 'succeeded': 0, 'failed': 0}

    def sync(
        self,
        show_id: Optional[int] = None,
        dry_run: bool = False,
        force: bool = False,
        backfill: Optional[int] = None
    ) -> dict:
        """
        Run sync for library shows.

        Args:
            show_id: Sync only specific show (None = all shows)
            dry_run: Preview mode, no actual processing
            force: Reprocess episodes even if already successful
            backfill: Override global backfill limit

        Returns:
            Stats dict with processed, succeeded, failed counts
        """
        shows = [self.library.get_show(show_id)] if show_id else self.library.list_shows()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:

            for show in shows:
                task = progress.add_task(f"[cyan]{show.title}", total=None)

                try:
                    episodes = self._fetch_new_episodes(show, backfill, force)
                    progress.update(task, total=len(episodes))

                    for episode in episodes:
                        if dry_run:
                            print(f"[DRY RUN] Would process: {episode.title}")
                            continue

                        self._process_episode(show, episode)
                        progress.advance(task)

                    # Update last checked timestamp
                    self.library.update_last_checked(show.id)

                except Exception as e:
                    print(f"[red]Error syncing {show.title}: {e}")
                    continue

        return self.stats

    def _fetch_new_episodes(
        self,
        show: Show,
        backfill: Optional[int],
        force: bool
    ) -> List[Episode]:
        """Fetch new episodes from show's RSS feed."""
        show_meta, episodes = RSSParser.parse_feed(show.feed_url)

        # Handle backfill for newly added shows
        if show.last_checked is None:
            limit = backfill or self.config.get('backfill_limit', 5)
            episodes = episodes[:limit]
        else:
            # Filter episodes published after last check
            episodes = [
                ep for ep in episodes
                if ep.pub_date > show.last_checked
            ]

        # Filter out already processed (unless force)
        if not force:
            episodes = [
                ep for ep in episodes
                if not self.history.is_processed(ep.guid)
            ]

        return episodes

    def _process_episode(self, show: Show, episode: Episode):
        """Process single episode through pipeline."""
        self.history.mark_in_progress(episode.guid, show.id, episode.title)

        try:
            # Download audio
            audio_path = download_audio(episode.audio_url)

            # Transcribe
            transcript = transcribe_audio(audio_path)

            # Analyze
            analysis = analyze_transcript(transcript)

            # Generate markdown
            output_file = generate_markdown(
                episode=episode,
                show=show,
                transcript=transcript,
                analysis=analysis
            )

            self.history.mark_success(episode.guid, output_file)
            self.stats['succeeded'] += 1

        except Exception as e:
            self.history.mark_failed(episode.guid, str(e))
            self.stats['failed'] += 1
            print(f"[red]Failed to process {episode.title}: {e}")

        finally:
            self.stats['processed'] += 1
```

**Key decisions**:
- Use rich Progress for visual feedback during long syncs
- Sequential processing for MVP (parallel in future optimization)
- Continue on error: One failed episode doesn't stop sync
- Separate fetch and process: Fetch all feeds first to calculate total progress

**Implementation steps**:
1. Create `SyncEngine` class with library, history, config dependencies
2. Implement `sync()` with show filtering and flags (dry-run, force, backfill)
3. Implement `_fetch_new_episodes()` with backfill logic and last_checked filtering
4. Implement `_process_episode()` calling existing modules (download, transcribe, analyze, markdown)
5. Add error handling with try/except per episode (don't fail entire sync)
6. Add rich progress bars for visual feedback
7. Return stats dict for summary display

### CLI Integration

**Pattern to follow**: Existing CLI subcommand structure

**Overview**: Add `sync` subcommand with various flags for control.

```python
def setup_sync_command(subparsers):
    """Add sync subcommand to CLI."""
    sync_parser = subparsers.add_parser(
        'sync',
        help='Sync library shows and process new episodes'
    )
    sync_parser.add_argument(
        '--show',
        type=int,
        metavar='ID',
        help='Sync only specific show by ID'
    )
    sync_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be processed without executing'
    )
    sync_parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess episodes even if already successful'
    )
    sync_parser.add_argument(
        '--backfill',
        type=int,
        metavar='N',
        help='Override global backfill limit for this sync'
    )

def cmd_sync(args):
    """Handle 'pidcast sync' command."""
    library = LibraryManager.load()
    history = ProcessingHistory(get_history_path())
    config = ConfigManager.load_config()

    engine = SyncEngine(library, history, config)
    stats = engine.sync(
        show_id=args.show,
        dry_run=args.dry_run,
        force=args.force,
        backfill=args.backfill
    )

    # Print summary
    print(f"\n[bold green]Sync complete!")
    print(f"Processed: {stats['processed']} episodes")
    print(f"Succeeded: {stats['succeeded']}")
    print(f"Failed: {stats['failed']}")
```

**Implementation steps**:
1. Add `sync` subcommand with flags to argparse
2. Implement `cmd_sync()` handler
3. Load library, history, config
4. Create SyncEngine and call sync()
5. Display summary stats at end

### Markdown Generation Enhancement

**Overview**: Add show metadata to YAML front matter for synced episodes.

```python
def generate_markdown_for_sync(
    episode: Episode,
    show: Show,
    transcript: str,
    analysis: dict
) -> str:
    """Generate markdown with show metadata."""
    front_matter = {
        'title': episode.title,
        'show': show.title,
        'show_id': show.id,
        'episode_guid': episode.guid,
        'pub_date': episode.pub_date.isoformat(),
        'duration': episode.duration,
        'audio_url': episode.audio_url,
        'processed_at': datetime.now().isoformat(),
        'tags': analysis.get('contextual_tags', []),
    }

    # Rest of markdown generation...
    return markdown_content
```

**Implementation steps**:
1. Modify `markdown.py` to accept show metadata parameter
2. Add show fields to YAML front matter
3. Ensure backward compatibility for direct URL transcriptions (show fields optional)

## Data Model

### Processing History JSON Structure

```json
{
  "episode-guid-12345": {
    "guid": "episode-guid-12345",
    "show_id": 1,
    "episode_title": "AI Safety and Alignment",
    "processed_at": "2026-01-19T12:30:00",
    "status": "success",
    "output_file": "data/transcripts/2026-01-19_AI-Safety-and-Alignment.md",
    "error_message": null
  },
  "episode-guid-67890": {
    "guid": "episode-guid-67890",
    "show_id": 2,
    "episode_title": "Bitcoin ETF Discussion",
    "processed_at": "2026-01-19T13:00:00",
    "status": "failed",
    "output_file": null,
    "error_message": "Network timeout downloading audio"
  }
}
```

## Testing Requirements

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_history.py` | ProcessingHistory CRUD and persistence |
| `tests/test_sync.py` | SyncEngine orchestration logic |

**Key test cases**:
- History: is_processed check, mark success/failed, incremental saves
- Sync: fetch new episodes (backfill vs last_checked), filter already processed, dry-run mode
- Error handling: corrupted history file, failed episode processing, network errors

**Edge cases**:
- Show never synced before (last_checked is None) → use backfill limit
- Show synced but no new episodes → skip gracefully
- Episode GUID collision (duplicate across feeds) → handle with show_id prefix
- Interrupted sync (Ctrl+C mid-episode) → history shows in_progress status
- Force flag with large backlog → reprocess all episodes

**Error cases**:
- RSS feed returns 404 during sync
- Audio download fails mid-sync
- Whisper transcription fails (bad audio format)
- Groq API rate limit during analysis
- Disk full when saving markdown

### Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_sync_integration.py` | End-to-end sync with mock feeds |

**Key scenarios**:
- Add two shows, run sync, verify both processed
- Run sync twice, verify no duplicate processing
- Run sync with --dry-run, verify no files created
- Run sync with --force, verify reprocessing
- Interrupt sync, verify history persisted, resume on next sync

### Manual Testing

- [ ] Add multiple shows with `pidcast add`
- [ ] Run `pidcast sync` and verify all new episodes processed
- [ ] Check `history.json` created with episode entries
- [ ] Run `pidcast sync` again and verify "No new episodes" message
- [ ] Publish new episode in test feed, run sync, verify only new episode processed
- [ ] Run `pidcast sync --show 1` and verify only that show synced
- [ ] Run `pidcast sync --dry-run` and verify no transcriptions created
- [ ] Run `pidcast sync --force` and verify episodes reprocessed
- [ ] Interrupt sync with Ctrl+C mid-processing, verify history saved, resume sync
- [ ] Test with slow/flaky network to verify retry logic

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| RSS feed 404 or network error | Log error for that show, continue syncing other shows. Retry feed 3x before failing. |
| Audio download fails | Mark episode as failed in history, log error, continue to next episode |
| Transcription fails | Mark episode as failed in history, log error with details, continue |
| LLM analysis fails | Mark episode as failed in history, save partial transcript markdown, continue |
| Disk full | Catch OSError, halt sync gracefully, display error message with disk space info |
| Corrupted history.json | Backup file, create new history, log warning about lost history |
| Invalid episode GUID | Fallback to episode URL as GUID, log warning |
| Ctrl+C interrupt | Catch KeyboardInterrupt, save history, display "Sync interrupted" message |

## Validation Commands

```bash
# Linting
uv run ruff check pidcast/

# Unit tests
uv run pytest tests/test_history.py
uv run pytest tests/test_sync.py

# Integration tests
uv run pytest tests/test_sync_integration.py

# Full test suite
uv run pytest

# Manual sync test
uv run pidcast add https://feeds.example.com/test1
uv run pidcast add https://feeds.example.com/test2
uv run pidcast sync --dry-run
uv run pidcast sync
uv run pidcast sync  # Should show "No new episodes"
```

## Rollout Considerations

- **Feature flag**: None - this is additive functionality
- **Monitoring**: Log sync operations (start, end, episode counts, errors) to `logs/sync-YYYY-MM-DD.log`
- **Backward compatibility**: Existing single-URL transcription and library management (Phase 1) continue working
- **Performance**: Sequential processing initially; add --parallel flag in future if needed
- **Rollback plan**: Users can continue manual transcription; sync command can be disabled if critical bugs found

## Open Items

- [ ] Should failed episodes auto-retry on next sync, or require --force? (Recommend: auto-retry failed, require --force for successful)
- [ ] Should we support parallel episode processing with concurrency limit? (Defer to future optimization)
- [ ] Should sync send desktop notification when complete? (Defer to future enhancement)

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
