# PRD: Pidcast Sync - Phase 2

**Contract**: ./contract.md
**Phase**: 2 of 3
**Focus**: Sync Pipeline and Automated Processing

## Phase Overview

Phase 2 implements the core sync automation that transforms pidcast from a manual tool into an automated monitoring system. This phase introduces the `pidcast sync` command that checks all library feeds for new episodes, tracks processing history to avoid duplicates, and orchestrates the download → transcribe → analyze pipeline for each new episode.

This phase is sequenced after Phase 1 because it requires the library and RSS parsing infrastructure to be functional. The sync command needs to iterate over library shows, fetch their feeds, and compare against processing history.

After Phase 2 completes, users can run a single command to automatically process all new podcast episodes from their library. Each episode gets the full treatment: audio download, Whisper transcription, Groq analysis, and individual markdown file output—all without manual URL input. This is where the time-saving automation promise is delivered.

Key architectural consideration: The processing history tracker must be efficient enough to handle hundreds of episodes per show without performance degradation. We'll use a simple JSON file with episode GUIDs as keys for O(1) lookup.

## User Stories

1. As a podcast listener, I want to run `pidcast sync` to automatically check all my subscribed shows for new episodes so that I don't have to manually monitor feeds
2. As a user, I want the system to track which episodes have been processed so that I don't waste compute re-transcribing the same content
3. As a user, I want to configure how many recent episodes to process when adding a new show so that I can catch up on back catalogs without processing everything
4. As a user, I want to see progress indicators during sync so that I know the system is working and how long it might take
5. As a developer, I want the sync command to be idempotent so that I can safely run it multiple times without side effects
6. As a user, I want partial failures (one feed down, one transcription fails) to not block the entire sync so that I still get results from successful episodes

## Functional Requirements

### Sync Command

- **FR-2.1**: System must support `pidcast sync` command that checks all library shows for new episodes
- **FR-2.2**: For each show, system must fetch RSS feed and compare episode GUIDs against processing history
- **FR-2.3**: System must identify episodes published since last sync (or within backfill limit for newly added shows)
- **FR-2.4**: System must update "last checked" timestamp for each show after fetching its feed
- **FR-2.5**: System must support `--show <show-id>` flag to sync only a specific show rather than entire library
- **FR-2.6**: System must support `--dry-run` flag to preview what would be processed without actually downloading/transcribing

### Processing History

- **FR-2.7**: System must track processed episodes in `~/.config/pidcast/history.json` with episode GUID as key
- **FR-2.8**: Each history entry must include: episode GUID, show ID, processed timestamp, output file path, processing status (success/failed)
- **FR-2.9**: System must skip episodes already marked as "success" in processing history
- **FR-2.10**: System must support `--force` flag to reprocess episodes even if already in history
- **FR-2.11**: System must mark failed episodes in history so they can be retried on next sync without being skipped
- **FR-2.12**: System must support `pidcast history` command to view processing history with filters (by show, by date range, by status)

### Automated Pipeline

- **FR-2.13**: For each new episode, system must download audio using existing `download.py` logic
- **FR-2.14**: System must transcribe audio using existing `transcription.py` Whisper integration
- **FR-2.15**: System must analyze transcript using existing `analysis.py` Groq LLM integration
- **FR-2.16**: System must generate markdown file using existing `markdown.py` with YAML front matter including show metadata
- **FR-2.17**: System must save individual episode markdown files to `data/transcripts/` with date-prefixed filenames
- **FR-2.18**: System must handle Obsidian save flag (`--save_to_obsidian`) for individual episodes if user wants vault integration

### Backfill Configuration

- **FR-2.19**: Global config must include `backfill_limit` setting (default: 5) for number of recent episodes to process on show add
- **FR-2.20**: When a show is newly added, system must process up to `backfill_limit` most recent episodes on first sync
- **FR-2.21**: After initial backfill, system must only process episodes published after the "last checked" timestamp
- **FR-2.22**: System must support `pidcast sync --backfill <N>` to override global limit for a specific sync run

### Progress and Logging

- **FR-2.23**: System must display progress bar or status indicator showing current episode being processed (e.g., "Processing 3/12: Show Name - Episode Title")
- **FR-2.24**: System must log summary at end of sync: X new episodes, Y processed successfully, Z failed
- **FR-2.25**: System must log errors for failed episodes (network issues, transcription errors, LLM failures) without halting entire sync
- **FR-2.26**: System must write detailed sync log to `~/.config/pidcast/logs/sync-YYYY-MM-DD.log`

## Non-Functional Requirements

- **NFR-2.1**: Sync command must handle libraries with 50+ shows and 1000+ total episodes without excessive memory usage (<500MB RAM)
- **NFR-2.2**: Processing history lookup (checking if episode was already processed) must be O(1) constant time using GUID-based dictionary
- **NFR-2.3**: Failed episodes must not corrupt processing history or block subsequent episode processing (fault isolation)
- **NFR-2.4**: System must recover gracefully from interruptions (Ctrl+C) by saving processing history incrementally, not just at end
- **NFR-2.5**: Network requests to RSS feeds must use connection pooling and respect rate limits (max 5 concurrent feed fetches)

## Dependencies

### Prerequisites

- Phase 1 complete: library management, RSS parsing, persistent storage
- Existing download, transcription, and analysis modules (`download.py`, `transcription.py`, `analysis.py`)
- Existing markdown generation (`markdown.py`)

### Outputs for Next Phase

- `sync.py` module with sync orchestration logic
- `history.py` module with processing history tracker
- `history.json` file populated with processed episodes
- Individual episode markdown files in `data/transcripts/`

## Acceptance Criteria

- [ ] User can run `pidcast sync` and see all new episodes from library shows processed automatically
- [ ] System skips episodes already in processing history (no duplicate transcriptions)
- [ ] User can run `pidcast sync --show <id>` to sync only one specific show
- [ ] User can run `pidcast sync --dry-run` to preview what would be processed without executing
- [ ] When adding a new show, first sync processes up to `backfill_limit` recent episodes (configurable, default 5)
- [ ] After initial backfill, subsequent syncs only process episodes newer than "last checked" timestamp
- [ ] Processing history persists in `~/.config/pidcast/history.json` with episode GUIDs, timestamps, and status
- [ ] Failed episodes are marked in history and automatically retried on next sync
- [ ] User can force reprocessing with `pidcast sync --force` to ignore history
- [ ] Progress indicator shows current episode being processed with count (3/12)
- [ ] Sync summary displayed at end: "Processed 8 new episodes: 7 successful, 1 failed"
- [ ] Partial failures (one feed down, one transcription error) don't halt entire sync
- [ ] Individual episode markdown files generated with show metadata in YAML front matter
- [ ] System handles Ctrl+C interruption gracefully by saving history incrementally
- [ ] All existing CLI commands (add, list, remove, show, single-episode processing) continue to work
- [ ] Integration tests for full sync pipeline with mock RSS feeds
- [ ] Unit tests for processing history edge cases (concurrent access, corrupted history file)

## Open Questions

- Should sync run episodes in parallel (multiple downloads/transcriptions concurrently) or sequentially? (Recommend: sequential initially for simplicity, add concurrency in future optimization)
- What should happen if an episode's audio URL (enclosure) is invalid or 404? (Recommend: mark failed in history, log error, continue sync)
- Should we enforce a maximum episode duration to avoid extremely long processing times? (Recommend: warn for episodes >3 hours, but don't skip)

---

*Review this PRD and provide feedback before spec generation.*
