# Implementation Spec: Pidcast Sync - Phase 1

**PRD**: ./prd-phase-1.md
**Estimated Effort**: M (Medium - 3-5 days)

## Technical Approach

Phase 1 introduces library management as a new module alongside existing pidcast functionality. The approach is to create a clean separation between library operations and single-episode processing, with both workflows coexisting in the CLI.

We'll use `feedparser` for RSS parsing (battle-tested, handles most podcast feed quirks) and `ruamel.yaml` for config persistence (preserves comments and formatting for human editability). Library data will be stored in `~/.config/pidcast/library.yaml` following XDG Base Directory specification.

The CLI will be extended with new subcommands (`add`, `list`, `remove`, `show`) that operate on the library, while the existing direct URL/file transcription workflow remains unchanged. All library operations will be implemented in a new `library.py` module with a `Show` dataclass and `LibraryManager` class for CRUD operations. RSS parsing will be isolated in `rss.py` to enable reuse in Phase 2 sync logic.

Key technical decisions:
- Show IDs: Simple auto-incrementing integers (easier CLI UX than UUIDs)
- RSS caching: Use `requests-cache` with ETag/Last-Modified support to minimize bandwidth
- Config location: `~/.config/pidcast/` on Unix, `%APPDATA%\pidcast\` on Windows

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `pidcast/library.py` | Library management with Show model and CRUD operations |
| `pidcast/rss.py` | RSS feed parsing and episode metadata extraction |
| `pidcast/config_manager.py` | Configuration file loading/saving with defaults |
| `tests/test_library.py` | Unit tests for library CRUD operations |
| `tests/test_rss.py` | Unit tests for RSS parsing edge cases |
| `tests/fixtures/sample_feed.xml` | Sample RSS feed for testing |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `pidcast/cli.py` | Add new subcommands (add, list, remove, show) with argument parsing |
| `pidcast/config.py` | Add library configuration constants (paths, defaults) |
| `requirements.txt` | Add `feedparser`, `ruamel.yaml`, `requests-cache` dependencies |
| `README.md` | Add library management command examples |

## Implementation Details

### LibraryManager Class

**Pattern to follow**: Similar to `utils.py` singleton pattern for centralized config access

**Overview**: Central class for managing the show library with CRUD operations and persistence.

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class Show:
    """Represents a podcast show in the library."""
    id: int
    title: str
    feed_url: str
    description: str = ""
    author: str = ""
    artwork_url: str = ""
    added_at: datetime = field(default_factory=datetime.now)
    last_checked: Optional[datetime] = None
    backfill_count: int = 5  # Uses global default

class LibraryManager:
    """Manages podcast library operations."""

    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.shows: List[Show] = []
        self._load()

    def add_show(self, feed_url: str) -> Show:
        """Add show to library from RSS feed URL."""
        # Validate feed, extract metadata, assign ID, save
        pass

    def get_show(self, show_id: int) -> Optional[Show]:
        """Get show by ID."""
        pass

    def list_shows(self) -> List[Show]:
        """Get all shows in library."""
        pass

    def remove_show(self, show_id: int) -> bool:
        """Remove show from library."""
        pass

    def _load(self):
        """Load library from YAML file."""
        pass

    def _save(self):
        """Save library to YAML file."""
        pass
```

**Key decisions**:
- Auto-incrementing IDs: Track max ID in memory, increment on add (simple, no collisions)
- YAML format: Human-readable, supports comments, git-friendly
- Lazy loading: Library loads on first access, not every command

**Implementation steps**:
1. Create `Show` dataclass with all metadata fields
2. Implement `LibraryManager.__init__` to load from YAML (create if missing)
3. Implement `add_show()` with RSS parsing and ID assignment
4. Implement `list_shows()`, `get_show()`, `remove_show()` operations
5. Implement `_save()` using ruamel.yaml to preserve formatting
6. Add validation: duplicate feed detection, URL format checking

### RSS Feed Parser

**Pattern to follow**: Existing `download.py` with error handling and retries

**Overview**: Parse podcast RSS feeds and extract episode metadata using feedparser library.

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import feedparser

@dataclass
class Episode:
    """Represents a podcast episode from RSS feed."""
    guid: str  # Unique episode identifier from RSS
    title: str
    description: str
    pub_date: datetime
    duration: Optional[int]  # Seconds
    audio_url: str  # Enclosure URL

class RSSParser:
    """Parse podcast RSS feeds."""

    @staticmethod
    def parse_feed(feed_url: str) -> tuple[dict, List[Episode]]:
        """
        Parse RSS feed and return (show_metadata, episodes).

        Raises:
            FeedFetchError: If feed can't be fetched
            FeedParseError: If feed format is invalid
        """
        feed = feedparser.parse(feed_url)

        # Validate feed
        if feed.bozo:
            raise FeedParseError(f"Invalid feed format: {feed.bozo_exception}")

        # Extract show metadata
        show_meta = {
            'title': feed.feed.get('title', 'Unknown'),
            'description': feed.feed.get('description', ''),
            'author': feed.feed.get('author', ''),
            'artwork_url': RSSParser._extract_artwork(feed.feed),
        }

        # Extract episodes
        episodes = [RSSParser._parse_episode(entry) for entry in feed.entries]

        return show_meta, episodes

    @staticmethod
    def _parse_episode(entry) -> Episode:
        """Parse single RSS entry into Episode."""
        # Handle iTunes extensions, multiple enclosures, etc.
        pass

    @staticmethod
    def _extract_artwork(feed_dict) -> str:
        """Extract artwork URL from feed (iTunes image, channel image, etc.)."""
        pass
```

**Key decisions**:
- Use `guid` as episode ID (RSS standard, guaranteed unique per feed)
- Handle missing fields gracefully with defaults
- Support iTunes podcast extensions for better metadata
- Retry logic: 3 retries with exponential backoff on network errors

**Implementation steps**:
1. Create `Episode` dataclass with required fields
2. Implement `parse_feed()` with feedparser integration
3. Add error handling for invalid feeds, network failures
4. Implement `_parse_episode()` with iTunes extension support
5. Add retry logic using `requests` with backoff
6. Handle edge cases: missing enclosures, malformed dates, HTML in descriptions

### CLI Integration

**Pattern to follow**: Existing `cli.py` with argparse subcommands

**Overview**: Extend CLI with library management subcommands while preserving existing transcription workflow.

```python
def setup_cli():
    """Setup argparse with subcommands."""
    parser = argparse.ArgumentParser(description="Podcast transcription tool")
    subparsers = parser.add_subparsers(dest='command')

    # Existing: direct transcription (backward compatible)
    parser.add_argument('url', nargs='?', help='YouTube URL or audio file path')
    parser.add_argument('--save_to_obsidian', action='store_true')

    # New: add show
    add_parser = subparsers.add_parser('add', help='Add podcast to library')
    add_parser.add_argument('feed_url', help='RSS feed URL')
    add_parser.add_argument('--preview', action='store_true', help='Preview episodes before adding')

    # New: list shows
    list_parser = subparsers.add_parser('list', help='List all shows in library')

    # New: show details
    show_parser = subparsers.add_parser('show', help='Show details for a podcast')
    show_parser.add_argument('show_id', type=int, help='Show ID')

    # New: remove show
    remove_parser = subparsers.add_parser('remove', help='Remove podcast from library')
    remove_parser.add_argument('show_id', type=int, help='Show ID')

    return parser

def cmd_add(args, library: LibraryManager):
    """Handle 'pidcast add' command."""
    if args.preview:
        # Fetch feed, show episodes, prompt for confirmation
        pass

    show = library.add_show(args.feed_url)
    print(f"Added: {show.title} (ID: {show.id})")

def cmd_list(args, library: LibraryManager):
    """Handle 'pidcast list' command."""
    shows = library.list_shows()
    # Format as table: ID, Title, Episodes, Last Checked
    pass
```

**Key decisions**:
- Backward compatibility: Positional URL argument still works for direct transcription
- Subcommands use verbs: `add`, `list`, `show`, `remove`
- Help text for each command with examples

**Implementation steps**:
1. Add subparsers to existing argparse setup
2. Implement command handlers: `cmd_add()`, `cmd_list()`, `cmd_show()`, `cmd_remove()`
3. Add `--preview` flag to `add` command with episode listing
4. Format `list` output as readable table (use rich library if available)
5. Add help text and usage examples
6. Ensure existing URL workflow still works (no breaking changes)

### Configuration Management

**Overview**: Centralized config loading with defaults for library path and settings.

```python
from pathlib import Path
from ruamel.yaml import YAML

class ConfigManager:
    """Manage pidcast configuration files."""

    @staticmethod
    def get_config_dir() -> Path:
        """Get config directory (XDG compliant)."""
        if os.name == 'posix':
            return Path.home() / '.config' / 'pidcast'
        else:  # Windows
            return Path(os.getenv('APPDATA')) / 'pidcast'

    @staticmethod
    def load_config() -> dict:
        """Load config.yaml with defaults."""
        config_path = ConfigManager.get_config_dir() / 'config.yaml'

        if not config_path.exists():
            return ConfigManager._default_config()

        yaml = YAML()
        return yaml.load(config_path)

    @staticmethod
    def _default_config() -> dict:
        """Default configuration."""
        return {
            'backfill_limit': 5,
            'output_dir': str(Path.cwd() / 'data' / 'transcripts'),
            'obsidian_vault': None,
        }
```

**Implementation steps**:
1. Create config directory on first run with default config.yaml
2. Add comments to config.yaml explaining each setting
3. Use ruamel.yaml to preserve comments on updates
4. Add config validation (invalid paths, negative limits)

## Data Model

### Library YAML Structure

```yaml
# Pidcast Library Configuration
# This file is auto-managed but safe to edit manually

shows:
  - id: 1
    title: "Lex Fridman Podcast"
    feed_url: "https://lexfridman.com/feed/podcast/"
    description: "Conversations about science, technology, history, philosophy and the nature of intelligence, consciousness, love, and power."
    author: "Lex Fridman"
    artwork_url: "https://example.com/artwork.jpg"
    added_at: "2026-01-19T10:30:00"
    last_checked: null
    backfill_count: 5

  - id: 2
    title: "The Joe Rogan Experience"
    feed_url: "https://feeds.megaphone.fm/JRE"
    description: "The official podcast of comedian Joe Rogan."
    author: "Joe Rogan"
    artwork_url: "https://example.com/jre.jpg"
    added_at: "2026-01-19T11:00:00"
    last_checked: null
    backfill_count: 5

# Auto-increment for next show ID
next_id: 3
```

### Config YAML Structure

```yaml
# Pidcast Global Configuration

# Number of recent episodes to process when adding a new show
backfill_limit: 5

# Directory for transcript output
output_dir: "./data/transcripts"

# Obsidian vault path (optional)
obsidian_vault: null

# RSS feed cache duration (hours)
feed_cache_hours: 1
```

## Testing Requirements

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_library.py` | LibraryManager CRUD operations |
| `tests/test_rss.py` | RSS parsing with various feed formats |
| `tests/test_config.py` | Config loading, defaults, validation |

**Key test cases**:
- Add show: valid feed, duplicate detection, invalid URL
- List shows: empty library, single show, multiple shows
- Remove show: existing ID, non-existent ID
- RSS parsing: valid feed, missing fields, malformed dates, iTunes extensions
- Config: default creation, XDG path resolution, Windows vs Unix

**Edge cases**:
- RSS feed with no episodes
- RSS feed with 1000+ episodes (pagination)
- Episode with missing enclosure
- Episode GUID missing (fallback to URL)
- Concurrent library modifications (file locking)

**Error cases**:
- Network timeout fetching RSS
- Invalid RSS XML
- Disk full when saving library
- Corrupted library.yaml file

### Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_cli_library.py` | End-to-end CLI commands |

**Key scenarios**:
- Add show via CLI, verify library.yaml updated
- List shows after adding multiple feeds
- Preview episodes before adding show
- Remove show and verify it's gone from library

### Manual Testing

- [ ] Run `pidcast add https://lexfridman.com/feed/podcast/` and verify show added
- [ ] Run `pidcast add --preview <url>` and confirm episodes displayed
- [ ] Run `pidcast list` and verify table formatting
- [ ] Run `pidcast show 1` and verify episode list displayed
- [ ] Run `pidcast remove 1` and verify show removed
- [ ] Edit library.yaml manually and verify CLI still works
- [ ] Test on both Unix (Linux/macOS) and Windows for path handling
- [ ] Test with slow network (simulate with proxy) to verify retry logic

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| Invalid RSS feed URL | Validate URL format, attempt fetch, return clear error message if 404/invalid |
| Network timeout fetching feed | Retry 3x with exponential backoff (1s, 2s, 4s), fail with error message if all retries fail |
| Duplicate feed URL | Check existing shows, reject with "Show already in library (ID: X)" message |
| Corrupted library.yaml | Backup current file, attempt to repair, fail with error and recovery instructions if unfixable |
| Show ID not found | Return "Show ID X not found. Run 'pidcast list' to see available shows." |
| Disk full when saving | Catch OSError, show error message, don't corrupt existing library file |
| Missing config directory permissions | Attempt to create, fail with permission error and manual instructions |

## Validation Commands

```bash
# Linting (ruff)
uv run ruff check pidcast/

# Unit tests
uv run pytest tests/test_library.py
uv run pytest tests/test_rss.py

# Integration tests
uv run pytest tests/test_cli_library.py

# Type checking (if using mypy)
uv run mypy pidcast/

# Full test suite
uv run pytest

# Manual CLI smoke test
uv run pidcast add https://feeds.example.com/test
uv run pidcast list
uv run pidcast show 1
uv run pidcast remove 1
```

## Rollout Considerations

- **Feature flag**: None - this is additive functionality
- **Monitoring**: Log library operations (add, remove) with timestamps
- **Backward compatibility**: Existing single-URL transcription workflow must continue working unchanged
- **Migration**: No migration needed - library.yaml created on first use
- **Rollback plan**: If issues arise, users can continue using direct URL transcription; library commands can be disabled via CLI if needed

## Open Items

- [ ] Should we support OPML import for bulk show additions? (Defer to future)
- [ ] Should artwork URLs be cached locally? (Defer to future)
- [ ] Should we validate feed URLs against known podcast directories? (Defer to future)

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
