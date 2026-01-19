# Implementation Spec: Pidcast Sync - Phase 3

**PRD**: ./prd-phase-3.md
**Estimated Effort**: M (Medium - 4-6 days)

## Technical Approach

Phase 3 adds the synthesis layer on top of the processing pipeline. After sync completes and individual episodes are analyzed, we use LLM-based aggregation to generate three tiers of summaries: one-liners per episode, rollups per show, and cross-show topic clustering.

The approach leverages existing Groq API integration (`analysis.py`) with new prompts for batch summarization. Rather than making one API call per episode for one-liners (expensive and slow), we'll batch 10-20 episodes per request using JSON mode to return structured arrays of summaries. Topic clustering uses a single API call with all episode summaries as input.

One-liners will be cached in the processing history to avoid regenerating on digest re-runs. The digest generator (`digest.py`) will read history entries, fetch cached one-liners or generate missing ones, aggregate by show, and perform topic clustering. Output is both terminal display (using rich for formatting) and markdown file with date prefix.

Key technical decisions:
- Batch API calls: Reduce cost by batching episodes (10-20 per request)
- Cache one-liners in history: Avoid regeneration, enable fast digest re-runs
- Terminal-first UX: Show digest immediately after sync, file is secondary
- Graceful degradation: If clustering fails, still show one-liners and rollups

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `pidcast/digest.py` | Digest generation orchestration and formatting |
| `pidcast/summarization.py` | LLM-based batch summarization and clustering logic |
| `tests/test_digest.py` | Unit tests for digest generation and formatting |
| `tests/test_summarization.py` | Unit tests for LLM summarization calls |
| `config/prompts.yaml` | Add one-liner and clustering prompt templates |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `pidcast/cli.py` | Add `digest` subcommand with --date and --range flags |
| `pidcast/sync.py` | Call digest generation at end of sync (unless --no-digest flag) |
| `pidcast/history.py` | Add `one_liner` field to HistoryEntry for caching |
| `pidcast/config.py` | Add digest output path configuration |
| `config/prompts.yaml` | Add prompts for one-liner generation and topic clustering |

## Implementation Details

### DigestGenerator Class

**Pattern to follow**: Similar to `SyncEngine` orchestration pattern

**Overview**: Orchestrate digest generation from processing history.

```python
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta

@dataclass
class DigestData:
    """Aggregated data for digest generation."""
    episodes: List[HistoryEntry]  # All episodes to include
    one_liners: Dict[str, str]  # guid -> one-liner
    show_rollups: Dict[int, ShowRollup]  # show_id -> rollup
    topic_clusters: List[TopicCluster]

@dataclass
class ShowRollup:
    """Show-level summary."""
    show: Show
    episode_count: int
    episodes: List[HistoryEntry]
    summary: str  # LLM-generated rollup

@dataclass
class TopicCluster:
    """Cross-show topic cluster."""
    topic: str
    episode_count: int
    episodes: List[tuple[Show, HistoryEntry]]  # (show, episode) pairs
    description: str

class DigestGenerator:
    """Generate multi-tier podcast digests."""

    def __init__(
        self,
        library: LibraryManager,
        history: ProcessingHistory,
        summarizer: Summarizer,
        config: dict
    ):
        self.library = library
        self.history = history
        self.summarizer = summarizer
        self.config = config

    def generate_digest(
        self,
        date_filter: Optional[datetime] = None,
        date_range: Optional[timedelta] = None
    ) -> DigestData:
        """
        Generate digest from processing history.

        Args:
            date_filter: Specific date to filter episodes
            date_range: Time range (e.g., last 7 days)

        Returns:
            DigestData with all summary tiers
        """
        # Filter episodes by date
        episodes = self._filter_episodes(date_filter, date_range)

        if not episodes:
            return DigestData([], {}, {}, [])

        # Generate or fetch cached one-liners
        one_liners = self._get_one_liners(episodes)

        # Generate show-level rollups
        show_rollups = self._generate_show_rollups(episodes, one_liners)

        # Generate topic clusters
        topic_clusters = self._generate_topic_clusters(episodes, one_liners)

        return DigestData(episodes, one_liners, show_rollups, topic_clusters)

    def _filter_episodes(
        self,
        date_filter: Optional[datetime],
        date_range: Optional[timedelta]
    ) -> List[HistoryEntry]:
        """Filter successful episodes by date."""
        episodes = [
            e for e in self.history.entries.values()
            if e.status == ProcessingStatus.SUCCESS
        ]

        if date_filter:
            episodes = [
                e for e in episodes
                if e.processed_at.date() == date_filter.date()
            ]
        elif date_range:
            cutoff = datetime.now() - date_range
            episodes = [
                e for e in episodes
                if e.processed_at >= cutoff
            ]

        return sorted(episodes, key=lambda e: e.processed_at, reverse=True)

    def _get_one_liners(self, episodes: List[HistoryEntry]) -> Dict[str, str]:
        """Get or generate one-liners for episodes."""
        one_liners = {}
        missing = []

        # Check cache first
        for episode in episodes:
            if episode.one_liner:
                one_liners[episode.guid] = episode.one_liner
            else:
                missing.append(episode)

        # Generate missing one-liners in batches
        if missing:
            new_liners = self.summarizer.generate_one_liners(missing)
            one_liners.update(new_liners)

            # Update history cache
            for guid, liner in new_liners.items():
                if guid in self.history.entries:
                    self.history.entries[guid].one_liner = liner
            self.history._save()

        return one_liners

    def _generate_show_rollups(
        self,
        episodes: List[HistoryEntry],
        one_liners: Dict[str, str]
    ) -> Dict[int, ShowRollup]:
        """Generate show-level rollups."""
        rollups = {}

        # Group episodes by show
        by_show = {}
        for episode in episodes:
            by_show.setdefault(episode.show_id, []).append(episode)

        # Generate rollup for each show
        for show_id, show_episodes in by_show.items():
            show = self.library.get_show(show_id)
            if not show:
                continue

            # Aggregate one-liners for this show
            show_liners = [one_liners.get(ep.guid, ep.episode_title) for ep in show_episodes]

            # Generate summary
            summary = self.summarizer.generate_show_rollup(show, show_liners)

            rollups[show_id] = ShowRollup(
                show=show,
                episode_count=len(show_episodes),
                episodes=show_episodes,
                summary=summary
            )

        return rollups

    def _generate_topic_clusters(
        self,
        episodes: List[HistoryEntry],
        one_liners: Dict[str, str]
    ) -> List[TopicCluster]:
        """Generate cross-show topic clusters."""
        if len(episodes) < 3:
            return []  # Not enough episodes for meaningful clustering

        # Prepare episode summaries for clustering
        episode_data = [
            {
                'guid': ep.guid,
                'show_id': ep.show_id,
                'title': ep.episode_title,
                'one_liner': one_liners.get(ep.guid, ep.episode_title)
            }
            for ep in episodes
        ]

        try:
            clusters = self.summarizer.generate_topic_clusters(episode_data)

            # Hydrate clusters with full episode objects
            result = []
            for cluster in clusters:
                episodes_in_cluster = []
                for guid in cluster['episode_guids']:
                    episode = next((e for e in episodes if e.guid == guid), None)
                    show = self.library.get_show(episode.show_id) if episode else None
                    if episode and show:
                        episodes_in_cluster.append((show, episode))

                result.append(TopicCluster(
                    topic=cluster['topic'],
                    episode_count=len(episodes_in_cluster),
                    episodes=episodes_in_cluster,
                    description=cluster['description']
                ))

            return result

        except Exception as e:
            print(f"[yellow]Warning: Topic clustering failed: {e}")
            return []  # Graceful degradation
```

**Key decisions**:
- Cache one-liners in history: Avoid redundant LLM calls on digest re-runs
- Graceful degradation: If clustering fails, still return one-liners and rollups
- Date filtering: Support both specific date and date range for flexible digest generation
- Batch processing: Generate one-liners in batches to reduce API calls

**Implementation steps**:
1. Create `DigestData`, `ShowRollup`, `TopicCluster` dataclasses
2. Implement `DigestGenerator` with library, history, summarizer dependencies
3. Implement `_filter_episodes()` with date filtering
4. Implement `_get_one_liners()` with cache checking and batch generation
5. Implement `_generate_show_rollups()` grouping by show
6. Implement `_generate_topic_clusters()` with error handling
7. Add one-liner field to `HistoryEntry` in `history.py`

### Summarizer Class

**Pattern to follow**: Existing `analysis.py` Groq API integration

**Overview**: LLM-based batch summarization for one-liners, rollups, and clustering.

```python
from typing import List, Dict
import yaml
from pidcast.analysis import call_groq_api

class Summarizer:
    """LLM-based summarization for digest generation."""

    def __init__(self, prompts_path: Path):
        with open(prompts_path) as f:
            self.prompts = yaml.safe_load(f)

    def generate_one_liners(self, episodes: List[HistoryEntry]) -> Dict[str, str]:
        """
        Generate one-line summaries for episodes in batch.

        Returns:
            Dict mapping episode GUID to one-liner
        """
        # Load episode transcripts/analyses
        episode_data = []
        for episode in episodes:
            if not episode.output_file:
                continue

            # Read markdown file for analysis content
            analysis_text = self._extract_analysis(episode.output_file)
            episode_data.append({
                'guid': episode.guid,
                'title': episode.episode_title,
                'analysis': analysis_text[:1000]  # Truncate for context limit
            })

        # Batch into groups of 10-20
        batch_size = 15
        all_liners = {}

        for i in range(0, len(episode_data), batch_size):
            batch = episode_data[i:i+batch_size]

            prompt = self.prompts['one_liner_batch'].format(
                episodes=yaml.dump(batch)
            )

            response = call_groq_api(
                prompt=prompt,
                model='llama-3.1-8b-instant',  # Fast model for simple task
                json_mode=True
            )

            # Parse JSON response: [{"guid": "...", "one_liner": "..."}, ...]
            for item in response:
                all_liners[item['guid']] = item['one_liner']

        return all_liners

    def generate_show_rollup(self, show: Show, one_liners: List[str]) -> str:
        """Generate show-level summary from episode one-liners."""
        prompt = self.prompts['show_rollup'].format(
            show_title=show.title,
            episode_count=len(one_liners),
            one_liners='\n'.join(f'- {liner}' for liner in one_liners)
        )

        response = call_groq_api(
            prompt=prompt,
            model='llama-3.1-8b-instant',
            json_mode=False  # Plain text response
        )

        return response.strip()

    def generate_topic_clusters(self, episode_data: List[dict]) -> List[dict]:
        """
        Generate cross-show topic clusters.

        Args:
            episode_data: List of {guid, show_id, title, one_liner}

        Returns:
            List of clusters: [{topic, description, episode_guids}, ...]
        """
        prompt = self.prompts['topic_clustering'].format(
            episode_data=yaml.dump(episode_data)
        )

        response = call_groq_api(
            prompt=prompt,
            model='llama-3.1-70b-versatile',  # More powerful model for clustering
            json_mode=True
        )

        # Expected response: [{"topic": "...", "description": "...", "episode_guids": [...]}]
        return response

    def _extract_analysis(self, markdown_path: str) -> str:
        """Extract analysis section from episode markdown."""
        with open(markdown_path) as f:
            content = f.read()

        # Extract analysis from JSON structure
        import json
        import re

        # Find JSON block in markdown
        match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return data.get('analysis', '')

        return ''
```

**Key decisions**:
- Batch one-liners: 10-20 episodes per API call to balance cost and context limits
- Use fast model (8B) for one-liners: Simple task doesn't need 70B
- Use powerful model (70B) for clustering: Complex reasoning across episodes
- Graceful fallback: If analysis extraction fails, use episode title

**Implementation steps**:
1. Create `Summarizer` class with prompts loading
2. Implement `generate_one_liners()` with batching logic
3. Implement `generate_show_rollup()` with one-liner aggregation
4. Implement `generate_topic_clusters()` with JSON mode parsing
5. Add `_extract_analysis()` to read episode markdown files
6. Add error handling with retries for API failures

### Digest Formatter

**Overview**: Format digest data for terminal display and markdown file output.

```python
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

class DigestFormatter:
    """Format digest data for output."""

    @staticmethod
    def format_terminal(digest: DigestData) -> None:
        """Display digest in terminal with rich formatting."""
        console = Console()

        console.print("\n[bold cyan]Podcast Digest", style="bold")
        console.print(f"[dim]{datetime.now().strftime('%Y-%m-%d')} · {len(digest.episodes)} episodes\n")

        # Topic Clusters
        if digest.topic_clusters:
            console.print("[bold yellow]📌 Topics Across Shows\n")
            for cluster in digest.topic_clusters:
                console.print(Panel(
                    f"[bold]{cluster.topic}[/bold]\n{cluster.description}\n\n"
                    f"Episodes: {cluster.episode_count}",
                    border_style="yellow"
                ))

        # Show Rollups
        console.print("\n[bold green]📻 Shows\n")
        for show_rollup in digest.show_rollups.values():
            console.print(f"[bold cyan]{show_rollup.show.title}[/bold] ({show_rollup.episode_count} episodes)")
            console.print(f"  {show_rollup.summary}\n")

            # Episode one-liners
            for episode in show_rollup.episodes:
                liner = digest.one_liners.get(episode.guid, episode.episode_title)
                console.print(f"  • {liner}")

            console.print()

    @staticmethod
    def format_markdown(digest: DigestData, shows: Dict[int, Show]) -> str:
        """Generate markdown file content."""
        lines = []

        # YAML front matter
        lines.append("---")
        lines.append(f"title: Podcast Digest")
        lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"episode_count: {len(digest.episodes)}")
        lines.append(f"show_count: {len(digest.show_rollups)}")
        lines.append("type: podcast-digest")
        lines.append("---")
        lines.append("")

        # Topic Clusters
        if digest.topic_clusters:
            lines.append("## Topics Across Shows")
            lines.append("")
            for cluster in digest.topic_clusters:
                lines.append(f"### {cluster.topic}")
                lines.append(f"{cluster.description}")
                lines.append(f"**Episodes:** {cluster.episode_count}")
                lines.append("")
                for show, episode in cluster.episodes:
                    liner = digest.one_liners.get(episode.guid, episode.episode_title)
                    link = f"[{episode.episode_title}]({episode.output_file})"
                    lines.append(f"- {show.title}: {liner} {link}")
                lines.append("")

        # Show Rollups
        lines.append("## Shows")
        lines.append("")
        for show_rollup in digest.show_rollups.values():
            lines.append(f"### {show_rollup.show.title}")
            lines.append(f"{show_rollup.summary}")
            lines.append("")
            lines.append(f"**Episodes:** {show_rollup.episode_count}")
            lines.append("")

            for episode in show_rollup.episodes:
                liner = digest.one_liners.get(episode.guid, episode.episode_title)
                link = f"[{episode.episode_title}]({episode.output_file})"
                lines.append(f"- {liner} {link}")

            lines.append("")

        return '\n'.join(lines)
```

**Implementation steps**:
1. Create `DigestFormatter` class with static methods
2. Implement `format_terminal()` using rich for colored output
3. Implement `format_markdown()` with YAML front matter and links
4. Add topic cluster formatting with episode lists
5. Add show rollup formatting with one-liners

### CLI Integration

**Overview**: Add `digest` subcommand and integrate with sync.

```python
def setup_digest_command(subparsers):
    """Add digest subcommand."""
    digest_parser = subparsers.add_parser(
        'digest',
        help='Generate or view podcast digest'
    )
    digest_parser.add_argument(
        '--date',
        help='Generate digest for specific date (YYYY-MM-DD)'
    )
    digest_parser.add_argument(
        '--range',
        help='Generate digest for date range (e.g., 7d, 30d)'
    )

def cmd_digest(args):
    """Handle 'pidcast digest' command."""
    library = LibraryManager.load()
    history = ProcessingHistory(get_history_path())
    summarizer = Summarizer(get_prompts_path())
    config = ConfigManager.load_config()

    generator = DigestGenerator(library, history, summarizer, config)

    # Parse date filters
    date_filter = parse_date(args.date) if args.date else None
    date_range = parse_range(args.range) if args.range else timedelta(days=1)

    digest = generator.generate_digest(date_filter, date_range)

    if not digest.episodes:
        print("No episodes found for specified date range.")
        return

    # Display in terminal
    DigestFormatter.format_terminal(digest)

    # Save to file
    output_path = get_digest_output_path(date_filter or datetime.now())
    markdown = DigestFormatter.format_markdown(digest, library.shows)
    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"\nDigest saved to: {output_path}")

# Modify sync command to generate digest at end
def cmd_sync(args):
    # ... existing sync logic ...

    if not args.no_digest:
        # Generate digest for today's processed episodes
        digest_generator = DigestGenerator(library, history, summarizer, config)
        digest = digest_generator.generate_digest(date_filter=datetime.now())

        if digest.episodes:
            DigestFormatter.format_terminal(digest)

            # Save digest
            output_path = get_digest_output_path(datetime.now())
            markdown = DigestFormatter.format_markdown(digest, library.shows_dict)
            with open(output_path, 'w') as f:
                f.write(markdown)
```

**Implementation steps**:
1. Add `digest` subcommand with --date and --range flags
2. Implement `cmd_digest()` handler
3. Modify `cmd_sync()` to call digest generator at end (unless --no-digest)
4. Add date parsing utilities (parse_date, parse_range)
5. Add digest output path generation (YYYY-MM-DD_podcast-digest.md)

## Data Model

### Prompts Configuration (prompts.yaml additions)

```yaml
# One-liner batch generation
one_liner_batch:
  system: "You are a podcast summarization expert. Generate concise one-line summaries (max 120 characters) for each episode."
  prompt: |
    Generate one-line summaries for these podcast episodes. Each summary should capture the key theme, guest, or topic.

    Episodes:
    {episodes}

    Return JSON array: [{"guid": "...", "one_liner": "..."}, ...]

# Show rollup
show_rollup:
  system: "You are a podcast curator. Synthesize episode summaries into a show-level overview."
  prompt: |
    Podcast Show: {show_title}
    Episode Count: {episode_count}

    Episode Summaries:
    {one_liners}

    Generate a 2-3 sentence summary of what this show covered across these episodes. Identify common themes if present.

# Topic clustering
topic_clustering:
  system: "You are a podcast analyst. Identify recurring topics across multiple shows and episodes."
  prompt: |
    Analyze these podcast episodes and identify 3-7 major themes or topics discussed across shows.

    Episodes:
    {episode_data}

    For each topic cluster:
    - Topic name (short, clear)
    - Description (1-2 sentences explaining the theme)
    - Episode GUIDs that discuss this topic

    Return JSON: [{"topic": "...", "description": "...", "episode_guids": [...]}, ...]
```

### Updated HistoryEntry (history.py)

```python
@dataclass
class HistoryEntry:
    guid: str
    show_id: int
    episode_title: str
    processed_at: datetime
    status: ProcessingStatus
    output_file: Optional[str]
    error_message: Optional[str] = None
    one_liner: Optional[str] = None  # NEW: Cached one-liner summary
```

## Testing Requirements

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_digest.py` | DigestGenerator logic |
| `tests/test_summarization.py` | Summarizer LLM calls |
| `tests/test_digest_formatter.py` | Markdown and terminal formatting |

**Key test cases**:
- Digest generation: filter episodes by date, aggregate by show, topic clustering
- Summarization: batch one-liners, show rollup, topic clustering with mock API
- Formatting: markdown structure, YAML front matter, terminal output
- Caching: one-liners cached in history, reused on digest re-run

**Edge cases**:
- Single episode digest (no clustering)
- Zero episodes for date range
- One-liner generation fails (fallback to title)
- Topic clustering returns empty (graceful degradation)
- Episode markdown file missing (skip that episode)

**Error cases**:
- Groq API rate limit during one-liner generation
- Corrupted episode markdown file
- Invalid date format in --date flag
- Disk full when saving digest markdown

### Integration Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_digest_integration.py` | End-to-end digest generation |

**Key scenarios**:
- Run sync, verify digest generated and displayed
- Run sync with --no-digest, verify no digest created
- Run `pidcast digest` after sync, verify cached one-liners used
- Run `pidcast digest --range 7d`, verify weekly digest

### Manual Testing

- [ ] Run `pidcast sync` and verify digest displayed in terminal
- [ ] Check digest markdown file created in `data/transcripts/`
- [ ] Run `pidcast digest` again and verify fast execution (cached one-liners)
- [ ] Run `pidcast digest --range 7d` and verify weekly aggregation
- [ ] Check topic clustering with diverse episodes
- [ ] Test with single episode (verify no clustering section)
- [ ] Test with episodes from same show (verify show rollup)
- [ ] Verify markdown file has links to individual episode files
- [ ] Test terminal output formatting on narrow terminal (80 columns)

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| One-liner generation fails (API error) | Fallback to episode title, log warning, continue digest |
| Topic clustering fails | Skip clustering section, still show one-liners and rollups, log warning |
| Episode markdown file not found | Skip that episode from digest, log warning |
| Corrupted episode markdown | Skip that episode, log error with file path |
| No episodes for date range | Display "No episodes found" message, don't create empty digest file |
| Groq API rate limit | Retry with exponential backoff, fall back to titles if all retries fail |
| Disk full when saving digest | Display error message, show digest in terminal only |

## Validation Commands

```bash
# Linting
uv run ruff check pidcast/

# Unit tests
uv run pytest tests/test_digest.py
uv run pytest tests/test_summarization.py

# Integration tests
uv run pytest tests/test_digest_integration.py

# Full test suite
uv run pytest

# Manual digest test
uv run pidcast sync  # Should display digest at end
uv run pidcast digest  # Should regenerate from cache (fast)
uv run pidcast digest --range 7d  # Weekly digest
```

## Rollout Considerations

- **Feature flag**: `--no-digest` flag to disable digest generation in sync
- **Monitoring**: Log digest generation (duration, episode count, LLM API calls)
- **Cost optimization**: Batch one-liner generation to minimize API costs
- **Caching**: One-liners cached in history to enable free digest re-runs
- **Graceful degradation**: Each summary tier (one-liners, rollups, clustering) can fail independently without breaking entire digest
- **Rollback plan**: Users can continue using sync without digests via --no-digest flag

## Open Items

- [ ] Should we support custom digest templates (user-defined formatting)? (Defer to future)
- [ ] Should topic clustering use embeddings instead of pure LLM? (Defer to future optimization for cost)
- [ ] Should we add episode duration estimates to digests? (Recommend: yes, pull from RSS metadata)
- [ ] Should digests support filtering by show or topic? (Defer to future enhancement)

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
