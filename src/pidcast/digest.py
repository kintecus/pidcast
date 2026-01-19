"""Digest generation for podcast sync."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .history import HistoryEntry, ProcessingHistory, ProcessingStatus
from .library import LibraryManager, Show
from .summarization import Summarizer

logger = logging.getLogger(__name__)


@dataclass
class ShowRollup:
    """Show-level summary."""

    show: Show
    episode_count: int
    episodes: list[HistoryEntry]
    summary: str


@dataclass
class TopicCluster:
    """Cross-show topic cluster."""

    topic: str
    episode_count: int
    episodes: list[tuple[Show, HistoryEntry]]
    description: str


@dataclass
class DigestData:
    """Aggregated data for digest generation."""

    episodes: list[HistoryEntry]
    one_liners: dict[str, str]
    show_rollups: dict[int, ShowRollup]
    topic_clusters: list[TopicCluster]


class DigestGenerator:
    """Generate multi-tier podcast digests."""

    def __init__(
        self,
        library: LibraryManager,
        history: ProcessingHistory,
        summarizer: Summarizer,
    ):
        """Initialize DigestGenerator.

        Args:
            library: Library manager instance
            history: Processing history instance
            summarizer: Summarizer instance
        """
        self.library = library
        self.history = history
        self.summarizer = summarizer

    def generate_digest(
        self,
        date_filter: datetime | None = None,
        date_range: timedelta | None = None,
    ) -> DigestData:
        """Generate digest from processing history.

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
        date_filter: datetime | None,
        date_range: timedelta | None,
    ) -> list[HistoryEntry]:
        """Filter successful episodes by date.

        Args:
            date_filter: Specific date to filter
            date_range: Time range to filter

        Returns:
            Filtered and sorted list of episodes
        """
        episodes = [
            e for e in self.history.entries.values() if e.status == ProcessingStatus.SUCCESS
        ]

        if date_filter:
            episodes = [e for e in episodes if e.processed_at.date() == date_filter.date()]
        elif date_range:
            cutoff = datetime.now() - date_range
            episodes = [e for e in episodes if e.processed_at >= cutoff]

        return sorted(episodes, key=lambda e: e.processed_at, reverse=True)

    def _get_one_liners(self, episodes: list[HistoryEntry]) -> dict[str, str]:
        """Get or generate one-liners for episodes.

        Args:
            episodes: List of episodes to process

        Returns:
            Dict mapping GUID to one-liner
        """
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
            logger.info(f"Generating one-liners for {len(missing)} episodes...")
            try:
                new_liners = self.summarizer.generate_one_liners(missing)
                one_liners.update(new_liners)

                # Update history cache
                for guid, liner in new_liners.items():
                    if guid in self.history.entries:
                        self.history.entries[guid].one_liner = liner
                self.history._save()

            except Exception as e:
                logger.error(f"Failed to generate one-liners: {e}")
                # Fallback to episode titles
                for episode in missing:
                    one_liners[episode.guid] = episode.episode_title

        return one_liners

    def _generate_show_rollups(
        self,
        episodes: list[HistoryEntry],
        one_liners: dict[str, str],
    ) -> dict[int, ShowRollup]:
        """Generate show-level rollups.

        Args:
            episodes: List of episodes
            one_liners: Dict of one-liners

        Returns:
            Dict mapping show ID to rollup
        """
        rollups = {}

        # Group episodes by show
        by_show: dict[int, list[HistoryEntry]] = {}
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
            try:
                summary = self.summarizer.generate_show_rollup(show.title, show_liners)
            except Exception as e:
                logger.error(f"Failed to generate rollup for {show.title}: {e}")
                summary = f"{len(show_episodes)} new episodes"

            rollups[show_id] = ShowRollup(
                show=show,
                episode_count=len(show_episodes),
                episodes=show_episodes,
                summary=summary,
            )

        return rollups

    def _generate_topic_clusters(
        self,
        episodes: list[HistoryEntry],
        one_liners: dict[str, str],
    ) -> list[TopicCluster]:
        """Generate cross-show topic clusters.

        Args:
            episodes: List of episodes
            one_liners: Dict of one-liners

        Returns:
            List of topic clusters
        """
        if len(episodes) < 3:
            return []  # Not enough episodes for meaningful clustering

        # Prepare episode summaries for clustering
        episode_data = [
            {
                "guid": ep.guid,
                "show_id": ep.show_id,
                "title": ep.episode_title,
                "one_liner": one_liners.get(ep.guid, ep.episode_title),
            }
            for ep in episodes
        ]

        try:
            clusters = self.summarizer.generate_topic_clusters(episode_data)

            # Hydrate clusters with full episode objects
            result = []
            for cluster in clusters:
                episodes_in_cluster = []
                for guid in cluster.get("episode_guids", []):
                    episode = next((e for e in episodes if e.guid == guid), None)
                    if episode:
                        show = self.library.get_show(episode.show_id)
                        if show:
                            episodes_in_cluster.append((show, episode))

                if episodes_in_cluster:
                    result.append(
                        TopicCluster(
                            topic=cluster.get("topic", "Unknown Topic"),
                            episode_count=len(episodes_in_cluster),
                            episodes=episodes_in_cluster,
                            description=cluster.get("description", ""),
                        )
                    )

            return result

        except Exception as e:
            logger.warning(f"Topic clustering failed: {e}")
            return []  # Graceful degradation


class DigestFormatter:
    """Format digest data for output."""

    @staticmethod
    def format_terminal(digest: DigestData) -> None:
        """Display digest in terminal with rich formatting.

        Args:
            digest: Digest data to format
        """
        console = Console()

        console.print("\n[bold cyan]Podcast Digest[/bold cyan]")
        console.print(
            f"[dim]{datetime.now().strftime('%Y-%m-%d')} · {len(digest.episodes)} episodes[/dim]\n"
        )

        # Topic Clusters
        if digest.topic_clusters:
            console.print("[bold yellow]Topics Across Shows[/bold yellow]\n")
            for cluster in digest.topic_clusters:
                panel_content = (
                    f"[bold]{cluster.topic}[/bold]\n{cluster.description}\n\n"
                    f"Episodes: {cluster.episode_count}"
                )
                console.print(Panel(panel_content, border_style="yellow"))
                console.print()

        # Show Rollups
        console.print("[bold green]Shows[/bold green]\n")
        for show_rollup in digest.show_rollups.values():
            console.print(
                f"[bold cyan]{show_rollup.show.title}[/bold cyan] ({show_rollup.episode_count} episodes)"
            )
            console.print(f"  {show_rollup.summary}\n")

            # Episode one-liners
            for episode in show_rollup.episodes:
                liner = digest.one_liners.get(episode.guid, episode.episode_title)
                console.print(f"  • {liner}")

            console.print()

    @staticmethod
    def format_markdown(digest: DigestData) -> str:
        """Generate markdown file content.

        Args:
            digest: Digest data to format

        Returns:
            Markdown content
        """
        lines = []

        # YAML front matter
        lines.append("---")
        lines.append("title: Podcast Digest")
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
                    if episode.output_file:
                        link = f"[{episode.episode_title}]({Path(episode.output_file).name})"
                    else:
                        link = episode.episode_title
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
                if episode.output_file:
                    link = f"[{episode.episode_title}]({Path(episode.output_file).name})"
                else:
                    link = episode.episode_title
                lines.append(f"- {liner} {link}")

            lines.append("")

        return "\n".join(lines)
