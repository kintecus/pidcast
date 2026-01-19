# Pidcast Sync Contract

**Created**: 2026-01-19
**Confidence Score**: 96/100
**Status**: Draft

## Problem Statement

Podcast listeners who follow multiple shows face a time-intensive manual workflow to stay current with new content. Currently, users must:
1. Manually check each podcast feed for new episodes
2. Decide which episodes warrant their limited listening time
3. Invest hours listening to full episodes to extract key insights
4. Potentially miss important episodes due to information overload

This creates a poor signal-to-noise ratio where valuable content is buried in hours of audio. For users who follow 5-10+ active shows releasing multiple episodes weekly, this becomes unsustainable. The result is either falling behind on content, listening inefficiently, or abandoning shows entirely.

The existing pidcast tool solves transcription and analysis for *individual* episodes but requires manual URL/file input. There's no automated discovery, library management, or cross-episode synthesis capability.

## Goals

1. **Automated Episode Discovery**: System monitors RSS feeds and detects new episodes from followed shows without manual intervention
2. **Library Management**: Users can add/remove shows to a persistent library with configurable catch-up settings (e.g., "process last 5 episodes on add")
3. **Batch Processing Pipeline**: Single command (`pidcast sync`) downloads, transcribes, and analyzes all new episodes from library shows
4. **Holistic Summarization**: Generate cross-show digests with episode one-liners, show-level rollups, and topic clustering to surface trends and priorities
5. **Preserve Existing Workflow**: Individual episode analysis files remain available while adding aggregate summaries for quick scanning

## Success Criteria

- [ ] User can add a podcast RSS feed to their library with a simple command (e.g., `pidcast add <feed-url>`)
- [ ] Library persists across sessions in a human-readable config file (YAML/JSON)
- [ ] `pidcast sync` command checks all library feeds, detects new episodes, and processes them end-to-end
- [ ] On initial add, system processes configurable number of recent episodes (global setting like `backfill: 5`)
- [ ] System tracks which episodes have been processed to avoid re-transcription
- [ ] Digest file is generated after sync with three summary levels: episode one-liners, show groupings, and cross-show topic clusters
- [ ] Digest is displayed in terminal AND saved to markdown file with date prefix (e.g., `2026-01-19_podcast-digest.md`)
- [ ] Individual episode analysis markdown files are still generated per existing workflow
- [ ] User can list, view, and remove shows from their library
- [ ] System handles RSS parsing errors gracefully (invalid feeds, network issues)

## Scope Boundaries

### In Scope

- RSS/XML feed parsing for podcast episodes
- Library management (add, list, remove shows) stored in local config file
- Episode tracking to prevent duplicate processing
- Manual sync command (`pidcast sync`) to check and process new episodes
- Configurable backfill limit when adding new shows (global setting)
- Three-tier digest generation: episode one-liners, show rollups, topic clustering
- Terminal output + markdown file output for digests
- Preserve existing per-episode analysis workflow
- Basic show metadata storage (title, feed URL, last checked timestamp)

### Out of Scope

- Podcast directory API integration (Apple Podcasts, Spotify) - start with direct RSS only
- Automated scheduling (cron jobs, daemon/watch mode) - manual sync only for MVP
- Per-show backfill configuration - global setting only initially
- Smart pre-filtering or interest-based ranking - process all new episodes automatically
- Web UI or dashboard - CLI only
- Episode playback or audio streaming
- User authentication or multi-user support
- Cloud sync or cross-device library sharing

### Future Considerations

- **Smart pre-filtering**: LLM reads episode descriptions and only fully processes episodes likely to match user interests based on learned preferences
- **Interest-based priority scoring**: Rank episodes by relevance and surface most interesting first
- **Podcast directory integration**: Search and subscribe via Apple Podcasts/Spotify APIs rather than manual RSS URLs
- **Automated scheduling**: Cron-based or daemon mode for hands-off operation
- **Per-show configuration**: Individual backfill limits, processing rules, or priority levels per show
- **Notification system**: Alert when digest is ready or high-priority episodes detected
- **Obsidian daily note integration**: Append digests directly to daily notes rather than standalone files
- **Episode recommendation engine**: Suggest new shows based on topics from existing library

---

*This contract was generated from brain dump input. Review and approve before proceeding to PRD generation.*
