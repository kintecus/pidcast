# PRD: Pidcast Sync - Phase 1

**Contract**: ./contract.md
**Phase**: 1 of 3
**Focus**: Library Management and RSS Foundation

## Phase Overview

Phase 1 establishes the foundational infrastructure for podcast library management. This phase focuses on giving users the ability to add, list, and remove podcast shows from a persistent library, along with the RSS parsing capabilities needed to fetch episode metadata.

This phase is sequenced first because all subsequent functionality (sync, processing, digests) depends on having a working library system and RSS parser. By completing Phase 1, users can build their podcast library and verify feed connectivity before committing to the heavier processing pipeline.

After Phase 1 completes, users will have a functional CLI for managing their podcast subscriptions, viewing available episodes, and validating RSS feeds—but actual transcription and analysis remain manual (existing workflow) until Phase 2.

Key constraints: Must maintain backward compatibility with existing pidcast URL/file workflow. Library should be human-readable and easily editable outside the CLI.

## User Stories

1. As a podcast listener, I want to add a show to my library by providing its RSS feed URL so that I can build a collection of shows to monitor
2. As a user, I want to view all shows in my library with basic metadata (title, feed URL, episode count) so that I can see what I'm tracking
3. As a user, I want to remove shows from my library so that I can stop tracking podcasts I no longer follow
4. As a user, I want the system to validate RSS feeds when adding shows so that I catch invalid URLs early
5. As a user, I want to preview available episodes from a feed before adding it so that I can confirm it's the right show
6. As a developer, I want library data stored in a human-readable config file so that I can manually edit or back it up easily

## Functional Requirements

### Library Management

- **FR-1.1**: System must support `pidcast add <feed-url>` command to add a podcast RSS feed to the library
- **FR-1.2**: System must parse RSS feed on add to extract show metadata (title, description, author, artwork URL)
- **FR-1.3**: System must validate RSS feed format and connectivity before adding to library (fail gracefully with clear error messages)
- **FR-1.4**: System must prevent duplicate feeds from being added (check if feed URL already exists)
- **FR-1.5**: System must support `pidcast list` command to display all shows in library with key metadata (title, feed URL, episode count, last checked timestamp)
- **FR-1.6**: System must support `pidcast remove <show-id>` command to remove a show from the library
- **FR-1.7**: System must support `pidcast show <show-id>` command to view detailed information about a specific show including recent episodes

### RSS Parsing

- **FR-1.8**: System must parse standard podcast RSS 2.0 feeds with support for iTunes podcast extensions
- **FR-1.9**: System must extract episode metadata: title, description, publication date, duration, audio URL (enclosure)
- **FR-1.10**: System must handle common RSS edge cases: missing fields, multiple enclosures, malformed HTML in descriptions
- **FR-1.11**: System must respect HTTP caching headers (ETag, Last-Modified) to avoid unnecessary re-fetching
- **FR-1.12**: System must support `--preview` flag on add command to show recent episodes before confirming addition

### Persistent Storage

- **FR-1.13**: Library must be stored in YAML format at `~/.config/pidcast/library.yaml` (or XDG_CONFIG_HOME equivalent)
- **FR-1.14**: Each show entry must include: unique ID, title, feed URL, added timestamp, last checked timestamp, backfill count (defaults to global config)
- **FR-1.15**: System must create config directory and file on first use with sensible defaults
- **FR-1.16**: Library file must be human-readable and editable with comments explaining key fields
- **FR-1.17**: Global configuration (backfill limit, output paths) stored in `~/.config/pidcast/config.yaml`

## Non-Functional Requirements

- **NFR-1.1**: RSS feed parsing must complete within 5 seconds for typical feeds (<500 episodes) under normal network conditions
- **NFR-1.2**: Library operations (add, list, remove) must complete in under 1 second for libraries with up to 50 shows
- **NFR-1.3**: System must handle network failures gracefully with retry logic (3 retries with exponential backoff) and user-friendly error messages
- **NFR-1.4**: Configuration files must use UTF-8 encoding and support international characters (podcast titles in any language)
- **NFR-1.5**: CLI commands must provide clear help text and usage examples (`pidcast add --help`)

## Dependencies

### Prerequisites

- Existing pidcast codebase with CLI infrastructure (`cli.py`)
- Python RSS parsing library (recommend `feedparser`)
- YAML library for config persistence (recommend `pyyaml` or `ruamel.yaml`)
- Existing utils for file handling and logging

### Outputs for Next Phase

- `library.py` module with Show model and library CRUD operations
- `rss.py` module with RSS feed parsing and episode extraction
- `config.yaml` with global settings (backfill limit, paths)
- `library.yaml` with show collection (may be empty or populated by user)

## Acceptance Criteria

- [ ] User can add a podcast RSS feed with `pidcast add <url>` and see confirmation with show title
- [ ] User can run `pidcast add --preview <url>` to see last 5 episodes before confirming
- [ ] System validates RSS feeds and rejects invalid URLs with helpful error messages
- [ ] User can list all shows with `pidcast list` showing title, episode count, and last checked date
- [ ] User can view detailed show info with `pidcast show <id>` including recent episodes
- [ ] User can remove a show with `pidcast remove <id>` and see confirmation
- [ ] Library persists across CLI sessions in `~/.config/pidcast/library.yaml`
- [ ] Library file is human-readable YAML with comments explaining structure
- [ ] System prevents adding the same feed twice (duplicate detection)
- [ ] System handles network errors gracefully (timeouts, DNS failures, 404s) with retry logic
- [ ] Global config file created with sensible defaults on first run
- [ ] All existing pidcast functionality (single URL/file transcription) continues to work unchanged
- [ ] Unit tests for RSS parsing edge cases (missing fields, malformed data)
- [ ] Integration tests for library operations (add, list, remove, show)

## Open Questions

- Should show IDs be auto-incrementing integers or UUIDs? (Recommend: simple integers for CLI usability)
- How should we handle feed URLs that redirect to different URLs? (Recommend: follow redirects, store canonical URL)
- Should `pidcast add` without preview be the default, or require confirmation? (Recommend: default immediate add, preview is optional)

---

*Review this PRD and provide feedback before spec generation.*
