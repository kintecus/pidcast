# PRD: Pidcast Sync - Phase 3

**Contract**: ./contract.md
**Phase**: 3 of 3
**Focus**: Digest Generation and Holistic Summaries

## Phase Overview

Phase 3 delivers the high-level synthesis feature that transforms pidcast from an episode processor into an intelligent content curator. This phase introduces multi-tier digest generation that aggregates insights across all processed episodes, providing episode one-liners, show-level rollups, and cross-show topic clustering.

This phase is sequenced last because it depends on having processed episode data (from Phase 2) to synthesize. The digest generation reads individual episode analyses and uses LLM-based summarization to extract patterns, themes, and highlights across the user's entire podcast consumption.

After Phase 3 completes, users get the "pre-listen" experience described in the original vision: run `pidcast sync`, receive a concise digest showing what's new across all shows, scan one-liners to identify priority episodes, and spot recurring themes without listening to hours of audio. The digest is both displayed in terminal for quick scanning and saved to a dated markdown file for reference.

Key design choice: Digests are generated *after* all episodes are processed (end of sync), not incrementally. This allows for cross-episode analysis and topic clustering that requires seeing the full batch of new content.

## User Stories

1. As a podcast listener, I want to see a concise summary of all new episodes at the end of sync so that I can quickly scan what's available without reading full analyses
2. As a user, I want episode one-liners that capture the key theme or topic so that I can decide which episodes to prioritize
3. As a user, I want episodes grouped by show so that I can see what's new from each podcast I follow
4. As a user, I want cross-show topic clustering so that I can identify recurring themes across my library (e.g., "5 episodes discussed AI regulation this week")
5. As a user, I want the digest saved to a dated markdown file so that I can reference it later or track trends over time
6. As a user, I want the digest displayed in my terminal after sync completes so that I get immediate visibility without opening files

## Functional Requirements

### Digest Generation

- **FR-3.1**: System must generate a digest after sync completes that includes all newly processed episodes
- **FR-3.2**: Digest must include three summary tiers: episode one-liners, show-level rollups, and cross-show topic clusters
- **FR-3.3**: System must use LLM (Groq API) to generate concise one-line summaries for each episode (max 120 characters)
- **FR-3.4**: System must group episodes by show and generate show-level rollup (e.g., "Lex Fridman: 3 new episodes covering AI, physics, startups")
- **FR-3.5**: System must analyze all episode summaries to extract cross-show topics and cluster episodes by theme
- **FR-3.6**: System must support `--no-digest` flag on sync command to skip digest generation (for users who only want individual files)

### Episode One-Liners

- **FR-3.7**: One-liner must be a single sentence capturing the episode's key theme, guest, or topic
- **FR-3.8**: One-liner must include episode title and show name for context
- **FR-3.9**: One-liner prompt must use existing Groq API integration with JSON mode for structured output
- **FR-3.10**: System must gracefully handle one-liner generation failures (network issues, LLM errors) by falling back to episode title

### Show-Level Rollups

- **FR-3.11**: Rollup must include show name, episode count, and synthesized summary of topics covered
- **FR-3.12**: Rollup must list all episode titles for that show with their one-liners
- **FR-3.13**: Rollup should identify if multiple episodes share a common theme (e.g., "All three episodes feature AI researchers")

### Topic Clustering

- **FR-3.14**: System must use LLM to identify 3-7 major themes across all new episodes
- **FR-3.15**: Each cluster must include: topic name, episode count, and list of relevant episodes
- **FR-3.16**: Clusters should surface cross-show patterns (e.g., "Crypto regulation discussed in 3 shows: Bankless, Unchained, All-In")
- **FR-3.17**: System must handle cases where no clear clusters emerge (few episodes, diverse topics) by skipping this section

### Output Formatting

- **FR-3.18**: Digest must be saved to `data/transcripts/YYYY-MM-DD_podcast-digest.md` with date prefix
- **FR-3.19**: Digest markdown must include YAML front matter with sync metadata: date, episode count, show count, processing duration
- **FR-3.20**: Digest must be printed to terminal in a readable format with clear section headings
- **FR-3.21**: Terminal output must use color/formatting (if supported) to highlight show names and topics
- **FR-3.22**: Digest must include links to individual episode markdown files for deeper reading

### Digest Command

- **FR-3.23**: System must support `pidcast digest` command to regenerate digest from recent processing history without re-running sync
- **FR-3.24**: `pidcast digest --date YYYY-MM-DD` should regenerate digest for a specific date's processed episodes
- **FR-3.25**: `pidcast digest --range 7d` should generate a weekly digest aggregating last 7 days of episodes

## Non-Functional Requirements

- **NFR-3.1**: Digest generation must complete within 30 seconds for batches up to 20 episodes
- **NFR-3.2**: LLM API calls for one-liners and clustering must be batched to minimize API requests and costs (target: 1 API call per summary tier)
- **NFR-3.3**: Digest markdown must be human-readable and follow consistent formatting for easy scanning
- **NFR-3.4**: Terminal output must gracefully handle narrow terminals (80 columns) without breaking formatting
- **NFR-3.5**: System must cache one-liners in processing history to avoid regenerating on digest re-runs

## Dependencies

### Prerequisites

- Phase 2 complete: sync pipeline, processing history, individual episode analyses
- Existing `analysis.py` with Groq API integration and JSON mode
- Individual episode markdown files with LLM analysis results

### Outputs for Next Phase

- `digest.py` module with digest generation and formatting logic
- `prompts.yaml` entries for one-liner and clustering prompts
- Dated digest markdown files in `data/transcripts/`
- (No next phase - this completes the contract scope)

## Acceptance Criteria

- [ ] User sees digest printed to terminal at end of `pidcast sync` showing all new episodes
- [ ] Digest includes episode one-liners (single sentence summaries) for each processed episode
- [ ] Digest includes show-level rollups grouping episodes by show with synthesized summary
- [ ] Digest includes cross-show topic clustering identifying 3-7 major themes across all episodes
- [ ] Digest is saved to `data/transcripts/YYYY-MM-DD_podcast-digest.md` with date prefix
- [ ] Digest markdown includes YAML front matter with sync metadata (date, counts, duration)
- [ ] Digest markdown includes links to individual episode analysis files
- [ ] Terminal output uses clear formatting with section headings and readable layout
- [ ] User can skip digest with `pidcast sync --no-digest` (only generates individual files)
- [ ] User can run `pidcast digest` to regenerate digest from recent history without re-syncing
- [ ] User can generate weekly digest with `pidcast digest --range 7d` aggregating last week's episodes
- [ ] One-liner generation failures fallback gracefully to episode titles without breaking digest
- [ ] Topic clustering handles edge cases (no episodes, single episode, highly diverse topics) without errors
- [ ] LLM API calls are batched to minimize cost (not one API call per episode)
- [ ] One-liners are cached in processing history to avoid redundant LLM calls on digest re-runs
- [ ] Integration tests for digest generation with various episode counts and topic distributions
- [ ] Unit tests for fallback logic (LLM failures, network errors)

## Open Questions

- Should topic clustering use embeddings + clustering algorithms or pure LLM-based synthesis? (Recommend: LLM-based for MVP simplicity, embeddings for future optimization)
- Should digests include episode duration estimates to help users prioritize? (Recommend: yes, pull from RSS metadata)
- Should we support custom digest templates or formatting preferences? (Recommend: defer to future, single format initially)

---

*Review this PRD and provide feedback before spec generation.*
