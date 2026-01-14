# PRD: Pidcast UX Overhaul - Phase 3

**Contract**: ./contract.md
**Phase**: 3 of 3
**Focus**: Long content handling with semantic chunking and synthesis

## Phase Overview

This phase solves the core problem of 45+ minute videos failing due to rate limits. By splitting transcripts into semantic chunks and synthesizing the results, we can handle arbitrarily long content while staying within API limits.

This is the final phase because it's the most complex and builds on both the prompt system (Phase 1) and model fallback (Phase 2). Chunking will make multiple API calls per video, so robust rate limit handling is essential.

Value delivered: Hour-long podcasts and lectures can be analyzed reliably, producing coherent summaries.

## User Stories

1. As a user, I want 60+ minute videos to be analyzed successfully so that I can triage any content
2. As a user, I want chunked analysis to produce a coherent summary so that the output doesn't feel fragmented
3. As a user, I want to know when content is being chunked so that I understand why it takes longer

## Functional Requirements

### Transcript Chunking

- **FR-3.1**: Detect when transcript exceeds safe token limit for current model
- **FR-3.2**: Split transcripts at semantic boundaries (paragraph breaks, topic shifts)
- **FR-3.3**: Each chunk shall include context overlap (~100 words) to maintain coherence
- **FR-3.4**: Chunks shall target 80% of model's context limit to leave room for prompt

### Chunk Analysis

- **FR-3.5**: Each chunk shall be analyzed independently with chunk-aware prompt
- **FR-3.6**: Chunk prompts shall instruct model this is "part N of M"
- **FR-3.7**: Chunk analysis shall use same model fallback logic as single analysis

### Synthesis

- **FR-3.8**: After all chunks analyzed, synthesize into single coherent output
- **FR-3.9**: Synthesis prompt shall combine chunk summaries into unified summary
- **FR-3.10**: Final key points shall be deduplicated and prioritized across all chunks
- **FR-3.11**: Synthesis shall indicate total content length in output

### Progress Indication

- **FR-3.12**: Display progress during chunked analysis ("Analyzing chunk 2 of 5...")
- **FR-3.13**: Show estimated time remaining based on chunk count
- **FR-3.14**: Log total token usage across all chunks

### Prompts

- **FR-3.15**: Add "chunk_analysis" prompt type to prompts.yaml
- **FR-3.16**: Add "synthesis" prompt type to prompts.yaml
- **FR-3.17**: Chunk prompt shall emphasize extracting key information for later synthesis

## Non-Functional Requirements

- **NFR-3.1**: 60-minute video shall complete analysis in under 3 minutes
- **NFR-3.2**: Semantic chunking shall preserve sentence boundaries (no mid-sentence splits)
- **NFR-3.3**: Memory usage shall not exceed 500MB for any video length

## Dependencies

### Prerequisites

- Phase 1 complete (prompts.yaml system)
- Phase 2 complete (model fallback for handling rate limits across multiple calls)

### Outputs for Next Phase

- None (final phase)

## Acceptance Criteria

- [ ] 60-minute transcript successfully produces coherent summary
- [ ] Chunks split at paragraph/topic boundaries, not mid-sentence
- [ ] Synthesis produces unified summary (not just concatenated chunks)
- [ ] Progress indicator shows "Analyzing chunk N of M"
- [ ] Key points are deduplicated across chunks
- [ ] `config/prompts.yaml` includes chunk_analysis and synthesis prompts
- [ ] Model fallback works correctly across chunked calls

---

*Review this PRD and provide feedback before spec generation.*
