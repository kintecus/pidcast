# Pidcast UX Overhaul

Simplify the pidcast CLI for the primary use case: quickly triaging YouTube videos before watching.

## Problem

The current CLI requires too many flags for common workflows, fails on long videos, and has confusing prompt management.

## Solution

Three phases of improvements:

### Phase 1: Simplified Defaults (M)
- `pidcast URL` analyzes and shows results in terminal (no flags needed)
- New `executive_summary` as default analysis type
- Migrate prompts from JSON to YAML
- Remove auto-generation magic

### Phase 2: Smart Model Fallback (M)
- Quality-prioritized model chain: gptoss120b → compound → gpt20b → llama70b → llama8b
- Automatic fallback on rate limits
- Retry with exponential backoff

### Phase 3: Long Content Handling (L)
- Semantic chunking for 45+ minute videos
- Chunk analysis with synthesis
- Progress indicators

## Quick Start

After implementation:

```bash
# Quick triage (default: analyze + terminal output)
pidcast "https://youtube.com/watch?v=VIDEO_ID"

# Skip analysis (transcription only)
pidcast "VIDEO_URL" --no-analyze

# Save to file
pidcast "VIDEO_URL" --save
```

## Artifacts

- `contract.md` - Project scope and success criteria
- `prd-phase-{1,2,3}.md` - Requirements for each phase
- `spec-phase-{1,2,3}.md` - Implementation specifications

## Status

- [x] Contract approved
- [x] PRDs approved
- [x] Specs generated
- [ ] Phase 1 implementation
- [ ] Phase 2 implementation
- [ ] Phase 3 implementation
