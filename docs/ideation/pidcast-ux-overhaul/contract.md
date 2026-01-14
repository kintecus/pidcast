# Pidcast UX Overhaul Contract

**Created**: 2026-01-14
**Confidence Score**: 96/100
**Status**: Draft

## Problem Statement

The current pidcast CLI has friction that slows down the primary use case: quickly triaging YouTube videos before deciding to watch them. The user browses YouTube, finds an interesting video, and wants a fast AI summary to decide if it's worth their time—all without leaving the terminal.

Current pain points:
1. **Too many flags**: The CLI requires explicit `--analyze` flag and other options for what should be the default workflow
2. **Unreliable on long content**: Videos 45+ minutes consistently fail due to rate limits, with no graceful recovery
3. **Brittle model selection**: When the preferred model hits rate limits, the fallback behavior is unpredictable
4. **Confusing prompt management**: Prompts are stored in JSON with auto-generation logic that's hard to understand and modify

## Goals

1. **Simplified default workflow**: Running `pidcast URL` should analyze and display results in terminal with zero additional flags
2. **Reliable long-content handling**: 45+ minute videos should succeed via semantic chunking with synthesis
3. **Smart model fallback**: Quality-prioritized fallback chain (gptoss120b → compound → llama-3.3-70b → llama-3.1-8b) that handles rate limits gracefully
4. **Cleaner prompt management**: Single YAML file with human-readable prompts, no auto-generation magic
5. **Better default output**: Executive summary + key points as default analysis type

## Success Criteria

- [ ] `pidcast URL` (no flags) produces executive summary + key points in terminal
- [ ] 60-minute video transcripts are successfully analyzed without rate limit failures
- [ ] Model fallback occurs automatically with clear logging when rate limits are hit
- [ ] All prompts live in a single `prompts.yaml` file with no auto-generation code
- [ ] Chunked analysis produces coherent synthesized output (not just concatenated chunks)
- [ ] Existing flags still work for power users who want different behavior

## Scope Boundaries

### In Scope

- Simplified CLI defaults (analyze by default, terminal output)
- Model fallback chain with quality priority
- Semantic transcript chunking for long videos
- Chunk synthesis for coherent output
- Prompt migration from JSON to YAML
- Removal of prompt auto-generation logic
- New default analysis type (exec summary + key points)
- Dedicated prompts for chunked analysis

### Out of Scope

- New LLM providers (staying with Groq) - complexity not needed now
- Evals system changes - separate project, already implemented
- Obsidian integration changes - works fine as-is
- Web UI or non-CLI interfaces - out of project scope
- Video download improvements - not the current pain point

### Future Considerations

- Resume interrupted analyses
- Caching analyzed results
- Batch processing multiple URLs
- Cost tracking for main tool (currently only in evals)

---

*This contract was generated from brain dump input. Review and approve before proceeding to PRD generation.*
