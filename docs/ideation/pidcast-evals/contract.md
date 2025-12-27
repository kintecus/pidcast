# Pidcast-Evals Contract

**Created**: 2025-12-27
**Confidence Score**: 95/100
**Status**: Draft

## Problem Statement

Currently, improving the LLM analysis feature in pidcast requires manual, time-consuming experimentation. When trying different prompts or comparing LLM models, there's no systematic way to evaluate quality or track what works. Each iteration involves:

1. Manually running analysis on a transcript
2. Reading the output and trying to remember how previous versions compared
3. Losing track of which prompt version or model produced which results
4. No reproducible way to validate that changes actually improve quality

This ad-hoc approach slows down iteration velocity and makes it difficult to make data-driven decisions about prompt engineering or model selection. Without a structured eval system, prompt improvements rely on gut feeling rather than systematic comparison.

## Goals

1. **Enable rapid iteration**: Run comprehensive prompt/model comparisons in <5 minutes, allowing quick experimentation cycles
2. **Make quality differences obvious**: Generate side-by-side markdown comparisons that make it immediately clear which prompt or model produces better results
3. **Ensure reproducibility**: Same inputs (transcript + prompt + model) always produce comparable outputs, enabling fair A/B testing
4. **Reduce friction for new experiments**: Adding a new prompt version or testing a new LLM should require minimal configuration changes

## Success Criteria

- [ ] Can run evals comparing 3+ prompt versions against 2+ reference transcripts in under 5 minutes
- [ ] Generated comparison markdown files clearly show differences between outputs in editor/Obsidian
- [ ] Sequential prompt versioning (v1, v2, v3) is stored and tracked in a structured format
- [ ] Reference transcripts representing different content types and lengths are checked into git
- [ ] Eval runs are reproducible: re-running same eval produces comparable results
- [ ] Failed API calls (rate limits, timeouts) automatically retry with backoff and report failures at end
- [ ] Eval results are stored in `data/evals/` with clear organization by run/prompt/model
- [ ] Adding a new Groq model to test requires <5 lines of configuration

## Scope Boundaries

### In Scope

- Standalone eval CLI tool (separate from main pidcast command)
- Sequential prompt versioning system (v1, v2, v3, etc.)
- Support for Groq models initially (llama-3.3-70b, llama-3.1-8b, mixtral, etc.)
- 2-3 curated reference transcripts checked into git (different content types and lengths)
- Systematic eval runner that tests combinations of prompts × models × transcripts
- Side-by-side markdown file comparison output
- Automatic retry with backoff for failed API calls
- Results stored in `data/evals/` directory
- Basic run metadata tracking (timestamp, model, prompt version, costs, tokens)

### Out of Scope

- Web UI for comparison (CLI/markdown files only for now)
- Integration with main pidcast CLI (standalone tool only)
- CI/CD integration or automated regression testing
- Support for non-Groq LLMs (Claude, GPT, Ollama) - deferred to future
- Human evaluation/rating system for outputs
- Statistical analysis or automated quality scoring
- Export to formats other than markdown

### Future Considerations

- Anthropic Claude API support (Claude 3.5 Sonnet, Opus)
- OpenAI GPT support (GPT-4, GPT-3.5)
- Local model support via Ollama
- Automated quality metrics or scoring
- HTML report generation with interactive comparison UI
- GitHub Actions integration for regression testing on prompt changes
- Human evaluation workflow (rating outputs, A/B preference voting)

---

*This contract was generated from brain dump input. Review and approve before proceeding to PRD generation.*
