# Pidcast-Evals: LLM Analysis Evaluation System

This directory contains the complete implementation plan for the pidcast-evals system - a tool for systematically evaluating and comparing different prompts and LLM models for transcript analysis.

## Documents

1. **[contract.md](./contract.md)** - Lean contract defining the problem, goals, success criteria, and scope
2. **[prd-phase-1.md](./prd-phase-1.md)** - Phase 1 requirements: Foundation infrastructure
3. **[prd-phase-2.md](./prd-phase-2.md)** - Phase 2 requirements: Matrix runner and comparisons
4. **[prd-phase-3.md](./prd-phase-3.md)** - Phase 3 requirements: Robustness and reliability
5. **[spec-phase-1.md](./spec-phase-1.md)** - Phase 1 implementation spec
6. **[spec-phase-2.md](./spec-phase-2.md)** - Phase 2 implementation spec
7. **[spec-phase-3.md](./spec-phase-3.md)** - Phase 3 implementation spec

## Quick Start Guide

### Read First
1. Start with **contract.md** to understand the overall vision
2. Review the phase PRDs in order to understand requirements
3. Dive into implementation specs when ready to code

### Implementation Order
Follow the phases sequentially:

**Phase 1** (2-3 days): Build foundation
- Prompt versioning system
- Reference transcript management
- Basic single-eval runner
- Results storage

**Phase 2** (2-3 days): Add comparison capabilities
- Matrix eval runner (prompts × models × transcripts)
- Side-by-side comparison generation
- Batch processing with progress tracking
- <5 minute comprehensive eval runs

**Phase 3** (1-2 days): Production hardening
- Retry logic with exponential backoff
- Error handling and reporting
- Cost tracking and confirmation
- Resume capability for interrupted batches

### Key Files Created

After full implementation, the project structure will include:

```
pidcast/
├── src/pidcast/evals/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point
│   ├── prompt_manager.py         # Prompt versioning
│   ├── reference_transcripts.py  # Reference transcript registry
│   ├── runner.py                 # Single eval runner
│   ├── batch_runner.py           # Matrix eval orchestration
│   ├── comparison.py             # Comparison generation
│   ├── results.py                # Results storage
│   ├── retry.py                  # Retry decorator
│   ├── cost_tracker.py           # Cost tracking
│   └── validation.py             # Pre-flight validation
├── config/
│   ├── eval_prompts.json         # Versioned prompts
│   └── reference_transcripts.json # Reference transcript registry
├── data/evals/
│   ├── references/               # Reference transcript .md files
│   ├── runs/                     # Individual eval results
│   ├── batches/                  # Batch run summaries
│   ├── comparisons/              # Side-by-side comparisons
│   └── cost_tracking.json        # Cost log
└── docs/ideation/pidcast-evals/  # This directory
```

## Usage Examples

### Phase 1: Single Eval
```bash
uv run pidcast-eval \
  --prompt_version v1 \
  --model llama-3.3-70b-versatile \
  --transcript_id tech-talk-01
```

### Phase 2: Matrix Eval
```bash
# Run all combinations
uv run pidcast-eval --run-matrix

# Run specific subset
uv run pidcast-eval --run-matrix \
  --prompts v1,v2,v3 \
  --models llama-3.3-70b,mixtral-8x7b \
  --transcripts tech-talk-01,interview-02
```

### Phase 3: Resume & Cost Tracking
```bash
# Resume interrupted batch
uv run pidcast-eval --resume-batch 20251227_143500_matrix

# View cost summary
uv run pidcast-eval --cost-summary

# Skip confirmation for automation
uv run pidcast-eval --run-matrix --skip-confirmation
```

## Success Criteria (from Contract)

- [ ] Run evals comparing 3+ prompt versions against 2+ reference transcripts in <5 minutes
- [ ] Generated comparison markdown files clearly show differences in editor/Obsidian
- [ ] Sequential prompt versioning (v1, v2, v3) is tracked in structured format
- [ ] Reference transcripts representing different content types and lengths are in git
- [ ] Eval runs are reproducible: same inputs produce comparable results
- [ ] Failed API calls automatically retry with backoff
- [ ] Eval results stored in `data/evals/` with clear organization
- [ ] Adding a new Groq model requires <5 lines of configuration

## Next Steps

1. **Review and approve** this plan
2. **Start with Phase 1** - build the foundation
3. **Validate each phase** before moving to the next
4. **Iterate on prompts** once the system is working

## Questions or Feedback

This plan was generated using the ideation workflow. If you need adjustments:
- Review the contract and PRDs
- Identify specific changes needed
- Update the relevant document
- Proceed with implementation

---

**Generated**: 2025-12-27
**Status**: Ready for implementation
