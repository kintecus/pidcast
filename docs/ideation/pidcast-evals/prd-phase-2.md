# PRD: Pidcast-Evals - Phase 2

**Contract**: ./contract.md
**Phase**: 2 of 3
**Focus**: Comparison & Scaling - Matrix eval runner and side-by-side comparison generation

## Phase Overview

Phase 2 transforms the eval system from a basic single-run tool into a powerful comparison engine. This phase enables the core value proposition: running multiple prompts and models against reference transcripts simultaneously and generating clear side-by-side comparisons.

After this phase, you'll be able to:
- Run matrix evals testing all combinations of prompts × models × transcripts
- Generate side-by-side markdown comparison files showing differences
- Quickly identify which prompt versions or models produce better results
- Complete comprehensive eval runs in <5 minutes

This phase is sequenced second because it depends on the stable data structures and single-run workflow from Phase 1. Now that we can reliably run one eval, we can scale to batch processing and build comparison tools. This is where the system becomes genuinely useful for rapid iteration.

Key value: After Phase 2, the eval system delivers on the promise of "<5 minute comprehensive evals with obvious quality differences."

## User Stories

1. As a prompt engineer, I want to test multiple prompt versions against the same transcript so that I can quickly identify improvements
2. As a researcher, I want to compare different LLM models on identical inputs so that I can select the best model for the task
3. As a developer, I want to run full eval matrices (all prompts × all models × all transcripts) so that I can comprehensively validate changes
4. As a user, I want side-by-side markdown comparisons so that I can visually spot quality differences in my editor

## Functional Requirements

### Matrix Eval Runner

- **FR-2.1**: CLI SHALL accept `--run-matrix` flag to run all combinations of prompts × models × transcripts
- **FR-2.2**: CLI SHALL accept `--prompts` parameter to specify subset of prompt versions (e.g., `--prompts v1,v2,v3`)
- **FR-2.3**: CLI SHALL accept `--models` parameter to specify subset of models (e.g., `--models llama-3.3-70b,mixtral-8x7b`)
- **FR-2.4**: CLI SHALL accept `--transcripts` parameter to specify subset of transcript IDs (e.g., `--transcripts tech-talk-01,interview-02`)
- **FR-2.5**: If no subsets specified with `--run-matrix`, system SHALL run ALL registered prompts × models × transcripts
- **FR-2.6**: System SHALL display progress during matrix runs (e.g., "Running 2/12: v2 + llama-3.3-70b + interview-02")
- **FR-2.7**: System SHALL track total estimated time and cost before starting matrix run
- **FR-2.8**: System SHALL show running total of tokens and cost as evals complete

### Side-by-Side Comparison Generation

- **FR-2.9**: System SHALL generate comparison files in `data/evals/comparisons/{comparison_id}/`
- **FR-2.10**: Comparison ID format SHALL be: `{timestamp}_{comparison_type}` (e.g., `20251227_prompt_comparison`, `20251227_model_comparison`)
- **FR-2.11**: For prompt comparisons (same model + transcript, different prompts), system SHALL generate markdown with sections for each prompt version showing output side-by-side
- **FR-2.12**: For model comparisons (same prompt + transcript, different models), system SHALL generate markdown with sections for each model showing output side-by-side
- **FR-2.13**: Comparison markdown SHALL use horizontal rules (---) to clearly separate outputs
- **FR-2.14**: Each output section SHALL include a header showing: prompt version OR model name, tokens used, estimated cost
- **FR-2.15**: Comparison files SHALL include YAML front matter with: comparison_type, timestamp, models compared, prompts compared, transcript used
- **FR-2.16**: System SHALL generate a comparison index file `data/evals/comparisons/index.md` linking to all comparisons

### Batch Result Management

- **FR-2.17**: Matrix runs SHALL create a batch directory: `data/evals/batches/{batch_id}/`
- **FR-2.18**: Batch directory SHALL contain: `summary.json` (batch metadata), `runs/` subdirectory (individual run results)
- **FR-2.19**: Batch summary SHALL track: batch_id, timestamp, total_runs, successful_runs, failed_runs, total_tokens, total_cost, duration_seconds
- **FR-2.20**: After batch completion, system SHALL auto-generate comparison files for logical groupings (same transcript, different prompts/models)

## Non-Functional Requirements

- **NFR-2.1**: Matrix runs SHALL complete in <5 minutes for 12 eval combinations (3 prompts × 2 models × 2 transcripts) with medium-length transcripts
- **NFR-2.2**: Comparison markdown files SHOULD be readable in any markdown editor without special rendering
- **NFR-2.3**: System SHOULD run evals in parallel where possible to minimize total time (up to 3 concurrent API calls)
- **NFR-2.4**: Comparison generation SHOULD complete in <10 seconds regardless of number of results being compared

## Dependencies

### Prerequisites

- Phase 1 complete (prompt versioning, reference transcripts, basic eval runner)
- Working single eval execution with results storage
- At least 2 prompt versions and 2 Groq models configured

### Outputs for Next Phase

- Matrix eval runner capable of batch processing
- Side-by-side comparison generation for prompts and models
- Batch result tracking structure
- Proven ability to complete evals in <5 minutes

## Acceptance Criteria

- [ ] Can run: `uv run pidcast-eval --run-matrix` and it executes all combinations
- [ ] Can run: `uv run pidcast-eval --run-matrix --prompts v1,v2 --models llama-3.3-70b` to test subset
- [ ] Matrix run shows progress indicator: "Running 5/12: v2 + mixtral + tech-talk-01"
- [ ] Before starting matrix, system displays estimated total tokens and cost
- [ ] After completion, system displays actual total tokens and cost
- [ ] Batch directory created in `data/evals/batches/` with summary.json
- [ ] Comparison files auto-generated in `data/evals/comparisons/` after batch
- [ ] Prompt comparison markdown clearly shows differences between v1, v2, v3 outputs for same model+transcript
- [ ] Model comparison markdown clearly shows differences between models for same prompt+transcript
- [ ] Comparison files include YAML front matter with all metadata
- [ ] `data/evals/comparisons/index.md` lists all generated comparisons with links
- [ ] Matrix eval of 12 combinations (3 prompts × 2 models × 2 transcripts) completes in <5 minutes
- [ ] Generated comparison files are readable in VS Code / Obsidian without issues
- [ ] All code passes `uv run ruff check`

## Open Questions

- Should we parallelize API calls during matrix runs? Risk of rate limits vs. speed benefit
- What's the best markdown format for side-by-side comparison? Tables vs. sections vs. diff-style?
- Should comparison generation happen automatically after batch runs or require explicit flag?

---

*Review this PRD and provide feedback before spec generation.*
