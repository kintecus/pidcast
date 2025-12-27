# PRD: Pidcast-Evals - Phase 1

**Contract**: ./contract.md
**Phase**: 1 of 3
**Focus**: Foundation - Prompt versioning, reference transcripts, and basic eval runner

## Phase Overview

Phase 1 establishes the foundational infrastructure for systematic prompt and model evaluation. This phase creates the core data structures and minimal viable eval workflow before adding complexity.

After this phase, you'll be able to:
- Store prompts with sequential version numbers (v1, v2, v3)
- Maintain a curated set of reference transcripts for reproducible testing
- Run a single eval (one prompt × one model × one transcript) and save results
- View basic eval output as markdown

This phase is sequenced first because it establishes the data model and file structure that subsequent phases will build upon. Without versioned prompts and reference transcripts, there's nothing to compare. The basic eval runner proves the core workflow before scaling to matrix comparisons.

Key constraint: Phase 1 intentionally limits scope to single eval runs to validate the approach before investing in batch processing.

## User Stories

1. As a prompt engineer, I want to store prompt variants with version numbers so that I can track iterations over time
2. As a developer, I want a curated set of reference transcripts so that I have consistent test data for evaluations
3. As a researcher, I want to run a single eval (prompt + model + transcript) so that I can validate the basic workflow
4. As a user, I want eval results saved to a predictable location so that I can review outputs later

## Functional Requirements

### Prompt Versioning

- **FR-1.1**: System SHALL store prompts in JSON format with fields: `id`, `version`, `name`, `description`, `system_prompt`, `user_prompt`, `max_output_tokens`
- **FR-1.2**: Prompt versions SHALL use sequential numbering (v1, v2, v3, etc.) within each prompt type
- **FR-1.3**: System SHALL validate that prompt version numbers are unique within a prompt type
- **FR-1.4**: Prompts JSON file SHALL be stored at `config/eval_prompts.json`
- **FR-1.5**: System SHALL support the existing analysis types from main pidcast: `summary`, `key_points`, `action_items`

### Reference Transcripts

- **FR-1.6**: System SHALL maintain a registry of reference transcripts in `config/reference_transcripts.json`
- **FR-1.7**: Each reference transcript SHALL have: `id`, `name`, `description`, `file_path`, `content_type`, `duration_seconds`, `source_url`
- **FR-1.8**: Reference transcript markdown files SHALL be stored in `data/evals/references/`
- **FR-1.9**: System SHALL validate that reference transcript files exist and are readable
- **FR-1.10**: Initial reference set SHALL include 2-3 transcripts covering different content types (e.g., technical talk, interview, tutorial) and lengths (short <10min, medium 20-40min)

### Basic Eval Runner

- **FR-1.11**: CLI SHALL accept parameters: `--prompt_version`, `--model`, `--transcript_id`
- **FR-1.12**: System SHALL load the specified prompt version from `config/eval_prompts.json`
- **FR-1.13**: System SHALL load the specified reference transcript from registry
- **FR-1.14**: System SHALL call the Groq API with the selected model and prompt
- **FR-1.15**: System SHALL substitute `{transcript}`, `{title}`, `{duration}`, `{url}` variables in prompts (reusing existing logic from `analysis.py`)

### Results Storage

- **FR-1.16**: Eval results SHALL be saved to `data/evals/runs/{run_id}/`
- **FR-1.17**: Run ID SHALL use format: `{timestamp}_{prompt_version}_{model}_{transcript_id}` (e.g., `20251227_143022_v1_llama-3.3-70b_tech-talk-01`)
- **FR-1.18**: Each run directory SHALL contain: `result.md` (LLM output), `metadata.json` (run params, tokens, cost, timestamp)
- **FR-1.19**: Metadata JSON SHALL track: run_id, timestamp, prompt_version, model, transcript_id, tokens_input, tokens_output, estimated_cost, duration_seconds, success status
- **FR-1.20**: Result markdown SHALL include YAML front matter with metadata for Obsidian compatibility

## Non-Functional Requirements

- **NFR-1.1**: Prompt versioning system SHOULD be extensible to support semantic versioning in future phases
- **NFR-1.2**: Reference transcript registry SHOULD validate JSON schema on load to catch configuration errors early
- **NFR-1.3**: Eval runner SHOULD complete single runs in <60 seconds for medium-length transcripts (<40min)
- **NFR-1.4**: Code SHOULD reuse existing abstractions from `analysis.py` and `config.py` to minimize duplication

## Dependencies

### Prerequisites

- Existing `pidcast` codebase with working Groq API integration
- `analysis.py` functions for LLM API calls and prompt substitution
- `config.py` with Groq model definitions and pricing
- Groq API key configured

### Outputs for Next Phase

- Working prompt versioning format in `config/eval_prompts.json`
- Reference transcript registry in `config/reference_transcripts.json`
- 2-3 curated reference transcripts in `data/evals/references/`
- Results storage structure in `data/evals/runs/`
- Basic eval runner that proves the workflow

## Acceptance Criteria

- [ ] `config/eval_prompts.json` exists with v1 of `summary`, `key_points`, `action_items` prompts
- [ ] `config/reference_transcripts.json` registry exists with 2-3 entries
- [ ] 2-3 reference transcript markdown files exist in `data/evals/references/`
- [ ] Can run: `uv run pidcast-eval --prompt_version v1 --model llama-3.3-70b --transcript_id tech-talk-01`
- [ ] Eval creates a run directory in `data/evals/runs/` with result.md and metadata.json
- [ ] Result markdown contains LLM analysis output with YAML front matter
- [ ] Metadata JSON contains all required fields (run_id, tokens, cost, etc.)
- [ ] Prompt variable substitution works correctly ({transcript}, {title}, etc.)
- [ ] Loading non-existent prompt version or transcript ID shows clear error message
- [ ] All code passes `uv run ruff check`

## Open Questions

- Should we extract actual reference transcripts from existing `data/transcripts/` or create fresh ones?
- Do we want a helper command to bootstrap `config/eval_prompts.json` from existing `config/analysis_prompts.json`?
- Should the eval runner be a separate script or integrated into the pidcast package as a module?

---

*Review this PRD and provide feedback before spec generation.*
