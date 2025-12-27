# PRD: Pidcast-Evals - Phase 3

**Contract**: ./contract.md
**Phase**: 3 of 3
**Focus**: Robustness - Retry logic, error handling, and production-ready reliability

## Phase Overview

Phase 3 hardens the eval system for reliable production use. While Phases 1 and 2 built the core functionality, Phase 3 ensures the system handles real-world conditions: API rate limits, transient failures, network issues, and cost tracking.

After this phase, you'll be able to:
- Run large eval batches without manual intervention when API calls fail
- Automatically retry failed requests with exponential backoff
- Track precise token usage and costs across all eval runs
- Get clear error reports showing which evals succeeded vs. failed and why

This phase is sequenced last because retry logic and error handling are only valuable once the core workflow is proven. Building robustness into an unstable system wastes effort. Now that Phases 1-2 validate the approach, we can invest in production-grade reliability.

Key value: Phase 3 makes the eval system reliable enough for daily use without constant babysitting.

## User Stories

1. As a researcher, I want failed API calls to automatically retry so that transient errors don't break my eval runs
2. As a developer, I want clear error reports showing which evals failed so that I can debug issues quickly
3. As a cost-conscious user, I want accurate token and cost tracking so that I can budget my API usage
4. As a user running large batches, I want partial results saved so that I don't lose progress if something fails midway

## Functional Requirements

### Retry Logic with Backoff

- **FR-3.1**: System SHALL retry failed API calls up to 3 times with exponential backoff (2s, 4s, 8s)
- **FR-3.2**: System SHALL retry on these error types: `RateLimitError`, `APIConnectionError`, `Timeout`, `ServiceUnavailable` (5xx)
- **FR-3.3**: System SHALL NOT retry on these error types: `AuthenticationError`, `InvalidRequestError`, `BadRequest` (4xx except 429)
- **FR-3.4**: Each retry attempt SHALL be logged with attempt number and wait time
- **FR-3.5**: After exhausting retries, system SHALL mark eval as failed and continue with next eval in batch
- **FR-3.6**: System SHALL respect Groq API rate limit headers (if available) and adjust backoff accordingly

### Error Handling and Reporting

- **FR-3.7**: Each eval result metadata SHALL include `success: true/false` and `error_message` field
- **FR-3.8**: Failed evals SHALL save partial metadata (prompt, model, transcript, error details) to `data/evals/runs/{run_id}/error.json`
- **FR-3.9**: Batch summary SHALL track: `successful_runs`, `failed_runs`, `runs_with_retries`
- **FR-3.10**: After batch completion, system SHALL display summary: "Completed 10/12 evals successfully. 2 failed (see errors below)"
- **FR-3.11**: System SHALL list failed evals with run_id, prompt, model, transcript, and error message
- **FR-3.12**: CLI SHALL exit with non-zero code if any evals failed (for CI/CD integration)

### Cost and Token Tracking

- **FR-3.13**: System SHALL calculate actual cost using token counts and model-specific pricing from `config.GROQ_PRICING`
- **FR-3.14**: Each eval metadata SHALL include: `tokens_input`, `tokens_output`, `tokens_total`, `cost_usd`
- **FR-3.15**: Batch summary SHALL include: `total_tokens_input`, `total_tokens_output`, `total_cost_usd`
- **FR-3.16**: Before starting matrix run, system SHALL display estimated total cost based on average transcript length
- **FR-3.17**: After batch completion, system SHALL display actual total cost and compare to estimate
- **FR-3.18**: System SHALL maintain a running cost log in `data/evals/cost_tracking.json` with daily/weekly/monthly aggregates

### Partial Results & Resume

- **FR-3.19**: System SHALL save each eval result immediately after completion (not batch at end)
- **FR-3.20**: CLI SHALL accept `--resume-batch {batch_id}` to continue incomplete batch run
- **FR-3.21**: Resume mode SHALL skip already-completed evals and only run missing/failed evals
- **FR-3.22**: Resumed batch SHALL update the original batch summary with new totals

### Validation and Safety

- **FR-3.23**: Before starting batch, system SHALL validate all prompt versions, models, and transcript IDs exist
- **FR-3.24**: If matrix would exceed a cost threshold (e.g., $5), system SHALL require confirmation: "Estimated cost: $6.50. Continue? [y/N]"
- **FR-3.25**: System SHALL validate Groq API key is set before starting any eval
- **FR-3.26**: System SHALL check for sufficient disk space before starting large batches (warn if <100MB available)

## Non-Functional Requirements

- **NFR-3.1**: Retry logic SHALL add <15 seconds overhead per failed request in worst case (3 retries with 2s+4s+8s backoff)
- **NFR-3.2**: Error messages SHALL be actionable (e.g., "Invalid API key" not "Request failed")
- **NFR-3.3**: Token counting SHOULD use the same estimation method as existing `analysis.py` for consistency
- **NFR-3.4**: Cost tracking SHOULD be accurate within 5% of actual Groq billing

## Dependencies

### Prerequisites

- Phase 2 complete (matrix eval runner, comparison generation)
- Working batch result storage
- Existing retry patterns from `download.py` to reference

### Outputs for Next Phase

- Production-ready eval system with robust error handling
- Cost tracking infrastructure for budgeting
- Resume capability for long-running batches

## Acceptance Criteria

- [ ] Simulating a rate limit error triggers 3 retry attempts with exponential backoff
- [ ] Simulating an auth error (invalid API key) fails immediately without retrying
- [ ] Failed eval creates `error.json` in run directory with error details
- [ ] Batch summary correctly reports successful vs. failed run counts
- [ ] After batch with failures, CLI displays clear error summary with run IDs and messages
- [ ] Cost calculation matches expected values based on token counts and `GROQ_PRICING`
- [ ] Batch summary includes accurate `total_cost_usd` field
- [ ] Before matrix run, system displays: "Estimated cost: $X.XX (based on Y tokens per transcript)"
- [ ] After matrix run, system displays: "Actual cost: $X.XX (Z total tokens)"
- [ ] `data/evals/cost_tracking.json` exists and tracks daily/weekly/monthly totals
- [ ] Can interrupt a batch run (Ctrl+C) and resume with: `uv run pidcast-eval --resume-batch {batch_id}`
- [ ] Resume skips already-completed evals and only runs missing ones
- [ ] Starting a matrix that exceeds $5 estimated cost prompts for confirmation
- [ ] Starting eval without `GROQ_API_KEY` set shows clear error: "Groq API key not configured"
- [ ] All code passes `uv run ruff check`
- [ ] Retry logic reuses existing patterns from `download.py` where applicable

## Open Questions

- Should we add a `--max-retries` CLI flag to override default of 3?
- Do we want Slack/email notifications for failed batches in long-running scenarios?
- Should cost tracking be opt-in or always-on?

---

*Review this PRD and provide feedback before spec generation.*
