# Implementation Spec: Pidcast-Evals - Phase 3

**PRD**: ./prd-phase-3.md
**Estimated Effort**: S (Small - ~1-2 days)

## Technical Approach

Phase 3 hardens the eval system with production-grade reliability features: retry logic, error handling, cost tracking, and resume capability. Much of this builds on existing patterns from `download.py` which already implements robust retry logic.

We'll extract retry logic into a reusable decorator, enhance error tracking in eval results, add cost tracking aggregation, and implement checkpoint-based resume functionality for batch runs.

Key technical decisions:
- **Retry pattern**: Decorator-based retry with exponential backoff (reuse pattern from `download.py`)
- **Error classification**: Distinguish retryable (rate limits, timeouts) from non-retryable (auth, validation) errors
- **Cost tracking**: Separate JSON log file with daily/weekly/monthly aggregates
- **Resume logic**: Check existing batch directory for completed runs, skip those
- **Validation**: Pre-flight checks before starting batch (API key, disk space, config validity)

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `src/pidcast/evals/retry.py` | Retry decorator and error classification logic |
| `src/pidcast/evals/cost_tracker.py` | Cost tracking and aggregation |
| `src/pidcast/evals/validation.py` | Pre-flight validation checks |
| `data/evals/cost_tracking.json` | Cost tracking log (gitignored) |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `src/pidcast/evals/runner.py` | Add retry decorator to API calls, enhanced error handling |
| `src/pidcast/evals/batch_runner.py` | Add resume logic, validation, cost confirmation |
| `src/pidcast/evals/cli.py` | Add `--resume-batch`, `--skip-confirmation` flags |
| `src/pidcast/evals/results.py` | Add error tracking fields to metadata |
| `.gitignore` | Add `data/evals/cost_tracking.json` |

### Deleted Files

None.

## Implementation Details

### Retry Logic (`retry.py`)

**Pattern to follow**: `src/pidcast/download.py:is_retryable_error()` and retry loops

**Overview**: Decorator for automatic retry with exponential backoff on transient errors.

```python
import time
import functools
from typing import Callable, Type, Tuple
from groq import RateLimitError, APIConnectionError, APITimeoutError

# Retryable error types
RETRYABLE_ERRORS: Tuple[Type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    # Add more as needed
)

# Non-retryable error types
NON_RETRYABLE_ERRORS: Tuple[Type[Exception], ...] = (
    # AuthenticationError, InvalidRequestError, etc.
)

def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exponential_base: float = 2.0,
):
    """Decorator to retry function with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (exponential_base ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")
                        raise
                except NON_RETRYABLE_ERRORS as e:
                    logger.error(f"Non-retryable error: {e}")
                    raise
            raise last_exception
        return wrapper
    return decorator

def is_retryable_error(error: Exception) -> bool:
    """Check if an error should trigger a retry."""
    return isinstance(error, RETRYABLE_ERRORS)
```

**Key decisions**:
- Exponential backoff: 2s, 4s, 8s for 3 retries
- Log each retry attempt with delay
- Raise immediately on non-retryable errors (don't waste time)
- Reuse existing logging infrastructure

**Implementation steps**:
1. Define retryable and non-retryable error tuples
2. Implement decorator with exponential backoff
3. Add logging for retry attempts
4. Apply decorator to `EvalRunner.run_eval()` API call
5. Write unit tests with mocked API errors

### Cost Tracking (`cost_tracker.py`)

**Pattern to follow**: `src/pidcast/utils.py:save_statistics()`

**Overview**: Track eval costs over time with daily/weekly/monthly aggregates.

```python
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List

@dataclass
class CostEntry:
    """Single cost entry for an eval run."""
    timestamp: datetime
    run_id: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_usd: float

@dataclass
class CostAggregates:
    """Aggregated cost statistics."""
    total_runs: int
    total_tokens: int
    total_cost_usd: float
    by_model: Dict[str, float]  # Model -> total cost

class CostTracker:
    """Tracks and aggregates eval costs."""

    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self._load_tracking_data()

    def _load_tracking_data(self) -> None:
        """Load existing cost tracking data from JSON."""
        pass

    def record_eval(self, result: EvalResult) -> None:
        """Record cost for a single eval run."""
        pass

    def get_daily_total(self, day: date) -> float:
        """Get total cost for a specific day."""
        pass

    def get_weekly_total(self, week_start: date) -> float:
        """Get total cost for a week."""
        pass

    def get_monthly_total(self, year: int, month: int) -> float:
        """Get total cost for a month."""
        pass

    def get_aggregates(self, start_date: date, end_date: date) -> CostAggregates:
        """Get aggregated statistics for a date range."""
        pass

    def _save_tracking_data(self) -> None:
        """Save tracking data to JSON."""
        pass
```

**Key decisions**:
- Append-only log structure for simplicity
- Calculate aggregates on-demand (no pre-aggregation)
- Store in `data/evals/cost_tracking.json`
- Track per-model costs for comparison

**Implementation steps**:
1. Create `CostEntry` and `CostAggregates` dataclasses
2. Implement JSON loading and saving
3. Implement append logic for new entries
4. Implement aggregate calculation (filter by date range, sum costs)
5. Add CLI command to display cost summary: `pidcast-eval --cost-summary`
6. Write unit tests with fixture data

### Validation (`validation.py`)

**Pattern to follow**: Pre-flight checks similar to CLI validation in `cli.py`

**Overview**: Validate configuration and environment before starting batch runs.

```python
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class ValidationResult:
    """Result of pre-flight validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class EvalValidator:
    """Validates eval configuration and environment."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        transcript_manager: ReferenceTranscriptManager,
    ):
        self.prompt_manager = prompt_manager
        self.transcript_manager = transcript_manager

    def validate_batch_config(
        self,
        config: BatchConfig,
        groq_api_key: Optional[str],
    ) -> ValidationResult:
        """Validate batch configuration before starting."""
        errors = []
        warnings = []

        # Check API key
        if not groq_api_key:
            errors.append("GROQ_API_KEY not configured")

        # Check prompt versions exist
        for version in config.prompt_versions:
            # Check each prompt type has this version
            pass

        # Check models are valid
        for model in config.models:
            if model not in GROQ_PRICING:
                errors.append(f"Unknown model: {model}")

        # Check transcript IDs exist
        for tid in config.transcript_ids:
            # Validate transcript exists
            pass

        # Check disk space
        available_space = self._check_disk_space()
        if available_space < 100_000_000:  # 100MB
            warnings.append(
                f"Low disk space: {available_space / 1e6:.1f}MB available"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_disk_space(self) -> int:
        """Check available disk space in bytes."""
        import shutil
        stat = shutil.disk_usage(Path.cwd())
        return stat.free
```

**Key decisions**:
- Fail fast on configuration errors before API calls
- Warn on non-critical issues (low disk space)
- Validate all prompt versions, models, transcripts exist
- Check API key is set

**Implementation steps**:
1. Create `ValidationResult` dataclass
2. Implement API key check
3. Implement prompt/model/transcript existence checks
4. Implement disk space check
5. Integrate into batch runner (validate before starting)
6. Write unit tests for each validation check

### Batch Resume Logic (`batch_runner.py` updates)

**Pattern to follow**: Checkpoint/resume pattern

**Overview**: Resume incomplete batch runs by checking for existing results.

```python
class BatchRunner:
    # ... existing code ...

    def resume_batch(self, batch_id: str, groq_api_key: str) -> BatchSummary:
        """Resume an incomplete batch run."""
        # 1. Load batch summary
        # 2. Load original batch config
        # 3. Identify completed runs (check for metadata.json)
        # 4. Generate remaining tasks (skip completed)
        # 5. Run remaining tasks
        # 6. Update batch summary with new totals
        # 7. Return updated BatchSummary
        pass

    def _load_batch_summary(self, batch_id: str) -> BatchSummary:
        """Load existing batch summary from JSON."""
        pass

    def _find_completed_runs(self, batch_dir: Path) -> Set[str]:
        """Find run IDs that have completed (have metadata.json)."""
        pass

    def _filter_remaining_tasks(
        self,
        all_tasks: List[EvalConfig],
        completed_runs: Set[str],
    ) -> List[EvalConfig]:
        """Filter out tasks that are already completed."""
        pass
```

**Key decisions**:
- Use presence of `metadata.json` as completion indicator
- Allow re-running failed evals in resume mode
- Update original batch summary (don't create new batch)
- Skip comparisons if no new results generated

**Implementation steps**:
1. Implement batch summary loading
2. Implement completed run detection (scan for metadata.json files)
3. Implement task filtering (skip completed run IDs)
4. Implement batch summary update (add new results to existing)
5. Integrate into CLI with `--resume-batch {batch_id}` flag
6. Write integration tests with partial batch fixtures

### Cost Confirmation (`batch_runner.py` updates)

**Pattern to follow**: Interactive confirmation prompts

**Overview**: Require user confirmation for expensive batch runs.

```python
COST_THRESHOLD_USD = 5.0

class BatchRunner:
    # ... existing code ...

    def _confirm_batch_cost(self, estimated_cost: float, skip_confirmation: bool) -> bool:
        """Prompt user to confirm if estimated cost exceeds threshold."""
        if skip_confirmation:
            return True

        if estimated_cost < COST_THRESHOLD_USD:
            return True

        print(f"\n⚠️  Estimated cost: ${estimated_cost:.2f}")
        response = input("Continue with batch run? [y/N]: ").strip().lower()
        return response in ("y", "yes")
```

**Key decisions**:
- Default threshold: $5.00 (configurable via constant)
- Allow `--skip-confirmation` flag for automation
- Block until user responds (not async)
- Clear display of estimated cost before prompt

**Implementation steps**:
1. Calculate estimated cost before starting batch
2. Display cost with emoji/formatting for visibility
3. Prompt for confirmation (y/N)
4. Return early if user declines
5. Add `--skip-confirmation` flag for CI/automation

### Enhanced Error Tracking (`results.py` updates)

**Pattern to follow**: Extend existing metadata schema

**Overview**: Add error tracking fields to eval result metadata.

```python
@dataclass
class EvalResult:
    # ... existing fields ...
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0  # How many retries occurred
    error_type: Optional[str] = None  # e.g., "RateLimitError"

def save_eval_result(result: EvalResult, output_dir: Path) -> None:
    """Save eval result with error metadata."""
    run_dir = output_dir / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if result.success:
        # Save result.md + metadata.json (existing logic)
        pass
    else:
        # Save error.json with detailed error info
        error_data = {
            "run_id": result.run_id,
            "timestamp": result.timestamp.isoformat(),
            "error_type": result.error_type,
            "error_message": result.error_message,
            "retry_count": result.retry_count,
            "config": {
                "prompt_version": result.prompt_version,
                "model": result.model,
                "transcript_id": result.transcript_id,
            },
        }
        error_file = run_dir / "error.json"
        error_file.write_text(json.dumps(error_data, indent=2))
```

**Key decisions**:
- Save `error.json` for failed runs (in addition to partial metadata)
- Track retry count to identify flaky vs. hard failures
- Store error type for categorization
- Include full config in error.json for debugging

**Implementation steps**:
1. Add error fields to `EvalResult` dataclass
2. Update `save_eval_result()` to handle failures
3. Create `error.json` format
4. Capture error type and message in exception handler
5. Update batch summary to report error types

## Data Model

### Cost Tracking Schema

```json
{
  "entries": [
    {
      "timestamp": "2025-12-27T14:30:00Z",
      "run_id": "20251227_143000_v1_llama_tech-talk",
      "model": "llama-3.3-70b-versatile",
      "tokens_input": 12500,
      "tokens_output": 850,
      "cost_usd": 0.0098
    }
  ]
}
```

### Error Tracking Schema

```json
{
  "run_id": "20251227_143000_v1_llama_tech-talk",
  "timestamp": "2025-12-27T14:30:00Z",
  "error_type": "RateLimitError",
  "error_message": "Rate limit exceeded. Retry after 60s.",
  "retry_count": 3,
  "config": {
    "prompt_version": "v1",
    "model": "llama-3.3-70b-versatile",
    "transcript_id": "tech-talk-01"
  }
}
```

## Testing Requirements

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/evals/test_retry.py` | Retry decorator, error classification |
| `tests/evals/test_cost_tracker.py` | Cost logging, aggregation, date filtering |
| `tests/evals/test_validation.py` | Pre-flight checks, error/warning detection |
| `tests/evals/test_batch_runner.py` | Resume logic, completed run detection |

**Key test cases**:
- Retry decorator retries on RateLimitError
- Retry decorator does NOT retry on AuthenticationError
- Retry uses exponential backoff (2s, 4s, 8s)
- Cost tracker appends entries correctly
- Cost tracker calculates daily/monthly totals
- Validator detects missing API key
- Validator detects invalid prompt versions
- Resume logic skips completed runs
- Resume logic re-runs failed runs
- Cost confirmation prompts when threshold exceeded

### Manual Testing

- [ ] Simulate rate limit error (mock), verify 3 retries with backoff
- [ ] Simulate auth error, verify immediate failure (no retry)
- [ ] Run batch that exceeds $5, verify confirmation prompt
- [ ] Run batch with `--skip-confirmation`, verify no prompt
- [ ] Check `data/evals/cost_tracking.json` after batch
- [ ] Run `pidcast-eval --cost-summary`, verify correct totals
- [ ] Interrupt batch mid-run (Ctrl+C)
- [ ] Resume with `pidcast-eval --resume-batch {batch_id}`
- [ ] Verify resume skips completed runs
- [ ] Run batch without GROQ_API_KEY, verify clear error
- [ ] Run batch with low disk space, verify warning displayed

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| Groq rate limit (429) | Retry 3x with exponential backoff (2s, 4s, 8s), then fail |
| API timeout | Retry 3x with exponential backoff, then fail |
| Invalid API key | Fail immediately with clear message (no retry) |
| Disk space <100MB | Display warning, continue unless user cancels |
| Invalid prompt version | Fail pre-flight validation before any API calls |
| Batch cost >$5 | Prompt user for confirmation, cancel if declined |
| Resume with invalid batch_id | Error: "Batch '{id}' not found in {path}" |

## Validation Commands

```bash
# Linting
uv run ruff check src/pidcast/evals/
uv run ruff format src/pidcast/evals/

# Unit tests
uv run pytest tests/evals/ -v

# Run batch with cost confirmation
uv run pidcast-eval --run-matrix

# Run batch without confirmation (for automation)
uv run pidcast-eval --run-matrix --skip-confirmation

# Resume incomplete batch
uv run pidcast-eval --resume-batch 20251227_143500_matrix

# View cost summary
uv run pidcast-eval --cost-summary
uv run pidcast-eval --cost-summary --start-date 2025-12-01 --end-date 2025-12-31
```

## Rollout Considerations

- **Feature flag**: None needed
- **Monitoring**: Track retry rates, error types, cost trends
- **Alerting**: Not applicable (local CLI tool)
- **Rollback plan**: Phase 3 is additive, doesn't break existing functionality

## Open Items

- [ ] Decide: Should `--skip-confirmation` be named `--yes` for brevity?
- [ ] Decide: Should cost threshold be configurable via CLI flag?
- [ ] Decide: Add `--dry-run` flag to show what would be run without executing?

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
