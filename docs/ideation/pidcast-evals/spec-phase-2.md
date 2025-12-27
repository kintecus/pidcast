# Implementation Spec: Pidcast-Evals - Phase 2

**PRD**: ./prd-phase-2.md
**Estimated Effort**: M (Medium - ~2-3 days)

## Technical Approach

Phase 2 extends the single-eval runner from Phase 1 into a matrix evaluation system with batch processing and comparison generation. The core challenge is orchestrating multiple API calls efficiently while generating useful comparison outputs.

We'll add matrix orchestration logic to `runner.py`, create a new comparison generator module, and extend the CLI to support batch operations. The comparison generator will parse eval results and generate side-by-side markdown files organized by comparison type (prompt vs. model).

Key technical decisions:
- **Parallelization**: Use `concurrent.futures.ThreadPoolExecutor` for parallel API calls (max 3 concurrent to respect rate limits)
- **Batch tracking**: Create batch metadata structure similar to individual run structure
- **Comparison format**: Generate markdown with horizontal sections (not tables) for readability
- **Auto-generation**: Auto-create comparisons after batch completion based on result set

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `src/pidcast/evals/batch_runner.py` | Matrix eval orchestration and batch management |
| `src/pidcast/evals/comparison.py` | Generate side-by-side comparison markdown files |
| `data/evals/batches/.gitkeep` | Ensure directory exists |
| `data/evals/comparisons/.gitkeep` | Ensure directory exists |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `src/pidcast/evals/cli.py` | Add `--run-matrix`, `--prompts`, `--models`, `--transcripts` flags |
| `src/pidcast/evals/runner.py` | Extract reusable components for batch runner |
| `.gitignore` | Add `data/evals/batches/*/` and `data/evals/comparisons/*/` |

### Deleted Files

None.

## Implementation Details

### Matrix Batch Runner (`batch_runner.py`)

**Pattern to follow**: Similar to Phase 1 `EvalRunner`, but with batch orchestration

**Overview**: Orchestrates multiple eval runs in parallel, tracks batch metadata, and saves batch summaries.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

@dataclass
class BatchConfig:
    """Configuration for a batch eval run."""
    prompt_versions: List[str]  # e.g., ["v1", "v2", "v3"]
    models: List[str]  # e.g., ["llama-3.3-70b", "mixtral"]
    transcript_ids: List[str]  # e.g., ["tech-talk-01", "interview-02"]
    max_concurrent: int = 3

@dataclass
class BatchSummary:
    """Summary of a batch eval run."""
    batch_id: str
    timestamp: datetime
    total_runs: int
    successful_runs: int
    failed_runs: int
    total_tokens_input: int
    total_tokens_output: int
    total_cost_usd: float
    duration_seconds: float
    eval_results: List[EvalResult] = field(default_factory=list)

class BatchRunner:
    """Runs matrix evals in parallel."""

    def __init__(
        self,
        eval_runner: EvalRunner,
        results_dir: Path,
        batches_dir: Path,
    ):
        self.eval_runner = eval_runner
        self.results_dir = results_dir
        self.batches_dir = batches_dir

    def run_batch(self, config: BatchConfig, groq_api_key: str) -> BatchSummary:
        """Execute matrix of evals: prompts × models × transcripts."""
        # 1. Generate batch_id
        # 2. Create eval task list (all combinations)
        # 3. Estimate total cost and display
        # 4. Run evals in parallel with ThreadPoolExecutor
        # 5. Track progress with rich progress bar
        # 6. Aggregate results
        # 7. Save batch summary
        # 8. Return BatchSummary
        pass

    def _generate_eval_tasks(self, config: BatchConfig) -> List[EvalConfig]:
        """Generate list of EvalConfigs for all combinations."""
        pass

    def _estimate_batch_cost(self, tasks: List[EvalConfig]) -> float:
        """Estimate total cost based on average transcript length."""
        pass

    def _run_evals_parallel(
        self,
        tasks: List[EvalConfig],
        max_concurrent: int,
    ) -> List[EvalResult]:
        """Run evals in parallel using ThreadPoolExecutor."""
        pass

    def _save_batch_summary(self, summary: BatchSummary, batch_dir: Path) -> None:
        """Save batch summary JSON."""
        pass
```

**Key decisions**:
- Limit concurrent API calls to 3 to avoid rate limiting
- Use `rich.progress.Progress` for live progress updates
- Save each individual eval result immediately (reuse Phase 1 logic)
- Aggregate results at end for batch summary
- Store batch results in separate directory structure

**Implementation steps**:
1. Create `BatchConfig` and `BatchSummary` dataclasses
2. Implement task generation (Cartesian product of prompts × models × transcripts)
3. Implement cost estimation using existing `GROQ_PRICING` data
4. Implement parallel execution with `ThreadPoolExecutor`
5. Add rich progress bar showing "Running X/Y: {prompt} + {model} + {transcript}"
6. Implement batch summary aggregation and saving
7. Write unit tests with mocked API calls

### Comparison Generator (`comparison.py`)

**Pattern to follow**: Similar to `src/pidcast/markdown.py:create_markdown_file()`

**Overview**: Generates side-by-side comparison markdown files from batch results. Supports prompt comparisons and model comparisons.

```python
from dataclasses import dataclass
from enum import Enum
from typing import List

class ComparisonType(Enum):
    """Type of comparison being generated."""
    PROMPT = "prompt_comparison"
    MODEL = "model_comparison"

@dataclass
class ComparisonConfig:
    """Configuration for generating a comparison."""
    comparison_type: ComparisonType
    eval_results: List[EvalResult]
    output_dir: Path

class ComparisonGenerator:
    """Generates side-by-side comparison markdown files."""

    def generate_comparisons_from_batch(
        self,
        batch_summary: BatchSummary,
        output_dir: Path,
    ) -> List[Path]:
        """Auto-generate logical comparisons from batch results."""
        # 1. Group results by (transcript, model) -> prompt comparison
        # 2. Group results by (transcript, prompt) -> model comparison
        # 3. Generate comparison markdown for each grouping
        # 4. Generate comparison index file
        # 5. Return list of generated files
        pass

    def generate_comparison(self, config: ComparisonConfig) -> Path:
        """Generate a single comparison markdown file."""
        # 1. Sort results by version/model name
        # 2. Generate markdown with sections for each result
        # 3. Include YAML front matter with metadata
        # 4. Save to comparison directory
        # 5. Return path to generated file
        pass

    def _create_comparison_markdown(
        self,
        comparison_type: ComparisonType,
        results: List[EvalResult],
    ) -> str:
        """Generate comparison markdown content."""
        pass

    def _create_comparison_index(
        self,
        comparison_files: List[Path],
        index_file: Path,
    ) -> None:
        """Generate index.md listing all comparisons."""
        pass
```

**Key decisions**:
- Auto-detect comparison opportunities from batch results
- Use horizontal sections separated by `---` for readability
- Include metadata headers (tokens, cost) for each output
- Generate index file for easy navigation
- YAML front matter includes comparison metadata

**Implementation steps**:
1. Create `ComparisonType` enum and `ComparisonConfig` dataclass
2. Implement result grouping logic (by transcript+model, by transcript+prompt)
3. Implement markdown generation with clear section headers
4. Implement YAML front matter generation
5. Implement index file generation
6. Write unit tests with fixture eval results

### Comparison Markdown Format

**Example: Prompt Comparison**

```markdown
---
comparison_type: prompt_comparison
timestamp: 2025-12-27T14:30:00Z
transcript: tech-talk-01
model: llama-3.3-70b-versatile
prompts_compared: [v1, v2, v3]
---

# Prompt Comparison: tech-talk-01

**Transcript**: Example Tech Talk
**Model**: llama-3.3-70b-versatile
**Compared**: v1, v2, v3

---

## Version v1 (summary)

**Tokens**: 12,500 input / 850 output
**Cost**: $0.0098
**Duration**: 4.2s

[LLM output from v1...]

---

## Version v2 (summary)

**Tokens**: 12,500 input / 920 output
**Cost**: $0.0105
**Duration**: 4.5s

[LLM output from v2...]

---

## Version v3 (summary)

**Tokens**: 12,500 input / 780 output
**Cost**: $0.0092
**Duration**: 3.8s

[LLM output from v3...]
```

**Example: Model Comparison**

```markdown
---
comparison_type: model_comparison
timestamp: 2025-12-27T14:35:00Z
transcript: interview-02
prompt_version: v1
models_compared: [llama-3.3-70b, mixtral-8x7b]
---

# Model Comparison: interview-02

**Transcript**: Example Interview
**Prompt**: v1 (summary)
**Compared**: llama-3.3-70b-versatile, mixtral-8x7b-32768

---

## Model: llama-3.3-70b-versatile

**Tokens**: 15,200 input / 1,020 output
**Cost**: $0.0129
**Duration**: 5.1s

[LLM output from llama-3.3-70b...]

---

## Model: mixtral-8x7b-32768

**Tokens**: 15,200 input / 980 output
**Cost**: $0.0062
**Duration**: 3.2s

[LLM output from mixtral...]
```

### CLI Updates (`cli.py`)

**Pattern to follow**: Extend existing argparse parser

**Overview**: Add flags for matrix eval mode and subset selection.

```python
def parse_eval_arguments() -> argparse.Namespace:
    """Parse CLI arguments for pidcast-eval."""
    parser = argparse.ArgumentParser(description="Run LLM analysis evals")

    # Existing single-eval flags
    parser.add_argument("--prompt_version", help="Prompt version (e.g., v1)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--transcript_id", help="Reference transcript ID")

    # New batch flags
    parser.add_argument("--run-matrix", action="store_true", help="Run matrix of all combinations")
    parser.add_argument("--prompts", help="Comma-separated prompt versions (e.g., v1,v2,v3)")
    parser.add_argument("--models", help="Comma-separated models")
    parser.add_argument("--transcripts", help="Comma-separated transcript IDs")

    # Existing flags
    parser.add_argument("--groq_api_key", help="Groq API key")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()

def eval_main() -> None:
    """Main entry point - route to single or batch mode."""
    args = parse_eval_arguments()

    if args.run_matrix:
        # Batch mode
        run_batch_evals(args)
    else:
        # Single mode (Phase 1 logic)
        run_single_eval(args)
```

**Key decisions**:
- Mutually exclusive modes: single eval XOR matrix eval
- Comma-separated lists for subset selection
- Default to "all" if subset not specified with `--run-matrix`

**Implementation steps**:
1. Add new arguments to parser
2. Implement mode routing logic
3. Implement `run_batch_evals()` function
4. Parse comma-separated lists into Python lists
5. Display pre-batch summary (estimated cost, total runs)
6. Display post-batch summary (actual cost, success/failure counts)

## Data Model

### Batch Summary Schema

```json
{
  "batch_id": "20251227_143500_matrix",
  "timestamp": "2025-12-27T14:35:00Z",
  "total_runs": 12,
  "successful_runs": 11,
  "failed_runs": 1,
  "total_tokens_input": 150000,
  "total_tokens_output": 10500,
  "total_cost_usd": 0.1234,
  "duration_seconds": 45.2,
  "config": {
    "prompts": ["v1", "v2"],
    "models": ["llama-3.3-70b", "mixtral"],
    "transcripts": ["tech-talk-01", "interview-02", "tutorial-03"]
  },
  "runs": [
    "20251227_143500_v1_llama-3.3-70b_tech-talk-01",
    "20251227_143502_v1_llama-3.3-70b_interview-02"
  ]
}
```

### Comparison Index Format

```markdown
# Eval Comparisons

Generated: 2025-12-27T14:40:00Z

## Prompt Comparisons

- [tech-talk-01: v1 vs v2 vs v3 (llama-3.3-70b)](./20251227_143500_prompt_tech-talk-01/comparison.md)
- [interview-02: v1 vs v2 (mixtral)](./20251227_143600_prompt_interview-02/comparison.md)

## Model Comparisons

- [tech-talk-01: llama vs mixtral (v1)](./20251227_143700_model_tech-talk-01/comparison.md)
- [interview-02: llama vs mixtral (v2)](./20251227_143800_model_interview-02/comparison.md)
```

## Testing Requirements

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/evals/test_batch_runner.py` | Matrix task generation, parallel execution, aggregation |
| `tests/evals/test_comparison.py` | Result grouping, markdown generation, index creation |

**Key test cases**:
- Generate correct eval tasks for 2×2×2 matrix (8 combinations)
- Estimate batch cost correctly based on token pricing
- Run evals in parallel (mock ThreadPoolExecutor)
- Aggregate results with correct totals
- Group results by (transcript, model) for prompt comparison
- Group results by (transcript, prompt) for model comparison
- Generate comparison markdown with correct sections
- Generate index file with all comparison links
- Handle empty batch results gracefully

### Manual Testing

- [ ] Run `uv run pidcast-eval --run-matrix`
- [ ] Verify progress bar shows: "Running 3/12: v2 + mixtral + interview-02"
- [ ] Verify pre-batch display shows estimated cost
- [ ] Verify post-batch display shows actual cost and success/failure counts
- [ ] Verify batch directory created in `data/evals/batches/`
- [ ] Verify batch summary JSON has correct totals
- [ ] Verify comparison files generated in `data/evals/comparisons/`
- [ ] Verify comparison markdown is readable in VS Code/Obsidian
- [ ] Verify index file lists all comparisons with working links
- [ ] Run `uv run pidcast-eval --run-matrix --prompts v1,v2 --models llama-3.3-70b --transcripts tech-talk-01`
- [ ] Verify only specified subset runs (2 evals total)

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| Matrix run exceeds 100 evals | Warn user: "This will run 120 evals. Estimated cost: $X. Continue? [y/N]" |
| One eval fails in batch | Log error, save error metadata, continue with remaining evals |
| All evals fail in batch | Display clear summary: "0/12 evals succeeded. Check batch summary for errors" |
| Invalid subset specification | Parse error: "--prompts must be comma-separated versions (e.g., v1,v2,v3)" |

## Validation Commands

```bash
# Linting
uv run ruff check src/pidcast/evals/
uv run ruff format src/pidcast/evals/

# Unit tests
uv run pytest tests/evals/ -v

# Run single eval (Phase 1 still works)
uv run pidcast-eval --prompt_version v1 --model llama-3.3-70b --transcript_id tech-talk-01

# Run matrix eval
uv run pidcast-eval --run-matrix

# Run subset matrix
uv run pidcast-eval --run-matrix --prompts v1,v2 --models llama-3.3-70b,mixtral --transcripts tech-talk-01
```

## Rollout Considerations

- **Feature flag**: None needed
- **Monitoring**: Track batch run counts, success rates, average duration
- **Alerting**: Not applicable for Phase 2
- **Rollback plan**: Phase 2 is additive, doesn't change Phase 1 single-eval mode

## Open Items

- [ ] Decide: Should we parallelize with threads or asyncio? (Threads simpler for Groq SDK)
- [ ] Decide: Max concurrent limit of 3 - validate with Groq rate limits documentation
- [ ] Decide: Should comparison generation happen automatically or require `--generate-comparisons` flag?

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
