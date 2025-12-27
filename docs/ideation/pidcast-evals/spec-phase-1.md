# Implementation Spec: Pidcast-Evals - Phase 1

**PRD**: ./prd-phase-1.md
**Estimated Effort**: M (Medium - ~2-3 days)

## Technical Approach

Phase 1 creates the foundational eval infrastructure by extending the existing pidcast codebase with new modules for prompt versioning, reference transcript management, and eval orchestration. The approach prioritizes reuse of existing abstractions from `analysis.py` and `config.py` to minimize duplication.

We'll create a new module `src/pidcast/evals/` to house eval-specific logic, keeping it separate from the main transcription pipeline. The eval runner will be invoked via a new CLI entry point (`pidcast-eval` command) defined in `pyproject.toml`.

Key technical decisions:
- **Prompt storage format**: JSON with versioning metadata (similar to existing `config/analysis_prompts.json` structure)
- **Reference transcript registry**: Separate JSON config that references existing transcript markdown files
- **Results storage**: File-based storage in `data/evals/` using run IDs as directory names (no database needed)
- **CLI framework**: Extend existing argparse pattern from `cli.py`

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `src/pidcast/evals/__init__.py` | Package initialization for evals module |
| `src/pidcast/evals/prompt_manager.py` | Load, validate, and retrieve versioned prompts from JSON |
| `src/pidcast/evals/reference_transcripts.py` | Load and validate reference transcript registry |
| `src/pidcast/evals/runner.py` | Core eval runner logic - orchestrates single eval execution |
| `src/pidcast/evals/results.py` | Results storage and metadata management |
| `src/pidcast/evals/cli.py` | CLI entry point for pidcast-eval command |
| `config/eval_prompts.json` | Versioned prompt definitions |
| `config/reference_transcripts.json` | Reference transcript registry |
| `data/evals/references/.gitkeep` | Ensure directory exists in git |
| `data/evals/runs/.gitkeep` | Ensure directory exists in git |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `pyproject.toml` | Add `pidcast-eval` CLI entry point in `[project.scripts]` section |
| `.gitignore` | Add `data/evals/runs/*/` to ignore eval results (keep references) |

### Deleted Files

None.

## Implementation Details

### Prompt Manager (`prompt_manager.py`)

**Pattern to follow**: `src/pidcast/analysis.py:load_analysis_prompts()`

**Overview**: Manages loading, validating, and retrieving versioned prompts from `config/eval_prompts.json`. Supports querying prompts by type and version.

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class EvalPrompt:
    """Represents a versioned eval prompt."""
    prompt_type: str  # e.g., "summary", "key_points"
    version: str  # e.g., "v1", "v2"
    name: str
    description: str
    system_prompt: str
    user_prompt: str
    max_output_tokens: int

class PromptManager:
    """Manages versioned eval prompts."""

    def __init__(self, prompts_file: Path):
        self.prompts_file = prompts_file
        self._prompts: Dict[str, List[EvalPrompt]] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from JSON, validate structure."""
        # Similar to load_analysis_prompts() logic
        pass

    def get_prompt(self, prompt_type: str, version: str) -> EvalPrompt:
        """Retrieve a specific prompt by type and version."""
        pass

    def list_versions(self, prompt_type: str) -> List[str]:
        """List all versions for a given prompt type."""
        pass

    def list_prompt_types(self) -> List[str]:
        """List all available prompt types."""
        pass
```

**Key decisions**:
- Use dataclass for type safety and IDE support
- Validate JSON schema on load to catch config errors early
- Raise `ConfigurationError` for missing/invalid prompts (reuse existing exception from `exceptions.py`)

**Implementation steps**:
1. Create `EvalPrompt` dataclass with all required fields
2. Implement `_load_prompts()` to parse JSON and validate structure
3. Implement `get_prompt()` with clear error messages for not found
4. Add helper methods to list available types and versions
5. Write unit tests for load validation and retrieval

### Reference Transcript Manager (`reference_transcripts.py`)

**Pattern to follow**: Similar structure to `PromptManager`

**Overview**: Loads and validates reference transcript registry from `config/reference_transcripts.json`. Verifies that referenced markdown files exist.

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ReferenceTranscript:
    """Metadata about a reference transcript for evals."""
    transcript_id: str  # e.g., "tech-talk-01"
    name: str
    description: str
    file_path: Path  # Path to .md file
    content_type: str  # e.g., "technical_talk", "interview"
    duration_seconds: int
    source_url: str

class ReferenceTranscriptManager:
    """Manages reference transcript registry."""

    def __init__(self, registry_file: Path, project_root: Path):
        self.registry_file = registry_file
        self.project_root = project_root
        self._transcripts: Dict[str, ReferenceTranscript] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from JSON, validate files exist."""
        pass

    def get_transcript(self, transcript_id: str) -> ReferenceTranscript:
        """Retrieve reference transcript by ID."""
        pass

    def list_transcript_ids(self) -> List[str]:
        """List all available transcript IDs."""
        pass

    def read_transcript_content(self, transcript_id: str) -> str:
        """Read the full markdown content of a reference transcript."""
        pass
```

**Key decisions**:
- Validate that markdown files exist on load (fail fast)
- Store absolute paths internally but accept relative paths in JSON
- Extract transcript content by parsing markdown (strip YAML front matter if needed)

**Implementation steps**:
1. Create `ReferenceTranscript` dataclass
2. Implement JSON loading with file existence validation
3. Implement transcript retrieval and content reading
4. Add helper to extract plain transcript text from markdown (handle YAML front matter)
5. Write unit tests with fixture transcript files

### Eval Runner (`runner.py`)

**Pattern to follow**: `src/pidcast/cli.py:main()` orchestration pattern

**Overview**: Orchestrates a single eval run - loads prompt, loads transcript, calls Groq API, saves results.

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class EvalConfig:
    """Configuration for a single eval run."""
    prompt_type: str
    prompt_version: str
    model: str
    transcript_id: str
    groq_api_key: str

@dataclass
class EvalResult:
    """Result of an eval run."""
    run_id: str
    timestamp: datetime
    prompt_type: str
    prompt_version: str
    model: str
    transcript_id: str
    output_text: str
    tokens_input: int
    tokens_output: int
    estimated_cost: float
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None

class EvalRunner:
    """Runs individual evals and saves results."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        transcript_manager: ReferenceTranscriptManager,
        results_dir: Path,
    ):
        self.prompt_manager = prompt_manager
        self.transcript_manager = transcript_manager
        self.results_dir = results_dir

    def run_eval(self, config: EvalConfig) -> EvalResult:
        """Execute a single eval run."""
        # 1. Generate run_id
        # 2. Load prompt and transcript
        # 3. Substitute prompt variables (reuse from analysis.py)
        # 4. Call Groq API (reuse from analysis.py)
        # 5. Parse response, extract tokens
        # 6. Calculate cost
        # 7. Save results
        # 8. Return EvalResult
        pass

    def _generate_run_id(self, config: EvalConfig) -> str:
        """Generate run ID: {timestamp}_{version}_{model}_{transcript}."""
        pass

    def _save_results(self, result: EvalResult, output_dir: Path) -> None:
        """Save result.md and metadata.json to run directory."""
        pass
```

**Key decisions**:
- Reuse `analyze_transcript_with_llm()` from `analysis.py` for API calls
- Reuse `substitute_prompt_variables()` for {transcript}, {title}, etc.
- Generate run_id before API call so we know where to save on failure
- Save metadata immediately after API response (don't batch)

**Implementation steps**:
1. Create `EvalConfig` and `EvalResult` dataclasses
2. Implement `_generate_run_id()` with timestamp formatting
3. Implement main `run_eval()` orchestration
4. Reuse existing Groq API call logic from `analysis.py`
5. Implement `_save_results()` to write markdown + JSON
6. Add error handling and logging

### Results Storage (`results.py`)

**Pattern to follow**: `src/pidcast/utils.py:save_statistics()`

**Overview**: Handles saving eval results to filesystem with proper directory structure and metadata formatting.

```python
def save_eval_result(result: EvalResult, output_dir: Path) -> None:
    """Save eval result as markdown + metadata JSON."""
    # Create run directory: data/evals/runs/{run_id}/
    # Write result.md with YAML front matter
    # Write metadata.json
    pass

def create_result_markdown(result: EvalResult) -> str:
    """Generate markdown with YAML front matter."""
    # YAML front matter: run_id, timestamp, prompt, model, transcript, tokens, cost
    # Markdown body: LLM output
    pass

def create_metadata_json(result: EvalResult) -> dict:
    """Convert EvalResult to JSON-serializable dict."""
    pass
```

**Key decisions**:
- YAML front matter format matches existing analysis markdown files
- Store ISO timestamp in metadata for sorting
- Include all run parameters in metadata for reproducibility

**Implementation steps**:
1. Implement directory creation with parents
2. Implement YAML front matter generation (use existing pattern from `markdown.py`)
3. Implement JSON serialization of EvalResult
4. Add error handling for file write failures

### CLI Entry Point (`evals/cli.py`)

**Pattern to follow**: `src/pidcast/cli.py:parse_arguments()`

**Overview**: Argument parsing and main entry point for `pidcast-eval` command.

```python
import argparse
from pathlib import Path

def parse_eval_arguments() -> argparse.Namespace:
    """Parse CLI arguments for pidcast-eval."""
    parser = argparse.ArgumentParser(description="Run LLM analysis evals")
    parser.add_argument("--prompt_version", required=True, help="Prompt version (e.g., v1)")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama-3.3-70b)")
    parser.add_argument("--transcript_id", required=True, help="Reference transcript ID")
    parser.add_argument("--groq_api_key", help="Groq API key (or use env var)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()

def eval_main() -> None:
    """Main entry point for pidcast-eval command."""
    # 1. Parse arguments
    # 2. Load config (API key from env if not provided)
    # 3. Initialize managers (PromptManager, ReferenceTranscriptManager)
    # 4. Initialize EvalRunner
    # 5. Create EvalConfig
    # 6. Run eval
    # 7. Display results summary
    pass
```

**Key decisions**:
- Separate CLI from main pidcast command (standalone tool)
- Reuse existing config loading from `config.py` (PROJECT_ROOT, GROQ_API_KEY)
- Display rich terminal output using existing `rich` integration

**Implementation steps**:
1. Create argument parser with required flags
2. Implement config loading and validation
3. Initialize all managers and runner
4. Execute eval and handle errors
5. Display summary with rich formatting

### Bootstrap Scripts

Create helper scripts to populate initial config files:

**`config/eval_prompts.json` (initial content)**:
```json
{
  "prompts": {
    "summary": [
      {
        "version": "v1",
        "name": "Summary Analysis",
        "description": "Generate concise summary with key points",
        "system_prompt": "You are an expert at analyzing and summarizing content.",
        "user_prompt": "Analyze this transcript and provide:\n\n1. One-paragraph summary\n2. Key points (3-5 bullets)\n3. Main takeaways\n\nTitle: {title}\nDuration: {duration} seconds\nURL: {url}\n\nTranscript:\n{transcript}",
        "max_output_tokens": 1000
      }
    ],
    "key_points": [
      {
        "version": "v1",
        "name": "Key Points Extraction",
        "description": "Extract main topics and insights",
        "system_prompt": "You are an expert at identifying key information.",
        "user_prompt": "Extract key points from this transcript:\n\nTitle: {title}\n\n{transcript}",
        "max_output_tokens": 1500
      }
    ],
    "action_items": [
      {
        "version": "v1",
        "name": "Action Items",
        "description": "Identify actionable steps",
        "system_prompt": "You are an expert at extracting actionable information.",
        "user_prompt": "Identify action items and recommendations from:\n\n{transcript}",
        "max_output_tokens": 1000
      }
    ]
  }
}
```

**`config/reference_transcripts.json` (template)**:
```json
{
  "transcripts": [
    {
      "transcript_id": "example-tech-talk",
      "name": "Example Tech Talk",
      "description": "Sample technical presentation for eval testing",
      "file_path": "data/evals/references/example-tech-talk.md",
      "content_type": "technical_talk",
      "duration_seconds": 1200,
      "source_url": "https://youtube.com/watch?v=example"
    }
  ]
}
```

**Implementation steps**:
1. Create JSON schema files with v1 prompts
2. Document how to add new versions (increment version number)
3. Create example reference transcript markdown
4. Add README in `data/evals/` explaining directory structure

## Data Model

### Prompt Versioning Schema

```json
{
  "prompts": {
    "{prompt_type}": [
      {
        "version": "v{N}",
        "name": "Human-readable name",
        "description": "What this prompt does",
        "system_prompt": "System instructions",
        "user_prompt": "User message with {variables}",
        "max_output_tokens": 1000
      }
    ]
  }
}
```

### Reference Transcript Registry Schema

```json
{
  "transcripts": [
    {
      "transcript_id": "unique-id",
      "name": "Display name",
      "description": "What makes this a good eval transcript",
      "file_path": "relative/path/to/transcript.md",
      "content_type": "technical_talk | interview | tutorial | etc",
      "duration_seconds": 1234,
      "source_url": "https://youtube.com/..."
    }
  ]
}
```

### Eval Result Metadata Schema

```json
{
  "run_id": "20251227_143022_v1_llama-3.3-70b_tech-talk-01",
  "timestamp": "2025-12-27T14:30:22Z",
  "prompt_type": "summary",
  "prompt_version": "v1",
  "model": "llama-3.3-70b-versatile",
  "transcript_id": "tech-talk-01",
  "tokens_input": 12500,
  "tokens_output": 850,
  "tokens_total": 13350,
  "estimated_cost": 0.0098,
  "duration_seconds": 4.2,
  "success": true,
  "error_message": null
}
```

## Testing Requirements

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/evals/test_prompt_manager.py` | Prompt loading, validation, retrieval |
| `tests/evals/test_reference_transcripts.py` | Registry loading, file validation |
| `tests/evals/test_runner.py` | Eval orchestration, result generation |
| `tests/evals/test_results.py` | Result storage, markdown/JSON generation |

**Key test cases**:
- Load valid prompt JSON successfully
- Raise error on malformed prompt JSON
- Raise error when prompt version doesn't exist
- Load reference registry and validate file paths
- Raise error when reference transcript file missing
- Generate correct run_id format
- Create result markdown with valid YAML front matter
- Handle API errors gracefully

### Manual Testing

- [ ] Run `uv run pidcast-eval --prompt_version v1 --model llama-3.3-70b-versatile --transcript_id tech-talk-01`
- [ ] Verify run directory created in `data/evals/runs/`
- [ ] Verify `result.md` has YAML front matter and LLM output
- [ ] Verify `metadata.json` contains all required fields
- [ ] Test with invalid prompt version, expect clear error
- [ ] Test with invalid transcript ID, expect clear error
- [ ] Test without GROQ_API_KEY set, expect clear error

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| Prompt version not found | Raise `ConfigurationError` with message: "Prompt version 'v5' not found for type 'summary'. Available: v1, v2, v3" |
| Reference transcript missing | Raise `ConfigurationError` with message: "Reference transcript 'unknown-id' not found. Available: {list}" |
| Transcript file doesn't exist | Raise `FileProcessingError` with message: "Transcript file not found: {path}" |
| Groq API key not set | Raise `ConfigurationError` with message: "GROQ_API_KEY not configured. Set env var or use --groq_api_key" |
| Groq API error | Catch exception, save error metadata, display clear message |
| File write error | Raise `FileProcessingError` with message about permissions/disk space |

## Validation Commands

```bash
# Linting
uv run ruff check src/pidcast/evals/
uv run ruff format src/pidcast/evals/

# Type checking (if using mypy in future)
# uv run mypy src/pidcast/evals/

# Unit tests
uv run pytest tests/evals/ -v

# Run eval (manual test)
uv run pidcast-eval --prompt_version v1 --model llama-3.3-70b-versatile --transcript_id example-tech-talk
```

## Rollout Considerations

- **Feature flag**: None needed (standalone tool)
- **Monitoring**: Track eval run counts and success rates in future phase
- **Alerting**: Not applicable for Phase 1
- **Rollback plan**: Phase 1 is additive only, no changes to main pidcast - can simply not use the new command

## Open Items

- [ ] Decide: Should we bootstrap `eval_prompts.json` from existing `analysis_prompts.json` automatically or manually curate?
- [ ] Decide: Extract 2-3 good reference transcripts from existing `data/transcripts/` or create new ones?
- [ ] Decide: Should eval CLI be `pidcast-eval` or `pidcast eval` (subcommand)?

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
