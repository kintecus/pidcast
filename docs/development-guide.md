# Development guide

Generated: 2026-03-12 | Scan level: Quick

## Prerequisites

- **Python** >=3.10
- **uv** - Fast Python package manager ([install](https://astral.sh/uv/install.sh))
- **ffmpeg** - Audio processing (install via system package manager)
- **whisper.cpp** - Transcription engine ([build from source](https://github.com/ggerganov/whisper.cpp))

## Environment setup

1. Clone the repository:

   ```bash
   git clone https://github.com/kintecus/pidcast
   cd pidcast
   ```

2. Install dependencies:

   ```bash
   uv sync          # Runtime dependencies
   uv sync --group dev  # Include dev dependencies (ruff, pytest)
   ```

3. Configure environment:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and tool paths
   ```

   Required variables:
   - `GROQ_API_KEY` - Groq API key ([console.groq.com](https://console.groq.com/))
   - `WHISPER_CPP_PATH` - Path to whisper.cpp binary
   - `WHISPER_MODEL` - Path to Whisper model file

   Optional variables:
   - `OBSIDIAN_VAULT_PATH` - For `--save-to-obsidian` flag
   - `HUGGINGFACE_TOKEN` - For speaker diarization (`--diarize`)
   - `FFMPEG_PATH` - Custom ffmpeg path (defaults to `ffmpeg`)

4. Set up pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Running the tool

```bash
# Main transcription
uv run pidcast "https://youtube.com/watch?v=VIDEO_ID"

# With analysis type
uv run pidcast "VIDEO_URL" -a key_points

# Skip analysis
uv run pidcast "VIDEO_URL" --no-analyze

# Local audio file
uv run pidcast "/path/to/file.mp3"

# Evaluation CLI
uv run pidcast-eval --compare groq,claude --transcript-file transcript.txt
```

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_chunking.py

# Verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "test_model"
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Code quality

```bash
# Lint
uv run ruff check src/

# Auto-fix lint issues
uv run ruff check --fix src/

# Format
uv run ruff format src/

# Check formatting without changes
uv run ruff format --check src/
```

Ruff configuration is in `pyproject.toml` under `[tool.ruff]`:

- Target: Python 3.10
- Line length: 100
- Rules: E, W, F, I, N, UP, B, C4, SIM

## Adding dependencies

```bash
# Runtime dependency
uv add package-name

# Dev dependency
uv add --dev package-name

# Optional dependency group
# Edit pyproject.toml [project.optional-dependencies] manually

# Update all
uv sync --upgrade
```

## Project layout

```
src/pidcast/          # Main package (src layout)
config/               # External config (prompts, models)
tests/                # Test suite
data/                 # Runtime outputs (transcripts, evals, logs)
docs/                 # Documentation
scripts/              # Utility scripts
```

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/PR to main:

1. **Lint job** - `ruff check` + `ruff format --check`
2. **Test job** - `pytest`

Both jobs use `uv sync --group dev` for dependency installation.

## Conventions

- Use `ruff` for all linting and formatting
- Follow src layout (`src/pidcast/`)
- Audio pipeline standardizes to 16kHz mono WAV before transcription
- All LLM responses use JSON mode with `analysis` and `contextual_tags` fields
- Transcripts >120k chars use semantic chunking with synthesis
- Smart filename filtering applies date prefixes (`YYYY-MM-DD_Title.md`)
