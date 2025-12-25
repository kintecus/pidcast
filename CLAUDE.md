# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YouTube transcription tool (`pidcast`) that downloads audio from YouTube videos and transcribes them using Whisper. It generates smart, concise filenames and creates Markdown files with YAML front matter for Obsidian integration. Optional LLM-based analysis is available via Groq API.

## Running the Tool

Basic usage:
```bash
# Using the installed command
uv run pidcast "https://www.youtube.com/watch?v=VIDEO_ID"

# Or using python -m
uv run python -m pidcast "VIDEO_URL"
```

Common options:
```bash
# Save to Obsidian vault
uv run pidcast "VIDEO_URL" --save_to_obsidian

# Verbose output (useful for debugging)
uv run pidcast "VIDEO_URL" --verbose

# Specify output directory
uv run pidcast "VIDEO_URL" --output_dir ./transcripts

# Keep transcript .txt file alongside .md file
uv run pidcast "VIDEO_URL" --keep_transcript

# Use PO Token for restricted videos
uv run pidcast "VIDEO_URL" --po_token "client.type+TOKEN"

# Enable LLM-based transcript analysis (requires Groq API key)
uv run pidcast "VIDEO_URL" --analyze

# Analyze with specific prompt template
uv run pidcast "VIDEO_URL" --analyze --analysis_type key_points

# Transcribe local audio file
uv run pidcast "/path/to/audio/file.mp3"
```

## Dependencies

The tool requires external binaries to be installed and in PATH:
- `yt-dlp` - YouTube audio download
- `ffmpeg` - Audio processing
- `whisper.cpp` - Transcription engine

Python packages (managed by uv):
- `yt-dlp` - YouTube downloader library
- `groq` - Groq API client for transcript analysis
- `python-dotenv` - Environment variable loading
- `rich` - Terminal formatting

Set Groq API key (choose one method):

**Option 1: .env file (recommended)**:
```bash
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=your-api-key-here
```

**Option 2: Environment variable**:
```bash
export GROQ_API_KEY="your-api-key-here"
```

**Option 3: CLI flag**:
```bash
uv run pidcast "VIDEO_URL" --groq_api_key "your-api-key-here"
```

Get a free API key at: https://console.groq.com/

## Development Setup

This project uses `uv` for fast dependency management and `ruff` for linting/formatting.

### Installation

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Configure environment variables (optional):
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   ```

4. Run the tool:
   ```bash
   uv run pidcast "VIDEO_URL"
   ```

### Code Quality

Format code:
```bash
uv run ruff format src/
```

Lint code:
```bash
uv run ruff check src/
uv run ruff check --fix src/  # Auto-fix issues
```

Run tests:
```bash
uv run pytest
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency (like a new linter)
uv add --dev package-name

# Update all dependencies
uv sync --upgrade
```

## Configuration

Configuration is managed via environment variables (with fallbacks) in `src/pidcast/config.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `WHISPER_CPP_PATH` | Path to whisper-cli binary | `/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli` |
| `WHISPER_MODEL` | Path to Whisper model (.bin file) | `/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin` |
| `OBSIDIAN_VAULT_PATH` | Path to Obsidian vault folder | (hardcoded iCloud path) |
| `FFMPEG_PATH` | Path to ffmpeg binary | `ffmpeg` |
| `GROQ_API_KEY` | Groq API key for LLM analysis | None |

LLM Analysis configuration constants in `config.py`:
- `DEFAULT_GROQ_MODEL` - Default model for analysis (`llama-3.3-70b-versatile`)
- `GROQ_PRICING` - Token pricing for cost estimation
- `MAX_TRANSCRIPT_LENGTH` - Maximum transcript length (120k chars)
- `ANALYSIS_TIMEOUT` - API call timeout (300s)

## Architecture

### Package Structure

The codebase is organized as a Python package with modular components:

```
pidcast/
├── __init__.py       # Package initialization, version
├── __main__.py       # Entry point for python -m pidcast
├── cli.py            # Argument parsing and main pipeline orchestration
├── config.py         # Constants, environment variables, dataclasses
├── download.py       # YouTube download with multi-strategy fallback
├── transcription.py  # Whisper transcription with progress display
├── markdown.py       # Markdown file creation with YAML front matter
├── analysis.py       # LLM-based transcript analysis via Groq
├── utils.py          # Logging, filename generation, JSON I/O
└── exceptions.py     # Custom exception hierarchy
```

### Dataclasses

The package uses dataclasses for type-safe structured data (`config.py`):
- `VideoInfo` - Metadata about a video or audio source
- `TranscriptionStats` - Statistics for a transcription run
- `AnalysisResult` - Result from LLM analysis
- `PromptTemplate` - A prompt template for LLM analysis
- `PromptsConfig` - Configuration for analysis prompts

### Exception Hierarchy

Custom exceptions in `exceptions.py`:
- `PidcastError` - Base exception
- `DownloadError` - Failed to download audio
- `TranscriptionError` - Whisper transcription failed
- `AnalysisError` - LLM analysis failed
- `ConfigurationError` - Configuration or setup error
- `FileProcessingError` - Error processing local audio file

### Download Strategy System

The tool uses a multi-strategy approach for downloading YouTube audio (`download.py`):

1. **Android client** (default, most reliable)
2. **Web client** with aggressive retry settings
3. **Mixed clients** (Android + Web fallback)
4. **iOS client** (only if PO token provided)

Each strategy has MAX_DOWNLOAD_RETRIES (3) attempts with RETRY_SLEEP_SECONDS (10s) between retries.

### Smart Filename Generation

The `create_smart_filename()` function in `utils.py`:
- Strips common filler patterns (Episode numbers, interview markers, etc.)
- Prioritizes capitalized words and non-stopwords
- Truncates to max_length (default 60 chars)
- Prepends YYYY-MM-DD date prefix
- Uses underscores as word separators

### Output Format

Generated Markdown files include:
- YAML front matter with metadata (title, date, URL, duration, channel, tags)
- Full transcript text from Whisper
- Automatic versioning (_v2, _v3) if filename exists

### Transcription Progress Indicator

During transcription, a real-time progress indicator is displayed:
- **With estimated time:** Shows a progress bar, percentage, elapsed time, and estimated time remaining
- **Without estimated time:** Shows a spinner with elapsed time
- Progress is disabled in verbose mode to avoid interfering with Whisper output
- Uses threading to update the display while transcription runs

### Statistics Tracking

Transcription statistics are saved to `data/transcripts/transcription_stats.json`:
- Run metadata (UUID, timestamp, video info)
- Duration metrics (total run time, transcription time, audio duration)
- Success/failure status
- Analysis metadata (if --analyze used): type, model, tokens, cost

### LLM-Based Transcript Analysis

Optional post-transcription analysis using Groq API (`analysis.py`):

**Workflow:**
1. After transcript markdown created, optionally run LLM analysis
2. Load prompt templates from `config/analysis_prompts.json` (auto-created if missing)
3. Truncate transcript if >120k chars (at sentence boundary)
4. Substitute variables in prompt template: {transcript}, {title}, {channel}, {duration}, {url}
5. Call Groq API with system + user prompts
6. Create separate analysis markdown file: `[original]_analysis_[type].md`
7. Track tokens, cost, and metadata in statistics

**Key Functions:**
- `load_analysis_prompts()` - Load and validate prompts JSON
- `create_default_prompts_file()` - Create default templates (summary, key_points, action_items)
- `substitute_prompt_variables()` - Replace {variable} placeholders
- `truncate_transcript()` - Truncate long transcripts at sentence boundary
- `estimate_analysis_cost()` - Calculate cost from GROQ_PRICING
- `analyze_transcript_with_llm()` - Core API integration
- `render_analysis_to_terminal()` - Rich terminal output

**Prompts Configuration:**
Default templates in `config/analysis_prompts.json`:
- `summary`: One-paragraph overview + key points + takeaways
- `key_points`: Main topics + bullet points + quotes + insights
- `action_items`: Direct actions + resources + best practices + tips

## Common Patterns

### Extending Download Strategies

Add new strategies in `download.py` using `build_download_strategy()`:
```python
strategies.append(
    build_download_strategy(
        "New Strategy Name",
        "config_key",  # Key in DOWNLOAD_STRATEGY_CONFIGS
        "bestaudio/best",  # yt-dlp format string
        ["client_type"],
        output_template,
        verbose,
    )
)
```

### Modifying Filename Generation

Edit `LOW_PRIORITY_WORDS` and `TITLE_FILLER_PATTERNS` in `utils.py` to adjust what gets stripped from titles.

### Customizing Analysis Prompts

Edit `config/analysis_prompts.json` or create a custom prompts file:

```json
{
  "prompts": {
    "custom_analysis": {
      "name": "Custom Analysis",
      "description": "Your custom analysis description",
      "system_prompt": "Your system instructions for the LLM",
      "user_prompt": "Analyze this transcript.\n\nTitle: {title}\n\n{transcript}",
      "max_output_tokens": 2000
    }
  }
}
```

Use custom prompts:
```bash
uv run pidcast "VIDEO_URL" --analyze --analysis_type custom_analysis
```

## File Organization

```
pidcast/
├── .env                          # Environment variables (gitignored)
├── .env.example                  # Template for .env
├── CLAUDE.md                     # This file
├── README.md                     # User-facing documentation
├── pyproject.toml                # Package configuration
├── uv.lock                       # Dependency lockfile
├── config/
│   └── analysis_prompts.json     # LLM analysis prompt templates
├── data/
│   ├── audio/                    # Audio files (gitignored)
│   └── transcripts/              # Default output directory
│       ├── YYYY-MM-DD_*.md       # Generated transcript Markdown files
│       ├── YYYY-MM-DD_*_analysis_*.md  # Generated analysis Markdown files
│       └── transcription_stats.json    # Performance tracking
├── src/
│   └── pidcast/                  # Python package
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── download.py
│       ├── transcription.py
│       ├── markdown.py
│       ├── analysis.py
│       ├── utils.py
│       └── exceptions.py
└── tests/                        # Unit tests (TODO)
```

## Notes

- The tool creates temporary audio files (`temp_audio.wav`) that are cleaned up automatically
- External tool paths can be configured via environment variables
- LLM analysis is optional and requires `groq` package + API key
- Analysis prompts file is auto-created with default templates on first use
- Long transcripts (>120k chars) are automatically truncated at sentence boundaries
- Analysis costs are estimated and displayed before API calls
- Use `--skip_analysis_on_error` to continue even if analysis fails
