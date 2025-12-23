# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YouTube transcription tool (`pidcast`) that downloads audio from YouTube videos and transcribes them using Whisper. It generates smart, concise filenames and creates Markdown files with YAML front matter for Obsidian integration.

## Running the Tool

Basic usage:
```bash
./yt_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Common options:
```bash
# Save to Obsidian vault
./yt_transcribe.py "VIDEO_URL" --save_to_obsidian

# Verbose output (useful for debugging)
./yt_transcribe.py "VIDEO_URL" --verbose

# Specify output directory
./yt_transcribe.py "VIDEO_URL" --output_dir ./my_transcripts

# Keep transcript .txt file alongside .md file
./yt_transcribe.py "VIDEO_URL" --keep_transcript

# Use PO Token for restricted videos
./yt_transcribe.py "VIDEO_URL" --po_token "client.type+TOKEN"

# Enable LLM-based transcript analysis (requires Groq API key)
./yt_transcribe.py "VIDEO_URL" --analyze

# Analyze with specific prompt template
./yt_transcribe.py "VIDEO_URL" --analyze --analysis_type key_points

# Use different Groq model
./yt_transcribe.py "VIDEO_URL" --analyze --groq_model llama-3.1-8b-instant
```

## Dependencies

The tool requires external binaries to be installed and in PATH:
- `yt-dlp` - YouTube audio download
- `ffmpeg` - Audio processing
- `whisper.cpp` - Transcription engine

Python packages for LLM analysis (optional):
- `groq` - Groq API client for transcript analysis

Install with:
```bash
pip install groq
```

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
./yt_transcribe.py "VIDEO_URL" --groq_api_key "your-api-key-here"
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

4. Run the script:
   ```bash
   uv run python src/yt_transcribe.py "VIDEO_URL"
   # Or activate the venv first:
   source .venv/bin/activate
   ./src/yt_transcribe.py "VIDEO_URL"
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

Hardcoded paths in `yt_transcribe.py:15-17`:
- `WHISPER_CPP_PATH` - Path to whisper-cli binary
- `WHISPER_MODEL` - Path to Whisper model (.bin file)
- `OBSIDIAN_PATH` - Path to Obsidian vault folder

LLM Analysis configuration (yt_transcribe.py:28-42):
- `DEFAULT_ANALYSIS_PROMPTS_FILE` - Path to analysis prompts JSON
- `DEFAULT_GROQ_MODEL` - Default model for analysis (llama-3.3-70b-versatile)
- `GROQ_PRICING` - Token pricing for cost estimation
- `MAX_TRANSCRIPT_LENGTH` - Maximum transcript length (120k chars)
- `ANALYSIS_TIMEOUT` - API call timeout (300s)

## Architecture

### Download Strategy System

The tool uses a multi-strategy approach for downloading YouTube audio (yt_transcribe.py:143-324):

1. **Android client** (default, most reliable)
2. **Web client** with aggressive retry settings
3. **Mixed clients** (Android + Web fallback)
4. **iOS client** (only if PO token provided)

Each strategy has MAX_DOWNLOAD_RETRIES (3) attempts with RETRY_SLEEP_SECONDS (10s) between retries. If all strategies fail, the download is aborted.

### Smart Filename Generation

The `create_smart_filename()` function (yt_transcribe.py:34-93):
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

### Statistics Tracking

Transcription statistics are saved to `transcripts/transcription_stats.json`:
- Run metadata (UUID, timestamp, video info)
- Duration metrics (total run time, transcription time, audio duration)
- Success/failure status
- Analysis metadata (if --analyze used): type, model, tokens, cost
- Used for estimating future transcription times via historical ratio analysis

### LLM-Based Transcript Analysis

Optional post-transcription analysis using Groq API (yt_transcribe.py:541-929):

**Workflow:**
1. After transcript markdown created, optionally run LLM analysis
2. Load prompt templates from `analysis_prompts.json` (auto-created if missing)
3. Truncate transcript if >120k chars (at sentence boundary)
4. Substitute variables in prompt template: {transcript}, {title}, {channel}, {duration}, {url}
5. Call Groq API with system + user prompts
6. Create separate analysis markdown file: `[original]_analysis_[type].md`
7. Track tokens, cost, and metadata in statistics

**Key Functions:**
- `load_analysis_prompts()` (line 593): Load and validate prompts JSON
- `create_default_prompts_file()` (line 541): Create default templates (summary, key_points, action_items)
- `substitute_prompt_variables()` (line 651): Replace {variable} placeholders
- `truncate_transcript()` (line 668): Truncate long transcripts at sentence boundary
- `estimate_analysis_cost()` (line 699): Calculate cost from GROQ_PRICING
- `analyze_transcript_with_llm()` (line 721): Core API integration
- `create_analysis_markdown_file()` (line 847): Create analysis output with full metadata

**Analysis Output Format:**
Markdown files with YAML front matter including:
- Analysis type and name
- Source transcript reference
- Model and provider (groq)
- Token usage (input, output, total)
- Estimated cost
- Analysis duration
- Truncation status
- Tags: ['analysis', 'ai-generated', type, 'youtube']

**Error Handling:**
- Missing API key: Abort (or skip with --skip_analysis_on_error)
- API failures: Print error, respect skip flag
- Invalid analysis type: Show available types, abort (or skip)
- Long transcripts: Auto-truncate with warning

**Prompts Configuration:**
Default templates in `analysis_prompts.json`:
- `summary`: One-paragraph overview + key points + takeaways
- `key_points`: Main topics + bullet points + quotes + insights
- `action_items`: Direct actions + resources + best practices + tips

Each template has:
- `name`: Display name
- `description`: What it does
- `system_prompt`: LLM system message
- `user_prompt`: User message with {variables}
- `max_output_tokens`: Token limit (default: 2000)

## Common Patterns

### Extending Download Strategies

Add new strategies to the `strategies` list in `download_audio_with_retry()`. Each strategy requires:
- `name`: Description for logging
- `opts`: yt-dlp options dict with `format`, `outtmpl`, `extractor_args`, etc.

### Modifying Filename Generation

Edit filler patterns in `create_smart_filename()` line 36-44 to adjust what gets stripped from titles.

### Changing Output Format

Whisper supports multiple formats (`--output_format`):
- `txt` - Plain text
- `vtt` - WebVTT subtitles
- `srt` - SubRip subtitles
- `json` - JSON with timestamps

Prefix with 'o' (e.g., `otxt`) to use original filename from Whisper.

### Customizing Analysis Prompts

Edit `analysis_prompts.json` or create a custom prompts file:

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

Available variables for substitution:
- `{transcript}`: Full transcript text
- `{title}`: Video title
- `{channel}`: Channel/uploader name
- `{duration}`: Video duration (e.g., "1h 23m 45s")
- `{url}`: Video URL

Use custom prompts:
```bash
./yt_transcribe.py "VIDEO_URL" --analyze --analysis_type custom_analysis
```

## File Organization

```
src/
├── yt_transcribe.py              # Main script (all functionality in single file)
├── analysis_prompts.json         # LLM analysis prompt templates (auto-created)
├── transcripts/                  # Default output directory
│   ├── YYYY-MM-DD_*.md          # Generated transcript Markdown files
│   ├── YYYY-MM-DD_*_analysis_*.md  # Generated analysis Markdown files
│   └── transcription_stats.json  # Performance tracking (includes analysis metadata)
└── .gitignore                    # Excludes .env, transcripts/, temp files
```

## Notes

- The tool creates temporary audio files (`temp_audio.wav`) that are cleaned up automatically
- Obsidian path is hardcoded to iCloud Drive location
- All configuration constants are at the top of yt_transcribe.py (lines 12-42)
- The script is executable and uses `#!/usr/bin/env python3` shebang
- LLM analysis is optional and requires `groq` package + API key
- Analysis prompts file is auto-created with default templates on first use
- Long transcripts (>120k chars) are automatically truncated at sentence boundaries
- Analysis costs are estimated and displayed before API calls
- Use `--skip_analysis_on_error` to continue even if analysis fails
