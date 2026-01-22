# Project: pidcast

YouTube transcription and analysis tool using Whisper and Groq. Generates Obsidian-ready Markdown with smart metadata.

## Tech Stack

- **Runtime:** Python (managed via `uv`)
- **Audio/Video:** `yt-dlp`, `ffmpeg`, `ffprobe`
- **Transcription:** `whisper.cpp` (via `pidcast/transcription.py`)
- **LLM Analysis:** Groq API with JSON mode for structured output

## Core Commands

**Transcription:**

- **Run tool:** `uv run pidcast "URL_OR_PATH"`
- **Save to Obsidian:** `uv run pidcast "URL" --save_to_obsidian` (saves analysis only)
- **Analyze existing:** `uv run pidcast --analyze_existing transcript.md`

**Library Management:**

- **Add show:** `uv run pidcast lib add "FEED_URL"`
- **List shows:** `uv run pidcast lib list`
- **Show details:** `uv run pidcast lib show ID`
- **Remove show:** `uv run pidcast lib remove ID`
- **Sync library:** `uv run pidcast lib sync`
- **Generate digest:** `uv run pidcast lib digest`

**Environment:** Requires `.env` (see `.env.example`)

## Module Architecture

**Core Modules:**

- `cli.py` - Command-line interface, argument parsing, main transcription flow
- `analysis.py` - LLM-based transcript analysis using Groq API with JSON mode
- `transcription.py` - Whisper.cpp integration for audio-to-text transcription
- `download.py` - YouTube video downloading via yt-dlp
- `markdown.py` - Markdown file creation with YAML front matter
- `chunking.py` - Semantic transcript chunking for long content
- `model_selector.py` - LLM model selection and fallback logic
- `config.py` - Configuration constants, dataclasses, environment variables
- `utils.py` - Utility functions (filenames, logging, JSON I/O, duplicate detection)
- `exceptions.py` - Custom exception classes

**Configuration:**

- `config/prompts.yaml` - LLM analysis prompts with JSON response format
- `config/models.yaml` - LLM model definitions and rate limits
- `data/transcripts/transcription_stats.json` - Historical run statistics

## Engineering Conventions

- **Workflow:** Always "Plan then execute." Propose changes before editing.
- **File Naming:** Smart filtering applies date prefixes (`YYYY-MM-DD_Title.md`).
- **Audio Pipeline:** Standardize to 16kHz mono WAV before transcription.
- **Constraints:** Transcripts >120k chars use semantic chunking with synthesis.
- **LLM Responses:** All analysis prompts now return JSON with `analysis` and `contextual_tags` fields.
- Use ruff for linting and syntax

## Data Files & Testing

**IMPORTANT:** DO NOT read transcript files in `/data/transcripts/` unless critically necessary and explicitly approved by the user. These files contain large amounts of text and will consume significant context. If you need sample transcripts for testing, request the user to specify which files to use or create a dedicated test fixtures file.

## Progressive Disclosure

- **API Details:** Refer to `analysis.py` for LLM API integration and JSON parsing.
- **Prompting:** Templates defined in `config/prompts.yaml` with JSON structure.
- **Duplicate Detection:** See `utils.py` for `find_existing_transcription()` logic.
