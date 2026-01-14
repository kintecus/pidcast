# Project: pidcast

YouTube transcription and analysis tool using Whisper and Groq. Generates Obsidian-ready Markdown with smart metadata.

## Tech Stack
- **Runtime:** Python (managed via `uv`)
- **Audio/Video:** `yt-dlp`, `ffmpeg`, `ffprobe`
- **Transcription:** `whisper.cpp` (via `pidcast/transcription.py`)
- **LLM Analysis:** Groq API

## Core Commands
- **Run tool:** `uv run pidcast "URL_OR_PATH"`
- **With analysis:** `uv run pidcast "URL" --analyze`
- **Environment:** Requires `.env` (see `.env.example`)

## Engineering Conventions
- **Workflow:** Always "Plan then execute." Propose changes before editing.
- **File Naming:** Smart filtering applies date prefixes (`YYYY-MM-DD_Title.md`).
- **Audio Pipeline:** Standardize to 16kHz mono WAV before transcription.
- **Constraints:** Transcripts >120k chars are truncated at the nearest sentence boundary.

## Progressive Disclosure
- **API Details:** Refer to `pidcast/transcription.py` for audio processing logic.
- **Prompting:** Templates are auto-generated in `analysis_prompts.yaml`.
