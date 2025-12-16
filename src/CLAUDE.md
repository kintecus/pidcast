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
```

## Dependencies

The tool requires external binaries to be installed and in PATH:
- `yt-dlp` - YouTube audio download
- `ffmpeg` - Audio processing
- `whisper.cpp` - Transcription engine

Hardcoded paths in `yt_transcribe.py:15-17`:
- `WHISPER_CPP_PATH` - Path to whisper-cli binary
- `WHISPER_MODEL` - Path to Whisper model (.bin file)
- `OBSIDIAN_PATH` - Path to Obsidian vault folder

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
- Used for estimating future transcription times via historical ratio analysis

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

## File Organization

```
src/
├── yt_transcribe.py      # Main script (all functionality in single file)
├── transcripts/          # Default output directory
│   ├── *.md             # Generated Markdown files
│   └── transcription_stats.json  # Performance tracking
└── .gitignore           # Excludes .env, transcripts/, temp files
```

## Notes

- The tool creates temporary audio files (`temp_audio.wav`) that are cleaned up automatically
- Obsidian path is hardcoded to iCloud Drive location
- All configuration constants are at the top of yt_transcribe.py (lines 12-26)
- The script is executable and uses `#!/usr/bin/env python3` shebang
