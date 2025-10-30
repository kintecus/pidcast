# YouTube Transcription Tool

Automated YouTube video transcription using yt-dlp and Whisper, with smart filename generation and Obsidian integration.

## Requirements

- Python 3.x
- `yt-dlp` (audio download)
- `ffmpeg` (audio processing)
- `whisper.cpp` (transcription)

## Usage

```bash
./yt_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Key Features

- **Smart Filenames**: Auto-generates concise, dated filenames from video titles
- **Obsidian Integration**: Direct save to Obsidian vault with front matter
- **Statistics Tracking**: Records transcription performance metrics
- **Retry Logic**: Handles transient download failures
- **Time Estimation**: Predicts transcription duration from historical data

## Options

- `--output_dir`: Output directory (default: `./transcripts`)
- `--save_to_obsidian`: Save directly to Obsidian vault
- `--keep_transcript`: Keep .txt transcript alongside .md file
- `--whisper_model`: Path to Whisper model file
- `--front_matter`: JSON string for custom Obsidian metadata
- `--stats_file`: Statistics file path (default: `./transcripts/transcription_stats.json`)
- `--po_token`: PO Token for bypassing YouTube restrictions
- `--verbose`: Enable detailed output

## Output

- Markdown files with YAML front matter (title, date, URL, channel, duration)
- Full transcript text
- Automatic versioning for duplicate filenames
