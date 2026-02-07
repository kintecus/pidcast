# Pidcast - Podcast & YouTube Transcription Tool

Transcription and LLM-powered analysis tool for podcasts, YouTube videos, and local audio files. Uses whisper.cpp for local transcription and Groq for structured analysis. Outputs Obsidian-ready Markdown with YAML front matter.

## Features

- **Multiple input sources** - YouTube videos, podcast RSS feeds, and local audio files
- **Whisper transcription** using whisper.cpp (local, fast)
- **LLM analysis** with Groq AI (enabled by default)
  - Automatic model fallback and retry logic
  - Smart chunking for long transcripts with semantic boundaries
  - JSON-validated structured output
- **Library management** - Manage podcast RSS feeds with persistent storage
  - Add/remove shows from your library
  - Sync and process new episodes automatically
  - Generate digests from processing history
- **Markdown output** with YAML front matter and contextual tags
- **Smart filenames** with date prefixes
- **Fast dependencies** managed with uv

![Pidcast example run](assets/screenshots/pidcast-example.png)

## Quick start

1. **Install uv**:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:

   ```bash
   git clone https://github.com/kintecus/pidcast
   cd pidcast
   uv sync
   ```

3. **Configure** (copy `.env.example` to `.env` and set):
   - `GROQ_API_KEY` - Get free key at <https://console.groq.com/>
   - `WHISPER_CPP_PATH` - Path to whisper.cpp main binary
   - `WHISPER_MODEL` - Path to Whisper model file
   - `OBSIDIAN_VAULT_PATH` - (Optional) For `--save_to_obsidian`

4. **Run**:

   ```bash
   # Transcribe a YouTube video with analysis (default)
   uv run pidcast "https://youtube.com/watch?v=VIDEO_ID"

   # Transcribe a local audio file
   uv run pidcast "/path/to/audio/file.mp3"

   # Skip LLM analysis
   uv run pidcast "VIDEO_URL" --no-analyze

   # Save to Obsidian vault
   uv run pidcast "VIDEO_URL" -o
   ```

## External dependencies

The following tools must be installed separately (all Python dependencies are handled by `uv sync`):

- **ffmpeg** - Audio processing
- **whisper.cpp** - Transcription engine

### Installing whisper.cpp

1. Clone and build [whisper.cpp](https://github.com/ggerganov/whisper.cpp):

   ```bash
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   ```

2. Download a Whisper model:

   ```bash
   bash ./models/download-ggml-model.sh base.en
   ```

3. Configure paths in `.env`:

   ```bash
   WHISPER_CPP_PATH=/path/to/whisper.cpp/main
   WHISPER_MODEL=/path/to/whisper.cpp/models/ggml-base.en.bin
   ```

## Usage examples

### Single video/episode transcription

```bash
# Basic transcription with analysis (default)
uv run pidcast "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcription only (skip analysis)
uv run pidcast "VIDEO_URL" --no-analyze

# Different analysis type
uv run pidcast "VIDEO_URL" -a key_points

# Analyze existing transcript without re-transcribing
uv run pidcast --analyze_existing transcript.md

# Transcribe local audio file
uv run pidcast "/path/to/audio/file.mp3"

# Force re-transcription (skip duplicate detection)
uv run pidcast "VIDEO_URL" -f

# Verbose output
uv run pidcast "VIDEO_URL" -v

# Use PO Token for restricted YouTube videos
uv run pidcast "VIDEO_URL" --po_token "client.type+TOKEN"
```

### Library management

Manage a persistent library of podcast shows for batch processing:

```bash
# Add a podcast to your library
uv run pidcast lib add "https://feeds.example.com/podcast.xml"

# Preview episodes before adding
uv run pidcast lib add "https://feeds.example.com/podcast.xml" --preview

# List all shows in library
uv run pidcast lib list

# Show details for a specific podcast (with recent episodes)
uv run pidcast lib show 1

# Show more episodes
uv run pidcast lib show 1 --episodes 10

# Remove a show from library
uv run pidcast lib remove 1

# Sync library and process new episodes
uv run pidcast lib sync

# Process a specific episode from a show
uv run pidcast lib process "show name" --latest
uv run pidcast lib process "show name" --match "episode title"

# Generate digest from processing history
uv run pidcast lib digest
```

The library is stored at `~/.config/pidcast/library.yaml` (or `%APPDATA%\pidcast\library.yaml` on Windows) and is human-readable and editable.

## Analysis prompts and configuration

### Prompt templates

Prompts are configured in `config/prompts.yaml`. Each prompt template defines:

- System and user prompts with variable substitution
- Max output tokens
- JSON response format for structured output with `analysis` and `contextual_tags`

Available analysis types:

- `executive_summary` (default) - Concise overview with key insights
- `key_points` - Bulleted highlights
- `action_items` - Actionable takeaways

### Model configuration

Models and fallback chains are defined in `config/models.yaml`:

- Automatic fallback on rate limits or failures
- Token-based model selection for long transcripts
- Smart chunking for content exceeding context windows

### Chunking strategy

Long transcripts are automatically chunked with:

- Semantic boundary detection (paragraph/sentence breaks)
- Overlap between chunks for context preservation
- Synthesis step to combine chunk analyses

## Development

### Code quality

Pre-commit hooks run ruff linting and formatting automatically on each commit. To set up:

```bash
pre-commit install
```

Manual usage:

```bash
uv run ruff format src/
uv run ruff check src/
uv run ruff check --fix src/
```

### Adding dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade
```

## License

MIT
