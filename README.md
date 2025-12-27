# Pidcast - YouTube Transcription Tool

A powerful YouTube transcription tool that downloads audio from YouTube videos and transcribes them using Whisper, with optional LLM-based analysis using Groq AI.

## Features

- 🎙️ **YouTube Audio Download** with multiple fallback strategies
- 📝 **Whisper Transcription** using whisper.cpp
- 🤖 **LLM Analysis** with Groq AI (summary, key points, action items)
- 📄 **Markdown Output** with YAML front matter for Obsidian
- 📊 **Smart Filenames** with automatic date prefixes
- ⚡ **Fast Dependencies** managed with uv
- ✨ **Code Quality** enforced with ruff

## Quick Start

1. **Install uv** (fast Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <repo>
   cd pidcast
   uv sync
   ```

3. **Set API key** (optional, for LLM analysis):

   **Option 1: .env file (recommended)**:
   ```bash
   cp .env.example .env
   # Edit .env and set: GROQ_API_KEY=your-key-here
   ```

   **Option 2: Environment variable**:
   ```bash
   export GROQ_API_KEY="your-key"
   ```

   Get a free API key at: https://console.groq.com/

4. **Configure paths**:
   Create a `.env` file (copy from `.env.example`) and set paths for `WHISPER_CPP_PATH`, `WHISPER_MODEL`, and `OBSIDIAN_VAULT_PATH`.

5. **Run**:
   ```bash
   # Basic transcription
   uv run pidcast "https://youtube.com/watch?v=VIDEO_ID"

   # With LLM analysis
   uv run pidcast "VIDEO_URL" --analyze

   # Save to Obsidian vault
   uv run pidcast "VIDEO_URL" --save_to_obsidian --analyze
   ```

## External Dependencies

The following tools must be installed separately:

- **yt-dlp** - YouTube audio download
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

## Usage Examples

```bash
# Transcribe a YouTube video
uv run pidcast "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcribe with verbose output
uv run pidcast "VIDEO_URL" --verbose

# Analyze transcript with LLM
uv run pidcast "VIDEO_URL" --analyze

# Extract key points instead of summary
uv run pidcast "VIDEO_URL" --analyze --analysis_type key_points

# Transcribe local audio file
uv run pidcast "/path/to/audio/file.mp3"

# Keep the raw transcript file
uv run pidcast "VIDEO_URL" --keep_transcript

# Use PO Token for restricted videos
uv run pidcast "VIDEO_URL" --po_token "client.type+TOKEN"
```

## Development

### Code Quality

```bash
# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/
uv run ruff check --fix src/  # Auto-fix issues
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update all dependencies
uv sync --upgrade
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation including:
- Architecture overview
- Download strategies
- LLM analysis configuration
- Custom prompt templates
- File organization
- Common patterns

## License

MIT
