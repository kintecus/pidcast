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

4. **Run**:
   ```bash
   # Basic transcription
   uv run python src/yt_transcribe.py "https://youtube.com/watch?v=VIDEO_ID"

   # With LLM analysis
   uv run python src/yt_transcribe.py "VIDEO_URL" --analyze

   # Save to Obsidian vault
   uv run python src/yt_transcribe.py "VIDEO_URL" --save_to_obsidian --analyze
   ```

## External Dependencies

The following tools must be installed separately:

- **yt-dlp** - YouTube audio download
- **ffmpeg** - Audio processing
- **whisper.cpp** - Transcription engine (set `WHISPER_CPP_PATH` and `WHISPER_MODEL` in script)

## Usage Examples

```bash
# Transcribe a YouTube video
./src/yt_transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcribe with verbose output
./src/yt_transcribe.py "VIDEO_URL" --verbose

# Analyze transcript with LLM
./src/yt_transcribe.py "VIDEO_URL" --analyze

# Extract key points instead of summary
./src/yt_transcribe.py "VIDEO_URL" --analyze --analysis_type key_points

# Transcribe local audio file
./src/yt_transcribe.py "/path/to/audio/file.mp3"

# Keep the raw transcript file
./src/yt_transcribe.py "VIDEO_URL" --keep_transcript

# Use PO Token for restricted videos
./src/yt_transcribe.py "VIDEO_URL" --po_token "client.type+TOKEN"
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

See [CLAUDE.md](src/CLAUDE.md) for detailed documentation including:
- Architecture overview
- Download strategies
- LLM analysis configuration
- Custom prompt templates
- File organization
- Common patterns

## License

MIT
