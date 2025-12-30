# Pidcast - YouTube Transcription Tool

A powerful YouTube transcription tool that downloads audio from YouTube videos and transcribes them using Whisper, with optional LLM-based analysis using Groq AI.

## Features

- 🎙️ **YouTube Audio Download** with multiple fallback strategies
- 📝 **Whisper Transcription** using whisper.cpp (local)
- 🤖 **LLM Analysis** with Groq AI (summary, key points, action items)
- 📄 **Markdown Output** with YAML front matter for Obsidian
- 📊 **Smart Filenames** with automatic date prefixes
- ⚡ **Fast Dependencies** managed with uv
- ✨ **Code Quality** enforced with ruff

![Pidcast example run](assets/screenshots/pidcast-example.png)

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

## Evaluating LLM Analysis (Evals)

The `pidcast-eval` tool allows you to systematically test and compare different prompts and LLM models for transcript analysis.

### Quick Start

```bash
# Run a single eval
uv run pidcast-eval \
  --prompt-version v1 \
  --model llama-3.3-70b-versatile \
  --transcript claude-skills-tech-talk

# Run matrix eval (all combinations)
uv run pidcast-eval --run-matrix \
  --prompts v1,v2 \
  --models llama-3.3-70b-versatile,llama-3.1-8b-instant \
  --transcripts veal-stew-recipe,agents-vs-skills-interview
```

### Features

- **Prompt Versioning**: Test different prompt versions side-by-side
- **Model Comparison**: Compare outputs from different LLM models
- **Matrix Evaluation**: Run all combinations of prompts × models × transcripts
- **Side-by-Side Comparisons**: Auto-generated markdown comparisons
- **Cost Tracking**: Track and aggregate API costs over time
- **Retry Logic**: Automatic retry with exponential backoff for transient errors
- **Pre-flight Validation**: Validates configuration before running expensive batches

### Available Reference Transcripts

The evals system includes 3 curated reference transcripts:
- `claude-skills-tech-talk` - Technical podcast (33 min)
- `agents-vs-skills-interview` - Conference talk (16 min)
- `veal-stew-recipe` - Tutorial (6 min)

### Managing Prompts

Prompts are stored in `config/eval_prompts.json` with versioning:

```json
{
  "prompts": {
    "summary": [
      {
        "version": "v1",
        "name": "Summary Analysis v1",
        "system_prompt": "...",
        "user_prompt": "...",
        "max_output_tokens": 2000
      },
      {
        "version": "v2",
        "name": "Summary Analysis v2 - Executive Brief",
        "system_prompt": "...",
        "user_prompt": "...",
        "max_output_tokens": 1500
      }
    ]
  }
}
```

### Viewing Results

Results are saved to `data/evals/`:
- `runs/` - Individual eval runs (markdown + JSON metadata)
- `comparisons/` - Side-by-side comparison files
- `cost_tracking.json` - Append-only cost log

### CLI Options

```bash
# Single eval mode
uv run pidcast-eval \
  --prompt-version v1 \
  --model llama-3.3-70b-versatile \
  --transcript claude-skills-tech-talk \
  [--verbose]

# Matrix mode
uv run pidcast-eval --run-matrix \
  --prompts v1,v2 \
  --models llama-3.3-70b-versatile,llama-3.1-8b-instant \
  --transcripts transcript-id-1,transcript-id-2 \
  [--max-concurrent 3] \
  [--skip-confirmation] \
  [--verbose]
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
