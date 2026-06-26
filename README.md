# pidcast

> [!TIP]
> ✨ ***A URL or audio file in, an Obsidian-ready Markdown transcript with smart LLM analysis out.***

pidcast is a single-shot CLI for podcast and YouTube transcription with optional LLM analysis. Built for engineers and knowledge-workers who want **searchable, source-attributed notes** out of long-form audio — without leaving the terminal. Whisper.cpp or ElevenLabs handles speech-to-text; Groq or Claude handles the analysis; output lands as Markdown with YAML front matter, ready for Obsidian.

> [!NOTE]
> [📦 Installation](#installation) · [🔧 Setup](#setup) · [⚡ Usage](#usage) · [🎙️ Library](#library) · [🗣️ Speaker diarization](#diarization) · [📊 Evals](#evals) · [📚 Documentation](#documentation)

## 🎬 Demo <a name="demo"></a>

![pidcast transcribing and analyzing a YouTube episode](assets/screenshots/pidcast-example.png)

```bash
$ uv run pidcast "https://youtube.com/watch?v=VIDEO_ID"
[1/5] Downloading audio ............................. ok (3m 21s)
[2/5] Normalizing to 16kHz mono WAV ................. ok
[3/5] Transcribing (whisper.cpp, large-v3-turbo) .... ok (4m 02s)
[4/5] Analyzing (groq, llama-3.3-70b-versatile) ..... ok
[5/5] Writing Markdown .............................. ok
→ 2026-05-11_Episode_Title.md  (12,847 words, 3 speakers)
```

## 📦 Installation <a name="installation"></a>

**Requirements:** Python ≥ 3.10, `ffmpeg`, and (for local transcription) a built `whisper.cpp`.

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and sync
git clone https://github.com/kintecus/pidcast && cd pidcast && uv sync

# 3. Optional: ffmpeg
brew install ffmpeg              # macOS — or: apt install ffmpeg
```

## 🔧 Setup <a name="setup"></a>

```bash
uv run pidcast setup
```

The wizard configures:

- **Transcription backend** — local `whisper.cpp` (private, free) or ElevenLabs Scribe v2 (cloud, faster)
- **API keys** — Groq for analysis, optionally ElevenLabs for cloud transcription, optionally HuggingFace for diarization
- **Output paths** — transcript directory and optional Obsidian vault

At any time, check the live state of your tooling and env:

```bash
uv run pidcast doctor
```

Configuration is read from `.env` (see `.env.example`). Full variable reference: [docs/development-guide.md](docs/development-guide.md#environment-variables).

## ⚡ Usage <a name="usage"></a>

```bash
# YouTube, Apple Podcasts URL, or local audio file
uv run pidcast "https://youtube.com/watch?v=VIDEO_ID"
uv run pidcast "https://podcasts.apple.com/.../id123?i=456"
uv run pidcast "/path/to/audio.mp3"

# Pick a transcription provider
uv run pidcast "URL" --transcription-provider elevenlabs

# Pick an LLM provider for analysis
uv run pidcast "URL" --provider claude --claude-model opus

# Pick an analysis type (or skip analysis entirely)
uv run pidcast "URL" -a key_points
uv run pidcast "URL" --no-analyze

# Custom front matter tags (overrides auto-inferred source tags)
uv run pidcast "/path/to/meeting.mp3" --tags meeting,standup,weekly

# Test settings on a 2-minute slice before committing to a long run
uv run pidcast "URL" --test-segment

# Bias transcription toward domain terms (proper nouns, jargon) via whisper --prompt.
# Cheapest accuracy win for niche vocabulary - beats jumping to a bigger model.
uv run pidcast "/path/to/interview.mp3" --glossary adtech-ai     # named glossary from config/glossaries.yaml
uv run pidcast "URL" --whisper-prompt "Names: Acme, Foobar, gRPC."  # raw inline prompt

# Reuse an existing transcript
uv run pidcast --analyze-existing transcript.md          # re-analyze
uv run pidcast --diarize-existing transcript.md          # retry diarization

# Discovery flags
uv run pidcast -L     # list analysis types
uv run pidcast -M     # list LLM models
uv run pidcast -W     # list Whisper models
uv run pidcast -P     # list presets
```

Available analysis types: `executive_summary` (default), `summary`, `key_points`, `action_items`, `comprehensive`. Every type ends with a **Shareable Brief** — a punchy headline (≤ 15 words) plus a 3–5 sentence quick-take suitable for sending to a friend.

Claude model aliases: `sonnet` (claude-sonnet-4-6, default), `opus` (claude-opus-4-6), `haiku` (claude-haiku-4-5).

## 🎙️ Library <a name="library"></a>

Subscribe to podcast feeds and process new episodes in one command.

```bash
# Add a show by name (searches Apple Podcasts + iTunes) or by RSS URL
uv run pidcast lib add "99% Invisible"
uv run pidcast lib add "https://feeds.example.com/podcast.xml"

# Inspect
uv run pidcast lib list                    # all shows
uv run pidcast lib show 1 --episodes 10    # recent episodes for show ID 1

# Process one episode
uv run pidcast lib process "99% Invisible" --latest
uv run pidcast lib process "99% Invisible" --match "episode title"

# Process all new episodes across the library
uv run pidcast lib sync

# Roll up recent processing history into a digest
uv run pidcast lib digest
```

Library file: `~/.config/pidcast/library.yaml` (macOS/Linux) or `%APPDATA%\pidcast\library.yaml` (Windows). Human-readable, hand-editable.

## 🗣️ Speaker diarization <a name="diarization"></a>

Identify who said what. Two paths:

- **whisper + pyannote** (`--diarize` with `--transcription-provider whisper`): runs locally, needs a HuggingFace token to download the model on first use. Install with `uv pip install 'pidcast[diarize]'`, set `HUGGINGFACE_TOKEN` in `.env`, and accept the license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
- **ElevenLabs built-in** (`--diarize` with `--transcription-provider elevenlabs`): no extra setup, runs as part of the transcription API call.

```bash
uv run pidcast "URL" --diarize
```

Output adds `**Speaker 1**` / `**Speaker 2**` labels inline and front matter fields `speaker_count` and `diarized: true`.

## 📊 Evals <a name="evals"></a>

Compare provider output quality on the same transcript, judged by Claude Opus:

```bash
uv run pidcast-eval --compare groq,claude --transcript-file transcript.txt --title "Episode"
uv run pidcast-eval --run-matrix           # all prompt × model × reference combos
```

Reports land in `data/evals/comparisons/`. See [docs/development-guide.md](docs/development-guide.md#provider-comparison-evals) for the full eval matrix flags.

## 📚 Documentation <a name="documentation"></a>

- [Architecture](docs/architecture.md) — system structure, data flow, component boundaries
- [Development guide](docs/development-guide.md) — env setup, testing, CI, conventions
- [Architecture Decision Records](docs/adr/) — why-we-chose decisions
- [CLAUDE.md](CLAUDE.md) — Claude Code agent instructions for this repo
