# Project: pidcast

Single-shot CLI that turns a URL or audio file into an Obsidian-ready Markdown transcript with optional LLM analysis. See [README.md](README.md) for user-facing overview and [docs/architecture.md](docs/architecture.md) for system structure.

## Tech stack

- **Runtime:** Python ≥ 3.10, managed via `uv`
- **Audio:** `yt-dlp`, `ffmpeg`, `ffprobe`
- **Transcription providers:** `whisper.cpp` (local, default) and ElevenLabs Scribe v2 (cloud)
- **Diarization:** `pyannote.audio` (whisper path, needs HF token) or built-in (ElevenLabs path)
- **LLM providers:** Groq (default, JSON mode) or Claude via the Claude Code CLI

## Core commands

Run with `uv run pidcast <input>`. Useful entry points:

| Command | Purpose |
|---------|---------|
| `pidcast <URL_OR_PATH>` | Transcribe + analyze |
| `pidcast setup` | Interactive setup wizard (env, deps, API keys) |
| `pidcast doctor` | Diagnose tooling/env configuration |
| `pidcast --analyze-existing transcript.md` | Re-analyze without re-transcribing |
| `pidcast --diarize-existing transcript.md [--audio file]` | Retry diarization only |
| `pidcast --test-segment [MIN] [--start-at MIN]` | Dry-run transcription on a slice |
| `pidcast --transcription-provider {whisper,elevenlabs}` | Pick transcription backend |
| `pidcast --vad [--vad-threshold N]` | Whisper anti-hallucination: strip silence via VAD (needs `WHISPER_VAD_MODEL`) |
| `pidcast --provider {groq,claude}` | Pick LLM analysis backend |
| `pidcast lib {add,list,show,remove,sync,process,digest}` | Manage podcast subscriptions |
| `pidcast -L \| -M \| -W \| -P` | List analysis types / LLM models / Whisper models / presets |

Environment: copy `.env.example` → `.env`. See [docs/development-guide.md](docs/development-guide.md) for the full var list.

## Build, test, lint

```bash
uv sync --group dev          # install runtime + dev deps
uv run pytest                # run tests
uv run ruff check src/       # lint
uv run ruff format src/      # format
pre-commit install           # enable pre-commit hooks (ruff, mermaid)
```

## Module map

Source lives under `src/pidcast/`. Hot-path modules (read these first when debugging a workflow):

- `cli.py` — argparse surface; dispatches to `workflow.py` and the `lib` subcommands
- `workflow.py` — orchestration: download → transcribe → diarize → analyze → write
- `transcription.py` + `providers/` — provider dispatch for whisper/ElevenLabs
- `diarization.py` — pyannote integration; consumes whisper JSON or ElevenLabs speakers
- `analysis.py` + `summarization.py` — Groq/Claude prompt execution, JSON validation
- `chunking.py` — semantic chunking for transcripts > 120k chars
- `markdown.py` — front matter assembly, smart filename construction
- `library.py` + `sync.py` + `rss.py` + `apple_podcasts.py` + `discovery.py` — podcast library subsystem
- `digest.py` + `history.py` — processing history + digest generation
- `download.py` + `cookies.py` — yt-dlp wrapper, browser cookie extraction
- `config.py` + `config_manager.py` + `model_selector.py` + `setup.py` — config, model fallback chain, setup wizard
- `evals/` — provider comparison evals (`pidcast-eval` entry point)
- `exceptions.py` + `utils.py` — shared error types, filename/duplicate helpers

## Engineering conventions

- **Workflow:** "plan then execute" — propose changes before editing.
- **Audio pipeline:** standardize to 16kHz mono WAV before transcription regardless of source format.
- **Whisper anti-hallucination:** `--suppress-nst` is ON by default; `--vad` (opt-in, needs a Silero model) strips silence pre-decode; repeated identical lines (3+) are collapsed in post-processing. These attack Whisper's silence-hallucination at the source rather than censoring phrases.
- **Chunking threshold:** transcripts > 120k chars use semantic chunking with synthesis.
- **LLM responses:** every analysis prompt returns JSON with `analysis` and `contextual_tags` fields.
- **Filenames:** smart-filtered with `YYYY-MM-DD_Title.md` date prefix.
- **Transcripts location:** the canonical store is the XDG data dir, `$XDG_DATA_HOME/pidcast/transcripts/` (default `~/.local/share/pidcast/transcripts/`; override with `PIDCAST_DATA_DIR`; run `pidcast paths` to print the resolved dirs). Audio, logs, and the unified run history (`state/runs.json`) live alongside it under the data dir. Never drop stray transcripts in the repo root (ignored via `.gitignore`).
- **Linting:** ruff is the single source of truth (config in `pyproject.toml` under `[tool.ruff]`).

## Data handling

The transcripts dir (`$XDG_DATA_HOME/pidcast/transcripts/`, run `pidcast paths`) contains large text files (some > 200 KB). The repo's `data/transcripts/` may still hold pre-migration files. **Do NOT read these into context** unless the user explicitly approves a specific file. For test fixtures, ask the user to point at one or create a dedicated minimal fixture under `tests/`.

## Further reading

- [docs/architecture.md](docs/architecture.md) — components, data flow, diagrams
- [docs/development-guide.md](docs/development-guide.md) — dev environment, testing, CI
- [docs/adr/](docs/adr/) — Architecture Decision Records (provider abstractions)
- `config/prompts.yaml` — analysis prompt templates
- `config/models.yaml` — LLM model definitions and fallback chains
