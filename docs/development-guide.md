# Development guide

How to set up a dev environment, run tests, and ship a change. For user-facing usage see [README.md](../README.md); for architecture see [architecture.md](architecture.md).

## Prerequisites

- **Python ≥ 3.10**
- **`uv`** — package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **`ffmpeg`** — required for all audio paths (system package manager)
- **`whisper.cpp`** — only if you want local transcription ([build from source](https://github.com/ggerganov/whisper.cpp)). Skip if you'll use ElevenLabs.

## First-time setup

```bash
git clone https://github.com/kintecus/pidcast
cd pidcast
uv sync --group dev          # runtime + dev deps (ruff, pytest)
cp .env.example .env         # fill in keys
uv run pidcast setup         # interactive wizard for paths and models
pre-commit install           # framework hooks: ruff + markdownlint (see Plugin-driven automation)
```

`pidcast setup` walks through whisper.cpp paths, model selection, and API keys. Run `pidcast doctor` at any time to check the current state of your tooling and env.

## Environment variables

Set in `.env`. See `.env.example` for the up-to-date list.

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `GROQ_API_KEY` | Default LLM provider for analysis | Required unless `--no-analyze` or `--provider claude` |
| `ELEVENLABS_API_KEY` | ElevenLabs Scribe v2 transcription provider | Required if `--transcription-provider elevenlabs` |
| `HUGGINGFACE_TOKEN` | Downloads pyannote diarization model on first use | Required for `--diarize` on the whisper path |
| `WHISPER_CPP_PATH` | Path to `whisper-cli` binary | Required for local whisper transcription |
| `WHISPER_MODEL` | Path to a `ggml-*.bin` model file | Required for local whisper transcription |
| `WHISPER_VAD_MODEL` | Path to a Silero VAD model (`ggml-silero-*.bin`); enables `--vad` to strip silence before decoding | Optional (required only when passing `--vad`) |
| `FFMPEG_PATH` | Custom ffmpeg location | Optional (defaults to PATH lookup) |
| `OBSIDIAN_VAULT_PATH` | Target vault root for `--save-to-obsidian` | Optional |

## Running tests

```bash
uv run pytest                              # full suite
uv run pytest tests/test_chunking.py       # one file
uv run pytest -k "test_model"              # name pattern
uv run pytest -v                           # verbose
```

Test config is in `pyproject.toml` under `[tool.pytest.ini_options]`. Provider-specific tests are skipped when their API key is absent — set `ELEVENLABS_API_KEY=dummy` and `GROQ_API_KEY=dummy` in the test env if you want to run the offline shape checks without making real calls.

## Lint and format

```bash
uv run ruff check src/                # lint
uv run ruff check --fix src/          # autofix
uv run ruff format src/               # format
uv run ruff format --check src/       # CI-style check
```

Ruff config lives in `pyproject.toml` under `[tool.ruff]`:

- Target Python 3.10
- Line length 100
- Rule set: `E, W, F, I, N, UP, B, C4, SIM`

Pre-commit hook runs ruff and Mermaid syntax validation. Don't bypass with `--no-verify` — fix the underlying issue.

## Adding dependencies

```bash
uv add package-name              # runtime
uv add --dev package-name        # dev only
uv sync --upgrade                # bump all pins
```

Optional dependency groups (e.g. `[diarize]`) live in `pyproject.toml` under `[project.optional-dependencies]`; edit manually.

## Project layout

```
src/pidcast/          # main package (src layout)
  providers/          # transcription provider implementations
  evals/              # pidcast-eval CLI for provider comparison
config/               # prompts.yaml, models.yaml
tests/                # pytest suite
data/                 # repo-local data: eval fixtures (committed) + any pre-migration transcripts
  evals/              # eval runs, references, comparisons (pidcast-eval only)
docs/                 # this directory
  adr/                # Architecture Decision Records
scripts/              # diarize-existing.sh and other helpers
```

## Project conventions

- **Audio pipeline:** every audio input is normalized to 16 kHz mono WAV before transcription. Downstream modules MUST assume this invariant.
- **LLM responses:** all analysis prompts return JSON with `analysis` and `contextual_tags` fields. Add fields via `config/prompts.yaml`, not via prompt-string surgery in code.
- **Chunking threshold:** 120 000 characters triggers semantic chunking with synthesis. Threshold lives in `config.py`.
- **Filenames:** smart-prefixed `YYYY-MM-DD_Title.md`. Logic in `utils.py`.
- **Transcripts canonical location:** the XDG data dir, `$XDG_DATA_HOME/pidcast/transcripts/` (default `~/.local/share/pidcast/`; override with `PIDCAST_DATA_DIR`; run `pidcast paths`). Audio (`audio/`), logs (`logs/`), and the unified run history (`state/runs.json`) live alongside it; config and the podcast library stay in `~/.config/pidcast/`. The repo gitignores stray `YYYY-MM-DD_*.md` files at the root.

## Provider comparison evals

`pidcast-eval` (entry point installed by `uv sync`) runs the same transcript through multiple providers and judges output quality with Claude Opus:

```bash
uv run pidcast-eval --compare groq,claude --transcript-file transcript.txt --title "Episode Title"
uv run pidcast-eval --run-matrix                              # all prompt × model × reference combos
uv run pidcast-eval --run-matrix --models "llama-3.3-70b-versatile,mixtral-8x7b-32768"
```

Results land in `data/evals/comparisons/` as Markdown reports. See `src/pidcast/evals/` for the underlying machinery.

> **Source-checkout only.** Unlike the main CLI (whose data moved to the XDG data dir), `pidcast-eval` reads its fixtures and writes its output under the repo's `data/evals/` and `config/` (committed reference data). It is a development tool and is not expected to work from a bare `pip`-installed wheel — run it from a cloned checkout via `uv run`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main`:

1. **Lint** — `ruff check` + `ruff format --check`
2. **Test** — `pytest`

Both jobs install with `uv sync --group dev`.

## Plugin-driven automation

This repo opts into several Claude Code plugins. Their config lives in `.claude-plugin/`:

- `semver.json` — auto bumps `pyproject.toml` version on feature/fix commits (excludes `*.md`, `docs/**`)
- `kb-grooming.json` — documentation health check scope

The session entry points (`/semver:setup`, `/kb-grooming:kb-groom`, `/playbook:playbook-browse`) are usable from Claude Code but not from regular shells.

### Commit-time validation

Two independent hook mechanisms run on commit — keep both in mind:

- **Native git hook** — `.githooks/pre-commit`, wired in via `core.hooksPath = .githooks` (already set in the repo). Runs `ruff check` + `ruff format --check` on staged `*.py`, then PlantUML URL-sync and Mermaid syntax validation on staged `*.md`. This is the authoritative gate and runs even without the pre-commit framework installed.
- **pre-commit framework** — `.pre-commit-config.yaml`, activated by running `pre-commit install` (see First-time setup). Runs `ruff` (with `--fix`), `ruff-format`, and `markdownlint --fix`. Complements the native hook; mainly autofixes formatting.

Don't bypass either with `--no-verify` — fix the underlying issue.
