# ADR 0001 — Two transcription providers behind one CLI flag

**Status:** Accepted (2026-03-18, ElevenLabs landed in PR #10)

## Context

Local `whisper.cpp` was the only transcription path until early 2026. It is private and free, but it has two real costs:

1. **Onboarding friction.** New users need to build whisper.cpp from source, download a model file, and wire paths into `.env`. The interactive `pidcast setup` wizard helps, but a non-trivial fraction of users abandon at this step.
2. **Speed.** A 60-minute episode at `large-v3-turbo` takes 3–5 minutes on Apple Silicon and longer on older hardware. Cloud transcription is faster on the wall clock for the same hardware.

ElevenLabs Scribe v2 emerged as a credible cloud alternative with built-in speaker diarization (no separate pyannote setup) and a forgiving rate limit.

## Decision

Support both providers behind a single `--transcription-provider {whisper,elevenlabs}` flag, with whisper as the default. Provider implementations live under `src/pidcast/providers/`. A common interface returns `(segments, speaker_turns_or_None)` so downstream code (`diarization.py`, `markdown.py`) is provider-agnostic.

Diarization follows the same pluggability: whisper uses pyannote post-hoc, ElevenLabs reuses its own speakers — both surface through the same `--diarize` flag.

## Consequences

**Positive:**

- New users can be productive in under 5 minutes with just an ElevenLabs API key. Local-first users keep their existing flow.
- Workflow code is provider-agnostic. Future providers (Deepgram, AssemblyAI) plug into `providers/` without touching `workflow.py`.
- The eval matrix (output under the data dir's `evals/`; run `pidcast paths`) can compare providers head-to-head on the same audio.

**Negative:**

- Two code paths to maintain and test. The `providers/` interface had to settle two divergent return shapes (whisper JSON vs. ElevenLabs API response). The reconciliation logic in `transcription.py` is non-obvious.
- Provider-specific failure modes (whisper: missing model file; ElevenLabs: 401, 413 payload-too-large) need separate handling, including the chunked-synthesis retry path for 413 errors (commit `74cc00e`).
- `pidcast doctor` has to know about both providers to give useful diagnostics.

## Related code

- `src/pidcast/providers/` — provider implementations
- `src/pidcast/transcription.py` — dispatch and result normalization
- `src/pidcast/diarization.py` — diarization merge across both paths
- `tests/` — cross-provider speaker label consistency tests (commit `f340525`)
