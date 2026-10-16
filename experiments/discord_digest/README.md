# Discord recap lab

Eval harness behind `pidcast discord`. It runs one preset across several models on the same picked stories, has Claude Opus judge the outputs blind, and writes a side-by-side `report.md`. Use it to tune a preset or to check whether a new model (especially a free Groq one) is good enough before changing the product.

It reuses the product's isolated Claude runner (`run_claude`), segmenter, and story parser, so lab results transfer directly to `pidcast discord`.

## Run

```bash
# 1. List stories (default model: Groq gpt-oss-120b, free; or --model claude:haiku)
uv run python experiments/discord_digest/digest_lab.py stories ~/.local/share/pidcast/transcripts/<file>.md

# 2. Write the recap for picked stories across models, then judge
uv run python experiments/discord_digest/digest_lab.py digest experiments/discord_digest/runs/<file> \
  --pick 3,9 \
  --preset experiments/discord_digest/presets/discord-play-uk.yaml \
  --models groq:openai/gpt-oss-120b claude:sonnet claude:opus

# Re-judge an existing run
uv run python experiments/discord_digest/digest_lab.py judge experiments/discord_digest/runs/<file>/<run>
```

Model specs: `groq:<model-id>` or `claude:<haiku|sonnet|opus>`. Append `@<effort>` to override a Groq model's reasoning effort (for example `groq:qwen/qwen3.8-27b@none`). In zsh, list models as separate arguments; an unquoted `$VAR` holding several specs is not split.

Output lands in `runs/` (gitignored): `stories.json`, `segments.json`, and one folder per digest run with `prompt.txt`, `results.json`, `judge.json`, and `report.md`.

## Files

- `presets/`: lab copies of the styles, plus the Groq-only knobs (`seed`, `temperature`, `reasoning_effort`). The product presets live in `config/discord/`; copy changes over once they win here.
- `examples/`: reference examples approved by the user. `play_gemini-4-argon-hn.md` is the gold standard for the play voice.

## Findings (2026-10-16, Tech Brew Ride Home, stories 3 + 9)

Opus judge scores out of 10:

| Model | Play | Prose digest |
|---|---|---|
| Claude Opus 5.5 | 8 | 6-7 |
| Claude Sonnet 5.5 | 7 | 5 |
| Claude Haiku 5.5 | 4 | 2 |
| Groq gpt-oss-120b (medium) | 2 | 2 |
| Groq Qwen 3.8 27B (reasoning off) | 2 | - |

Groq free tier limits: 8k tokens per minute and 1000 requests per day per model. Qwen also has a 1000 output-tokens-per-minute cap, which shows up only in the body of a 429 error. With reasoning on (gpt-oss@high, qwen@medium), the whole output budget goes to reasoning and the reply comes back empty. With reasoning off, Ukrainian quality collapses. The story pass itself works fine on Groq. That's why the product is Claude-only.

Known caveat: Opus judges its own output. The candidates are anonymized and shuffled, but read the outputs yourself before trusting a ranking.
