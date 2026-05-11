# ADR 0002 — Two LLM analysis providers behind one CLI flag

**Status:** Accepted (2026-03-06, Claude provider landed in PR #9)

## Context

Analysis (summarization, key points, action items, comprehensive briefing) was Groq-only since the project's first release. Groq is fast and cheap, but two problems pushed for a second option:

1. **Quality ceiling.** On long, dense audio (technical talks, multi-speaker debates), Groq's Llama 3.x outputs lose nuance compared to frontier models. A handful of evals (`data/evals/comparisons/`) confirmed the gap on accuracy and completeness scores.
2. **No use of local Claude Code.** Many users already have an authenticated Claude Code CLI installed locally for other workflows. Reusing that subscription for pidcast analysis sidesteps yet another API key.

## Decision

Add `--provider claude` as a peer to `--provider groq` (default). The Claude path shells out to the local `claude` CLI, so it requires no Anthropic API key — it uses whatever auth the user already has set up for Claude Code. Model selection via `--claude-model {sonnet,opus,haiku}` (aliases that resolve to current Claude 4.x model IDs).

Cost estimation is implemented per-provider (`config/models.yaml` carries pricing/rate-limit metadata) so the CLI can warn before a run rather than after.

## Consequences

**Positive:**

- Quality ceiling lifted for users who care. Eval matrix scores on `comprehensive` analyses improved noticeably with `claude-opus`.
- Zero-config for Claude Code users — no new API key, no new dashboard.
- The dispatch layer in `analysis.py` is now provider-agnostic; adding a third provider (e.g. direct Anthropic API, OpenAI) is a small change.

**Negative:**

- Two prompt-execution code paths. Groq goes through the SDK; Claude shells out and parses the CLI's structured output. Errors look different and surface differently — `model_selector.py` has to handle both fallback chains.
- Claude CLI must be installed and authenticated. `pidcast doctor` does not currently check Claude CLI presence — a known gap.
- Cost estimation for the Claude path depends on the user's subscription model (Max, Pro, API), which the CLI cannot reliably introspect; estimates are approximate.

## Related code

- `src/pidcast/analysis.py`, `src/pidcast/summarization.py` — prompt execution
- `src/pidcast/model_selector.py` — fallback chains and per-provider model lists
- `config/models.yaml` — model definitions, rate limits, pricing
- `src/pidcast/evals/` — provider comparison eval machinery
