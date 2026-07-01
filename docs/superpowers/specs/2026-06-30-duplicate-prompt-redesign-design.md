# Duplicate-detection prompt redesign

## Context

After the Phase 1-3 CLI/UX refactor, the transcription run output is polished (single owned rich Live display, clean phase lines). But the interactive **duplicate-detection prompt** — shown when a recording was already transcribed — was carried over untouched and now looks dated next to the rest of the run. The dated feel comes from three things:

1. A heavy stacked layout: a bordered panel, then a loose key/value `rich.Table`, then a loose numbered list.
2. "Type a number then Enter" interaction (`Enter choice [1/2/3/4] (4):`), the giveaway of an old CLI.
3. Inconsistent density vs. the compact `✓ phase`-line style the run now uses.

The current implementation lives in `src/pidcast/duplicate.py` (`prompt_duplicate_detected` + `_prompt_duplicate_basic`). It returns a `DuplicateAction` enum consumed by `commands/transcribe.py`; that contract is unchanged by this work.

**Intended outcome:** a compact, modern prompt — one rounded panel containing the info and the choices — with instant single-keypress selection, no new dependency, and a clean non-TTY fallback.

## Decisions (locked)

- **Layout:** one rounded `rich` panel containing a title line, a one-line metadata summary, and a bracketed-key choices line.
- **Interaction:** single keypress (`r`/`a`/`f`/`c`), acts instantly, no Enter. Case-insensitive. Enter alone selects the default; Esc / Ctrl-C / EOF select cancel.
- **No new dependency:** `rich` (already present) + a tiny `termios`/`tty` raw-keypress helper from the stdlib.
- **Fallback:** when stdin is not a TTY, or `termios` is unavailable (e.g. Windows), or rich is missing, fall back to a line-based `input()` prompt. (Non-interactive duplicate handling already errors out upstream in `commands/transcribe.py` via `is_interactive()`, so the prompt is only reached interactively; the fallback is the safety net for odd terminals.)
- **Scope:** only the duplicate prompt changes now. The `lib process` episode picker and `discovery.prompt_user_selection` keep their current behavior. The new `select_key` helper is placed where they could adopt it later, but converting them is out of scope (YAGNI).

## Components

### `ui.select_key(prompt, choices, *, default)` — new, in `src/pidcast/ui.py`

A reusable single-keypress selector. `ui.py` is the established home for terminal-UI concerns (it already owns the run's Console/Live).

- **Signature:** `select_key(prompt: str, choices: list[tuple[str, str]], *, default: str) -> str`
  - `choices`: ordered `(key, label)` pairs, e.g. `[("r", "re-transcribe"), ("a", "analyze"), ("f", "force"), ("c", "cancel")]`. `key` is a single lowercase character.
  - `default`: the key returned on Enter / empty input. Must be one of the choice keys.
  - Returns the selected key (lowercase).
- **TTY path:** read one raw byte via `termios`/`tty` (set cbreak, restore in `finally`). Match case-insensitively against the choice keys. Enter/`\r`/`\n` → `default`. Esc (`\x1b`), Ctrl-C (`\x03`), EOF → return the `"c"`/cancel key if present, else `default`. An unrecognized key re-reads (no echo spam) until a valid key or an exit key arrives.
- **Non-TTY / no-termios path:** print the bracketed-key hint line once, then loop on `input()` reading a line; first character is matched the same way; empty line → default; EOF → cancel/default.
- **Isolation:** pure function of stdin/stdout. The raw-getch is a small inner helper so tests can monkeypatch it. No dependency on `RunReporter` state.

### `prompt_duplicate_detected(prev, verbose=False)` — rewritten, in `src/pidcast/duplicate.py`

- Build the metadata summary line from `prev`: `formatted_date` + (analysis type if `prev.analysis_performed`) + transcript status (`on disk` vs `file not found`), separated by ` · `.
- Render ONE `rich` panel (rounded box, `title="Already transcribed"`, yellow border) whose body is:
  - line 1: `prev.video_title` (bold)
  - line 2: the dimmed metadata summary
  - blank line
  - the choices line: `[r] re-transcribe   [a] analyze   [f] force   [c] cancel` with the bracketed key letters highlighted (cyan), the `a` option included only when the transcript file exists.
- Call `select_key("", choices, default="c")` where `choices` omits `a` when `prev.transcript_path` is missing.
- Map the returned key to `DuplicateAction`:
  - `r` → `RE_TRANSCRIBE`, `a` → `ANALYZE_EXISTING`, `f` → `FORCE_CONTINUE`, `c` → `CANCEL`.
- Remove `_prompt_duplicate_basic`. When rich is missing, render the info as plain `print()` lines (title + metadata) and call `select_key`, which itself handles the no-rich/non-TTY case via its `input()` fallback. This deletes the separate numbered-prompt code path entirely (one selection mechanism, not two).

The function signature, return type, and the four `DuplicateAction` values are unchanged, so `commands/transcribe.py`'s dispatch needs no edits.

## Data flow

`commands/transcribe.cmd_transcribe` (unchanged) → on duplicate + interactive → `prompt_duplicate_detected(prev, verbose)` → builds panel, calls `ui.select_key(...)` → `select_key` reads one key (TTY) or a line (fallback) → returns key → mapped to `DuplicateAction` → returned to the existing dispatch.

## Error handling

- `termios`/`tty` import or `tcgetattr` failure → fall straight to the `input()` fallback (wrapped in try/except).
- Terminal state is always restored in a `finally` (cbreak → original), so a raised `KeyboardInterrupt` mid-read doesn't leave the terminal in raw mode.
- Ctrl-C / Esc / EOF resolve to cancel — never crash the prompt.

## Testing

New `tests/test_duplicate_prompt.py` (the existing `test_duplicate_detection.py` covers detection logic and stays untouched):

- `select_key`:
  - TTY keypress matches the right key (monkeypatch the inner getch to feed `"r"`, assert `"r"`).
  - case-insensitive (`"R"` → `"r"`).
  - Enter / empty → default.
  - Esc / Ctrl-C / EOF → cancel key.
  - invalid key then valid key → returns the valid one (re-read loop).
  - non-TTY: monkeypatch `isatty` False, feed `input()` → returns first-char match.
- `prompt_duplicate_detected`:
  - each key maps to the correct `DuplicateAction` (monkeypatch `select_key`).
  - `a` choice omitted (and `a` keypress not offered) when `prev.transcript_path` does not exist.

Verification: `uv run pytest`, `uv run ruff check src/ tests/`, and a manual run under a real terminal (trigger a duplicate, confirm the panel renders and a single keypress acts instantly) plus a piped run (confirm the `input()` fallback engages without raw-mode artifacts).

## Out of scope

- The `lib process` episode picker (`commands/lib.py`) and `discovery.prompt_user_selection` — they may adopt `select_key` later but are not changed here.
- Arrow-key navigation / any new prompt dependency (explicitly rejected: rich-only).
