"""Single-owner run reporter for polished, stable CLI output.

One run owns ONE rich ``Console`` + ``Live`` + ``Progress`` via ``RunReporter``.
The Live region shows the current phase and a progress bar that updates in place;
completed steps are logged ABOVE it so the bar never gets "pushed down" by
interleaved prints. In a non-TTY environment (pipes, CI, pytest) the reporter
degrades to plain ``logging`` and never starts a Live region.

Producers (whisper streaming, analysis chunking) and the ``log_*`` helpers reach
the in-flight reporter via the module-global ``active_reporter()`` so they render
through the one display instead of spawning competing ones.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The reporter currently driving the run (or None). Producers consult this to
# attach to the shared progress display instead of creating their own.
_ACTIVE: RunReporter | None = None


def active_reporter() -> RunReporter | None:
    """Return the reporter driving the current run, or None."""
    return _ACTIVE


def _stdout_is_tty() -> bool:
    import sys

    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


class _TaskHandle:
    """Handle to one task on the shared Progress.

    Always safe to call: when there's no live Progress (non-TTY), update() is a
    no-op so producers can drive it unconditionally.
    """

    def __init__(self, progress=None, task_id=None):
        self._progress = progress
        self._task_id = task_id

    def update(self, *, completed=None, total=None, description=None) -> None:
        if self._progress is None or self._task_id is None:
            return
        kwargs = {}
        if completed is not None:
            kwargs["completed"] = completed
        if total is not None:
            kwargs["total"] = total
        if description is not None:
            kwargs["description"] = description
        if kwargs:
            self._progress.update(self._task_id, **kwargs)

    def finish(self) -> None:
        """Hide this task so a finished bar isn't redrawn under later output."""
        if self._progress is None or self._task_id is None:
            return
        with contextlib.suppress(Exception):
            self._progress.update(self._task_id, visible=False)


class RunReporter:
    """Owns the single Live display for one run.

    Use as a context manager. While active it is registered as the global
    ``active_reporter()``. On a TTY it starts a rich ``Live`` wrapping a single
    ``Progress``; off a TTY it stays plain and routes everything through
    ``logging`` so piped/CI output is clean and never carries control codes.
    """

    def __init__(self, verbose: bool = False, force_plain: bool = False):
        self.verbose = verbose
        self._force_plain = force_plain
        self._prev_active: RunReporter | None = None
        self._live = None
        self._progress = None
        self.is_live = False

    # -- context management ---------------------------------------------------

    def __enter__(self) -> RunReporter:
        global _ACTIVE
        self._prev_active = _ACTIVE
        _ACTIVE = self

        if not self._force_plain and _stdout_is_tty():
            self._start_live()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _ACTIVE
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.stop()
            self._live = None
        self.is_live = False
        _ACTIVE = self._prev_active

    def _start_live(self) -> None:
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError:
            return  # rich missing -> stay plain

        self._console = Console()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]elapsed[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]eta[/dim]"),
            TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._live = Live(self._progress, console=self._console, refresh_per_second=10)
        self._live.start()
        self.is_live = True

    # -- producer API ---------------------------------------------------------

    def task(self, description: str, total: int | None = None) -> _TaskHandle:
        """Add a task to the shared Progress and return a safe handle."""
        if self._progress is None:
            return _TaskHandle()
        task_id = self._progress.add_task(description, total=total)
        return _TaskHandle(self._progress, task_id)

    # -- log API (above the Live region) --------------------------------------

    def log(self, message: str, style: str | None = None) -> None:
        """Emit a completed-step line above the Live region."""
        if self.is_live and self._console is not None:
            self._console.print(message if style is None else f"[{style}]{message}[/{style}]")
        else:
            logger.info(message)

    def phase(self, label: str) -> None:
        """Mark the start of a phase (a clean log line / rule)."""
        self.log(label, style="bold")

    def warn(self, message: str) -> None:
        if self.is_live and self._console is not None:
            self._console.print(f"[yellow]⚠ {message}[/yellow]")
        else:
            logger.warning(message)

    def error(self, message: str) -> None:
        if self.is_live and self._console is not None:
            self._console.print(f"[red]✗ {message}[/red]")
        else:
            logger.error(message)


# ---------------------------------------------------------------------------
# Single-keypress selector
# ---------------------------------------------------------------------------


def _read_raw_key() -> str:
    """Read one keypress from stdin in cbreak mode and return it as a string.

    Restores the terminal in a ``finally`` so a ``KeyboardInterrupt`` mid-read
    never leaves stdin in raw mode. Returns ``""`` on EOF. Raises if termios is
    unavailable or the terminal can't enter cbreak (caller falls back to line
    input).
    """
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def select_key(prompt: str, choices: list[tuple[str, str]], *, default: str) -> str:
    """Single-keypress selector. Returns the chosen key (lowercase).

    Args:
        prompt: Hint line printed once in the non-TTY fallback (may be empty).
        choices: Ordered ``(key, label)`` pairs; each ``key`` is one lowercase char.
        default: Key returned on Enter / empty input. Must be a choice key.

    Behaviour:
        - TTY: reads one raw key (no Enter). Case-insensitive match. Enter selects
          the default. Esc / Ctrl-C / EOF select the cancel key (``"c"`` if present,
          else the default). An unrecognized key re-reads until a valid/exit key.
        - Non-TTY or no termios: prints the hint once, then loops on ``input()``;
          the first character is matched the same way. Empty line → default,
          EOF → cancel/default.
    """
    import sys

    keys = {k.lower() for k, _ in choices}
    cancel = "c" if "c" in keys else default

    def _resolve(ch: str) -> str | None:
        """Map a single character to a choice key, or None to re-read."""
        if ch in ("\r", "\n"):
            return default
        if ch in ("\x1b", "\x03"):  # Esc, Ctrl-C
            return cancel
        low = ch.lower()
        if low in keys:
            return low
        return None

    use_tty = False
    try:
        use_tty = bool(sys.stdin.isatty())
    except Exception:
        use_tty = False

    if use_tty:
        try:
            while True:
                ch = _read_raw_key()
                if ch == "":  # EOF
                    return cancel
                resolved = _resolve(ch)
                if resolved is not None:
                    return resolved
        except Exception:
            # termios import / tcgetattr failure (e.g. Windows, odd terminal) ->
            # fall through to the line-based input() fallback below.
            pass

    # Non-TTY / no-termios fallback: line-based input.
    if prompt:
        print(prompt)
    while True:
        try:
            line = input()
        except EOFError:
            return cancel
        if line == "":
            return default
        resolved = _resolve(line[0])
        if resolved is not None:
            return resolved


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def spinner(label: str):
    """Transient spinner with elapsed time for a blocking call; a log line off-TTY."""
    if not _stdout_is_tty():
        logger.info(f"{label}...")
        yield
        return
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task(label, total=None)
        yield


# ---------------------------------------------------------------------------
# Multi-select checklist
# ---------------------------------------------------------------------------


@dataclass
class ChecklistItem:
    label: str
    detail: str = ""
    aside: str = ""


_KEY_ALIASES = {
    "\x1b[A": "up",
    "\x1bOA": "up",
    "k": "up",
    "\x1b[B": "down",
    "\x1bOB": "down",
    "j": "down",
    " ": "space",
    "\r": "enter",
    "\n": "enter",
    "\x1b": "quit",
    "\x03": "quit",
    "\x04": "quit",
    "q": "quit",
    "a": "all",
}


def _split_keys(raw: str) -> list[str]:
    """Turn one read into keys: an escape sequence is one key, anything else is per char."""
    if raw.startswith("\x1b") and len(raw) > 1:
        return [raw]
    return list(raw)


@dataclass
class Checklist:
    """Cursor + selection state behind ``select_many``, kept IO-free for tests."""

    selected: list[bool]
    cursor: int = 0
    hint: str = ""

    def press(self, key: str) -> str | None:
        """Apply one key. Returns "done" or "quit" to finish, None to keep reading."""
        n = len(self.selected)
        action = _KEY_ALIASES.get(key, _KEY_ALIASES.get(key.lower(), key))
        self.hint = ""
        if action == "up":
            self.cursor = (self.cursor - 1) % n
        elif action == "down":
            self.cursor = (self.cursor + 1) % n
        elif action == "space":
            self.selected[self.cursor] = not self.selected[self.cursor]
        elif action == "all":
            target = not all(self.selected)
            self.selected = [target] * n
        elif action.isdigit() and 1 <= int(action) <= n:
            self.cursor = int(action) - 1
            self.selected[self.cursor] = not self.selected[self.cursor]
        elif action == "enter":
            if any(self.selected):
                return "done"
            self.hint = "Pick at least one"
        elif action == "quit":
            return "quit"
        return None

    @property
    def picked(self) -> list[int]:
        return [i for i, on in enumerate(self.selected) if on]


def checklist_view(
    items: list[ChecklistItem],
    state: Checklist | None = None,
    notes: list[str] | None = None,
    help_line: str = "",
):
    """Render items as a numbered list; with ``state``, add checkboxes and a cursor."""
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1)
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", style="dim", no_wrap=True)
    for i, item in enumerate(items):
        on = state.selected[i] if state else True
        pointer = Text("❯", style="cyan") if state and state.cursor == i else Text("")
        box = f"[{'x' if on else ' '}] " if state else ""
        num = Text(f"{box}{i + 1}", style="cyan" if on else "dim")
        grid.add_row(pointer, num, Text(item.label, style="bold" if on else "dim"), item.aside)
        if item.detail:
            grid.add_row("", "", Text(item.detail, style="dim"), "")
    parts: list[Any] = ["", grid, ""]
    parts.extend(Text(n, style="dim") for n in notes or [])
    if state and state.hint:
        parts.append(Text(state.hint, style="yellow"))
    if help_line:
        parts.append(Text(help_line, style="dim"))
    return Group(*parts)


def _read_raw_bytes(fd: int) -> str:
    """One read from a cbreak stdin: a whole arrow-key escape sequence arrives together."""
    import os

    return os.read(fd, 16).decode(errors="ignore")


def select_many(items: list[ChecklistItem], notes: list[str] | None = None) -> list[int] | None:
    """Interactive checklist on a TTY, all items preselected.

    Returns the picked 0-based indices in list order, or None if the user quit.
    The caller handles the non-TTY case.
    """
    import sys
    import termios
    import tty

    from rich.console import Console
    from rich.live import Live

    state = Checklist(selected=[True] * len(items))
    help_line = "↑↓/jk move  space/1-9 toggle  a all  enter confirm  esc quit"
    console = Console()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    result: str | None = None
    with Live(
        checklist_view(items, state, notes, help_line), console=console, auto_refresh=False
    ) as live:
        try:
            tty.setcbreak(fd)
            while result is None:
                raw = _read_raw_bytes(fd)
                if raw == "":
                    result = "quit"
                for key in _split_keys(raw):
                    result = state.press(key)
                    if result:
                        break
                live.update(checklist_view(items, state, notes, help_line), refresh=True)
        except KeyboardInterrupt:
            result = "quit"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        final = Checklist(selected=state.selected, cursor=-1)
        live.update(checklist_view(items, final, notes), refresh=True)
    return state.picked if result == "done" else None
