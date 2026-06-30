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
