"""Find and pick among the most recent transcripts (for verbs run without a path)."""

import fnmatch
import sys
from datetime import datetime
from pathlib import Path

from . import ui
from .config import resolve_output_dir
from .exceptions import PidcastError

DEFAULT_RECENT = 8

# Artifacts derived from a transcript, never a transcript themselves
_DERIVED_PATTERNS = ("*.discord-*.md", "*.notes-*.md", "*_analysis_*.md", "*_podcast-digest.md")


def list_recent_transcripts(directory: Path, limit: int = DEFAULT_RECENT) -> list[Path]:
    """Newest-first transcript files in ``directory``, derived artifacts excluded."""
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.glob("*.md")
        if p.is_file() and not any(fnmatch.fnmatch(p.name, pat) for pat in _DERIVED_PATTERNS)
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def _item(path: Path) -> ui.ChecklistItem:
    stat = path.stat()
    when = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return ui.ChecklistItem(path.stem, aside=f"{when}  {stat.st_size // 1024} KB")


def pick_transcript(args, limit: int | None = None) -> Path | None:
    """Let the user choose among recent transcripts; None means they quit.

    Raises PidcastError when there is nothing to pick from or no TTY to ask on.
    """
    directory = resolve_output_dir(args)
    recent = list_recent_transcripts(directory, limit or DEFAULT_RECENT)
    if not recent:
        raise PidcastError(f"No transcripts found in {directory}")
    if not sys.stdin.isatty():
        raise PidcastError(
            "No transcript given (pass a path; the recent-transcripts picker needs a TTY)"
        )
    chosen = ui.select_one([_item(p) for p in recent], [f"Recent transcripts in {directory}"])
    return recent[chosen] if chosen is not None else None
