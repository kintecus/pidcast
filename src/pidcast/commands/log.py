"""Handler for ``pidcast log`` — list the last N recorded runs.

Run dicts in the unified store are not uniformly shaped: legacy/pre-migration
entries can be as sparse as 5 keys. Every field below is read defensively;
only ``run_timestamp`` is guaranteed present on every entry.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_TITLE_WIDTH = 50


def cmd_log(args: argparse.Namespace) -> None:
    """Print the last ``args.limit`` recorded runs, newest first."""
    from ..config import RUNS_FILE
    from ..history import RunHistory

    runs = RunHistory(RUNS_FILE).get_runs_for_estimation()
    if not runs:
        print("No run history yet. Transcribe something to get started.")
        return

    runs = sorted(runs, key=lambda r: r.get("run_timestamp") or "", reverse=True)
    runs = runs[: args.limit]

    try:
        from rich.console import Console
        from rich.table import Table

        has_rich = True
    except ImportError:
        has_rich = False

    if has_rich:
        console = Console()
        table = Table(show_header=True, title=f"Run History (last {len(runs)})")
        table.add_column("Date", style="dim", width=16, no_wrap=True)
        table.add_column("Title", style="white", width=_TITLE_WIDTH, no_wrap=True)
        table.add_column("Provider", style="cyan", width=10, no_wrap=True)
        table.add_column("Duration", style="yellow", justify="right", width=10, no_wrap=True)
        table.add_column("Status", justify="center", width=6, no_wrap=True)
        table.add_column("Transcript", style="green", width=20, no_wrap=True)

        for entry in runs:
            table.add_row(*_row(entry))
        console.print(table)
    else:
        print(f"\nRun History (last {len(runs)}):")
        print("-" * 100)
        for entry in runs:
            date, title, provider, duration, status, transcript = _row(entry)
            print(f"{date}  [{status}]  {title}")
            print(f"    provider: {provider}  duration: {duration}  transcript: {transcript}")


def _row(entry: dict) -> tuple[str, str, str, str, str, str]:
    """Format one run dict into display columns, tolerating sparse entries."""
    return (
        _format_date(entry.get("run_timestamp")),
        _format_title(entry.get("video_title")),
        entry.get("transcription_provider") or "-",
        _format_duration(entry.get("audio_duration")),
        _format_status(entry.get("success")),
        _format_transcript(entry.get("transcript_path")),
    )


def _format_date(raw: str | None) -> str:
    if not raw:
        return "-"
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return raw


def _format_title(title: str | None) -> str:
    title = title or "(untitled)"
    if len(title) > _TITLE_WIDTH:
        return title[: _TITLE_WIDTH - 1] + "…"
    return title


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    from ..utils import format_duration

    return format_duration(seconds)


def _format_status(success: bool | None) -> str:
    if success is True:
        return "✓"
    if success is False:
        return "✗"
    return "?"


def _format_transcript(transcript_path: str | None) -> str:
    if not transcript_path:
        return "-"
    return Path(transcript_path).name
