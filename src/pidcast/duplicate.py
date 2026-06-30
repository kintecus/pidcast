"""Duplicate-detection prompt UI.

When a recording was already transcribed, ``prompt_duplicate_detected`` renders
a compact rounded panel (title, one-line metadata summary, bracketed-key
choices) and resolves the user's choice with a single keypress via
``ui.select_key``. Lifted out of the CLI so the parser/dispatch layer stays thin
and this UI concern lives on its own.
"""

from enum import Enum

from . import ui


class DuplicateAction(Enum):
    """User's choice when duplicate transcription is detected."""

    RE_TRANSCRIBE = "retranscribe"
    ANALYZE_EXISTING = "analyze"
    FORCE_CONTINUE = "force"
    CANCEL = "cancel"


# Returned key -> action. ``a`` is only offered when the transcript file exists.
_KEY_TO_ACTION = {
    "r": DuplicateAction.RE_TRANSCRIBE,
    "a": DuplicateAction.ANALYZE_EXISTING,
    "f": DuplicateAction.FORCE_CONTINUE,
    "c": DuplicateAction.CANCEL,
}


def _build_summary(prev, transcript_exists: bool) -> str:
    """One-line metadata summary: date · [analysis type] · transcript status."""
    parts = [prev.formatted_date]
    if prev.analysis_performed:
        parts.append(prev.analysis_type or "analyzed")
    parts.append("on disk" if transcript_exists else "file not found")
    return " · ".join(parts)


def prompt_duplicate_detected(
    prev,
    verbose: bool = False,
) -> DuplicateAction:
    """Display the duplicate-detection prompt and return the user's choice.

    Args:
        prev: Information about the previous transcription (PreviousTranscription)
        verbose: Enable verbose output (currently unused; kept for call-site compat)

    Returns:
        User's selected action
    """
    transcript_exists = prev.transcript_path.exists()
    summary = _build_summary(prev, transcript_exists)

    # Ordered choices; ``a`` (analyze existing) only when the file is on disk.
    choices: list[tuple[str, str]] = [("r", "re-transcribe")]
    if transcript_exists:
        choices.append(("a", "analyze"))
    choices.extend([("f", "force"), ("c", "cancel")])

    try:
        from rich.box import ROUNDED
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        # No rich: plain info lines, then select_key handles the keypress/fallback.
        print("Already transcribed:")
        print(f"  {prev.video_title}")
        print(f"  {summary}")
        choices_line = "   ".join(f"[{k}] {label}" for k, label in choices)
        key = ui.select_key(choices_line, choices, default="c")
        return _KEY_TO_ACTION[key]

    console = Console()

    body = Text()
    body.append(prev.video_title, style="bold")
    body.append("\n")
    body.append(summary, style="dim")
    body.append("\n\n")
    for i, (k, label) in enumerate(choices):
        if i:
            body.append("   ")
        body.append(f"[{k}]", style="cyan")
        body.append(f" {label}")

    console.print(
        Panel(
            body,
            title="Already transcribed",
            border_style="yellow",
            box=ROUNDED,
        )
    )

    # Plain hint string only used by the non-TTY fallback path inside select_key.
    choices_line = "   ".join(f"[{k}] {label}" for k, label in choices)
    key = ui.select_key(choices_line, choices, default="c")
    return _KEY_TO_ACTION[key]
