"""Duplicate-detection prompt UI.

When a recording was already transcribed, ``prompt_duplicate_detected`` renders
an interactive menu (re-transcribe / analyze existing / force / cancel). Lifted
out of the CLI so the parser/dispatch layer stays thin and this UI concern lives
on its own.
"""

from enum import Enum


class DuplicateAction(Enum):
    """User's choice when duplicate transcription is detected."""

    RE_TRANSCRIBE = "retranscribe"
    ANALYZE_EXISTING = "analyze"
    FORCE_CONTINUE = "force"
    CANCEL = "cancel"


def prompt_duplicate_detected(
    prev,
    verbose: bool = False,
) -> DuplicateAction:
    """Display duplicate detection UI and get user's choice.

    Args:
        prev: Information about the previous transcription (PreviousTranscription)
        verbose: Enable verbose output

    Returns:
        User's selected action
    """

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
        from rich.table import Table
    except ImportError:
        return _prompt_duplicate_basic(prev)

    console = Console()

    # Build info panel
    console.print()
    console.print(
        Panel(
            "[yellow bold]Duplicate Detected![/yellow bold]\n\n"
            "This recording was previously transcribed.",
            border_style="yellow",
        )
    )

    # Show previous transcription details
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", width=20)
    table.add_column(style="white")

    table.add_row("Title", prev.video_title)
    table.add_row("Transcribed", prev.formatted_date)
    table.add_row("File", prev.smart_filename)

    # Check if transcript file still exists
    transcript_exists = prev.transcript_path.exists()
    if transcript_exists:
        table.add_row("Status", "[green]Transcript file exists[/green]")
    else:
        table.add_row("Status", "[red]Transcript file not found[/red]")

    if prev.analysis_performed:
        table.add_row("Previous Analysis", prev.analysis_type or "Yes")

    console.print(table)
    console.print()

    # Build options
    console.print("[bold]What would you like to do?[/bold]")
    console.print()
    console.print("  [cyan]1[/cyan] - Re-transcribe the recording")

    if transcript_exists:
        console.print("  [cyan]2[/cyan] - Analyze existing transcript (skip re-transcription)")
    else:
        console.print("  [dim]2 - Analyze existing (unavailable - file not found)[/dim]")

    console.print("  [cyan]3[/cyan] - Continue anyway (force re-transcription)")
    console.print("  [cyan]4[/cyan] - Cancel")
    console.print()

    # Get choice
    valid_choices = ["1", "2", "3", "4"] if transcript_exists else ["1", "3", "4"]
    while True:
        choice = Prompt.ask(
            "Enter choice",
            choices=valid_choices,
            default="4",
        )

        if choice == "1":
            return DuplicateAction.RE_TRANSCRIBE
        elif choice == "2" and transcript_exists:
            return DuplicateAction.ANALYZE_EXISTING
        elif choice == "3":
            return DuplicateAction.FORCE_CONTINUE
        elif choice == "4":
            return DuplicateAction.CANCEL


def _prompt_duplicate_basic(prev) -> DuplicateAction:
    """Basic fallback prompt without rich.

    Args:
        prev: Information about the previous transcription (PreviousTranscription)
    """
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTED!")
    print("=" * 60)
    print(f"Title: {prev.video_title}")
    print(f"Previously transcribed: {prev.formatted_date}")
    print(f"File: {prev.smart_filename}")
    print()
    print("Options:")
    print("  1 - Re-transcribe the recording")
    print("  2 - Analyze existing transcript")
    print("  3 - Continue anyway")
    print("  4 - Cancel")
    print()

    transcript_exists = prev.transcript_path.exists()
    valid_choices = {"1", "3", "4"}
    if transcript_exists:
        valid_choices.add("2")
    else:
        print("  (Option 2 unavailable - transcript file not found)")

    while True:
        choice = input("Enter choice [4]: ").strip() or "4"
        if choice in valid_choices:
            break
        print(f"Invalid choice. Please enter: {', '.join(sorted(valid_choices))}")

    mapping = {
        "1": DuplicateAction.RE_TRANSCRIBE,
        "2": DuplicateAction.ANALYZE_EXISTING,
        "3": DuplicateAction.FORCE_CONTINUE,
        "4": DuplicateAction.CANCEL,
    }
    return mapping[choice]
