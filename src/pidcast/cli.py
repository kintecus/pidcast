"Command-line interface for pidcast."

import argparse
import datetime
import logging
import time
import uuid
from enum import Enum
from pathlib import Path

from .config import (
    DEFAULT_PROMPTS_FILE,
    DEFAULT_STATS_FILE,
    OBSIDIAN_PATH,
    WHISPER_MODEL,
)
from .utils import (
    find_existing_transcription,
    is_interactive,
    log_error,
    log_section,
    setup_logging,
)
from .workflow import (
    process_input_source,
    run_analyze_existing_mode,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DUPLICATE DETECTION
# ============================================================================


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
            "This video was previously transcribed.",
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
    console.print("  [cyan]1[/cyan] - Re-transcribe the video")

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
    print("  1 - Re-transcribe the video")
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


# ============================================================================
# PRESET APPLICATION
# ============================================================================


def apply_preset(
    args: argparse.Namespace,
    explicitly_set: set[str] | None = None,
) -> None:
    """Apply a named preset to args, without overriding explicitly set flags.

    Args:
        args: Parsed arguments namespace
        explicitly_set: Set of arg names that were explicitly provided on CLI
    """
    from .config_manager import ConfigManager

    if explicitly_set is None:
        explicitly_set = set()

    preset_values = ConfigManager.load_preset(args.preset)

    for key, value in preset_values.items():
        if not hasattr(args, key):
            logger.warning(f"Unknown preset key '{key}', skipping")
            continue
        if key in explicitly_set:
            continue
        setattr(args, key, value)

    if getattr(args, "verbose", False):
        logger.info(f"Applied preset '{args.preset}': {preset_values}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Automate audio transcription with Whisper (YouTube URL or local file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common Workflows:
  # Quick transcription with defaults
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"

  # Save to Obsidian vault
  %(prog)s "VIDEO_URL" -o

  # Custom analysis type (supports fuzzy matching)
  %(prog)s "VIDEO_URL" -o -a exec          # Matches 'executive_summary'
  %(prog)s "VIDEO_URL" -o -a detailed -v   # Verbose output

  # Choose specific model
  %(prog)s "VIDEO_URL" -m llama33           # Matches 'llama-3.3-70b-versatile'

  # Test settings on a short segment before full transcription
  %(prog)s "VIDEO_URL" --test-segment --diarize
  %(prog)s "VIDEO_URL" --test-segment 5 --start-at 10

  # Analyze existing transcript
  %(prog)s --analyze-existing transcript.md -a summary

  # Retry diarization on existing transcript (without re-transcribing)
  %(prog)s --diarize-existing transcript.md

Discovery:
  %(prog)s -L                              # List analysis types
  %(prog)s -M                              # List LLM models
  %(prog)s -W                              # List Whisper models

Short Flags:
  -o  --save-to-obsidian    Save to Obsidian vault
  -a  --analysis-type       Analysis type (fuzzy matching enabled)
  -m  --groq-model          Model name (fuzzy matching enabled)
  -l  --language            Language code for transcription
  -f  --force               Force re-transcription
  -v  --verbose             Verbose output
  -p  --preset              Use a named preset
  -L  --list-analyses       List available analysis types
  -M  --list-models         List available LLM models
  -W  --list-whisper-models List available Whisper models
  -P  --list-presets        List available presets
        """,
    )

    # Input source group (mutually exclusive) - for original workflow
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "input_source", nargs="?", help="YouTube video URL or path to local audio file"
    )
    input_group.add_argument(
        "--analyze-existing",
        dest="analyze_existing",
        help="Path to existing transcript file (.md or .txt) to analyze without re-transcribing",
    )
    input_group.add_argument(
        "--diarize-existing",
        dest="diarize_existing",
        help="Path to existing transcript (.md) to run speaker diarization on. "
        "Requires .whisper.json next to the transcript (auto-saved during transcription).",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file for --diarize-existing (overrides auto-detected .wav).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for Markdown files (default: current directory)",
    )
    parser.add_argument(
        "-o",
        "--save-to-obsidian",
        dest="save_to_obsidian",
        action="store_true",
        help=f"Save analysis files to Obsidian vault (transcripts still saved to --output-dir) at: {OBSIDIAN_PATH}",
    )
    parser.add_argument(
        "--whisper-model",
        dest="whisper_model",
        default=WHISPER_MODEL,
        help="Whisper model name (e.g., 'medium', 'large-v3-turbo') or path to model file",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code for transcription (e.g., 'uk', 'en', 'de'). Default: auto-detect.",
    )
    parser.add_argument(
        "--transcription-provider",
        dest="transcription_provider",
        choices=["whisper", "elevenlabs"],
        default="whisper",
        help="Transcription backend: whisper (local) or elevenlabs (cloud). Default: whisper.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Run speaker diarization (whisper: pyannote.audio + HUGGINGFACE_TOKEN, elevenlabs: built-in)",
    )
    parser.add_argument(
        "--output-format",
        dest="output_format",
        default="otxt",
        help="Whisper output format (txt, vtt, srt, json). Prefix with 'o' for original filename.",
    )
    parser.add_argument(
        "--front-matter",
        dest="front_matter",
        default="{}",
        help="JSON string for Markdown front matter",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Comma-separated tags for front matter (overrides auto-inferred source tags). "
        "Example: --tags meeting,standup,weekly",
    )
    parser.add_argument(
        "--stats-file",
        dest="stats_file",
        default=None,
        help=f"File to store statistics (default: {DEFAULT_STATS_FILE})",
    )
    parser.add_argument(
        "--keep-transcript",
        dest="keep_transcript",
        action="store_true",
        help="Keep the .txt transcript file alongside the .md file",
    )
    parser.add_argument(
        "--keep-audio",
        dest="keep_audio",
        action="store_true",
        help="Save the converted WAV audio file to the output directory",
    )
    parser.add_argument(
        "--po-token",
        dest="po_token",
        default=None,
        help="PO Token for bypassing YouTube restrictions (format: 'client.type+TOKEN')",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser to extract cookies from (e.g., 'chrome', 'firefox', 'safari')",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to Netscape format cookies file",
    )
    parser.add_argument(
        "--chrome-profile",
        default=None,
        dest="chrome_profile",
        help="Chrome profile for cookie extraction (display name or directory name)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip duplicate detection and force transcription",
    )

    # Test segment arguments
    segment_group = parser.add_argument_group("Test Segment Options")
    segment_group.add_argument(
        "--test-segment",
        nargs="?",
        const=2,
        type=float,
        default=None,
        dest="test_segment",
        metavar="MINUTES",
        help="Test transcription settings on first N minutes of audio (default: 2). "
        "Skips markdown creation, analysis, and statistics.",
    )
    segment_group.add_argument(
        "--start-at",
        type=float,
        default=None,
        dest="start_at",
        metavar="MINUTES",
        help="Start offset in minutes for --test-segment (default: 0). "
        "Use to skip intros or jump to a specific section.",
    )

    # LLM Analysis arguments
    analysis_group = parser.add_argument_group("LLM Analysis Options")
    analysis_group.add_argument(
        "--no-analyze",
        action="store_true",
        dest="no_analyze",
        help="Skip LLM analysis (default: analyze is enabled)",
    )
    analysis_group.add_argument(
        "-a",
        "--analysis-type",
        dest="analysis_type",
        default="executive_summary",
        help="Analysis type/prompt template to use (default: executive_summary). Use -L to list available types.",
    )
    analysis_group.add_argument(
        "--prompts-file",
        dest="prompts_file",
        default=None,
        help=f"Path to prompts YAML file (default: {DEFAULT_PROMPTS_FILE})",
    )
    analysis_group.add_argument(
        "--groq-api-key",
        dest="groq_api_key",
        default=None,
        help="Groq API key (default: GROQ_API_KEY environment variable)",
    )
    analysis_group.add_argument(
        "-m",
        "--groq-model",
        dest="groq_model",
        default=None,
        help="Groq model to use for analysis (default: from config/models.yaml). Use -M to list available models.",
    )
    analysis_group.add_argument(
        "--skip-analysis-on-error",
        dest="skip_analysis_on_error",
        action="store_true",
        help="Continue if analysis fails instead of aborting",
    )
    analysis_group.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "claude"],
        help="LLM provider for analysis: 'groq' (default) or 'claude' (uses local Claude CLI)",
    )
    analysis_group.add_argument(
        "--claude-model",
        dest="claude_model",
        default=None,
        help="Claude model alias when --provider claude: 'sonnet' (default), 'opus', 'haiku'",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save",
        action="store_true",
        help="Save analysis output to file (default: terminal only)",
    )

    # Discoverability options
    discovery_group = parser.add_argument_group("Discovery Options")
    discovery_group.add_argument(
        "-L",
        "--list-analyses",
        action="store_true",
        dest="list_analyses",
        help="List available analysis types and exit",
    )
    discovery_group.add_argument(
        "-M",
        "--list-models",
        action="store_true",
        dest="list_models",
        help="List available Groq models and exit",
    )
    discovery_group.add_argument(
        "-W",
        "--list-whisper-models",
        action="store_true",
        dest="list_whisper_models",
        help="List available Whisper models and exit",
    )
    discovery_group.add_argument(
        "--list-chrome-profiles",
        action="store_true",
        dest="list_chrome_profiles",
        help="List available Chrome profiles for cookie extraction and exit",
    )

    # Preset options
    preset_group = parser.add_argument_group("Preset Options")
    preset_group.add_argument(
        "-p",
        "--preset",
        default=None,
        help="Use a named preset from config.yaml (e.g., 'daily', 'meeting'). "
        "Explicit flags override preset values. Use -P to list presets.",
    )
    preset_group.add_argument(
        "-P",
        "--list-presets",
        action="store_true",
        dest="list_presets",
        help="List available presets and exit",
    )

    return parser.parse_args()


def cmd_doctor() -> None:
    """Run health checks and display configuration status."""
    from .setup import determine_status, run_all_checks

    print("\npidcast doctor")
    print("=" * 40)

    checks = run_all_checks()
    max_name = max(len(c.name) for c in checks)

    for check in checks:
        dots = "." * (max_name + 6 - len(check.name))
        if check.ok:
            print(f"  {check.name} {dots} {check.detail}")
        else:
            label = "MISSING" if check.required else "not set"
            msg = check.detail if check.detail != "not set" else label
            print(f"  {check.name} {dots} {msg}")
            if check.hint:
                print(f"    -> {check.hint}")

    status, tip = determine_status(checks)
    print(f"\n  Status: {status}")
    if tip:
        print(f"  Tip: {tip}")
    print()


def cmd_setup() -> None:
    """Interactive setup wizard for first-time users."""

    from .setup import (
        ENV_FILE,
        check_env_var,
        check_ffmpeg,
        check_whisper,
        check_whisper_model,
        write_env_var,
    )

    print("\npidcast setup")
    print("=" * 40)

    # Step 1: System dependencies
    print("\nStep 1/3: Checking system dependencies...\n")

    ffmpeg = check_ffmpeg()
    if ffmpeg.ok:
        print(f"  ffmpeg: {ffmpeg.detail}")
    else:
        print("  ffmpeg: not found")
        print("    Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        print("    Then re-run: pidcast setup")
        return

    whisper = check_whisper()
    whisper_model = check_whisper_model()
    use_elevenlabs = False

    if whisper.ok and whisper_model.ok:
        print(f"  whisper.cpp: {whisper.detail}")
        print(f"  whisper model: {whisper_model.detail}")
    else:
        print("  whisper.cpp: not configured")
        print("\n  whisper.cpp is needed for local (private) transcription.")
        print("  You can skip it and use ElevenLabs (cloud) instead.\n")
        print("  [1] I'll set up whisper.cpp myself (see README for instructions)")
        print("  [2] Skip - use ElevenLabs cloud transcription")
        try:
            choice = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return
        if choice == "2":
            use_elevenlabs = True
        else:
            print("\n  To set up whisper.cpp:")
            print("    1. git clone https://github.com/ggerganov/whisper.cpp.git")
            print("    2. cd whisper.cpp && make")
            print("    3. bash models/download-ggml-model.sh base.en")
            print("    4. Set in .env: WHISPER_CPP_PATH=/path/to/whisper.cpp/build/bin/whisper-cli")
            print("    5. Set in .env: WHISPER_MODEL=/path/to/whisper.cpp/models/ggml-base.en.bin")
            print("    6. Re-run: pidcast setup")
            return

    # Step 2: API keys
    print("\nStep 2/3: API keys\n")

    # Groq (for analysis)
    groq = check_env_var("GROQ_API_KEY", "GROQ_API_KEY")
    if groq.ok:
        print(f"  GROQ_API_KEY: {groq.detail}")
    else:
        print("  GROQ_API_KEY: not set (needed for AI analysis)")
        print("  Get a free key at: https://console.groq.com/")
        try:
            key = input("  Paste your key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if key:
            write_env_var("GROQ_API_KEY", key)
            print("  Saved to .env")
        else:
            print("  Skipped - transcription will work, but AI analysis will be disabled")

    # ElevenLabs (if chosen or if no whisper)
    if use_elevenlabs:
        el = check_env_var("ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY")
        if el.ok:
            print(f"  ELEVENLABS_API_KEY: {el.detail}")
        else:
            print("\n  ELEVENLABS_API_KEY: not set (needed for cloud transcription)")
            print("  Get a key at: https://elevenlabs.io/")
            try:
                key = input("  Paste your key: ").strip()
            except (EOFError, KeyboardInterrupt):
                key = ""
            if key:
                write_env_var("ELEVENLABS_API_KEY", key)
                print("  Saved to .env")
            else:
                print("  Without this key, transcription won't work.")
                print("  Add it to .env later: ELEVENLABS_API_KEY=your_key")
                return

    # Step 3: Summary
    print("\nStep 3/3: Ready!\n")

    if not ENV_FILE.exists():
        print(f"  Note: .env file created at {ENV_FILE}")

    if use_elevenlabs:
        print("  Provider: ElevenLabs (cloud)")
        print("\n  Try it:")
        print('    pidcast "https://youtube.com/watch?v=VIDEO_ID"')
    else:
        print("  Provider: whisper.cpp (local)")
        print("\n  Try it:")
        print('    pidcast "https://youtube.com/watch?v=VIDEO_ID"')

    print()


def main() -> None:
    """Main entry point for pidcast CLI."""
    import sys

    # Handle setup/doctor subcommands before arg parsing
    if len(sys.argv) > 1:
        if sys.argv[1] == "doctor":
            from dotenv import load_dotenv

            load_dotenv()
            cmd_doctor()
            return
        if sys.argv[1] == "setup":
            from dotenv import load_dotenv

            load_dotenv()
            cmd_setup()
            return

    args = parse_arguments()

    # Set up logging
    setup_logging(getattr(args, "verbose", False))

    # Handle --list-presets
    if getattr(args, "list_presets", False):
        from .config_manager import ConfigManager

        presets = ConfigManager.list_presets()
        if not presets:
            print("No presets defined. Add presets to ~/.config/pidcast/config.yaml")
            print("\nExample:")
            print("  presets:")
            print("    daily:")
            print("      whisper_model: large-v3")
            print("      language: uk")
            print("      diarize: true")
            print("      no_analyze: true")
        else:
            print("Available presets:\n")
            for name, flags in presets.items():
                flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
                print(f"  {name}: {flag_str}")
            print("\nUsage: pidcast <input> -p <preset>")
        return

    # Apply preset if specified
    if getattr(args, "preset", None):
        import sys

        # Determine which args were explicitly set on CLI
        explicitly_set = set()
        for arg in vars(args):
            if any(
                a in (f"--{arg}", f"-{arg}", f"--{arg.replace('_', '-')}") for a in sys.argv[1:]
            ):
                explicitly_set.add(arg)
        try:
            apply_preset(args, explicitly_set=explicitly_set)
        except ValueError as e:
            log_error(str(e))
            return

    # Load chrome_profile from config if not explicitly set
    if not getattr(args, "chrome_profile", None):
        from .config_manager import ConfigManager

        config = ConfigManager.load_config()
        chrome_profile = config.get("chrome_profile")
        if chrome_profile:
            args.chrome_profile = chrome_profile

    # Handle discovery/list commands first (they exit immediately)
    if getattr(args, "list_analyses", False):
        from .utils import list_available_analyses

        list_available_analyses()
        return

    if getattr(args, "list_models", False):
        from .utils import list_available_models

        list_available_models()
        return

    if getattr(args, "list_whisper_models", False):
        from .transcription import list_whisper_models

        models = list_whisper_models()
        if not models:
            print("No whisper models found. Set WHISPER_MODELS_DIR or WHISPER_MODEL env var.")
        else:
            print("Available Whisper models:\n")
            for m in models:
                print(f"  {m['name']:<25} {m['size']:>10}")
            print(f"\nUsage: pidcast <input> --whisper_model {models[0]['name']}")
        return

    if getattr(args, "list_chrome_profiles", False):
        from .cookies import list_chrome_profiles

        profiles = list_chrome_profiles()
        if not profiles:
            print("No Chrome profiles found.")
        else:
            print("Available Chrome profiles:\n")
            print(f"  {'Display Name':<25} {'Directory':<20} {'Config Value'}")
            print(f"  {'-' * 25} {'-' * 20} {'-' * 30}")
            for dir_name, meta in profiles.items():
                print(f"  {meta['display_name']:<25} {dir_name:<20} {dir_name}")
            print(f'\nUsage: pidcast <input> --chrome-profile "{list(profiles.keys())[0]}"')
            print("   Or: Set 'chrome_profile' in ~/.config/pidcast/config.yaml")
        return

    # Validate that we have either input_source or analyze_existing for transcription workflow
    if (
        not args.input_source
        and not args.analyze_existing
        and not getattr(args, "diarize_existing", None)
    ):
        log_error("No input provided.")
        logger.info('  Usage: pidcast "https://youtube.com/watch?v=VIDEO_ID"')
        logger.info("         pidcast /path/to/audio.mp3")
        logger.info("  First time? Run: pidcast setup")
        logger.info("  Full help: pidcast --help")
        return

    # Resolve analysis type with fuzzy matching
    if args.analysis_type and args.analysis_type != "executive_summary":
        from .utils import resolve_analysis_type

        try:
            resolved_type = resolve_analysis_type(args.analysis_type, args.prompts_file)
            if resolved_type != args.analysis_type and args.verbose:
                logger.info(f"Matched '{args.analysis_type}' → '{resolved_type}'")
            args.analysis_type = resolved_type
        except ValueError as e:
            log_error(str(e))
            return

    # Resolve model name with fuzzy matching
    if args.groq_model:
        from .utils import resolve_model_name

        try:
            resolved_model = resolve_model_name(args.groq_model)
            if resolved_model != args.groq_model and args.verbose:
                logger.info(f"Matched '{args.groq_model}' → '{resolved_model}'")
            args.groq_model = resolved_model
        except ValueError as e:
            log_error(str(e))
            return

    # Resolve whisper model name to path (only needed for whisper provider)
    transcription_provider = getattr(args, "transcription_provider", "whisper")
    if args.whisper_model and not args.analyze_existing and transcription_provider == "whisper":
        from .transcription import resolve_whisper_model

        try:
            resolved = resolve_whisper_model(args.whisper_model)
            if resolved != args.whisper_model and args.verbose:
                logger.info(f"Whisper model: '{args.whisper_model}' -> {resolved}")
            args.whisper_model = resolved
        except Exception as e:
            log_error(str(e))
            return

    # Set defaults for paths
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    stats_file = Path(args.stats_file) if args.stats_file else DEFAULT_STATS_FILE

    # Determine where analysis files should go
    analysis_output_dir = Path(OBSIDIAN_PATH) if args.save_to_obsidian else output_dir

    if args.save_to_obsidian and args.verbose:
        logger.info(f"Analysis will be saved to Obsidian vault: {analysis_output_dir}")
        logger.info(f"Transcripts will be saved to: {output_dir}")

    # Initialize tracking variables
    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()

    # Validate test-segment options
    test_segment = getattr(args, "test_segment", None)
    start_at = getattr(args, "start_at", None)

    if test_segment is not None:
        if test_segment <= 0:
            log_error("--test-segment duration must be positive")
            return
        if test_segment > 30:
            log_error("--test-segment maximum is 30 minutes")
            return
        if args.analyze_existing:
            log_error("--test-segment cannot be used with --analyze_existing")
            return

    if start_at is not None:
        if test_segment is None:
            log_error("--start-at requires --test-segment")
            return
        if start_at < 0:
            log_error("--start-at must be non-negative")
            return

    # Validate --audio flag
    if getattr(args, "audio", None) and not getattr(args, "diarize_existing", None):
        log_error("--audio requires --diarize-existing")
        return

    # Handle diarize-existing mode
    diarize_existing = getattr(args, "diarize_existing", None)
    if diarize_existing:
        from .workflow import run_diarize_existing_mode

        run_diarize_existing_mode(
            diarize_existing,
            audio_override=getattr(args, "audio", None),
            verbose=args.verbose,
        )
        return

    # Handle analyze-existing mode
    if args.analyze_existing:
        if args.no_analyze:
            log_error(
                "--no-analyze cannot be used with --analyze_existing. "
                "The purpose of --analyze_existing is to analyze a transcript."
            )
            return

        run_analyze_existing_mode(
            args.analyze_existing,
            args,
            output_dir,
            analysis_output_dir,
            stats_file,
            run_uid,
            run_timestamp,
            start_time,
        )
        return

    if args.verbose:
        log_section("Transcription Tool")
        logger.info(f"Input: {args.input_source}")
        logger.info(f"Run ID: {run_uid}")
        logger.info(f"Timestamp: {run_timestamp}")

    # Check for duplicate transcription (unless --force or --test-segment is used)
    if not args.force and test_segment is None:
        prev_transcription = find_existing_transcription(stats_file, args.input_source, output_dir)

        if prev_transcription:
            # Handle non-interactive mode
            if not is_interactive():
                log_error(
                    f"Duplicate detected: '{prev_transcription.video_title}' "
                    f"was already transcribed on {prev_transcription.formatted_date}. "
                    "Use --force to proceed anyway."
                )
                return

            # Interactive mode: prompt user for action
            action = prompt_duplicate_detected(prev_transcription, args.verbose)

            if action == DuplicateAction.CANCEL:
                logger.info("Operation cancelled.")
                return

            elif action == DuplicateAction.ANALYZE_EXISTING:
                run_analyze_existing_mode(
                    prev_transcription.transcript_path,
                    args,
                    output_dir,
                    analysis_output_dir,
                    stats_file,
                    run_uid,
                    run_timestamp,
                    start_time,
                )
                return

            elif action == DuplicateAction.RE_TRANSCRIBE:
                logger.info("Re-transcribing video...")

            elif action == DuplicateAction.FORCE_CONTINUE:
                logger.info("Continuing with transcription...")

    # Run the main transcription workflow
    process_input_source(
        args.input_source,
        args,
        output_dir,
        analysis_output_dir,
        stats_file,
        run_uid,
        run_timestamp,
        start_time,
    )


if __name__ == "__main__":
    main()
