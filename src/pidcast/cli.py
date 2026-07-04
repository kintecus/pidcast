"""Command-line interface for pidcast.

Thin parser + dispatch layer. Every verb is a real subparser with a ``func``
default; command bodies live in ``commands/`` and the domain modules
(``library``/``setup``/``resume``). ``main()`` loads env, injects the bare-input
shortcut, parses, configures logging, and calls ``args.func(args)``.
"""

import argparse
import logging

from .config import (
    DEFAULT_PROMPTS_FILE,
    OBSIDIAN_PATH,
    RUNS_FILE,
    WHISPER_MODEL,
    resolve_output_dir,  # re-exported for backwards-compatible imports/tests
)
from .utils import log_error, setup_logging

logger = logging.getLogger(__name__)

# Re-export so callers/tests that did `from .cli import resolve_output_dir`
# keep working after the move to config.py.
__all__ = ["main", "parse_arguments", "build_transcribe_namespace", "resolve_output_dir"]


# Verbs that the bare-input shortcut must NOT treat as a transcription input.
KNOWN_VERBS = frozenset(
    {
        "transcribe",
        "analyze",
        "diarize",
        "lib",
        "list",
        "setup",
        "doctor",
        "resume",
        "info",
        "history",
    }
)


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
        # Preset keys may be written with hyphens (matching CLI flags like
        # --transcription-provider) or underscores (matching argparse dests).
        # Normalize to the dest form so both styles resolve.
        dest = key.replace("-", "_")
        if not hasattr(args, dest):
            logger.warning(f"Unknown preset key '{key}', skipping")
            continue
        if dest in explicitly_set:
            continue
        setattr(args, dest, value)

    if getattr(args, "verbose", False):
        logger.info(f"Applied preset '{args.preset}': {preset_values}")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def _build_parent_parsers() -> dict[str, argparse.ArgumentParser]:
    """Shared flag groups, passed to verbs via ``parents=[...]``.

    Keeps the common analysis/whisper/output/global flags defined once instead
    of duplicated across transcribe / lib process / lib sync.
    """
    parents: dict[str, argparse.ArgumentParser] = {}

    # --- global (verbosity / force / preset) ---------------------------------
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    g.add_argument(
        "-f", "--force", action="store_true", help="Skip duplicate detection and force the run"
    )
    g.add_argument(
        "-p",
        "--preset",
        default=None,
        help="Use a named preset from config.yaml (e.g., 'daily'). Explicit flags override it.",
    )
    parents["global"] = g

    # --- output --------------------------------------------------------------
    o = argparse.ArgumentParser(add_help=False)
    o.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for Markdown files (default: config.yaml output_dir, "
        "else the XDG data dir's transcripts/; see 'pidcast info')",
    )
    o.add_argument(
        "-o",
        "--save-to-obsidian",
        dest="save_to_obsidian",
        action="store_true",
        help=f"Save analysis files to Obsidian vault at: {OBSIDIAN_PATH}",
    )
    o.add_argument(
        "--save",
        action="store_true",
        help="Save analysis output to file (default: terminal only)",
    )
    o.add_argument(
        "--front-matter",
        dest="front_matter",
        default="{}",
        help="JSON string for Markdown front matter",
    )
    o.add_argument(
        "--tags",
        default=None,
        help="Comma-separated tags for front matter (overrides auto-inferred source tags).",
    )
    o.add_argument(
        "--stats-file",
        dest="stats_file",
        default=None,
        help=f"File to store run history/statistics (default: {RUNS_FILE})",
    )
    o.add_argument(
        "--keep-transcript",
        dest="keep_transcript",
        action="store_true",
        help="Keep the .txt transcript file alongside the .md file",
    )
    o.add_argument(
        "--keep-audio",
        dest="keep_audio",
        action="store_true",
        help="Save the converted WAV audio file to the output directory",
    )
    parents["output"] = o

    # --- analysis ------------------------------------------------------------
    a = argparse.ArgumentParser(add_help=False)
    a.add_argument(
        "--no-analyze",
        action="store_true",
        dest="no_analyze",
        help="Skip LLM analysis (default: analyze is enabled)",
    )
    a.add_argument(
        "-a",
        "--analysis-type",
        dest="analysis_type",
        default="executive_summary",
        help="Analysis type/prompt template (default: executive_summary). "
        "See 'pidcast list analyses'.",
    )
    a.add_argument(
        "--prompts-file",
        dest="prompts_file",
        default=None,
        help=f"Path to prompts YAML file (default: {DEFAULT_PROMPTS_FILE})",
    )
    a.add_argument(
        "--groq-api-key",
        dest="groq_api_key",
        default=None,
        help="Groq API key (default: GROQ_API_KEY environment variable)",
    )
    a.add_argument(
        "-m",
        "--groq-model",
        dest="groq_model",
        default=None,
        help="Groq model for analysis (default: from config/models.yaml). "
        "See 'pidcast list models'.",
    )
    a.add_argument(
        "--skip-analysis-on-error",
        dest="skip_analysis_on_error",
        action="store_true",
        help="Continue if analysis fails instead of aborting",
    )
    a.add_argument(
        "--provider",
        default="groq",
        choices=["groq", "claude"],
        help="LLM provider for analysis: 'groq' (default) or 'claude' (uses local Claude CLI)",
    )
    a.add_argument(
        "--claude-model",
        dest="claude_model",
        default=None,
        help="Claude model alias when --provider claude: 'sonnet' (default), 'opus', 'haiku'",
    )
    parents["analysis"] = a

    # --- whisper / transcription ---------------------------------------------
    w = argparse.ArgumentParser(add_help=False)
    w.add_argument(
        "--whisper-model",
        dest="whisper_model",
        default=WHISPER_MODEL,
        help="Whisper model name (e.g., 'large-v3-turbo') or path to model file",
    )
    w.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code for transcription (e.g., 'uk', 'en', 'de'). Default: auto-detect.",
    )
    w.add_argument(
        "--transcription-provider",
        dest="transcription_provider",
        choices=["whisper", "elevenlabs"],
        default="whisper",
        help="Transcription backend: whisper (local) or elevenlabs (cloud). Default: whisper.",
    )
    w.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Run speaker diarization (whisper: pyannote.audio + HUGGINGFACE_TOKEN; "
        "elevenlabs: built-in)",
    )
    w.add_argument(
        "--output-format",
        dest="output_format",
        default="otxt",
        help="Whisper output format (txt, vtt, srt, json). Prefix with 'o' for original filename.",
    )

    # Expert / tuning knobs. Hidden from the default --help; still settable and
    # preset-supplied. Help text is SUPPRESSed to keep the surface slim.
    w.add_argument("--vad", action="store_true", default=False, help=argparse.SUPPRESS)
    w.add_argument("--vad-model", dest="vad_model", default=None, help=argparse.SUPPRESS)
    w.add_argument(
        "--vad-threshold", dest="vad_threshold", type=float, default=None, help=argparse.SUPPRESS
    )
    w.add_argument(
        "--no-speech-thold",
        dest="no_speech_thold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    w.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    w.add_argument(
        "--no-fallback",
        dest="no_fallback",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    w.add_argument(
        "--suppress-nst",
        "--no-suppress-nst",
        dest="suppress_nst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    w.add_argument(
        "--whisper-threads", dest="whisper_threads", type=int, default=8, help=argparse.SUPPRESS
    )
    parents["whisper"] = w

    # --- download / cookies (transcribe + lib process) -----------------------
    d = argparse.ArgumentParser(add_help=False)
    d.add_argument(
        "--po-token",
        dest="po_token",
        default=None,
        help="PO Token for bypassing YouTube restrictions (format: 'client.type+TOKEN')",
    )
    d.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser to extract cookies from (e.g., 'chrome', 'firefox', 'safari')",
    )
    d.add_argument("--cookies", default=None, help="Path to Netscape format cookies file")
    d.add_argument(
        "--chrome-profile",
        default=None,
        dest="chrome_profile",
        help="Chrome profile for cookie extraction (display name or directory name)",
    )
    parents["download"] = d

    return parents


def _add_lib_subparsers(lib_subparsers, parents) -> None:
    """Wire the ``pidcast lib <command>`` tree."""
    from .commands import lib as lib_cmds

    g = parents["global"]

    add_p = lib_subparsers.add_parser("add", parents=[g], help="Add podcast to library")
    add_p.add_argument("feed_url", help="RSS feed URL or podcast name to search for")
    add_p.add_argument("--preview", action="store_true", help="Preview episodes before adding")
    add_p.set_defaults(func=lib_cmds.handle_add)

    list_p = lib_subparsers.add_parser("list", parents=[g], help="List all shows in library")
    list_p.set_defaults(func=lib_cmds.handle_list)

    show_p = lib_subparsers.add_parser("show", parents=[g], help="Show details for a podcast")
    show_p.add_argument("show_id", type=int, help="Show ID")
    show_p.add_argument(
        "--episodes", type=int, default=5, help="Number of recent episodes to show (default: 5)"
    )
    show_p.set_defaults(func=lib_cmds.handle_show)

    remove_p = lib_subparsers.add_parser("remove", parents=[g], help="Remove podcast from library")
    remove_p.add_argument("show_id", type=int, help="Show ID")
    remove_p.set_defaults(func=lib_cmds.handle_remove)

    process_p = lib_subparsers.add_parser(
        "process",
        parents=[
            g,
            parents["whisper"],
            parents["analysis"],
            parents["output"],
            parents["download"],
        ],
        help="Process an episode from a library show",
    )
    process_p.add_argument("show_query", help="Show ID or partial name")
    process_p.add_argument("--latest", action="store_true", help="Process the latest episode")
    process_p.add_argument("--match", help="Process episode matching this title string")
    process_p.set_defaults(func=lib_cmds.handle_process)

    sync_p = lib_subparsers.add_parser(
        "sync",
        parents=[g, parents["whisper"], parents["analysis"], parents["output"]],
        help="Sync library shows and process new episodes",
    )
    sync_p.add_argument("--show", type=int, metavar="ID", help="Sync only specific show by ID")
    sync_p.add_argument("--dry-run", action="store_true", help="Preview only")
    sync_p.add_argument("--backfill", type=int, metavar="N", help="Override backfill limit")
    sync_p.add_argument("--no-digest", action="store_true", help="Skip digest generation")
    sync_p.set_defaults(func=lib_cmds.handle_sync)

    digest_p = lib_subparsers.add_parser(
        "digest",
        parents=[g, parents["analysis"], parents["output"]],
        help="Generate podcast digest",
    )
    digest_p.add_argument("--date", help="Specific date (YYYY-MM-DD)")
    digest_p.add_argument("--range", help="Date range (e.g., 7d)")
    digest_p.set_defaults(func=lib_cmds.handle_digest)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the full verb-first parser (unconditional subparser tree)."""
    from .commands.analyze import cmd_analyze
    from .commands.diarize import cmd_diarize
    from .commands.history import cmd_history
    from .commands.info import cmd_info
    from .commands.listing import cmd_list
    from .commands.transcribe import cmd_transcribe
    from .resume import run_resume
    from .setup import run_doctor, run_setup

    parents = _build_parent_parsers()

    parser = argparse.ArgumentParser(
        prog="pidcast",
        description="Turn a URL or audio file into an Obsidian-ready transcript with LLM analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common workflows:
  pidcast "https://youtube.com/watch?v=ID"          Transcribe + analyze (bare = transcribe)
  pidcast transcribe FILE.mp3 -o -a exec            Save to Obsidian, executive summary
  pidcast transcribe FILE.mp3 --test 5 --start-at 10  Dry-run a 5-min slice from 10:00
  pidcast analyze transcript.md -a summary          Re-analyze an existing transcript
  pidcast diarize transcript.md                     Retry diarization only
  pidcast list models                               Discover analyses/models/presets/...
  pidcast lib add "99% Invisible"                     Manage the podcast library
  pidcast history -n 20                             Show last 20 runs
""",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # --- transcribe (also the bare-input default) ----------------------------
    t = sub.add_parser(
        "transcribe",
        parents=[
            parents["global"],
            parents["whisper"],
            parents["analysis"],
            parents["output"],
            parents["download"],
        ],
        help="Transcribe a URL or audio file (and analyze)",
    )
    t.add_argument("input", help="YouTube/podcast URL or path to a local audio file")
    t.add_argument(
        "--test",
        nargs="?",
        const=2,
        type=float,
        default=None,
        dest="test_segment",
        metavar="MINUTES",
        help="Dry-run transcription on the first N minutes (default: 2). "
        "Skips markdown, analysis, and stats.",
    )
    t.add_argument(
        "--start-at",
        type=float,
        default=None,
        dest="start_at",
        metavar="MINUTES",
        help="Start offset in minutes for --test (default: 0).",
    )
    # Checkpoint / resume options.
    t.add_argument(
        "--no-checkpoint",
        dest="no_checkpoint",
        action="store_true",
        help="Disable resume checkpointing (one-shot transcription).",
    )
    t.add_argument(
        "--keep-checkpoint",
        dest="keep_checkpoint",
        action="store_true",
        help="Keep the checkpoint directory after a successful run (default: delete).",
    )
    # Internal: set by `pidcast resume` to re-enter a specific job.
    t.add_argument("--resume-job-id", dest="resume_job_id", default=None, help=argparse.SUPPRESS)
    t.set_defaults(func=cmd_transcribe)

    # --- analyze -------------------------------------------------------------
    an = sub.add_parser(
        "analyze",
        parents=[parents["global"], parents["analysis"], parents["output"]],
        help="Analyze an existing transcript without re-transcribing",
    )
    an.add_argument("transcript", help="Path to an existing transcript (.md or .txt)")
    an.set_defaults(func=cmd_analyze)

    # --- diarize -------------------------------------------------------------
    di = sub.add_parser(
        "diarize",
        parents=[parents["global"]],
        help="Run speaker diarization on an existing transcript",
    )
    di.add_argument("transcript", help="Path to an existing transcript (.md). Needs .whisper.json.")
    di.add_argument(
        "--audio", default=None, help="Path to audio file (overrides auto-detected .wav)."
    )
    di.set_defaults(func=cmd_diarize)

    # --- list ----------------------------------------------------------------
    li = sub.add_parser(
        "list", parents=[parents["global"]], help="List analyses/models/presets/..."
    )
    li.add_argument(
        "thing",
        choices=["analyses", "models", "whisper-models", "presets", "profiles"],
        help="What to list",
    )
    li.set_defaults(func=cmd_list)

    # --- lib -----------------------------------------------------------------
    lib_p = sub.add_parser("lib", help="Podcast library management")
    lib_sub = lib_p.add_subparsers(dest="lib_command", metavar="<lib-command>", required=True)
    _add_lib_subparsers(lib_sub, parents)

    # --- setup / doctor / resume / info --------------------------------------
    su = sub.add_parser("setup", parents=[parents["global"]], help="Interactive setup wizard")
    su.set_defaults(func=run_setup)

    do = sub.add_parser("doctor", parents=[parents["global"]], help="Diagnose tooling/env config")
    do.add_argument(
        "--prune-stats",
        dest="prune_stats",
        action="store_true",
        help="Remove phantom stats entries with a missing transcript",
    )
    do.set_defaults(func=run_doctor)

    re = sub.add_parser("resume", parents=[parents["global"]], help="Resume a paused transcription")
    re.add_argument("job_id", nargs="?", default=None, help="Job id to resume (optional)")
    re.set_defaults(func=run_resume)

    inf = sub.add_parser(
        "info", parents=[parents["global"]], help="Show resolved data/config paths"
    )
    inf.set_defaults(func=cmd_info)

    hi = sub.add_parser(
        "history", parents=[parents["global"]], help="List recent transcription runs"
    )
    hi.add_argument(
        "-n", "--limit", type=int, default=10, help="Number of runs to show (default: 10)"
    )
    hi.set_defaults(func=cmd_history)

    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Explicit argument list (without prog). Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed arguments namespace with ``command`` and ``func`` set.
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    return parser.parse_args(_inject_default_verb(argv))


def _suppress_all_defaults(parser: argparse.ArgumentParser) -> None:
    """Set every action's default to SUPPRESS, recursing into subparsers.

    After this, a parse only populates dests the user actually supplied — the
    basis for robust explicit-flag detection across any flag spelling.
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _suppress_all_defaults(sub)
        else:
            action.default = argparse.SUPPRESS


def _explicitly_set_dests(argv: list[str]) -> set[str]:
    """Return the dests the user explicitly passed, by any spelling.

    Re-parses ``argv`` through a shadow parser whose defaults are all SUPPRESS,
    so only supplied options appear in the result. This correctly catches short
    aliases (``-m``) and ``BooleanOptionalAction`` (``--no-suppress-nst``) that a
    naive ``sys.argv`` substring scan misses.
    """
    shadow = _build_parser()
    _suppress_all_defaults(shadow)
    try:
        ns, _ = shadow.parse_known_args(_inject_default_verb(argv))
    except SystemExit:
        # A parse error here shouldn't crash preset handling; fall back to "none
        # detected" so explicit flags simply aren't protected (preset still safe).
        return set()
    # `command`/`func` are structural, not user flags; drop them.
    return {k for k in vars(ns) if k not in ("command", "func")}


def _inject_default_verb(argv: list[str]) -> list[str]:
    """Inject the implicit ``transcribe`` verb for the bare-input shortcut.

    Rule: prepend ``transcribe`` only when the first token is a non-empty value
    that is NOT a known verb and does NOT start with ``-``. So ``pidcast URL``
    and ``pidcast file.mp3`` route to transcribe, while ``pidcast --help``,
    ``pidcast lib list``, and ``pidcast info`` are left untouched. A file
    literally named like a verb is shadowed (use ``transcribe ./info``).
    """
    if not argv:
        return argv
    first = argv[0]
    if not first or first.startswith("-") or first in KNOWN_VERBS:
        return argv
    return ["transcribe", *argv]


def build_transcribe_namespace(input_path: str) -> argparse.Namespace:
    """Materialize a fully-defaulted ``transcribe`` Namespace for a given input.

    Used by ``resume`` to rebuild the original run's args without mutating
    ``sys.argv``. Every dest the transcribe workflow reads is present with its
    default, ready to be overlaid with persisted checkpoint flags.
    """
    return parse_arguments(["transcribe", input_path])


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    """Main entry point for pidcast CLI."""
    from dotenv import load_dotenv

    # Env must load before parsing so env-backed defaults resolve.
    load_dotenv()

    args = parse_arguments()

    # No verb (and no bare input) -> show top-level help.
    if not getattr(args, "command", None) or not hasattr(args, "func"):
        _build_parser().print_help()
        return

    setup_logging(getattr(args, "verbose", False))

    # Apply preset if specified (transcribe/lib paths carry -p). Detect which
    # flags the user passed (by any spelling) so the preset never clobbers them.
    if getattr(args, "preset", None):
        import sys

        explicitly_set = _explicitly_set_dests(sys.argv[1:])
        try:
            apply_preset(args, explicitly_set=explicitly_set)
        except ValueError as e:
            log_error(str(e))
            return

    # Load chrome_profile from config if not explicitly set (download paths).
    if hasattr(args, "chrome_profile") and not getattr(args, "chrome_profile", None):
        from .config_manager import ConfigManager

        config = ConfigManager.load_config()
        chrome_profile = config.get("chrome_profile")
        if chrome_profile:
            args.chrome_profile = chrome_profile

    args.func(args)


if __name__ == "__main__":
    main()
