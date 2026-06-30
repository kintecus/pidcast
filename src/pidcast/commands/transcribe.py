"""Handler for ``pidcast transcribe`` (and the bare-input shortcut).

Owns fuzzy name resolution, test-segment validation, duplicate detection, and
the pause-aware run of the main transcription workflow. The analyze/diarize
existing-file paths are now their own verbs (see analyze.py / diarize.py).
"""

import argparse
import datetime
import logging
import time
import uuid
from pathlib import Path

from ..config import OBSIDIAN_PATH, RUNS_FILE, resolve_output_dir
from ..duplicate import DuplicateAction, prompt_duplicate_detected
from ..utils import (
    find_existing_transcription,
    is_interactive,
    log_error,
    log_section,
)
from ..workflow import process_input_source, run_analyze_existing_mode

logger = logging.getLogger(__name__)


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Handle ``pidcast transcribe <input>`` (and bare ``pidcast <input>``)."""
    # Resolve analysis type with fuzzy matching
    if args.analysis_type and args.analysis_type != "executive_summary":
        from ..utils import resolve_analysis_type

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
        from ..utils import resolve_model_name

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
    if args.whisper_model and transcription_provider == "whisper":
        from ..transcription import resolve_whisper_model

        try:
            resolved = resolve_whisper_model(args.whisper_model)
            if resolved != args.whisper_model and args.verbose:
                logger.info(f"Whisper model: '{args.whisper_model}' -> {resolved}")
            args.whisper_model = resolved
        except Exception as e:
            log_error(str(e))
            return

    # Set defaults for paths
    output_dir = resolve_output_dir(args)
    stats_file = Path(args.stats_file) if args.stats_file else RUNS_FILE

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
            log_error("--test duration must be positive")
            return
        if test_segment > 30:
            log_error("--test maximum is 30 minutes")
            return

    if start_at is not None:
        if test_segment is None:
            log_error("--start-at requires --test")
            return
        if start_at < 0:
            log_error("--start-at must be non-negative")
            return

    if args.verbose:
        log_section("Transcription Tool")
        logger.info(f"Input: {args.input}")
        logger.info(f"Run ID: {run_uid}")
        logger.info(f"Timestamp: {run_timestamp}")

    # Check for duplicate transcription (unless --force or --test is used)
    if not args.force and test_segment is None:
        prev_transcription = find_existing_transcription(stats_file, args.input, output_dir)

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

    # Run the main transcription workflow under a pause-aware SIGINT handler.
    _run_with_pause_handler(
        lambda: process_input_source(
            args.input,
            args,
            output_dir,
            analysis_output_dir,
            stats_file,
            run_uid,
            run_timestamp,
            start_time,
        )
    )


def _run_with_pause_handler(run_fn) -> None:
    """Run ``run_fn`` with Ctrl-C wired to a clean transcription pause.

    First Ctrl-C requests a pause (the whisper stream loop stops at the next
    segment boundary and the job is checkpointed). A second Ctrl-C restores the
    default handler and hard-quits.
    """
    import signal

    from ..workflow import request_pause

    state = {"count": 0}
    original = signal.getsignal(signal.SIGINT)

    def _handler(signum, frame):
        state["count"] += 1
        if state["count"] == 1:
            logger.info(
                "\nPausing transcription at the next segment boundary... "
                "(Ctrl-C again to force quit)"
            )
            request_pause()
        else:
            signal.signal(signal.SIGINT, original)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)
    try:
        run_fn()
    finally:
        signal.signal(signal.SIGINT, original)
