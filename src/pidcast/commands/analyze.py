"""Handler for ``pidcast analyze <transcript>`` (re-analyze an existing file)."""

import argparse
import datetime
import logging
import time
import uuid
from pathlib import Path

from ..config import OBSIDIAN_PATH, RUNS_FILE, resolve_output_dir
from ..utils import log_error
from ..workflow import run_analyze_existing_mode

logger = logging.getLogger(__name__)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze an existing transcript without re-transcribing."""
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

    output_dir = resolve_output_dir(args)
    stats_file = Path(args.stats_file) if args.stats_file else RUNS_FILE
    analysis_output_dir = Path(OBSIDIAN_PATH) if args.save_to_obsidian else output_dir

    run_analyze_existing_mode(
        args.transcript,
        args,
        output_dir,
        analysis_output_dir,
        stats_file,
        str(uuid.uuid4()),
        datetime.datetime.now().isoformat(),
        time.time(),
    )
