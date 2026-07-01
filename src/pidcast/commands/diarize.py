"""Handler for ``pidcast diarize <transcript>`` (retry diarization only)."""

import argparse
import logging

from ..workflow import run_diarize_existing_mode

logger = logging.getLogger(__name__)


def cmd_diarize(args: argparse.Namespace) -> None:
    """Run speaker diarization on an existing transcript (no re-transcription)."""
    run_diarize_existing_mode(
        args.transcript,
        audio_override=getattr(args, "audio", None),
        verbose=args.verbose,
    )
