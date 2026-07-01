"""Resume a paused/interrupted transcription job from its checkpoint manifest.

Reconstructs the original CLI arguments from the manifest, validates that the job
can actually be resumed (binary, model, audio present), points the input at the
job dir's ``source.wav`` so nothing is re-downloaded, skips duplicate detection,
and re-enters the normal workflow with the checkpoint wired in.
"""

from __future__ import annotations

import datetime
import logging
import time
import uuid
from pathlib import Path

from .checkpoint import DONE, JobManifest
from .config import RUNS_FILE, TRANSCRIPTS_DIR, WHISPER_CPP_PATH
from .utils import log_error, log_section

logger = logging.getLogger(__name__)


def _reconstruct_args(manifest: JobManifest):
    """Rebuild an argparse.Namespace: full defaults overlaid with persisted flags."""
    from .cli import build_transcribe_namespace

    # A fully-defaulted transcribe Namespace (no sys.argv mutation).
    args = build_transcribe_namespace(manifest.source_wav_path.as_posix())

    # Overlay only the flags that belong to the transcribe surface. Legacy
    # manifests may carry dropped dests (e.g. analyze_existing/diarize_existing)
    # from before the verb-grammar refactor; filtering against the live
    # namespace prevents those dead flags from resurrecting onto args.
    for key, value in manifest.cli_args.items():
        if hasattr(args, key):
            setattr(args, key, value)

    # Resume always points at the job-dir source WAV (a local file), forces past
    # duplicate detection, and disables test-segment.
    args.input = str(manifest.source_wav_path)
    args.force = True
    args.test_segment = None
    args.start_at = None
    args.resume_job_id = manifest.job_id
    args.no_checkpoint = False
    # Keep the checkpoint across a resume that itself gets paused again.
    args.keep_checkpoint = getattr(args, "keep_checkpoint", False)
    return args


def _preflight(manifest: JobManifest) -> bool:
    """Validate the job can be resumed; log and return False if not."""
    if not manifest.source_wav_path.exists():
        log_error(
            f"Checkpoint audio missing: {manifest.source_wav_path}\n"
            "  Cannot resume without the source WAV. Re-run the original command."
        )
        return False
    if manifest.provider == "whisper" and not WHISPER_CPP_PATH:
        log_error("WHISPER_CPP_PATH is not set - cannot resume a whisper job.")
        return False
    # Resolve the model by name (paths may have moved since the job was created).
    if manifest.provider == "whisper":
        from .transcription import resolve_whisper_model

        model_name = manifest.cli_args.get("whisper_model") or manifest.model
        try:
            resolve_whisper_model(model_name)
        except Exception as e:
            log_error(f"Cannot resolve whisper model '{model_name}' for resume: {e}")
            return False
    return True


def resume_job(manifest: JobManifest) -> None:
    """Resume the given job to completion."""
    phase = "diarization" if manifest.transcription.status == DONE else "transcription"
    title = (manifest.video_info or {}).get("title", manifest.input_source)
    log_section(f"Resuming {phase}: {title}")
    if phase == "diarization":
        logger.info(
            f"Job {manifest.job_id} - transcription already complete, "
            "resuming at the diarization step."
        )
    else:
        from .providers.whisper_provider import _fmt_clock

        # Count from the JSONL (the source of truth), not the manifest's counter,
        # so "N segments done" always agrees with the resume offset.
        segments = manifest.load_segments()
        offset_ms = manifest.resume_offset_ms()
        logger.info(
            f"Job {manifest.job_id} - {len(segments)} segments done, "
            f"continuing from {_fmt_clock(offset_ms)}."
        )

    if not _preflight(manifest):
        return

    args = _reconstruct_args(manifest)

    # Resolve the whisper model name -> path now (workflow expects a resolved path).
    if manifest.provider == "whisper":
        from .transcription import resolve_whisper_model

        args.whisper_model = resolve_whisper_model(
            manifest.cli_args.get("whisper_model") or manifest.model
        )

    from .commands.transcribe import _run_with_pause_handler
    from .workflow import process_input_source

    # Honor the manifest's pinned output_dir (reconstructed onto args); fall back
    # to the canonical transcripts dir, never the cwd. Resume deliberately does
    # NOT use resolve_output_dir - a resumed job must land where it originally did.
    output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else TRANSCRIPTS_DIR
    stats_file = Path(args.stats_file) if getattr(args, "stats_file", None) else RUNS_FILE
    analysis_output_dir = output_dir
    if getattr(args, "save_to_obsidian", False):
        from .config import OBSIDIAN_PATH

        if OBSIDIAN_PATH:
            analysis_output_dir = Path(OBSIDIAN_PATH)

    # Reuse the ORIGINAL recording's metadata (title, url, channel) so the resumed
    # transcript keeps its real name and front matter - otherwise the title would
    # be re-derived from the checkpoint's generic "source.wav" filename.
    video_info_override = manifest.video_info_obj()

    run_uid = uuid.uuid4().hex[:12]
    run_timestamp = datetime.datetime.now().isoformat()

    _run_with_pause_handler(
        lambda: process_input_source(
            args.input,
            args,
            output_dir,
            analysis_output_dir,
            stats_file,
            run_uid,
            run_timestamp,
            time.time(),
            video_info_override=video_info_override,
        )
    )


def run_resume(args=None) -> None:
    """Resume a paused/interrupted transcription job (``pidcast resume [job-id]``)."""
    from .checkpoint import DONE, find_resumable_jobs

    job_arg = getattr(args, "job_id", None)

    jobs = find_resumable_jobs()
    if not jobs:
        logger.info("No resumable jobs found.")
        return

    manifest: JobManifest | None = None
    if job_arg:
        manifest = next((j for j in jobs if j.job_id.startswith(job_arg)), None)
        if manifest is None:
            logger.error(f"No resumable job matching '{job_arg}'.")
            return
    elif len(jobs) == 1:
        manifest = jobs[0]
    else:
        logger.info("Multiple resumable jobs - pass a job id to pick one:\n")
        for j in jobs:
            phase = "diarization" if j.transcription.status == DONE else "transcription"
            title = (j.video_info or {}).get("title", j.input_source)
            logger.info(
                f"  {j.job_id}  [{phase}]  {title}  "
                f"({j.transcription.segment_count} segments, updated {j.updated_at})"
            )
        return

    resume_job(manifest)
