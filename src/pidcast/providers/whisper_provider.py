"""Whisper.cpp transcription provider."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..checkpoint import JobManifest

from ..config import HUGGINGFACE_TOKEN
from ..diarization import merge_whisper_with_diarization, run_diarization
from ..exceptions import DiarizationError, TranscriptionError
from ..transcription import (
    collapse_repeated_lines,
    dedupe_whisper_segments,
    merge_segments_to_whisper_json,
    run_whisper_transcription,
)
from . import TranscriptionResult

logger = logging.getLogger(__name__)

# When resuming a VAD run, back the offset off by this many ms before the last
# good segment. -ot + --vad can shift the boundary and swallow audio at the seam
# (verified in the phase-0 spike); re-decoding a few seconds and de-overlapping on
# merge avoids dropped words. Harmless for non-VAD resumes (de-overlap drops the
# re-decoded segments anyway).
_VAD_RESUME_BACKOFF_MS = 8000


def _fmt_clock(ms: int) -> str:
    """Render milliseconds as MM:SS (or H:MM:SS past an hour) for human-facing logs."""
    total_s = max(0, int(ms)) // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class WhisperTranscriptionProvider:
    """Transcription provider using local whisper.cpp."""

    def __init__(
        self,
        whisper_model: str,
        output_format: str,
        output_dir: str | Path,
        estimated_duration: float | None = None,
        save_whisper_json_to: Path | None = None,
        whisper_options: dict | None = None,
        checkpoint: JobManifest | None = None,
        pause_check: Callable[[], bool] | None = None,
        audio_duration: float | None = None,
    ) -> None:
        self._whisper_model = whisper_model
        self._output_format = output_format
        self._output_dir = Path(output_dir)
        self._estimated_duration = estimated_duration
        # True audio length (seconds) - drives the progress bar by audio position,
        # which stays correct across a resume offset (unlike the wall-clock estimate).
        self._audio_duration = audio_duration
        self._save_whisper_json_to = save_whisper_json_to
        # Decoding/quality knobs (threads, VAD, thresholds, suppress_nst) splatted
        # into run_whisper_transcription. Whisper-only, kept off the shared
        # TranscriptionProvider.transcribe() signature.
        self._opts = whisper_options or {}
        # Resume/pause: when set, segments stream into the manifest's JSONL and a
        # crash/pause can be resumed via -ot. None keeps the one-shot behavior
        # byte-identical to before.
        self._checkpoint = checkpoint
        self._pause_check = pause_check

    def transcribe(
        self,
        audio_file: str | Path,
        language: str | None = None,
        diarize: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using whisper.cpp."""
        temp_output = self._output_dir / f"temp_transcript_{uuid.uuid4().hex[:8]}"
        # Always output JSON when we need to save whisper data, diarize, or checkpoint.
        need_json = (
            diarize or self._save_whisper_json_to is not None or self._checkpoint is not None
        )
        output_format = "json" if need_json else self._output_format

        if diarize and not HUGGINGFACE_TOKEN:
            raise DiarizationError(
                "HUGGINGFACE_TOKEN environment variable not set. "
                "Required for speaker diarization with Whisper."
            )

        json_file = Path(f"{temp_output}.json")

        if self._checkpoint is not None:
            # Resumable path: stream segments into the manifest's JSONL, resume
            # from the persisted offset, then materialize the merged JSON. The
            # leg's compute time is accumulated into the manifest inside
            # _transcribe_checkpointed; report the CUMULATIVE total across all
            # resume sessions so a multi-session run's duration reflects the true
            # end-to-end transcription time (not just this leg).
            self._transcribe_checkpointed(audio_file, output_format, temp_output, language, verbose)
            duration = self._checkpoint.transcription.elapsed_seconds
        else:
            start_time = time.time()
            try:
                run_whisper_transcription(
                    audio_file,
                    self._whisper_model,
                    output_format,
                    str(temp_output),
                    verbose,
                    estimated_duration=self._estimated_duration,
                    audio_duration=self._audio_duration,
                    language=language,
                    pause_check=self._pause_check,
                    **self._opts,
                )
            except Exception as e:
                raise TranscriptionError(f"Whisper transcription failed: {e}") from e
            duration = time.time() - start_time

        # Save whisper JSON before diarization (so it survives diarization failures)
        whisper_json_path = None
        if self._save_whisper_json_to and json_file.exists():
            shutil.copy2(json_file, self._save_whisper_json_to)
            whisper_json_path = self._save_whisper_json_to
            if verbose:
                logger.info(f"Saved whisper JSON: {whisper_json_path}")

        # Handle diarization
        speaker_count = None
        if diarize:
            text, speaker_count = self._run_diarization(audio_file, temp_output, verbose)
            diarized = speaker_count is not None and speaker_count > 0
        else:
            # When we forced JSON for saving, also read plain text from it
            if need_json:
                segments = self._load_whisper_segments(json_file)
                text = "\n".join(seg["text"].strip() for seg in segments if seg["text"].strip())
            else:
                txt_file = Path(f"{temp_output}.txt")
                try:
                    text = collapse_repeated_lines(txt_file.read_text(encoding="utf-8"))
                except FileNotFoundError as e:
                    raise TranscriptionError(f"Whisper output file not found: {txt_file}") from e
            diarized = False

        # Clean up temp files
        self._cleanup_temp_files(temp_output)

        return TranscriptionResult(
            text=text,
            speaker_count=speaker_count,
            duration=duration,
            provider="whisper",
            language=language,
            diarized=diarized,
            whisper_json_path=whisper_json_path,
        )

    def _transcribe_checkpointed(
        self,
        audio_file: str | Path,
        output_format: str,
        temp_output: Path,
        language: str | None,
        verbose: bool,
    ) -> None:
        """Resumable transcription: stream segments to the JSONL, merge on completion.

        Resumes from the offset persisted in the manifest's JSONL. With VAD the
        offset is backed off a few seconds (the seam can swallow audio) and the
        re-decoded overlap is dropped at merge time. On completion the merged,
        de-overlapped whisper JSON is written to ``{temp_output}.json`` so the rest
        of the provider (text extraction, diarization, JSON save) is unchanged.
        """
        from ..checkpoint import DONE

        manifest = self._checkpoint
        assert manifest is not None

        # Transcription already finished in a prior run (job paused/crashed during
        # diarization or analysis). Skip whisper entirely and rebuild the JSON from
        # the persisted segments so we go straight to diarization/text.
        if manifest.transcription.status == DONE:
            logger.info("Transcription already complete - skipping whisper, rebuilding JSON...")
            self._materialize_json_from_segments(manifest, temp_output)
            return

        prior = manifest.load_segments()
        resume_offset = manifest.resume_offset_ms()
        if resume_offset > 0 and self._opts.get("vad_model"):
            resume_offset = max(0, resume_offset - _VAD_RESUME_BACKOFF_MS)

        if prior:
            done = _fmt_clock(resume_offset)
            if self._audio_duration and self._audio_duration > 0:
                total = _fmt_clock(int(self._audio_duration * 1000))
                pct = min(100, int(100 * resume_offset / (self._audio_duration * 1000)))
                logger.info(
                    f"Resuming from {done} of {total} ({pct}% done, "
                    f"{len(prior)} segments) - transcribing the remainder..."
                )
            else:
                logger.info(f"Resuming from {done} ({len(prior)} segments already done)...")

        def _on_segment(seg: dict) -> None:
            # Skip anything we already persisted (resume re-decode overlap).
            if seg["to_ms"] <= manifest.transcription.last_offset_ms:
                return
            manifest.append_segment(seg)
            manifest.transcription.last_offset_ms = seg["to_ms"]
            manifest.transcription.segment_count += 1

        # Time this leg and fold it into the manifest's cumulative total in a
        # finally so a pause (TranscriptionPaused) or crash still records the
        # active compute it consumed. Without this, a multi-session run would
        # report only the final leg's duration - skewing the estimate-vs-actual
        # line and poisoning the ETA training data with a falsely-fast ratio.
        leg_start = time.time()
        try:
            run_whisper_transcription(
                audio_file,
                self._whisper_model,
                output_format,
                str(temp_output),
                verbose,
                estimated_duration=self._estimated_duration,
                audio_duration=self._audio_duration,
                language=language,
                offset_ms=resume_offset,
                segment_callback=_on_segment,
                pause_check=self._pause_check,
                **self._opts,
            )
        except Exception:
            # TranscriptionPaused and real failures both propagate; the JSONL
            # holds everything decoded so far either way.
            raise
        finally:
            manifest.add_transcription_elapsed(time.time() - leg_start)

        # Materialize the full, de-overlapped whisper JSON from the JSONL (single
        # source of truth across the resume seam).
        self._materialize_json_from_segments(manifest, temp_output)

    @staticmethod
    def _materialize_json_from_segments(manifest, temp_output: Path) -> None:
        """Write the merged whisper JSON from the manifest's persisted segments.

        Preserves any metadata (model/params/systeminfo) from a freshly written
        per-run JSON if present, then overwrites it with the full de-overlapped set.
        """
        all_segments = manifest.load_segments()
        base_template: dict = {}
        raw_json = Path(f"{temp_output}.json")
        if raw_json.exists():
            try:
                loaded = json.loads(raw_json.read_text(encoding="utf-8"))
                base_template = {k: v for k, v in loaded.items() if k != "transcription"}
            except (json.JSONDecodeError, OSError):
                base_template = {}
        merged = merge_segments_to_whisper_json(all_segments, base_template)
        raw_json.parent.mkdir(parents=True, exist_ok=True)
        raw_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_diarization(
        self,
        audio_file: str | Path,
        temp_output: Path,
        verbose: bool,
    ) -> tuple[str, int]:
        """Run pyannote diarization and merge with whisper output."""
        logger.info("Running speaker diarization...")
        diarization_segments = run_diarization(audio_file, HUGGINGFACE_TOKEN, verbose)

        # Read (and dedup) whisper JSON output before aligning to speakers
        whisper_segments = self._load_whisper_segments(Path(f"{temp_output}.json"))
        text, speaker_count = merge_whisper_with_diarization(whisper_segments, diarization_segments)

        return text, speaker_count

    @staticmethod
    def _load_whisper_segments(json_file: Path) -> list[dict]:
        """Load whisper JSON segments and collapse hallucinated repeats.

        Shared by the plain-JSON and diarization paths so both get identical
        anti-hallucination dedup.
        """
        try:
            with open(json_file, encoding="utf-8") as f:
                whisper_data = json.load(f)
        except FileNotFoundError as e:
            raise TranscriptionError(f"Whisper JSON output not found: {json_file}") from e

        return dedupe_whisper_segments(whisper_data.get("transcription", []))

    def _cleanup_temp_files(self, temp_output: Path) -> None:
        """Remove temporary whisper output files."""
        for ext in (".txt", ".json", ".vtt", ".srt"):
            temp_file = Path(f"{temp_output}{ext}")
            if temp_file.exists():
                temp_file.unlink()
