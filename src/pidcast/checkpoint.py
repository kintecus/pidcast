"""Resume/pause checkpoint state for transcription jobs.

A *job* is one transcription run keyed by a content+config signature. Its state
lives in ``CHECKPOINT_DIR/<job_id>/``:

- ``manifest.json``           - phase status, resolved CLI args, video info.
- ``segments.jsonl``          - one completed whisper segment per line, appended
                                and flushed as it streams off whisper's stdout.
- ``source.wav``              - the 16kHz mono WAV whisper consumes, copied in
                                BEFORE transcription so a crash/pause is resumable
                                without re-downloading or re-converting.

The JSONL is the source of truth for the resume offset: the manifest's
``last_offset_ms`` is advisory (writes are throttled), so on load we always
recompute the offset from the last well-formed JSONL line. A partially written
trailing line (process killed mid-write) is tolerated and skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CHECKPOINT_DIR, VideoInfo

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
SEGMENTS_NAME = "segments.jsonl"
SOURCE_WAV_NAME = "source.wav"

# Phase statuses.
PENDING = "pending"
IN_PROGRESS = "in_progress"
PAUSED = "paused"
DONE = "done"

# Files smaller than this are hashed whole; larger ones use a head+tail sample so
# startup stays fast on multi-hundred-MB media.
_SIGNATURE_SAMPLE_THRESHOLD = 16 * 1024 * 1024
_SIGNATURE_SAMPLE_BYTES = 8 * 1024 * 1024


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def quick_file_signature(path: str | Path) -> str:
    """Cheap content signature for an audio file.

    For files under 16MB, hashes the whole file. For larger files, hashes the
    size plus the first and last 8MB - enough to detect a different recording
    without reading hundreds of MB on every run.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(p, "rb") as f:
        if size <= _SIGNATURE_SAMPLE_THRESHOLD:
            h.update(f.read())
        else:
            h.update(f.read(_SIGNATURE_SAMPLE_BYTES))
            f.seek(-_SIGNATURE_SAMPLE_BYTES, os.SEEK_END)
            h.update(f.read(_SIGNATURE_SAMPLE_BYTES))
    return h.hexdigest()


def compute_job_id(
    audio_signature: str,
    provider: str,
    model: str,
    language: str | None,
    vad_signature: str,
) -> str:
    """Stable id for a (content, config) pair.

    Config is part of the key on purpose: changing the model, language, or VAD
    settings produces a different job id, so a resume never merges segments
    decoded under incompatible settings.
    """
    raw = f"{audio_signature}:{provider}:{model}:{language or ''}:{vad_signature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class PhaseState:
    status: str = PENDING
    # Transcription-only; advisory mirror of the JSONL tail.
    last_offset_ms: int = 0
    segment_count: int = 0
    # Diarization-only.
    requested: bool = False
    speaker_count: int | None = None


@dataclass
class JobManifest:
    job_id: str
    input_source: str
    audio_signature: str
    smart_filename: str
    provider: str
    model: str
    language: str | None
    vad_signature: str
    # Resolved flags needed to re-run on `pidcast resume`. Stores the model NAME,
    # not a resolved absolute path - resume re-resolves the model and re-reads env
    # tokens, so a moved model file or rotated token still works.
    cli_args: dict[str, Any] = field(default_factory=dict)
    video_info: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    transcription: PhaseState = field(default_factory=PhaseState)
    diarization: PhaseState = field(default_factory=PhaseState)

    # -- paths -----------------------------------------------------------------

    @property
    def job_dir(self) -> Path:
        return CHECKPOINT_DIR / self.job_id

    @property
    def segments_path(self) -> Path:
        return self.job_dir / SEGMENTS_NAME

    @property
    def source_wav_path(self) -> Path:
        return self.job_dir / SOURCE_WAV_NAME

    # -- persistence -----------------------------------------------------------

    def save(self) -> None:
        """Atomically write the manifest (tmp + os.replace)."""
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now_iso()
        payload = asdict(self)
        tmp = self.job_dir / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.job_dir / MANIFEST_NAME)

    @classmethod
    def load(cls, job_dir: Path) -> JobManifest | None:
        """Load a manifest, or None if missing/corrupt (e.g. crashed mid-write)."""
        manifest_file = job_dir / MANIFEST_NAME
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping unreadable checkpoint manifest at {job_dir}: {e}")
            return None
        trans = PhaseState(**data.pop("transcription", {}) or {})
        diar = PhaseState(**data.pop("diarization", {}) or {})
        data.pop("schema_version", None)
        try:
            return cls(transcription=trans, diarization=diar, schema_version=SCHEMA_VERSION, **data)
        except TypeError as e:
            logger.warning(f"Checkpoint manifest schema mismatch at {job_dir}: {e}")
            return None

    def video_info_obj(self) -> VideoInfo | None:
        return VideoInfo.from_dict(self.video_info) if self.video_info else None

    # -- segment IO ------------------------------------------------------------

    def load_segments(self) -> list[dict]:
        """Read completed segments from the JSONL, skipping a partial last line."""
        if not self.segments_path.exists():
            return []
        segments: list[dict] = []
        with open(self.segments_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seg = json.loads(line)
                except json.JSONDecodeError:
                    # Trailing partial line from a kill mid-write - tolerate and stop.
                    logger.debug("Skipping partial trailing segment line in %s", self.segments_path)
                    break
                segments.append(seg)
        return segments

    def append_segment(self, segment: dict) -> None:
        """Append one segment record and flush so it survives an immediate kill."""
        self.job_dir.mkdir(parents=True, exist_ok=True)
        with open(self.segments_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(segment, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def resume_offset_ms(self) -> int:
        """Resume offset = end of the last well-formed persisted segment (0 if fresh)."""
        segments = self.load_segments()
        if not segments:
            return 0
        return int(segments[-1].get("to_ms", 0))


def find_resumable_jobs() -> list[JobManifest]:
    """Paused/in-progress jobs, newest first (ties broken by created_at, job_id)."""
    if not CHECKPOINT_DIR.exists():
        return []
    jobs: list[JobManifest] = []
    for job_dir in CHECKPOINT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        manifest = JobManifest.load(job_dir)
        if manifest is None:
            continue
        if manifest.transcription.status in (PAUSED, IN_PROGRESS) or (
            manifest.transcription.status == DONE and manifest.diarization.status != DONE
        ):
            jobs.append(manifest)
    jobs.sort(key=lambda m: (m.updated_at, m.created_at, m.job_id), reverse=True)
    return jobs


def cleanup_job_dir(job_id: str) -> None:
    """Remove a job's checkpoint directory (called on successful completion)."""
    job_dir = CHECKPOINT_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
