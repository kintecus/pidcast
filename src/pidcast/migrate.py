"""One-shot migration from the legacy in-repo layout to the XDG data dir.

Moves transcripts/audio/logs out of the repo's ``data/`` tree and merges the
two legacy history stores (``history.json`` keyed by guid, and the append-only
``transcription_stats.json`` list) into the unified ``runs.json``, rewriting
absolute paths that point into the old repo location.

A plain ``mv`` is unsafe: the histories store absolute paths used for duplicate
detection, so those must be rewritten in-file. External paths (transcripts that
live outside the repo, e.g. an interview-recordings dir) are left untouched.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_TRANSCRIPT_GLOBS = ("*.md", "*.whisper.json")
_AUDIO_GLOBS = ("*.wav",)
# Date-prefixed stray transcripts that landed at the repo root (the same
# YYYY-MM-DD_ pattern .gitignore guards). Restricting to this pattern avoids
# sweeping up README.md / CLAUDE.md / etc. at the root.
_STRAY_ROOT_GLOBS = (
    "20[0-9][0-9]-[01][0-9]-[0-3][0-9]_*.md",
    "20[0-9][0-9]-[01][0-9]-[0-3][0-9]_*.whisper.json",
)
# Path keys in a record that may hold an absolute in-repo path to rewrite.
_PATH_KEYS = ("transcript_path", "output_file", "analysis_file", "audio_path")


@dataclass
class MigrationReport:
    """Counts of what migrate_data did (or would do, for a dry run)."""

    transcripts_moved: int = 0
    audio_moved: int = 0
    logs_moved: int = 0
    skipped: int = 0
    history_records: int = 0
    stats_records: int = 0
    paths_rewritten: int = 0
    dry_run: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verb = "would move" if self.dry_run else "moved"
        lines = [
            f"  {verb} {self.transcripts_moved} transcript file(s)",
            f"  {verb} {self.audio_moved} audio file(s)",
            f"  {verb} {self.logs_moved} log file(s)",
            f"  skipped {self.skipped} file(s) (destination already exists)",
            f"  merged {self.history_records} sync + {self.stats_records} run record(s)",
            f"  rewrote {self.paths_rewritten} in-repo path(s)",
        ]
        return "\n".join(lines)


def rewrite_legacy_path(
    value: str | None,
    legacy_repo_root: Path,
    legacy_transcripts: Path,
    new_transcripts: Path,
) -> str | None:
    """Rewrite an absolute path that lived under the old repo to the new dir.

    Handles two legacy locations, both mapped to ``new_transcripts``:
      - ``<repo>/data/transcripts/<name>`` (the canonical old location)
      - ``<repo>/<name>`` (stray transcripts that landed at the repo root)

    Paths outside the repo (or empty/relative) are returned unchanged.
    """
    if not value:
        return value

    try:
        path = Path(value)
        if not path.is_absolute():
            return value
        resolved = path.resolve()
    except OSError:
        return value

    try:
        legacy_t = legacy_transcripts.resolve()
        legacy_r = legacy_repo_root.resolve()
    except OSError:
        return value

    # Compare case-insensitively so macOS APFS paths that differ only in case
    # (e.g. /Users/x/Code vs /Users/x/code, both valid aliases of the same dir)
    # still match the legacy anchors. Path.resolve() does NOT normalize case,
    # and os.path.normcase is a no-op on POSIX, so we lowercase explicitly.
    tail = _relative_tail(resolved, legacy_t)
    if tail is not None:
        return str(new_transcripts / tail) if tail else str(new_transcripts)

    # Stray-at-repo-root: a transcript file sitting directly in the repo root.
    if str(resolved.parent).lower() == str(legacy_r).lower():
        return str(new_transcripts / resolved.name)

    return value


def _relative_tail(path: Path, base: Path) -> str | None:
    """Return path's portion below base (case-insensitive), or None if not under base."""
    norm_path = str(path).lower()
    norm_base = str(base).lower()
    prefix = norm_base.rstrip(os.sep) + os.sep
    if norm_path == norm_base:
        return ""
    if norm_path.startswith(prefix):
        # Slice the ORIGINAL (case-preserving) string by the matched length.
        return str(path)[len(prefix) :]
    return None


def migrate_data(
    legacy_repo_root: Path,
    data_dir: Path,
    legacy_history_file: Path | None = None,
    legacy_stats_file: Path | None = None,
    dry_run: bool = False,
) -> MigrationReport:
    """Migrate legacy in-repo data to the XDG data dir.

    Args:
        legacy_repo_root: The pidcast repo root (holds the old ``data/`` tree).
        data_dir: The new XDG data dir (``DATA_DIR``).
        legacy_history_file: Old sync ``history.json`` (defaults to the config dir).
        legacy_stats_file: Old ``transcription_stats.json`` (defaults to the
            repo's ``data/transcripts/``).
        dry_run: If True, report what would happen without moving/writing anything.

    Returns:
        A MigrationReport with counts.
    """
    report = MigrationReport(dry_run=dry_run)

    legacy_transcripts = legacy_repo_root / "data" / "transcripts"
    legacy_logs = legacy_repo_root / "data" / "logs"
    new_transcripts = data_dir / "transcripts"
    new_audio = data_dir / "audio"
    new_logs = data_dir / "logs"
    new_state = data_dir / "state"

    if not dry_run:
        for d in (new_transcripts, new_audio, new_logs, new_state):
            d.mkdir(parents=True, exist_ok=True)

    # --- Move transcript + sidecar files -----------------------------------
    if legacy_transcripts.is_dir():
        for pattern in _TRANSCRIPT_GLOBS:
            for src in sorted(legacy_transcripts.glob(pattern)):
                moved = _move_one(src, new_transcripts / src.name, dry_run, report)
                if moved:
                    report.transcripts_moved += 1
        for pattern in _AUDIO_GLOBS:
            for src in sorted(legacy_transcripts.glob(pattern)):
                moved = _move_one(src, new_audio / src.name, dry_run, report)
                if moved:
                    report.audio_moved += 1

    # --- Move stray date-prefixed transcripts left at the repo root --------
    for pattern in _STRAY_ROOT_GLOBS:
        for src in sorted(legacy_repo_root.glob(pattern)):
            moved = _move_one(src, new_transcripts / src.name, dry_run, report)
            if moved:
                report.transcripts_moved += 1

    # --- Move error log ----------------------------------------------------
    legacy_errors = legacy_logs / "errors.jsonl"
    if legacy_errors.exists():
        moved = _move_one(legacy_errors, new_logs / "errors.jsonl", dry_run, report)
        if moved:
            report.logs_moved += 1

    # --- Merge histories ---------------------------------------------------
    _migrate_histories(
        legacy_repo_root=legacy_repo_root,
        legacy_transcripts=legacy_transcripts,
        new_transcripts=new_transcripts,
        runs_file=new_state / "runs.json",
        legacy_history_file=legacy_history_file,
        legacy_stats_file=legacy_stats_file or (legacy_transcripts / "transcription_stats.json"),
        dry_run=dry_run,
        report=report,
    )

    return report


def _move_one(src: Path, dst: Path, dry_run: bool, report: MigrationReport) -> bool:
    """Move src -> dst. Skip (don't overwrite) if dst exists. Returns True if moved."""
    if dst.exists():
        report.skipped += 1
        return False
    if dry_run:
        return True
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        report.notes.append(f"failed to move {src.name}: {e}")
        return False


def _migrate_histories(
    legacy_repo_root: Path,
    legacy_transcripts: Path,
    new_transcripts: Path,
    runs_file: Path,
    legacy_history_file: Path | None,
    legacy_stats_file: Path,
    dry_run: bool,
    report: MigrationReport,
) -> None:
    """Merge legacy history.json (dict) + transcription_stats.json (list) into runs.json."""
    from .history import RunHistory, RunRecord

    def rewrite(value: str | None) -> str | None:
        out = rewrite_legacy_path(value, legacy_repo_root, legacy_transcripts, new_transcripts)
        if out != value:
            report.paths_rewritten += 1
        return out

    # Load legacy sync history (dict keyed by guid)
    sync_entries: dict[str, dict] = {}
    if legacy_history_file and legacy_history_file.exists():
        try:
            sync_entries = json.loads(legacy_history_file.read_text(encoding="utf-8"))
        except Exception as e:
            report.notes.append(f"could not read legacy history.json: {e}")

    # Load legacy stats (append-only list)
    stats_entries: list[dict] = []
    if legacy_stats_file.exists():
        try:
            loaded = json.loads(legacy_stats_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                stats_entries = loaded
        except Exception as e:
            report.notes.append(f"could not read legacy transcription_stats.json: {e}")

    report.history_records = len(sync_entries)
    report.stats_records = len(stats_entries)

    if dry_run:
        # Still count the rewrites that would happen.
        for entry in sync_entries.values():
            rewrite(entry.get("output_file"))
        for entry in stats_entries:
            for key in _PATH_KEYS:
                rewrite(entry.get(key))
        return

    history = RunHistory(runs_file)

    # Merge sync entries into by_guid (idempotent: keyed by guid).
    for guid, entry in sync_entries.items():
        rewritten = dict(entry)
        rewritten["output_file"] = rewrite(entry.get("output_file"))
        history.entries[guid] = RunRecord.from_dict(rewritten)

    # Append stats entries to the runs list, rewriting any in-repo paths.
    # Dedup against what's already present (idempotent re-run) by source_id+timestamp.
    existing_keys = {
        (r.get("source_id"), r.get("run_timestamp"), r.get("transcript_path"))
        for r in history.get_runs_for_estimation()
    }
    for entry in stats_entries:
        rewritten = dict(entry)
        for key in _PATH_KEYS:
            if key in rewritten:
                rewritten[key] = rewrite(rewritten[key])
        run_key = (
            rewritten.get("source_id"),
            rewritten.get("run_timestamp"),
            rewritten.get("transcript_path"),
        )
        if run_key in existing_keys:
            continue
        existing_keys.add(run_key)
        history.append_run_dict(rewritten)

    history.save()
