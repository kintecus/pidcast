"""Tests for the one-shot data migration (pidcast migrate-data).

Moves the legacy in-repo layout to the XDG data dir and merges the two legacy
history stores into the unified runs.json, rewriting absolute paths that point
into the old repo location while leaving external paths untouched.
"""

import json
from pathlib import Path

from pidcast.migrate import MigrationReport, migrate_data, rewrite_legacy_path

# ---------------------------------------------------------------------------
# Path rewriting
# ---------------------------------------------------------------------------


def test_rewrite_path_under_legacy_transcripts(tmp_path):
    legacy_root = tmp_path / "repo"
    legacy_transcripts = legacy_root / "data" / "transcripts"
    new_transcripts = tmp_path / "xdg" / "transcripts"

    old = str(legacy_transcripts / "2026-01-01_Ep.md")
    new = rewrite_legacy_path(old, legacy_root, legacy_transcripts, new_transcripts)
    assert new == str(new_transcripts / "2026-01-01_Ep.md")


def test_rewrite_path_at_legacy_repo_root(tmp_path):
    legacy_root = tmp_path / "repo"
    legacy_transcripts = legacy_root / "data" / "transcripts"
    new_transcripts = tmp_path / "xdg" / "transcripts"

    old = str(legacy_root / "2026-01-01_Stray.md")  # stray at repo root
    new = rewrite_legacy_path(old, legacy_root, legacy_transcripts, new_transcripts)
    assert new == str(new_transcripts / "2026-01-01_Stray.md")


def test_rewrite_leaves_external_paths_untouched(tmp_path):
    legacy_root = tmp_path / "repo"
    legacy_transcripts = legacy_root / "data" / "transcripts"
    new_transcripts = tmp_path / "xdg" / "transcripts"

    external = "/Users/x/other-project/interview-recordings/2026-01-01_screen.md"
    out = rewrite_legacy_path(external, legacy_root, legacy_transcripts, new_transcripts)
    assert out == external


def test_rewrite_handles_none(tmp_path):
    assert rewrite_legacy_path(None, tmp_path, tmp_path / "t", tmp_path / "n") is None


def test_rewrite_is_case_insensitive_for_legacy_anchor():
    # Real-world wrinkle: the legacy sync history stored paths under
    # /Users/.../Code/... (capital C) while PROJECT_ROOT resolves to .../code/...
    # On a case-insensitive filesystem these are the same dir, so the rewrite
    # must still fire. Path.resolve() does NOT normalize case, hence this test.
    legacy_root = Path("/Users/me/code/pidcast")
    legacy_transcripts = legacy_root / "data" / "transcripts"
    new_transcripts = Path("/Users/me/.local/share/pidcast/transcripts")

    old = "/Users/me/Code/pidcast/data/transcripts/2026-01-01_Ep.md"  # capital Code
    new = rewrite_legacy_path(old, legacy_root, legacy_transcripts, new_transcripts)
    assert new == str(new_transcripts / "2026-01-01_Ep.md")


# ---------------------------------------------------------------------------
# File moves
# ---------------------------------------------------------------------------


def _legacy_repo(tmp_path):
    """Build a fake legacy repo layout. Returns (repo_root, data_dir)."""
    repo = tmp_path / "repo"
    transcripts = repo / "data" / "transcripts"
    logs = repo / "data" / "logs"
    transcripts.mkdir(parents=True)
    logs.mkdir(parents=True)
    (transcripts / "2026-01-01_Ep.md").write_text("transcript")
    (transcripts / "2026-01-01_Ep.whisper.json").write_text("{}")
    (transcripts / "2026-01-01_Ep.wav").write_text("audio")
    (transcripts / "2026-01-01_podcast-digest.md").write_text("digest")
    (logs / "errors.jsonl").write_text('{"e":1}\n')
    return repo, repo / "data"


def test_migrate_moves_files(tmp_path):
    repo, _ = _legacy_repo(tmp_path)
    xdg = tmp_path / "xdg"

    report = migrate_data(
        legacy_repo_root=repo,
        data_dir=xdg,
        dry_run=False,
    )

    assert (xdg / "transcripts" / "2026-01-01_Ep.md").exists()
    assert (xdg / "transcripts" / "2026-01-01_Ep.whisper.json").exists()
    assert (xdg / "transcripts" / "2026-01-01_podcast-digest.md").exists()
    assert (xdg / "audio" / "2026-01-01_Ep.wav").exists()
    assert (xdg / "logs" / "errors.jsonl").exists()
    assert report.transcripts_moved >= 2
    assert report.audio_moved == 1


def test_migrate_dry_run_moves_nothing(tmp_path):
    repo, _ = _legacy_repo(tmp_path)
    xdg = tmp_path / "xdg"

    report = migrate_data(legacy_repo_root=repo, data_dir=xdg, dry_run=True)

    # Source files still in place, destination not created.
    assert (repo / "data" / "transcripts" / "2026-01-01_Ep.md").exists()
    assert not (xdg / "transcripts" / "2026-01-01_Ep.md").exists()
    # Report still counts what WOULD move.
    assert report.transcripts_moved >= 2
    assert report.audio_moved == 1


def test_migrate_skips_existing_destination(tmp_path):
    repo, _ = _legacy_repo(tmp_path)
    xdg = tmp_path / "xdg"
    (xdg / "transcripts").mkdir(parents=True)
    (xdg / "transcripts" / "2026-01-01_Ep.md").write_text("ALREADY THERE")

    report = migrate_data(legacy_repo_root=repo, data_dir=xdg, dry_run=False)

    # Existing destination is not overwritten.
    assert (xdg / "transcripts" / "2026-01-01_Ep.md").read_text() == "ALREADY THERE"
    assert report.skipped >= 1


def test_migrate_idempotent(tmp_path):
    repo, _ = _legacy_repo(tmp_path)
    xdg = tmp_path / "xdg"

    migrate_data(legacy_repo_root=repo, data_dir=xdg, dry_run=False)
    # Second run: sources already gone, nothing to do, must not raise.
    report2 = migrate_data(legacy_repo_root=repo, data_dir=xdg, dry_run=False)
    assert report2.transcripts_moved == 0
    assert report2.audio_moved == 0


def test_migrate_missing_legacy_dirs(tmp_path):
    repo = tmp_path / "empty-repo"
    repo.mkdir()
    xdg = tmp_path / "xdg"
    # No data/ dir at all - must not raise.
    report = migrate_data(legacy_repo_root=repo, data_dir=xdg, dry_run=False)
    assert report.transcripts_moved == 0


# ---------------------------------------------------------------------------
# History merge
# ---------------------------------------------------------------------------


def test_migrate_merges_histories_and_rewrites_paths(tmp_path):
    repo, data = _legacy_repo(tmp_path)
    xdg = tmp_path / "xdg"
    new_transcripts = xdg / "transcripts"

    # Legacy sync history: dict keyed by guid, absolute output_file in repo.
    legacy_history = tmp_path / "history.json"
    legacy_history.write_text(
        json.dumps(
            {
                "guid-1": {
                    "guid": "guid-1",
                    "show_id": 1,
                    "episode_title": "Ep",
                    "processed_at": "2026-01-01T10:00:00",
                    "status": "success",
                    "output_file": str(data / "transcripts" / "2026-01-01_Ep.md"),
                    "one_liner": "a liner",
                }
            }
        )
    )

    # Legacy stats: list, one entry with an in-repo transcript_path, one external.
    legacy_stats = data / "transcripts" / "transcription_stats.json"
    legacy_stats.write_text(
        json.dumps(
            [
                {
                    "success": True,
                    "smart_filename": "2026-01-01_Ep.md",
                    "source_id": "yt:1",
                    "transcript_path": str(data / "transcripts" / "2026-01-01_Ep.md"),
                },
                {
                    "success": True,
                    "smart_filename": "ext.md",
                    "source_id": "yt:2",
                    "transcript_path": "/Users/x/other/interview-recordings/ext.md",
                },
            ]
        )
    )

    migrate_data(
        legacy_repo_root=repo,
        data_dir=xdg,
        legacy_history_file=legacy_history,
        legacy_stats_file=legacy_stats,
        dry_run=False,
    )

    runs = json.loads((xdg / "state" / "runs.json").read_text())
    # Sync entry preserved under by_guid with rewritten output_file.
    assert "guid-1" in runs["by_guid"]
    assert runs["by_guid"]["guid-1"]["output_file"] == str(new_transcripts / "2026-01-01_Ep.md")
    assert runs["by_guid"]["guid-1"]["one_liner"] == "a liner"
    # Stats entries preserved under runs; in-repo path rewritten, external left.
    paths = {r.get("transcript_path") for r in runs["runs"]}
    assert str(new_transcripts / "2026-01-01_Ep.md") in paths
    assert "/Users/x/other/interview-recordings/ext.md" in paths


def test_migrate_history_merge_is_idempotent(tmp_path):
    repo, data = _legacy_repo(tmp_path)
    xdg = tmp_path / "xdg"
    legacy_history = tmp_path / "history.json"
    legacy_history.write_text(
        json.dumps(
            {
                "guid-1": {
                    "guid": "guid-1",
                    "show_id": 1,
                    "episode_title": "Ep",
                    "processed_at": "2026-01-01T10:00:00",
                    "status": "success",
                    "output_file": str(data / "transcripts" / "2026-01-01_Ep.md"),
                }
            }
        )
    )

    migrate_data(
        legacy_repo_root=repo,
        data_dir=xdg,
        legacy_history_file=legacy_history,
        dry_run=False,
    )
    migrate_data(
        legacy_repo_root=repo,
        data_dir=xdg,
        legacy_history_file=legacy_history,
        dry_run=False,
    )

    runs = json.loads((xdg / "state" / "runs.json").read_text())
    # No duplicate guid entries after two runs.
    assert len(runs["by_guid"]) == 1


def test_report_summary_string(tmp_path):
    report = MigrationReport()
    report.transcripts_moved = 3
    report.audio_moved = 1
    s = report.summary()
    assert "3" in s and "transcript" in s.lower()
