"""Tests for duplicate detection, source-id matching, and stats reconciliation.

Covers the rework from issues #20-#23:
- compute_source_id normalizes every input type (not just YouTube)
- find_existing_transcription matches by source_id and validates the artifact
- phantom stats (success recorded, file gone) don't block re-transcription
- prune/find phantom helpers
"""

from __future__ import annotations

import json

from pidcast.utils import (
    compute_source_id,
    find_existing_transcription,
    find_phantom_stats,
    prune_phantom_stats,
)

# ---------------------------------------------------------------------------
# compute_source_id
# ---------------------------------------------------------------------------


def test_source_id_youtube_normalizes_formats():
    a = compute_source_id("https://youtu.be/dQw4w9WgXcQ?si=tracking")
    b = compute_source_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=shared")
    assert a == b == "yt:dQw4w9WgXcQ"


def test_source_id_generic_url_strips_query_and_fragment():
    a = compute_source_id("https://feeds.example.com/ep/123.mp3?token=abc")
    b = compute_source_id("https://feeds.example.com/ep/123.mp3?token=different#t=10")
    # Same episode, different tracking params -> same id (podcasts/RSS now dedup).
    assert a == b == "url:https://feeds.example.com/ep/123.mp3"


def test_source_id_local_file(tmp_path):
    f = tmp_path / "rec.wav"
    f.write_bytes(b"x")
    assert compute_source_id(str(f)) == f"file:{f.resolve()}"


def test_source_id_apple_podcasts():
    # Apple URL keyed by collection/track id, not the marketing slug.
    sid = compute_source_id("https://podcasts.apple.com/us/podcast/some-show/id12345?i=67890")
    assert sid.startswith("apple:12345")


# ---------------------------------------------------------------------------
# find_existing_transcription
# ---------------------------------------------------------------------------


def _write_stats(stats_file, entries):
    stats_file.write_text(json.dumps(entries), encoding="utf-8")


def test_finds_duplicate_for_non_youtube_url(tmp_path):
    # Regression for #21: a podcast/RSS URL must be detectable as a duplicate.
    out = tmp_path / "out"
    out.mkdir()
    md = out / "2026-06-24_episode.md"
    md.write_text("transcript", encoding="utf-8")
    stats = tmp_path / "stats.json"
    url = "https://feeds.example.com/ep/123.mp3"
    _write_stats(
        stats,
        [
            {
                "video_title": "Episode 123",
                "smart_filename": "2026-06-24_episode.md",
                "video_url": url,
                "source_id": compute_source_id(url),
                "transcript_path": str(md),
                "success": True,
                "run_timestamp": "2026-06-24T10:00:00",
            }
        ],
    )
    prev = find_existing_transcription(stats, url + "?utm=x", out)
    assert prev is not None
    assert prev.video_title == "Episode 123"


def test_missing_transcript_is_not_a_duplicate(tmp_path):
    # Regression for #20: a phantom entry must NOT block re-transcription.
    out = tmp_path / "out"
    out.mkdir()
    stats = tmp_path / "stats.json"
    url = "https://youtu.be/dQw4w9WgXcQ"
    _write_stats(
        stats,
        [
            {
                "video_title": "Gone",
                "smart_filename": "2026-06-24_gone.md",
                "video_url": url,
                "source_id": compute_source_id(url),
                "transcript_path": str(out / "2026-06-24_gone.md"),  # never created
                "success": True,
                "run_timestamp": "2026-06-24T10:00:00",
            }
        ],
    )
    assert find_existing_transcription(stats, url, out) is None


def test_full_path_survives_different_output_dir(tmp_path):
    # The stored transcript_path lets detection find the file even when the
    # current output_dir differs from where it was originally written.
    real_out = tmp_path / "real"
    real_out.mkdir()
    md = real_out / "2026-06-24_x.md"
    md.write_text("t", encoding="utf-8")
    other_out = tmp_path / "elsewhere"
    other_out.mkdir()
    stats = tmp_path / "stats.json"
    url = "https://youtu.be/abcdef12345"
    _write_stats(
        stats,
        [
            {
                "video_title": "X",
                "smart_filename": "2026-06-24_x.md",
                "video_url": url,
                "source_id": compute_source_id(url),
                "transcript_path": str(md),
                "success": True,
                "run_timestamp": "2026-06-24T10:00:00",
            }
        ],
    )
    # Running from a different output dir still finds the duplicate.
    prev = find_existing_transcription(stats, url, other_out)
    assert prev is not None
    assert prev.transcript_path == md


def test_legacy_entry_without_source_id_still_matches(tmp_path):
    # Older entries (no source_id) fall back to recomputing from video_url.
    out = tmp_path / "out"
    out.mkdir()
    md = out / "old.md"
    md.write_text("t", encoding="utf-8")
    stats = tmp_path / "stats.json"
    url = "https://youtu.be/legacy00000"
    _write_stats(
        stats,
        [
            {
                "video_title": "Old",
                "smart_filename": "old.md",
                "video_url": url,
                "success": True,
                "run_timestamp": "2026-06-24T10:00:00",
            }
        ],
    )
    prev = find_existing_transcription(stats, url, out)
    assert prev is not None


# ---------------------------------------------------------------------------
# phantom stats reconciliation
# ---------------------------------------------------------------------------


def test_find_and_prune_phantom_stats(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    present = out / "present.md"
    present.write_text("t", encoding="utf-8")
    stats = tmp_path / "stats.json"
    _write_stats(
        stats,
        [
            {"smart_filename": "present.md", "transcript_path": str(present), "success": True},
            {"smart_filename": "gone.md", "transcript_path": str(out / "gone.md"), "success": True},
            {"smart_filename": "failed.md", "transcript_path": None, "success": False},
        ],
    )
    phantoms = find_phantom_stats(stats)
    assert len(phantoms) == 1
    assert phantoms[0]["smart_filename"] == "gone.md"

    removed = prune_phantom_stats(stats)
    assert removed == 1
    # Pruning rewrites the file in the unified RunHistory shape; read the
    # surviving runs back through the store rather than as a bare list.
    from pidcast.history import RunHistory

    remaining = RunHistory(stats).get_runs_for_estimation()
    names = {e["smart_filename"] for e in remaining}
    assert names == {"present.md", "failed.md"}


def test_prune_noop_when_no_phantoms(tmp_path):
    stats = tmp_path / "stats.json"
    present = tmp_path / "a.md"
    present.write_text("t", encoding="utf-8")
    _write_stats(
        stats, [{"smart_filename": "a.md", "transcript_path": str(present), "success": True}]
    )
    assert prune_phantom_stats(stats) == 0
