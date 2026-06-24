"""Tests for transcription pause/resume checkpointing.

Covers the load-bearing pieces: whisper stdout segment parsing, timestamp
rendering, the de-overlapping merge across a resume seam, and the JobManifest
JSONL/manifest round-trip plus resumable-job discovery.
"""

from __future__ import annotations

import pytest

import pidcast.checkpoint as cp
from pidcast.transcription import (
    merge_segments_to_whisper_json,
    ms_to_timestamp,
    parse_whisper_segment_line,
)

# ---------------------------------------------------------------------------
# Segment line parsing
# ---------------------------------------------------------------------------


def test_parse_segment_line_dot_separator():
    seg = parse_whisper_segment_line("[00:00:05.000 --> 00:00:10.440]   Hello world.")
    assert seg == {"from_ms": 5000, "to_ms": 10440, "text": "Hello world."}


def test_parse_segment_line_comma_separator():
    # Some whisper.cpp builds render ms with a comma.
    seg = parse_whisper_segment_line("[00:01:02,500 --> 00:01:05,000]  hi")
    assert seg == {"from_ms": 62500, "to_ms": 65000, "text": "hi"}


def test_parse_segment_line_hours():
    seg = parse_whisper_segment_line("[01:02:03.004 --> 01:02:04.005]  x")
    assert seg["from_ms"] == ((1 * 60 + 2) * 60 + 3) * 1000 + 4


def test_parse_non_segment_lines_return_none():
    assert parse_whisper_segment_line("whisper_print_timings: load time = 5 ms") is None
    assert parse_whisper_segment_line("") is None
    assert parse_whisper_segment_line("processing 'audio.wav'") is None


def test_parse_malformed_bracket_line_warns_and_returns_none(caplog):
    # A line that looks like a segment but doesn't parse must not be silently dropped.
    with caplog.at_level("WARNING"):
        assert parse_whisper_segment_line("[bad --> line] text") is None
    assert any("format drift" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Timestamp rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ms,expected",
    [
        (0, "00:00:00,000"),
        (5000, "00:00:05,000"),
        (3661500, "01:01:01,500"),
        (-10, "00:00:00,000"),  # clamps negatives
    ],
)
def test_ms_to_timestamp(ms, expected):
    assert ms_to_timestamp(ms) == expected


# ---------------------------------------------------------------------------
# Merge / de-overlap across the resume seam
# ---------------------------------------------------------------------------


def test_merge_produces_offsets_dict_schema():
    # Diarization reads seg["offsets"]["from"]; the schema must carry it.
    merged = merge_segments_to_whisper_json([{"from_ms": 0, "to_ms": 5000, "text": "a"}])
    seg = merged["transcription"][0]
    assert seg["offsets"] == {"from": 0, "to": 5000}
    assert seg["timestamps"] == {"from": "00:00:00,000", "to": "00:00:05,000"}
    assert seg["text"] == "a"


def test_merge_preserves_base_template_metadata():
    merged = merge_segments_to_whisper_json(
        [{"from_ms": 0, "to_ms": 1000, "text": "a"}],
        base_template={"model": {"type": "base"}, "systeminfo": "x"},
    )
    assert merged["model"] == {"type": "base"}
    assert merged["systeminfo"] == "x"


def test_merge_de_overlaps_resume_seam():
    # Simulate a VAD-backed-off resume: the second invocation re-decodes a segment
    # that ends inside the already-persisted prefix. It must be dropped.
    segments = [
        {"from_ms": 0, "to_ms": 5000, "text": "first"},
        {"from_ms": 5000, "to_ms": 10000, "text": "second"},
        # re-decoded overlap (ends at/inside 10000) -> dropped
        {"from_ms": 8000, "to_ms": 10000, "text": "second-redecode"},
        {"from_ms": 10000, "to_ms": 15000, "text": "third"},
    ]
    merged = merge_segments_to_whisper_json(segments)
    texts = [s["text"] for s in merged["transcription"]]
    assert texts == ["first", "second", "third"]


def test_merge_offsets_monotonic_non_decreasing():
    segments = [
        {"from_ms": 10000, "to_ms": 15000, "text": "c"},
        {"from_ms": 0, "to_ms": 5000, "text": "a"},
        {"from_ms": 5000, "to_ms": 10000, "text": "b"},
    ]
    merged = merge_segments_to_whisper_json(segments)
    offs = [s["offsets"]["from"] for s in merged["transcription"]]
    assert offs == sorted(offs)


# ---------------------------------------------------------------------------
# JobManifest round-trip and discovery
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpoint_dir(tmp_path, monkeypatch):
    """Point CHECKPOINT_DIR at a temp dir for the duration of a test."""
    d = tmp_path / "checkpoints"
    monkeypatch.setattr(cp, "CHECKPOINT_DIR", d)
    return d


def _make_manifest(job_id="abc123def456"):
    return cp.JobManifest(
        job_id=job_id,
        input_source="interview.wav",
        audio_signature="sig",
        smart_filename="2026-06-24_Interview",
        provider="whisper",
        model="large-v3",
        language="en",
        vad_signature="novad",
    )


def test_manifest_save_load_round_trip(checkpoint_dir):
    m = _make_manifest()
    m.transcription.status = cp.PAUSED
    m.transcription.segment_count = 3
    m.save()

    loaded = cp.JobManifest.load(m.job_dir)
    assert loaded is not None
    assert loaded.job_id == m.job_id
    assert loaded.transcription.status == cp.PAUSED
    assert loaded.transcription.segment_count == 3
    assert loaded.smart_filename == "2026-06-24_Interview"


def test_segments_append_and_resume_offset(checkpoint_dir):
    m = _make_manifest()
    m.append_segment({"from_ms": 0, "to_ms": 5000, "text": "a"})
    m.append_segment({"from_ms": 5000, "to_ms": 9000, "text": "b"})
    assert m.resume_offset_ms() == 9000
    assert len(m.load_segments()) == 2


def test_load_segments_tolerates_partial_trailing_line(checkpoint_dir):
    m = _make_manifest()
    m.append_segment({"from_ms": 0, "to_ms": 5000, "text": "a"})
    # Simulate a kill mid-write: append a partial JSON line by hand.
    with open(m.segments_path, "a", encoding="utf-8") as f:
        f.write('{"from_ms": 5000, "to_ms": 90')
    segs = m.load_segments()
    assert len(segs) == 1
    assert segs[0]["text"] == "a"
    assert m.resume_offset_ms() == 5000


def test_find_resumable_jobs_lists_paused_and_partial(checkpoint_dir):
    paused = _make_manifest("paused0000000001")
    paused.transcription.status = cp.PAUSED
    paused.save()

    diar_pending = _make_manifest("diarpend00000002")
    diar_pending.transcription.status = cp.DONE
    diar_pending.diarization.status = cp.PENDING
    diar_pending.save()

    done = _make_manifest("alldone000000003")
    done.transcription.status = cp.DONE
    done.diarization.status = cp.DONE
    done.save()

    ids = {j.job_id for j in cp.find_resumable_jobs()}
    assert "paused0000000001" in ids
    assert "diarpend00000002" in ids
    assert "alldone000000003" not in ids  # fully done -> not resumable


def test_find_resumable_jobs_skips_corrupt_manifest(checkpoint_dir):
    good = _make_manifest("good000000000001")
    good.transcription.status = cp.PAUSED
    good.save()

    bad_dir = checkpoint_dir / "corrupt000000000"
    bad_dir.mkdir(parents=True)
    (bad_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")

    ids = {j.job_id for j in cp.find_resumable_jobs()}
    assert ids == {"good000000000001"}


def test_cleanup_job_dir_removes_everything(checkpoint_dir):
    m = _make_manifest()
    m.append_segment({"from_ms": 0, "to_ms": 1000, "text": "a"})
    m.save()
    assert m.job_dir.exists()
    cp.cleanup_job_dir(m.job_id)
    assert not m.job_dir.exists()


def test_compute_job_id_changes_with_config():
    base = cp.compute_job_id("sig", "whisper", "large-v3", "en", "novad")
    assert base == cp.compute_job_id("sig", "whisper", "large-v3", "en", "novad")
    # A different model or VAD signature yields a different job id.
    assert base != cp.compute_job_id("sig", "whisper", "medium", "en", "novad")
    assert base != cp.compute_job_id("sig", "whisper", "large-v3", "en", "vad:silero")


def test_quick_file_signature_small_file_hashes_whole(tmp_path):
    f = tmp_path / "small.wav"
    f.write_bytes(b"hello world" * 10)
    sig1 = cp.quick_file_signature(f)
    # Same content -> same signature; changed content -> different.
    assert sig1 == cp.quick_file_signature(f)
    f.write_bytes(b"different content")
    assert sig1 != cp.quick_file_signature(f)


# ---------------------------------------------------------------------------
# Streaming + pause loop (against a fake whisper binary)
# ---------------------------------------------------------------------------

_FAKE_WHISPER = """#!/bin/bash
of=""
while [ $# -gt 0 ]; do
  case "$1" in
    -of) of="$2"; shift 2;;
    *) shift;;
  esac
done
for i in 0 1 2 3 4 5 6 7 8 9; do
  start=$(printf "00:00:%02d.000" $((i*2)))
  end=$(printf "00:00:%02d.000" $(((i+1)*2)))
  echo "[$start --> $end]   segment $i"
  sleep 0.1
done
echo '{"transcription":[]}' > "${of}.json"
"""


@pytest.fixture
def fake_whisper(tmp_path, monkeypatch):
    import pidcast.transcription as t

    binp = tmp_path / "fake_whisper.sh"
    binp.write_text(_FAKE_WHISPER)
    binp.chmod(0o755)
    monkeypatch.setattr(t, "WHISPER_CPP_PATH", str(binp))
    return t


def test_streaming_invokes_segment_callback(fake_whisper):
    collected = []
    fake_whisper.run_whisper_transcription(
        "dummy.wav",
        "m",
        "json",
        "/tmp/fake_stream_out",
        verbose=False,
        show_progress=False,
        segment_callback=collected.append,
    )
    assert len(collected) == 10
    assert collected[0] == {"from_ms": 0, "to_ms": 2000, "text": "segment 0"}


def test_streaming_pause_raises_with_offset(fake_whisper):
    from pidcast.exceptions import TranscriptionPaused

    collected = []
    pause_flag = {"v": False}

    # Deterministic (no wall-clock race): request the pause from the callback once
    # the 3rd segment lands. The next pause_check poll then trips it.
    def cb(seg):
        collected.append(seg)
        if len(collected) == 3:
            pause_flag["v"] = True

    with pytest.raises(TranscriptionPaused) as exc:
        fake_whisper.run_whisper_transcription(
            "dummy.wav",
            "m",
            "json",
            "/tmp/fake_pause_out",
            verbose=False,
            show_progress=False,
            segment_callback=cb,
            pause_check=lambda: pause_flag["v"],
        )

    # Paused partway: at least the 3 we saw, fewer than all 10, offset matches last.
    assert 3 <= len(collected) < 10
    assert exc.value.last_offset_ms == collected[-1]["to_ms"]
