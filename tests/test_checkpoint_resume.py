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


def test_elapsed_seconds_defaults_to_zero(checkpoint_dir):
    m = _make_manifest()
    assert m.transcription.elapsed_seconds == 0.0


def test_add_transcription_elapsed_accumulates_across_legs(checkpoint_dir):
    m = _make_manifest()
    # Three resume legs: 12s + 8s + 5s = 25s of true cumulative compute.
    assert m.add_transcription_elapsed(12.0) == pytest.approx(12.0)
    assert m.add_transcription_elapsed(8.0) == pytest.approx(20.0)
    total = m.add_transcription_elapsed(5.0)
    assert total == pytest.approx(25.0)
    assert m.transcription.elapsed_seconds == pytest.approx(25.0)


def test_add_transcription_elapsed_persists(checkpoint_dir):
    m = _make_manifest()
    m.add_transcription_elapsed(7.5)
    # Reload from disk: the accumulator survived the save().
    loaded = cp.JobManifest.load(m.job_dir)
    assert loaded is not None
    assert loaded.transcription.elapsed_seconds == pytest.approx(7.5)


def test_add_transcription_elapsed_clamps_negative(checkpoint_dir):
    # A clock skew shouldn't subtract from the accumulated total.
    m = _make_manifest()
    m.add_transcription_elapsed(10.0)
    assert m.add_transcription_elapsed(-3.0) == pytest.approx(10.0)


def test_elapsed_seconds_survives_legacy_manifest_without_field(checkpoint_dir):
    # A pre-upgrade manifest JSON has no elapsed_seconds; it must load with the
    # 0.0 default rather than erroring.
    import json

    m = _make_manifest()
    m.save()
    raw = json.loads((m.job_dir / cp.MANIFEST_NAME).read_text())
    raw["transcription"].pop("elapsed_seconds", None)
    (m.job_dir / cp.MANIFEST_NAME).write_text(json.dumps(raw))

    loaded = cp.JobManifest.load(m.job_dir)
    assert loaded is not None
    assert loaded.transcription.elapsed_seconds == 0.0


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
# Resume arg/metadata reconstruction (regression: don't lose the original
# title / output dir / source url when resuming from a different directory)
# ---------------------------------------------------------------------------


def test_reconstruct_args_restores_metadata_and_output_dir(checkpoint_dir, monkeypatch):
    from pidcast.config import VideoInfo
    from pidcast.resume import _reconstruct_args

    m = _make_manifest()
    m.video_info = VideoInfo(
        title="Sabre Interview",
        webpage_url="file:///recordings/sabre.wav",
        channel="",
        uploader="",
        duration=600,
        duration_string="10:00",
        view_count=0,
        upload_date="",
        description="",
    ).to_dict()
    # Original run wrote to a specific dir; resume must restore it, not use cwd.
    m.cli_args = {"output_dir": "/some/original/out", "whisper_model": "base.en", "force": False}
    m.save()

    args = _reconstruct_args(m)

    # Input is redirected to the checkpoint's source.wav and duplicate detection is off.
    assert args.input_source == str(m.source_wav_path)
    assert args.force is True
    assert args.resume_job_id == m.job_id
    # The original output dir survives (so resume writes to the intended location).
    assert args.output_dir == "/some/original/out"
    # The persisted model name is carried through for re-resolution.
    assert args.whisper_model == "base.en"


def test_manifest_video_info_obj_round_trips():
    from pidcast.config import VideoInfo

    m = _make_manifest()
    vi = VideoInfo(
        title="Sabre Interview",
        webpage_url="file:///recordings/sabre.wav",
        channel="ACME",
        uploader="ACME",
        duration=600,
        duration_string="10:00",
        view_count=0,
        upload_date="20260624",
        description="desc",
    )
    m.video_info = vi.to_dict()
    restored = m.video_info_obj()
    assert restored.title == "Sabre Interview"
    assert restored.webpage_url == "file:///recordings/sabre.wav"
    assert restored.channel == "ACME"


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


def test_checkpointed_transcribe_reports_cumulative_duration(
    fake_whisper, checkpoint_dir, tmp_path
):
    """transcribe() on the checkpointed path returns total elapsed across legs.

    A resumed run must report leg1 + leg2 + ..., not just the final leg, so the
    estimate-vs-actual line and the persisted ETA stat reflect the true
    end-to-end transcription time rather than a falsely-fast partial.
    """
    from pidcast.providers.whisper_provider import WhisperTranscriptionProvider

    manifest = _make_manifest(job_id="cumuljob00001")
    # Simulate a prior resume leg that already burned 42s of compute.
    manifest.add_transcription_elapsed(42.0)

    provider = WhisperTranscriptionProvider(
        whisper_model="m",
        output_format="json",
        output_dir=tmp_path,
        checkpoint=manifest,
    )
    result = provider.transcribe("dummy.wav", verbose=False)

    # Reported duration includes the prior 42s plus this (small) leg's compute,
    # i.e. the cumulative total - not just the final leg.
    assert result.duration >= 42.0
    # And the manifest's accumulator advanced past the seeded 42s.
    reloaded = cp.JobManifest.load(manifest.job_dir)
    assert reloaded is not None
    assert reloaded.transcription.elapsed_seconds >= 42.0
    assert reloaded.transcription.elapsed_seconds == pytest.approx(result.duration, abs=0.01)
