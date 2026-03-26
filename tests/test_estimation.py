"""Tests for transcription time estimation with provider-aware filtering."""

import json
import tempfile

import pytest

from pidcast.transcription import _avg_ratio, estimate_transcription_time


def _make_stats(
    provider="whisper",
    audio_duration=1000,
    transcription_duration=500,
    diarization_performed=False,
    whisper_model=None,
    success=True,
):
    """Create a single stats record for testing."""
    record = {
        "success": success,
        "audio_duration": audio_duration,
        "transcription_duration": transcription_duration,
        "transcription_provider": provider,
        "diarization_performed": diarization_performed,
    }
    if whisper_model is not None:
        record["whisper_model"] = whisper_model
    return record


def _write_stats(records):
    """Write stats records to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(records, f)
        return f.name


class TestProviderFiltering:
    """Estimation should use records matching the requested provider."""

    def test_whisper_only_uses_whisper_records(self):
        records = [
            # 3 whisper records with ratio 2.0
            _make_stats(provider="whisper", audio_duration=100, transcription_duration=200),
            _make_stats(provider="whisper", audio_duration=200, transcription_duration=400),
            _make_stats(provider="whisper", audio_duration=300, transcription_duration=600),
            # 3 elevenlabs records with ratio 0.05
            _make_stats(provider="elevenlabs", audio_duration=100, transcription_duration=5),
            _make_stats(provider="elevenlabs", audio_duration=200, transcription_duration=10),
            _make_stats(provider="elevenlabs", audio_duration=300, transcription_duration=15),
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper")
        assert estimate == pytest.approx(2000.0)

    def test_elevenlabs_only_uses_elevenlabs_records(self):
        records = [
            _make_stats(provider="whisper", audio_duration=100, transcription_duration=200),
            _make_stats(provider="whisper", audio_duration=200, transcription_duration=400),
            _make_stats(provider="whisper", audio_duration=300, transcription_duration=600),
            _make_stats(provider="elevenlabs", audio_duration=100, transcription_duration=5),
            _make_stats(provider="elevenlabs", audio_duration=200, transcription_duration=10),
            _make_stats(provider="elevenlabs", audio_duration=300, transcription_duration=15),
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(stats_file, 1000, provider="elevenlabs")
        assert estimate == pytest.approx(50.0)


class TestNullProviderAsWhisper:
    """Records with null/missing transcription_provider should count as whisper."""

    def test_null_provider_treated_as_whisper(self):
        records = [
            # Legacy records with no provider field
            _make_stats(provider=None, audio_duration=100, transcription_duration=150),
            _make_stats(provider=None, audio_duration=200, transcription_duration=300),
            _make_stats(provider=None, audio_duration=300, transcription_duration=450),
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper")
        # ratio = 1.5
        assert estimate == pytest.approx(1500.0)

    def test_missing_provider_field_treated_as_whisper(self):
        records = [
            {"success": True, "audio_duration": 100, "transcription_duration": 100},
            {"success": True, "audio_duration": 200, "transcription_duration": 200},
            {"success": True, "audio_duration": 300, "transcription_duration": 300},
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(stats_file, 500, provider="whisper")
        assert estimate == pytest.approx(500.0)


class TestTieredFallback:
    """Should fall back through tiers when specific matches have < 3 records."""

    def test_falls_back_from_diarize_to_provider_only(self):
        records = [
            # Only 1 diarized whisper record (< 3 threshold)
            _make_stats(provider="whisper", diarization_performed=True,
                        audio_duration=100, transcription_duration=300),
            # 3 non-diarized whisper records (ratio = 1.0)
            _make_stats(provider="whisper", diarization_performed=False,
                        audio_duration=100, transcription_duration=100),
            _make_stats(provider="whisper", diarization_performed=False,
                        audio_duration=200, transcription_duration=200),
            _make_stats(provider="whisper", diarization_performed=False,
                        audio_duration=300, transcription_duration=300),
        ]
        stats_file = _write_stats(records)

        # Requesting diarized estimate - tier 2 has < 3 diarized records,
        # falls to tier 3 (provider only, 4 records, mixed ratios)
        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", diarize=True)
        # Tier 3: all 4 whisper records, avg ratio = (3+1+1+1)/4 = 1.5
        assert estimate == pytest.approx(1500.0)

    def test_whisper_model_tier1_used_when_enough_records(self):
        records = [
            # 3 records for large-v3 with diarize=False, ratio = 2.0
            _make_stats(provider="whisper", whisper_model="large-v3",
                        audio_duration=100, transcription_duration=200),
            _make_stats(provider="whisper", whisper_model="large-v3",
                        audio_duration=200, transcription_duration=400),
            _make_stats(provider="whisper", whisper_model="large-v3",
                        audio_duration=300, transcription_duration=600),
            # 3 records for base, ratio = 0.5
            _make_stats(provider="whisper", whisper_model="base",
                        audio_duration=100, transcription_duration=50),
            _make_stats(provider="whisper", whisper_model="base",
                        audio_duration=200, transcription_duration=100),
            _make_stats(provider="whisper", whisper_model="base",
                        audio_duration=300, transcription_duration=150),
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(
            stats_file, 1000, provider="whisper", whisper_model="large-v3"
        )
        assert estimate == pytest.approx(2000.0)

    def test_whisper_model_falls_back_when_insufficient(self):
        records = [
            # Only 1 record for large-v3 (< 3 threshold)
            _make_stats(provider="whisper", whisper_model="large-v3",
                        audio_duration=100, transcription_duration=200),
            # 3 records for base (ratio = 0.5)
            _make_stats(provider="whisper", whisper_model="base",
                        audio_duration=100, transcription_duration=50),
            _make_stats(provider="whisper", whisper_model="base",
                        audio_duration=200, transcription_duration=100),
            _make_stats(provider="whisper", whisper_model="base",
                        audio_duration=300, transcription_duration=150),
        ]
        stats_file = _write_stats(records)

        # Falls to tier 3 (all whisper records)
        estimate = estimate_transcription_time(
            stats_file, 1000, provider="whisper", whisper_model="large-v3"
        )
        # All 4 records: ratios = [2.0, 0.5, 0.5, 0.5], avg = 0.875
        assert estimate == pytest.approx(875.0)


class TestColdStart:
    """Edge cases with no or insufficient data."""

    def test_empty_stats_returns_none(self):
        stats_file = _write_stats([])
        assert estimate_transcription_time(stats_file, 1000) is None

    def test_no_successful_runs_returns_none(self):
        records = [_make_stats(success=False)]
        stats_file = _write_stats(records)
        assert estimate_transcription_time(stats_file, 1000) is None

    def test_cross_provider_cold_start_uses_all_records(self):
        """When requesting elevenlabs but only whisper records exist, use all as fallback."""
        records = [
            _make_stats(provider="whisper", audio_duration=100, transcription_duration=200),
            _make_stats(provider="whisper", audio_duration=200, transcription_duration=400),
            _make_stats(provider="whisper", audio_duration=300, transcription_duration=600),
        ]
        stats_file = _write_stats(records)

        # Elevenlabs has no records, falls through to tier 4 (all records)
        estimate = estimate_transcription_time(stats_file, 1000, provider="elevenlabs")
        assert estimate == pytest.approx(2000.0)


class TestAvgRatio:
    """Tests for the _avg_ratio helper."""

    def test_basic_ratio(self):
        runs = [
            {"audio_duration": 100, "transcription_duration": 50},
            {"audio_duration": 200, "transcription_duration": 100},
        ]
        assert _avg_ratio(runs, 1000) == pytest.approx(500.0)

    def test_skips_zero_audio_duration(self):
        runs = [
            {"audio_duration": 0, "transcription_duration": 50},
            {"audio_duration": 100, "transcription_duration": 50},
        ]
        assert _avg_ratio(runs, 1000) == pytest.approx(500.0)

    def test_all_zero_returns_none(self):
        runs = [{"audio_duration": 0, "transcription_duration": 50}]
        assert _avg_ratio(runs, 1000) is None
