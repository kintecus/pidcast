"""Tests for transcription time estimation with provider-aware filtering."""

import json
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from pidcast.transcription import _avg_ratio, estimate_transcription_time

# Fixed reference instant for deterministic recency-weighting tests. Most legacy
# fixtures below carry no run_timestamp, so they get a uniform floor weight and
# the weighted mean reduces to the simple mean (preserving their exact values).
_NOW = datetime(2026, 6, 30, 12, 0, 0)


def _make_stats(
    provider="whisper",
    audio_duration=1000,
    transcription_duration=500,
    diarization_performed=False,
    whisper_model=None,
    success=True,
    age_days=None,
):
    """Create a single stats record for testing.

    If ``age_days`` is given, stamp ``run_timestamp`` that many days before
    ``_NOW`` so recency-weighting tests can control decay. Omit it to leave the
    record untimestamped (uniform floor weight).
    """
    record = {
        "success": success,
        "audio_duration": audio_duration,
        "transcription_duration": transcription_duration,
        "transcription_provider": provider,
        "diarization_performed": diarization_performed,
    }
    if whisper_model is not None:
        record["whisper_model"] = whisper_model
    if age_days is not None:
        record["run_timestamp"] = (_NOW - timedelta(days=age_days)).isoformat()
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
            _make_stats(
                provider="whisper",
                diarization_performed=True,
                audio_duration=100,
                transcription_duration=300,
            ),
            # 3 non-diarized whisper records (ratio = 1.0)
            _make_stats(
                provider="whisper",
                diarization_performed=False,
                audio_duration=100,
                transcription_duration=100,
            ),
            _make_stats(
                provider="whisper",
                diarization_performed=False,
                audio_duration=200,
                transcription_duration=200,
            ),
            _make_stats(
                provider="whisper",
                diarization_performed=False,
                audio_duration=300,
                transcription_duration=300,
            ),
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
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=100,
                transcription_duration=200,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=200,
                transcription_duration=400,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=300,
                transcription_duration=600,
            ),
            # 3 records for base, ratio = 0.5
            _make_stats(
                provider="whisper",
                whisper_model="base",
                audio_duration=100,
                transcription_duration=50,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="base",
                audio_duration=200,
                transcription_duration=100,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="base",
                audio_duration=300,
                transcription_duration=150,
            ),
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(
            stats_file, 1000, provider="whisper", whisper_model="large-v3"
        )
        assert estimate == pytest.approx(2000.0)

    def test_whisper_model_falls_back_when_insufficient(self):
        records = [
            # Only 1 record for large-v3 (< 3 threshold)
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=100,
                transcription_duration=200,
            ),
            # 3 records for base (ratio = 0.5)
            _make_stats(
                provider="whisper",
                whisper_model="base",
                audio_duration=100,
                transcription_duration=50,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="base",
                audio_duration=200,
                transcription_duration=100,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="base",
                audio_duration=300,
                transcription_duration=150,
            ),
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
    """Tests for the _avg_ratio helper (unweighted; left intact by the refactor)."""

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


class TestRecencyWeighting:
    """Recent runs should dominate the estimate over older ones."""

    def test_recent_runs_bias_estimate(self):
        # Old cluster (slow, ratio 4.0) vs recent cluster (fast, ratio 1.0).
        # 30-day half-life: the 120-day-old runs are ~16x down-weighted, so the
        # blended estimate must land far below the unweighted mean of 2.5.
        records = [
            _make_stats(audio_duration=100, transcription_duration=400, age_days=120),
            _make_stats(audio_duration=100, transcription_duration=400, age_days=120),
            _make_stats(audio_duration=100, transcription_duration=400, age_days=120),
            _make_stats(audio_duration=100, transcription_duration=100, age_days=1),
            _make_stats(audio_duration=100, transcription_duration=100, age_days=1),
            _make_stats(audio_duration=100, transcription_duration=100, age_days=1),
        ]
        stats_file = _write_stats(records)

        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", now=_NOW)
        # Heavily weighted toward the recent ratio of 1.0 -> well under 1500
        # (the midpoint), comfortably above the pure-recent 1000.
        assert 1000.0 <= estimate < 1400.0

    def test_untimestamped_runs_reduce_to_simple_mean(self):
        # No run_timestamp anywhere -> uniform floor weight -> simple mean.
        records = [
            _make_stats(audio_duration=100, transcription_duration=100),
            _make_stats(audio_duration=100, transcription_duration=200),
            _make_stats(audio_duration=100, transcription_duration=300),
        ]
        stats_file = _write_stats(records)
        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", now=_NOW)
        # ratios [1,2,3], simple mean 2.0
        assert estimate == pytest.approx(2000.0)

    def test_thin_bucket_elevenlabs_biases_recent(self):
        # Only elevenlabs runs, split old-slow / recent-fast. Weighting must
        # still produce an estimate AND lean recent (exercises decay, not just
        # non-None).
        records = [
            _make_stats(
                provider="elevenlabs",
                audio_duration=100,
                transcription_duration=20,
                age_days=200,
                diarization_performed=True,
            ),
            _make_stats(
                provider="elevenlabs",
                audio_duration=100,
                transcription_duration=20,
                age_days=200,
                diarization_performed=True,
            ),
            _make_stats(
                provider="elevenlabs",
                audio_duration=100,
                transcription_duration=20,
                age_days=200,
                diarization_performed=True,
            ),
            _make_stats(
                provider="elevenlabs",
                audio_duration=100,
                transcription_duration=5,
                age_days=2,
                diarization_performed=True,
            ),
            _make_stats(
                provider="elevenlabs",
                audio_duration=100,
                transcription_duration=5,
                age_days=2,
                diarization_performed=True,
            ),
            _make_stats(
                provider="elevenlabs",
                audio_duration=100,
                transcription_duration=5,
                age_days=2,
                diarization_performed=True,
            ),
        ]
        stats_file = _write_stats(records)
        estimate = estimate_transcription_time(
            stats_file, 1000, provider="elevenlabs", diarize=True, now=_NOW
        )
        # Recent ratio 0.05, old ratio 0.20; weighted blend leans recent (< midpoint 125).
        assert estimate is not None
        assert estimate < 125.0


class TestOutlierRejection:
    """A single anomalous run should not skew a large-enough tier."""

    def test_outlier_dropped_in_large_tier(self):
        # 5 tight runs (ratio 1.0) + 1 wild outlier (ratio 10.0). N>=5 gate fires,
        # outlier is outside [median/3, median*3] = [0.333, 3.0] -> dropped.
        records = [
            _make_stats(audio_duration=100, transcription_duration=100),
            _make_stats(audio_duration=100, transcription_duration=100),
            _make_stats(audio_duration=100, transcription_duration=100),
            _make_stats(audio_duration=100, transcription_duration=100),
            _make_stats(audio_duration=100, transcription_duration=100),
            _make_stats(audio_duration=100, transcription_duration=1000),  # ratio 10.0
        ]
        stats_file = _write_stats(records)
        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", now=_NOW)
        # Outlier dropped -> mean ratio 1.0 -> 1000 (not the polluted ~2500).
        assert estimate == pytest.approx(1000.0)

    def test_small_tier_keeps_outlier(self):
        # Only 4 records -> below the N>=5 gate -> outlier rejection NOT applied,
        # so the spread-out value is retained (matches legacy behavior).
        records = [
            _make_stats(audio_duration=100, transcription_duration=200),  # 2.0
            _make_stats(audio_duration=100, transcription_duration=50),  # 0.5
            _make_stats(audio_duration=100, transcription_duration=50),  # 0.5
            _make_stats(audio_duration=100, transcription_duration=50),  # 0.5
        ]
        stats_file = _write_stats(records)
        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", now=_NOW)
        # All 4 kept: mean [2,0.5,0.5,0.5] = 0.875 -> 875 (no rejection at N=4).
        assert estimate == pytest.approx(875.0)


class TestWeightUnderflow:
    """All-ancient runs must not divide by ~zero."""

    def test_all_ancient_runs_still_estimate(self):
        # 10 years old: 0.5 ** (3650/30) underflows toward 0. Must fall back to
        # an unweighted mean rather than crashing or returning None.
        records = [
            _make_stats(audio_duration=100, transcription_duration=150, age_days=3650),
            _make_stats(audio_duration=100, transcription_duration=150, age_days=3650),
            _make_stats(audio_duration=100, transcription_duration=150, age_days=3650),
        ]
        stats_file = _write_stats(records)
        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", now=_NOW)
        assert estimate == pytest.approx(1500.0)


class TestTimezoneTolerance:
    """A tz-aware now must not crash naive-local timestamp subtraction."""

    def test_tz_aware_now_does_not_crash(self):
        records = [
            _make_stats(audio_duration=100, transcription_duration=100, age_days=1),
            _make_stats(audio_duration=100, transcription_duration=100, age_days=1),
            _make_stats(audio_duration=100, transcription_duration=100, age_days=1),
        ]
        stats_file = _write_stats(records)
        aware_now = _NOW.replace(tzinfo=timezone.utc)
        estimate = estimate_transcription_time(stats_file, 1000, provider="whisper", now=aware_now)
        assert estimate == pytest.approx(1000.0)


class TestUnlabeledModelExclusion:
    """Null-model runs must not pollute a named-model Tier 1."""

    def test_named_model_tier_excludes_null_model(self):
        records = [
            # 3 genuine large-v3 runs (ratio 2.0)
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=100,
                transcription_duration=200,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=100,
                transcription_duration=200,
            ),
            _make_stats(
                provider="whisper",
                whisper_model="large-v3",
                audio_duration=100,
                transcription_duration=200,
            ),
            # 3 unlabeled-model runs (ratio 0.1) that must NOT enter Tier 1
            _make_stats(
                provider="whisper",
                whisper_model=None,
                audio_duration=100,
                transcription_duration=10,
            ),
            _make_stats(
                provider="whisper",
                whisper_model=None,
                audio_duration=100,
                transcription_duration=10,
            ),
            _make_stats(
                provider="whisper",
                whisper_model=None,
                audio_duration=100,
                transcription_duration=10,
            ),
        ]
        stats_file = _write_stats(records)
        estimate = estimate_transcription_time(
            stats_file, 1000, provider="whisper", whisper_model="large-v3", now=_NOW
        )
        # Tier 1 uses only the 3 large-v3 runs -> ratio 2.0 -> 2000.
        assert estimate == pytest.approx(2000.0)
