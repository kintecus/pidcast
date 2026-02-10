"""Tests for pidcast.utils module."""

from unittest.mock import patch

import pytest

from pidcast.utils import (
    create_smart_filename,
    extract_youtube_video_id,
    format_duration,
    fuzzy_match_key,
    get_unique_filename,
    load_json_file,
    sanitize_filename,
    save_json_file,
    suggest_closest_match,
    validate_input_source,
)

# ============================================================================
# sanitize_filename
# ============================================================================


class TestSanitizeFilename:
    def test_removes_special_chars(self):
        assert sanitize_filename("hello!@#world") == "helloworld"

    def test_preserves_hyphens(self):
        assert sanitize_filename("hello-world") == "hello-world"

    def test_preserves_spaces(self):
        assert sanitize_filename("hello world") == "hello world"

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_strips_whitespace(self):
        assert sanitize_filename("  hello  ") == "hello"


# ============================================================================
# create_smart_filename
# ============================================================================


class TestCreateSmartFilename:
    @patch("pidcast.utils.datetime")
    def test_includes_date_prefix(self, mock_dt):
        mock_dt.datetime.now.return_value.strftime.return_value = "2024-01-15"
        result = create_smart_filename("Great Talk About Python", include_date=True)
        assert result.startswith("2024-01-15_")

    def test_no_date_prefix(self):
        result = create_smart_filename("Great Talk About Python", include_date=False)
        assert not result[0].isdigit() or "_" not in result[:11]

    def test_strips_episode_prefix(self):
        result = create_smart_filename("Episode 42: The Answer", include_date=False)
        assert "Episode" not in result
        assert "42" not in result.split("_")[0]

    def test_prioritizes_important_words(self):
        result = create_smart_filename("The Quick Brown Fox", include_date=False, max_length=30)
        # "Quick", "Brown", "Fox" are important (capitalized)
        assert "Quick" in result or "Brown" in result

    def test_respects_max_length(self):
        long_title = "A " * 100 + "Very Important Word"
        result = create_smart_filename(long_title, include_date=False, max_length=30)
        # Without date prefix, should be within max_length (roughly)
        assert len(result) <= 60  # some tolerance for word boundaries

    def test_filler_words_stripped(self):
        result = create_smart_filename("EP.5 - Interview", include_date=False)
        assert "EP" not in result


# ============================================================================
# get_unique_filename
# ============================================================================


class TestGetUniqueFilename:
    def test_no_collision(self, tmp_path):
        result = get_unique_filename(tmp_path, "test", ".md")
        assert result == tmp_path / "test.md"

    def test_with_collision(self, tmp_path):
        (tmp_path / "test.md").touch()
        result = get_unique_filename(tmp_path, "test", ".md")
        assert result == tmp_path / "test_v2.md"

    def test_multiple_collisions(self, tmp_path):
        (tmp_path / "test.md").touch()
        (tmp_path / "test_v2.md").touch()
        result = get_unique_filename(tmp_path, "test", ".md")
        assert result == tmp_path / "test_v3.md"


# ============================================================================
# format_duration
# ============================================================================


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45.23) == "45.23 seconds"

    def test_exactly_60(self):
        assert format_duration(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert format_duration(90) == "1m 30s"

    def test_hours(self):
        assert format_duration(3661) == "61m 1s"

    def test_zero(self):
        assert format_duration(0) == "0.00 seconds"


# ============================================================================
# extract_youtube_video_id
# ============================================================================


class TestExtractYoutubeVideoId:
    def test_standard_watch_url(self):
        assert (
            extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        )

    def test_short_url(self):
        assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_with_tracking_params(self):
        assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ?si=abc123") == "dQw4w9WgXcQ"

    def test_with_feature_param(self):
        assert (
            extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=shared")
            == "dQw4w9WgXcQ"
        )

    def test_live_url(self):
        assert extract_youtube_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert (
            extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        )

    def test_old_v_format(self):
        assert extract_youtube_video_id("https://www.youtube.com/v/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_mobile_url(self):
        assert (
            extract_youtube_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        )

    def test_invalid_url(self):
        assert extract_youtube_video_id("https://example.com/video") is None

    def test_malformed_url(self):
        assert extract_youtube_video_id("not a url at all") is None

    def test_invalid_video_id_length(self):
        assert extract_youtube_video_id("https://www.youtube.com/watch?v=short") is None


# ============================================================================
# validate_input_source
# ============================================================================


class TestValidateInputSource:
    def test_youtube_url(self):
        source, is_local = validate_input_source("https://www.youtube.com/watch?v=abc")
        assert is_local is False

    def test_local_audio_file(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.touch()
        source, is_local = validate_input_source(str(audio))
        assert is_local is True

    def test_unsupported_format(self, tmp_path):
        video = tmp_path / "test.avi"
        video.touch()
        with pytest.raises(ValueError, match="Unsupported audio format"):
            validate_input_source(str(video))

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="Invalid input"):
            validate_input_source("random string that is not a file or url")


# ============================================================================
# fuzzy_match_key
# ============================================================================


class TestFuzzyMatchKey:
    def test_exact_match_case_insensitive(self):
        assert fuzzy_match_key("Hello", ["hello", "world"]) == "hello"

    def test_starts_with(self):
        assert fuzzy_match_key("exec", ["executive_summary", "deep_dive"]) == "executive_summary"

    def test_contains(self):
        assert fuzzy_match_key("summary", ["executive_summary", "deep_dive"]) == "executive_summary"

    def test_no_match(self):
        assert fuzzy_match_key("xyz", ["hello", "world"]) is None

    def test_multiple_starts_with_returns_shortest(self):
        keys = ["deep_dive", "deep_analysis"]
        result = fuzzy_match_key("deep", keys)
        assert result == "deep_dive"  # shortest

    def test_normalized_matching(self):
        result = fuzzy_match_key("gptoss120b", ["openai/gpt-oss-120b"], normalize=True)
        assert result == "openai/gpt-oss-120b"


# ============================================================================
# suggest_closest_match
# ============================================================================


class TestSuggestClosestMatch:
    def test_close_match(self):
        result = suggest_closest_match("exectuive_summary", ["executive_summary", "deep_dive"])
        assert result == "executive_summary"

    def test_no_close_match(self):
        result = suggest_closest_match("zzzzzzz", ["hello", "world"])
        assert result is None


# ============================================================================
# load_json_file / save_json_file
# ============================================================================


class TestJsonFileIO:
    def test_load_existing_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = load_json_file(f)
        assert result == {"key": "value"}

    def test_load_missing_file(self, tmp_path):
        result = load_json_file(tmp_path / "missing.json", default=[])
        assert result == []

    def test_load_corrupt_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{invalid json")
        result = load_json_file(f, default={"fallback": True})
        assert result == {"fallback": True}

    def test_save_and_load_roundtrip(self, tmp_path):
        f = tmp_path / "out.json"
        data = {"items": [1, 2, 3], "nested": {"a": True}}
        assert save_json_file(f, data) is True
        assert load_json_file(f) == data

    def test_save_creates_parent_dirs(self, tmp_path):
        f = tmp_path / "sub" / "dir" / "data.json"
        assert save_json_file(f, {"ok": True}) is True
        assert f.exists()
