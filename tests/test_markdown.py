"""Tests for pidcast.markdown module."""

import pytest

from pidcast.markdown import (
    create_analysis_markdown_file,
    create_markdown_file,
    format_yaml_front_matter,
)

# ============================================================================
# format_yaml_front_matter
# ============================================================================


class TestFormatYamlFrontMatter:
    def test_simple_values(self):
        result = format_yaml_front_matter({"title": "Hello", "count": 42})
        assert result.startswith("---")
        assert result.endswith("---")
        assert "title: Hello" in result
        assert "count: 42" in result

    def test_value_with_colon_quoted(self):
        result = format_yaml_front_matter({"title": "Part 1: The Beginning"})
        assert 'title: "Part 1: The Beginning"' in result

    def test_value_with_hash_quoted(self):
        result = format_yaml_front_matter({"note": "issue #42"})
        assert 'note: "issue #42"' in result

    def test_value_starting_with_star(self):
        result = format_yaml_front_matter({"note": "*bold text*"})
        assert 'note: "*bold text*"' in result

    def test_list_values(self):
        result = format_yaml_front_matter({"tags": ["a", "b", "c"]})
        assert "tags: ['a', 'b', 'c']" in result

    def test_empty_dict(self):
        result = format_yaml_front_matter({})
        assert result == "---\n---"

    def test_boolean_value(self):
        result = format_yaml_front_matter({"active": True})
        assert "active: True" in result


# ============================================================================
# create_markdown_file
# ============================================================================


class TestCreateMarkdownFile:
    def test_creates_file(self, tmp_path, sample_video_info):
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("This is the transcript content.")
        output = tmp_path / "output.md"

        result = create_markdown_file(output, transcript, sample_video_info)
        assert result is True
        assert output.exists()

        content = output.read_text()
        assert content.startswith("---")
        assert "This is the transcript content." in content
        assert "How to Build Great Software" in content

    def test_missing_transcript(self, tmp_path, sample_video_info):
        output = tmp_path / "output.md"
        result = create_markdown_file(output, tmp_path / "missing.txt", sample_video_info)
        assert result is False

    def test_custom_front_matter(self, tmp_path, sample_video_info):
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Content")
        output = tmp_path / "output.md"

        create_markdown_file(
            output,
            transcript,
            sample_video_info,
            front_matter={"custom_key": "custom_value"},
        )
        content = output.read_text()
        assert "custom_key: custom_value" in content


# ============================================================================
# create_analysis_markdown_file
# ============================================================================


class TestCreateAnalysisMarkdownFile:
    @pytest.fixture
    def analysis_results(self):
        return {
            "analysis_type": "executive_summary",
            "analysis_name": "Executive Summary",
            "analysis_text": "# Summary\n\nThis is the analysis.",
            "provider": "groq",
            "model": "test-model",
            "tokens_input": 1000,
            "tokens_output": 500,
            "tokens_total": 1500,
            "estimated_cost": 0.001,
            "duration": 2.5,
            "truncated": False,
            "contextual_tags": ["python", "software"],
        }

    def test_creates_analysis_file(self, tmp_path, sample_video_info, analysis_results):
        source = tmp_path / "2024-01-01_test.md"
        source.touch()

        result = create_analysis_markdown_file(
            analysis_results,
            source,
            sample_video_info,
            tmp_path,
        )
        assert result is not None
        assert result.exists()

        content = result.read_text()
        assert "analysis_type: executive_summary" in content
        assert "# Summary" in content
        assert "python" in content

    def test_filename_includes_analysis_type(self, tmp_path, sample_video_info, analysis_results):
        source = tmp_path / "transcript.md"
        source.touch()

        result = create_analysis_markdown_file(
            analysis_results,
            source,
            sample_video_info,
            tmp_path,
        )
        assert "analysis_executive_summary" in result.name

    def test_contextual_tags_merged(self, tmp_path, sample_video_info, analysis_results):
        source = tmp_path / "test.md"
        source.touch()

        result = create_analysis_markdown_file(
            analysis_results,
            source,
            sample_video_info,
            tmp_path,
        )
        content = result.read_text()
        # Should have both static and contextual tags
        assert "analysis" in content
        assert "python" in content
        assert "software" in content

    def test_missing_contextual_tags(self, tmp_path, sample_video_info, analysis_results):
        analysis_results.pop("contextual_tags")
        source = tmp_path / "test.md"
        source.touch()

        result = create_analysis_markdown_file(
            analysis_results,
            source,
            sample_video_info,
            tmp_path,
        )
        assert result is not None
