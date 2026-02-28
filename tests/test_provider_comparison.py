"""Tests for pidcast.evals.provider_comparison module."""

import json
from unittest.mock import patch

import pytest

from pidcast.config import AnalysisResult
from pidcast.evals.provider_comparison import (
    ComparisonResult,
    ProviderScore,
    _judge_summaries,
    _score_row,
    save_comparison_report,
)
from pidcast.exceptions import AnalysisError

# ============================================================================
# Helpers
# ============================================================================


def _make_analysis_result(**kwargs) -> AnalysisResult:
    defaults = {
        "analysis_text": "Summary text",
        "analysis_type": "executive_summary",
        "analysis_name": "Executive Summary",
        "model": "llama-3.3-70b-versatile",
        "provider": "groq",
        "tokens_input": 100,
        "tokens_output": 50,
        "tokens_total": 150,
        "estimated_cost": 0.001,
        "duration": 2.5,
        "truncated": False,
        "contextual_tags": ["ai", "tech"],
    }
    defaults.update(kwargs)
    return AnalysisResult(**defaults)


def _make_comparison_result(**kwargs) -> ComparisonResult:
    defaults = {
        "transcript_id": "https://example.com/ep1",
        "analysis_type": "executive_summary",
        "run_id": "cmp_20260228_120000",
        "timestamp": "2026-02-28T12:00:00",
        "provider_a": "groq",
        "provider_b": "claude",
        "result_a": _make_analysis_result(provider="groq", analysis_text="Groq summary"),
        "result_b": _make_analysis_result(provider="claude", analysis_text="Claude summary"),
        "score_a": ProviderScore(accuracy=8, completeness=7, clarity=9, conciseness=8),
        "score_b": ProviderScore(accuracy=9, completeness=8, clarity=8, conciseness=7),
        "verdict": "B",
        "reasoning": "B was more accurate.",
        "judge_model": "claude-opus-4-6",
        "duration_total": 15.0,
    }
    defaults.update(kwargs)
    return ComparisonResult(**defaults)


# ============================================================================
# ProviderScore
# ============================================================================


class TestProviderScore:
    def test_total(self):
        score = ProviderScore(accuracy=8, completeness=7, clarity=9, conciseness=6)
        assert score.total == 30

    def test_average(self):
        score = ProviderScore(accuracy=8, completeness=8, clarity=8, conciseness=8)
        assert score.average == 8.0

    def test_average_with_remainder(self):
        score = ProviderScore(accuracy=7, completeness=8, clarity=9, conciseness=8)
        assert score.average == pytest.approx(8.0)


# ============================================================================
# _score_row
# ============================================================================


class TestScoreRow:
    def test_marks_winner_a(self):
        row = _score_row("Accuracy", 9, 7)
        assert "9 ✓" in row
        assert "7 ✓" not in row

    def test_marks_winner_b(self):
        row = _score_row("Accuracy", 6, 9)
        assert "9 ✓" in row
        assert "6 ✓" not in row

    def test_no_mark_on_tie(self):
        row = _score_row("Accuracy", 8, 8)
        assert "✓" not in row

    def test_includes_label(self):
        row = _score_row("Clarity", 7, 8)
        assert "Clarity" in row

    def test_both_scores_present(self):
        row = _score_row("Completeness", 7, 9)
        assert "7" in row
        assert "9" in row


# ============================================================================
# _judge_summaries
# ============================================================================


class TestJudgeSummaries:
    JUDGE_JSON = json.dumps({
        "scores": {
            "A": {"accuracy": 8, "completeness": 7, "clarity": 9, "conciseness": 8},
            "B": {"accuracy": 9, "completeness": 8, "clarity": 8, "conciseness": 7},
        },
        "verdict": "B",
        "reasoning": "B was more precise.",
    })

    def test_parses_valid_judge_response(self):
        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=self.JUDGE_JSON,
        ):
            score_a, score_b, verdict, reasoning = _judge_summaries(
                "Summary A", "Summary B", title="Episode", judge_model="claude-opus-4-6"
            )

        assert score_a.accuracy == 8
        assert score_b.accuracy == 9
        assert verdict == "B"
        assert "precise" in reasoning

    def test_extracts_json_from_fenced_block(self):
        wrapped = f"```json\n{self.JUDGE_JSON}\n```"
        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=wrapped,
        ):
            score_a, score_b, verdict, _ = _judge_summaries(
                "A", "B", title="Ep", judge_model="claude-opus-4-6"
            )

        assert score_a.accuracy == 8
        assert verdict == "B"

    def test_raises_on_unparseable_output(self):
        with (
            patch(
                "pidcast.providers.claude_provider.run_claude_subprocess",
                return_value="This is not JSON at all",
            ),
            pytest.raises(AnalysisError, match="invalid JSON"),
        ):
            _judge_summaries("A", "B", title="Ep", judge_model="claude-opus-4-6")

    def test_defaults_verdict_to_tie_when_missing(self):
        data = {
            "scores": {
                "A": {"accuracy": 8, "completeness": 8, "clarity": 8, "conciseness": 8},
                "B": {"accuracy": 8, "completeness": 8, "clarity": 8, "conciseness": 8},
            },
            "reasoning": "Too close to call.",
        }
        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=json.dumps(data),
        ):
            _, _, verdict, _ = _judge_summaries("A", "B", title="Ep", judge_model="claude-opus-4-6")

        assert verdict == "tie"


# ============================================================================
# save_comparison_report
# ============================================================================


class TestSaveComparisonReport:
    def test_creates_file_in_output_dir(self, tmp_path):
        result = _make_comparison_result()
        report = save_comparison_report(result, tmp_path)
        assert report.exists()
        assert report.suffix == ".md"

    def test_report_contains_providers(self, tmp_path):
        result = _make_comparison_result()
        report = save_comparison_report(result, tmp_path)
        content = report.read_text()
        assert "GROQ" in content
        assert "CLAUDE" in content

    def test_report_contains_verdict(self, tmp_path):
        result = _make_comparison_result(verdict="A", reasoning="A was better.")
        report = save_comparison_report(result, tmp_path)
        content = report.read_text()
        assert "A was better." in content

    def test_report_contains_scores_when_present(self, tmp_path):
        result = _make_comparison_result()
        report = save_comparison_report(result, tmp_path)
        content = report.read_text()
        assert "Accuracy" in content
        assert "Completeness" in content

    def test_report_contains_both_summaries(self, tmp_path):
        result = _make_comparison_result()
        report = save_comparison_report(result, tmp_path)
        content = report.read_text()
        assert "Groq summary" in content
        assert "Claude summary" in content

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "evals" / "comparisons"
        result = _make_comparison_result()
        save_comparison_report(result, nested)
        assert nested.exists()

    def test_report_filename_matches_run_id(self, tmp_path):
        result = _make_comparison_result(run_id="cmp_test_run_123")
        report = save_comparison_report(result, tmp_path)
        assert "cmp_test_run_123" in report.name

    def test_winner_checkmark_in_score_table(self, tmp_path):
        result = _make_comparison_result(
            score_a=ProviderScore(accuracy=9, completeness=9, clarity=9, conciseness=9),
            score_b=ProviderScore(accuracy=5, completeness=5, clarity=5, conciseness=5),
        )
        report = save_comparison_report(result, tmp_path)
        content = report.read_text()
        assert "✓" in content

    def test_handles_none_scores_gracefully(self, tmp_path):
        result = _make_comparison_result(score_a=None, score_b=None)
        report = save_comparison_report(result, tmp_path)
        content = report.read_text()
        # Should still produce a valid report without score rows
        assert "# Provider Comparison" in content
        assert "Accuracy" not in content
