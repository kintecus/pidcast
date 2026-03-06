"""Tests for pidcast.providers.claude_provider module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pidcast.config import VideoInfo
from pidcast.exceptions import AnalysisError
from pidcast.providers.claude_provider import (
    CLAUDE_MODELS,
    _find_claude_cli,
    _resolve_claude_model,
    analyze_with_claude_cli,
    run_claude_subprocess,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_prompts_config(analysis_type: str = "executive_summary"):
    """Build a minimal PromptsConfig with one prompt entry."""
    from pidcast.config import PromptsConfig, PromptTemplate

    template = PromptTemplate(
        name="Executive Summary",
        description="Test",
        system_prompt="You are a summarizer. Respond with JSON.",
        user_prompt="Summarize: {transcript}\nTitle: {title}\nChannel: {channel}\nDuration: {duration}",
        max_output_tokens=500,
    )
    return PromptsConfig(prompts={analysis_type: template})


def _make_video_info(**kwargs):
    defaults = {
        "title": "Test Episode",
        "webpage_url": "https://example.com",
        "channel": "Test Show",
        "duration": 3600.0,
    }
    defaults.update(kwargs)
    return VideoInfo(**defaults)


# ============================================================================
# _find_claude_cli
# ============================================================================


class TestFindClaudeCli:
    def test_returns_path_when_found(self):
        # Clear the lru_cache before testing to avoid stale state
        _find_claude_cli.cache_clear()
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            path = _find_claude_cli()
        assert path == "/usr/local/bin/claude"
        _find_claude_cli.cache_clear()

    def test_raises_when_not_found(self):
        _find_claude_cli.cache_clear()
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(AnalysisError, match="claude CLI not found"),
        ):
            _find_claude_cli()
        _find_claude_cli.cache_clear()


# ============================================================================
# _resolve_claude_model
# ============================================================================


class TestResolveClaudeModel:
    def test_resolves_alias(self):
        assert _resolve_claude_model("sonnet") == CLAUDE_MODELS["sonnet"]

    def test_resolves_opus_alias(self):
        assert _resolve_claude_model("opus") == CLAUDE_MODELS["opus"]

    def test_passthrough_full_model_id(self):
        full_id = "claude-custom-model-1"
        assert _resolve_claude_model(full_id) == full_id

    def test_defaults_to_sonnet_when_none(self):
        assert _resolve_claude_model(None) == CLAUDE_MODELS["sonnet"]


# ============================================================================
# run_claude_subprocess
# ============================================================================


class TestRunClaudeSubprocess:
    def _mock_proc(self, returncode=0, stdout="output", stderr=""):
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    def test_returns_stdout_on_success(self):
        with (
            patch("pidcast.providers.claude_provider._find_claude_cli", return_value="/bin/claude"),
            patch("subprocess.run", return_value=self._mock_proc(stdout="  hello  ")),
        ):
            result = run_claude_subprocess("prompt", "claude-sonnet-4-6")
        assert result == "hello"

    def test_raises_on_nonzero_exit(self):
        with (
            patch("pidcast.providers.claude_provider._find_claude_cli", return_value="/bin/claude"),
            patch(
                "subprocess.run", return_value=self._mock_proc(returncode=1, stderr="auth error")
            ),
            pytest.raises(AnalysisError, match="exited with code 1"),
        ):
            run_claude_subprocess("prompt", "claude-sonnet-4-6")

    def test_raises_on_empty_output(self):
        with (
            patch("pidcast.providers.claude_provider._find_claude_cli", return_value="/bin/claude"),
            patch("subprocess.run", return_value=self._mock_proc(stdout="")),
            pytest.raises(AnalysisError, match="empty output"),
        ):
            run_claude_subprocess("prompt", "claude-sonnet-4-6")

    def test_raises_on_timeout(self):
        with (
            patch("pidcast.providers.claude_provider._find_claude_cli", return_value="/bin/claude"),
            patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)
            ),
            pytest.raises(AnalysisError, match="timed out"),
        ):
            run_claude_subprocess("prompt", "claude-sonnet-4-6")

    def test_raises_on_os_error(self):
        with (
            patch("pidcast.providers.claude_provider._find_claude_cli", return_value="/bin/claude"),
            patch("subprocess.run", side_effect=OSError("no such file")),
            pytest.raises(AnalysisError, match="Failed to invoke"),
        ):
            run_claude_subprocess("prompt", "claude-sonnet-4-6")


# ============================================================================
# analyze_with_claude_cli
# ============================================================================


class TestAnalyzeWithClaudeCli:
    VALID_JSON = '{"analysis": "Great episode.", "contextual_tags": ["ai", "tech", "open-source"]}'

    def test_raises_on_unknown_analysis_type(self):
        prompts = _make_prompts_config("executive_summary")
        with pytest.raises(AnalysisError, match="not found"):
            analyze_with_claude_cli("transcript", _make_video_info(), "nonexistent", prompts)

    def test_returns_analysis_result_on_success(self):
        prompts = _make_prompts_config()
        video = _make_video_info()

        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=self.VALID_JSON,
            create=True,
        ):
            result = analyze_with_claude_cli("some transcript", video, "executive_summary", prompts)

        assert result.analysis_text == "Great episode."
        assert result.contextual_tags == ["ai", "tech", "open-source"]
        assert result.provider == "claude"
        assert result.analysis_type == "executive_summary"
        assert result.estimated_cost is not None

    def test_formats_duration_correctly(self):
        prompts = _make_prompts_config()
        video = _make_video_info(duration=3661.0)  # 1h 1m 1s -> "61:01"

        captured_prompt = {}

        def capture(prompt, model, **kwargs):
            captured_prompt["value"] = prompt
            return self.VALID_JSON

        with patch("pidcast.providers.claude_provider.run_claude_subprocess", side_effect=capture):
            analyze_with_claude_cli("transcript", video, "executive_summary", prompts)

        assert "61:01" in captured_prompt["value"]

    def test_uses_unknown_duration_when_zero(self):
        prompts = _make_prompts_config()
        video = _make_video_info(duration=0)

        captured_prompt = {}

        def capture(prompt, model, **kwargs):
            captured_prompt["value"] = prompt
            return self.VALID_JSON

        with patch("pidcast.providers.claude_provider.run_claude_subprocess", side_effect=capture):
            analyze_with_claude_cli("transcript", video, "executive_summary", prompts)

        assert "unknown" in captured_prompt["value"]

    def test_token_estimates_are_nonzero(self):
        prompts = _make_prompts_config()
        video = _make_video_info()

        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=self.VALID_JSON,
        ):
            result = analyze_with_claude_cli("a" * 1000, video, "executive_summary", prompts)

        assert result.tokens_input > 0
        assert result.tokens_output > 0
        assert result.tokens_total == result.tokens_input + result.tokens_output


# ============================================================================
# analyze_with_claude_cli - cost estimation
# ============================================================================


class TestAnalyzeWithClaudeCliCost:
    VALID_JSON = '{"analysis": "Great episode.", "contextual_tags": ["ai"]}'

    def test_estimated_cost_is_computed_when_model_known(self):
        """Cost should be non-None for known Claude models."""
        prompts = _make_prompts_config()
        video = _make_video_info()

        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=self.VALID_JSON,
        ):
            result = analyze_with_claude_cli(
                "a" * 4000, video, "executive_summary", prompts, model="sonnet"
            )

        assert result.estimated_cost is not None
        assert result.estimated_cost > 0

    def test_estimated_cost_is_none_for_unknown_model(self):
        """Cost should be None for models not in config."""
        prompts = _make_prompts_config()
        video = _make_video_info()

        with patch(
            "pidcast.providers.claude_provider.run_claude_subprocess",
            return_value=self.VALID_JSON,
        ):
            result = analyze_with_claude_cli(
                "a" * 4000, video, "executive_summary", prompts, model="claude-custom-999"
            )

        assert result.estimated_cost is None
