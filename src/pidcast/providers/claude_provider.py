"""Claude CLI provider for pidcast analysis.

Shells out to the local `claude` CLI (Claude Code) to run analysis.
Requires Claude Code to be installed and authenticated.
"""

from __future__ import annotations

import functools
import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from pidcast.analysis import parse_llm_json_response, substitute_prompt_variables
from pidcast.config import DEFAULT_MODELS_FILE, AnalysisResult, PromptsConfig, VideoInfo
from pidcast.exceptions import AnalysisError
from pidcast.model_selector import load_models_config

logger = logging.getLogger(__name__)

# Claude models available via CLI
CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "haiku": "claude-haiku-5-5",
}
DEFAULT_CLAUDE_MODEL = "sonnet"

# The CLI injects the account email, environment, and date even with --system-prompt
# (only --bare strips it, and --bare requires an API key instead of the subscription login).
HARNESS_CONTEXT_GUARD = (
    "Ignore any account, email, environment, or date context attached outside this prompt. "
    "It is not part of the task: never use the user's name or details in your output."
)


@dataclass
class ClaudeRun:
    """Result of one isolated claude CLI call."""

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@functools.lru_cache(maxsize=1)
def _find_claude_cli() -> str:
    """Return path to claude CLI, raising AnalysisError if not found."""
    path = shutil.which("claude")
    if not path:
        raise AnalysisError("claude CLI not found. Install Claude Code: https://claude.ai/code")
    return path


def run_claude(
    prompt: str, model: str, *, system_prompt: str | None = None, timeout: int = 300
) -> ClaudeRun:
    """Invoke the claude CLI in isolation and return text plus usage.

    Runs from a temp dir with the default system prompt replaced and tools, MCP,
    hooks-bearing settings, and session persistence disabled, so the user's
    CLAUDE.md/memory/plugins never reach the model. The prompt goes via stdin.

    Raises:
        AnalysisError: On timeout, OS error, non-zero exit, bad JSON, or empty output.
    """
    claude_bin = _find_claude_cli()
    system = (
        f"{system_prompt}\n\n{HARNESS_CONTEXT_GUARD}" if system_prompt else HARNESS_CONTEXT_GUARD
    )
    try:
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [
                    claude_bin,
                    "-p",
                    "--model",
                    model,
                    "--system-prompt",
                    system,
                    "--tools",
                    "",
                    "--strict-mcp-config",
                    "--no-session-persistence",
                    "--setting-sources",
                    "local",
                    "--output-format",
                    "json",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
    except subprocess.TimeoutExpired as e:
        raise AnalysisError(f"Claude CLI timed out after {timeout} seconds") from e
    except OSError as e:
        raise AnalysisError(f"Failed to invoke claude CLI: {e}") from e

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()[:500]
        raise AnalysisError(f"Claude CLI exited with code {proc.returncode}: {detail}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Claude CLI returned non-JSON output: {proc.stdout[:200]}") from e
    if data.get("is_error"):
        raise AnalysisError(f"Claude CLI reported an error: {data.get('result', '')[:500]}")

    text = (data.get("result") or "").strip()
    if not text:
        raise AnalysisError("Claude CLI returned empty output")
    usage = data.get("usage") or {}
    input_tokens = sum(
        usage.get(k) or 0
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return ClaudeRun(
        text=text,
        model=",".join(data.get("modelUsage") or {}) or model,
        input_tokens=input_tokens or None,
        output_tokens=usage.get("output_tokens"),
        cost_usd=data.get("total_cost_usd"),
    )


def run_claude_subprocess(
    prompt: str, model: str, *, system_prompt: str | None = None, timeout: int = 300
) -> str:
    """Invoke the claude CLI in isolation and return only the response text."""
    return run_claude(prompt, model, system_prompt=system_prompt, timeout=timeout).text


def _resolve_claude_model(model: str | None) -> str:
    """Resolve a short model alias or full model name."""
    if not model:
        return CLAUDE_MODELS[DEFAULT_CLAUDE_MODEL]
    return CLAUDE_MODELS.get(model, model)


class ClaudeProvider:
    """Analysis provider that shells out to the local Claude CLI."""

    def __init__(self, model: str | None = None) -> None:
        self._model = _resolve_claude_model(model)

    def analyze(
        self,
        transcript: str,
        video_info: VideoInfo,
        analysis_type: str,
        prompts_config: PromptsConfig,
        verbose: bool = False,
    ) -> AnalysisResult:
        """Analyze transcript using the local Claude CLI."""
        if analysis_type not in prompts_config.prompts:
            available = ", ".join(prompts_config.prompts.keys())
            raise AnalysisError(
                f"Analysis type '{analysis_type}' not found. Available: {available}"
            )

        prompt_template = prompts_config.prompts[analysis_type]

        duration_str = (
            f"{int(video_info.duration // 60)}:{int(video_info.duration % 60):02d}"
            if video_info.duration
            else "unknown"
        )

        user_prompt = substitute_prompt_variables(
            prompt_template.user_prompt,
            {
                "transcript": transcript,
                "title": video_info.title or "Unknown",
                "channel": video_info.channel or "Unknown",
                "duration": duration_str,
            },
        )
        system_prompt = prompt_template.system_prompt

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        if verbose:
            logger.info(f"Calling Claude CLI with model {self._model}...")

        start_time = time.time()
        raw_output = run_claude_subprocess(user_prompt, self._model, system_prompt=system_prompt)
        duration = time.time() - start_time

        if verbose:
            logger.debug(f"Claude CLI raw output ({len(raw_output)} chars)")

        analysis_text, contextual_tags = parse_llm_json_response(raw_output)

        # Rough token estimate (no usage data from CLI)
        estimated_tokens_in = len(full_prompt) // 4
        estimated_tokens_out = len(raw_output) // 4

        # Estimate cost from models config (best-effort)
        estimated_cost = None
        try:
            models_config = load_models_config(DEFAULT_MODELS_FILE)
            model_cfg = models_config.get_model(self._model)
            if model_cfg:
                estimated_cost = model_cfg.estimate_cost(estimated_tokens_in, estimated_tokens_out)
        except Exception:
            pass  # Cost estimation is best-effort

        return AnalysisResult(
            analysis_text=analysis_text,
            analysis_type=analysis_type,
            analysis_name=prompt_template.name,
            model=self._model,
            provider="claude",
            tokens_input=estimated_tokens_in,
            tokens_output=estimated_tokens_out,
            tokens_total=estimated_tokens_in + estimated_tokens_out,
            estimated_cost=estimated_cost,
            duration=duration,
            truncated=False,
            contextual_tags=contextual_tags,
        )


def analyze_with_claude_cli(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    model: str | None = None,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript using the local Claude CLI.

    Convenience wrapper around ClaudeProvider for backward compatibility.
    """
    return ClaudeProvider(model=model).analyze(
        transcript, video_info, analysis_type, prompts_config, verbose
    )
