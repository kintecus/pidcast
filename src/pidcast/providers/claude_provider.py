"""Claude CLI provider for pidcast analysis.

Shells out to the local `claude` CLI (Claude Code) to run analysis.
Requires Claude Code to be installed and authenticated.
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import time

from pidcast.analysis import parse_llm_json_response, substitute_prompt_variables
from pidcast.config import AnalysisResult, PromptsConfig, VideoInfo
from pidcast.exceptions import AnalysisError

logger = logging.getLogger(__name__)

# Claude models available via CLI
CLAUDE_MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}
DEFAULT_CLAUDE_MODEL = "sonnet"


@functools.lru_cache(maxsize=1)
def _find_claude_cli() -> str:
    """Return path to claude CLI, raising AnalysisError if not found."""
    path = shutil.which("claude")
    if not path:
        raise AnalysisError("claude CLI not found. Install Claude Code: https://claude.ai/code")
    return path


def run_claude_subprocess(prompt: str, model: str, timeout: int = 300) -> str:
    """Invoke the claude CLI with a prompt and return stdout.

    Raises:
        AnalysisError: On timeout, OS error, non-zero exit, or empty output.
    """
    claude_bin = _find_claude_cli()
    try:
        proc = subprocess.run(
            [claude_bin, "-p", prompt, "--model", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise AnalysisError(f"Claude CLI timed out after {timeout} seconds") from e
    except OSError as e:
        raise AnalysisError(f"Failed to invoke claude CLI: {e}") from e

    if proc.returncode != 0:
        raise AnalysisError(f"Claude CLI exited with code {proc.returncode}: {proc.stderr.strip()}")
    output = proc.stdout.strip()
    if not output:
        raise AnalysisError("Claude CLI returned empty output")
    return output


def _resolve_claude_model(model: str | None) -> str:
    """Resolve a short model alias or full model name."""
    if not model:
        return CLAUDE_MODELS[DEFAULT_CLAUDE_MODEL]
    return CLAUDE_MODELS.get(model, model)


def analyze_with_claude_cli(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    model: str | None = None,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript using the local Claude CLI.

    Args:
        transcript: Full transcript text
        video_info: Video metadata
        analysis_type: Which prompt template to use
        prompts_config: Loaded prompts configuration
        model: Claude model alias ('sonnet', 'opus', 'haiku') or full model ID
        verbose: Enable verbose output

    Returns:
        AnalysisResult with analysis data

    Raises:
        AnalysisError: If the CLI call fails or returns invalid JSON
    """
    resolved_model = _resolve_claude_model(model)

    if analysis_type not in prompts_config.prompts:
        available = ", ".join(prompts_config.prompts.keys())
        raise AnalysisError(f"Analysis type '{analysis_type}' not found. Available: {available}")

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

    # Build full prompt: system + user combined (claude CLI -p takes a single prompt)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    if verbose:
        logger.info(f"Calling Claude CLI with model {resolved_model}...")

    start_time = time.time()
    raw_output = run_claude_subprocess(full_prompt, resolved_model)
    duration = time.time() - start_time

    if verbose:
        logger.debug(f"Claude CLI raw output ({len(raw_output)} chars)")

    analysis_text, contextual_tags = parse_llm_json_response(raw_output)

    # Rough token estimate (no usage data from CLI)
    estimated_tokens_in = len(full_prompt) // 4
    estimated_tokens_out = len(raw_output) // 4

    return AnalysisResult(
        analysis_text=analysis_text,
        analysis_type=analysis_type,
        analysis_name=prompt_template.name,
        model=resolved_model,
        provider="claude",
        tokens_input=estimated_tokens_in,
        tokens_output=estimated_tokens_out,
        tokens_total=estimated_tokens_in + estimated_tokens_out,
        estimated_cost=None,  # Claude subscription - no per-token cost
        duration=duration,
        truncated=False,
        contextual_tags=contextual_tags,
    )
