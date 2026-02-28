"""Provider comparison eval: run same transcript through multiple providers and judge with Opus."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pidcast.analysis import (
    analyze_transcript_with_llm,
    load_analysis_prompts,
)
from pidcast.config import AnalysisResult, VideoInfo
from pidcast.exceptions import AnalysisError

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated podcast summaries.
You will receive two summaries (A and B) of the same podcast transcript.
Evaluate each on four criteria. Be objective and fair.

CRITICAL FORMATTING RULES:
1. You MUST respond with ONLY valid JSON - no other text before or after
2. Use this EXACT structure:
{
  "scores": {
    "A": {"accuracy": 0, "completeness": 0, "clarity": 0, "conciseness": 0},
    "B": {"accuracy": 0, "completeness": 0, "clarity": 0, "conciseness": 0}
  },
  "verdict": "A or B or tie",
  "reasoning": "2-3 sentence explanation of your verdict"
}
3. Scores are integers 1-10
4. Do NOT include any text outside the JSON structure"""

JUDGE_USER_PROMPT = """Evaluate these two podcast summaries.

Transcript title: {title}

## Summary A
{summary_a}

## Summary B
{summary_b}

Scoring criteria (1-10 each):
- **accuracy**: Does it correctly represent the podcast content?
- **completeness**: Does it cover the key topics and insights?
- **clarity**: Is it clear and easy to understand?
- **conciseness**: Is it appropriately concise without losing important info?

Return JSON only."""


@dataclass
class ProviderScore:
    accuracy: int
    completeness: int
    clarity: int
    conciseness: int

    @property
    def total(self) -> int:
        return self.accuracy + self.completeness + self.clarity + self.conciseness

    @property
    def average(self) -> float:
        return self.total / 4


@dataclass
class ComparisonResult:
    """Result of a provider comparison eval."""

    transcript_id: str
    analysis_type: str
    run_id: str
    timestamp: str

    provider_a: str
    provider_b: str
    result_a: AnalysisResult
    result_b: AnalysisResult

    score_a: ProviderScore | None
    score_b: ProviderScore | None
    verdict: str  # "A", "B", or "tie"
    reasoning: str
    judge_model: str

    duration_total: float
    error: str | None = None

    extra: dict = field(default_factory=dict)


def _call_claude_cli(prompt: str, model: str, timeout: int = 300) -> str:
    """Invoke the claude CLI and return stdout."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise AnalysisError("claude CLI not found. Install Claude Code.")

    proc = subprocess.run(
        [claude_bin, "-p", prompt, "--model", model, "--output-format", "text"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AnalysisError(f"Claude CLI error (code {proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def _run_groq_analysis(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config,
    groq_api_key: str,
    verbose: bool,
) -> AnalysisResult:
    return analyze_transcript_with_llm(
        transcript,
        video_info,
        analysis_type,
        prompts_config,
        groq_api_key,
        model=None,
        verbose=verbose,
    )


def _run_claude_analysis(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config,
    claude_model: str,
    verbose: bool,
) -> AnalysisResult:
    from pidcast.providers.claude_provider import analyze_with_claude_cli

    return analyze_with_claude_cli(
        transcript,
        video_info,
        analysis_type,
        prompts_config,
        model=claude_model,
        verbose=verbose,
    )


def _judge_summaries(
    summary_a: str,
    summary_b: str,
    title: str,
    judge_model: str,
) -> tuple[ProviderScore, ProviderScore, str, str]:
    """Call Claude Opus to judge two summaries. Returns (score_a, score_b, verdict, reasoning)."""
    user_prompt = JUDGE_USER_PROMPT.format(
        title=title,
        summary_a=summary_a,
        summary_b=summary_b,
    )
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

    raw = _call_claude_cli(full_prompt, model=judge_model)

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from fenced block
        import re

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise AnalysisError(f"Judge returned invalid JSON: {raw[:200]}") from None
        data = json.loads(match.group())

    scores = data.get("scores", {})
    a = scores.get("A", {})
    b = scores.get("B", {})

    score_a = ProviderScore(
        accuracy=a.get("accuracy", 0),
        completeness=a.get("completeness", 0),
        clarity=a.get("clarity", 0),
        conciseness=a.get("conciseness", 0),
    )
    score_b = ProviderScore(
        accuracy=b.get("accuracy", 0),
        completeness=b.get("completeness", 0),
        clarity=b.get("clarity", 0),
        conciseness=b.get("conciseness", 0),
    )
    verdict = data.get("verdict", "tie")
    reasoning = data.get("reasoning", "")
    return score_a, score_b, verdict, reasoning


def run_provider_comparison(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    providers: list[str],
    groq_api_key: str | None,
    judge_model: str,
    claude_model: str = "sonnet",
    verbose: bool = False,
) -> ComparisonResult:
    """Run the same transcript through two providers and judge the results.

    Args:
        providers: List of exactly two providers, e.g. ['groq', 'claude']
        groq_api_key: Required if 'groq' is in providers
        judge_model: Claude model alias for the judge (e.g. 'opus')
        claude_model: Claude model alias for analysis (e.g. 'sonnet')
    """
    if len(providers) != 2:
        raise ValueError("Exactly two providers must be specified for comparison")

    from pidcast.config import DEFAULT_PROMPTS_FILE

    prompts_config = load_analysis_prompts(DEFAULT_PROMPTS_FILE, verbose)

    run_id = f"cmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()

    provider_a, provider_b = providers[0], providers[1]

    from pidcast.providers.claude_provider import CLAUDE_MODELS

    resolved_judge = CLAUDE_MODELS.get(judge_model, judge_model)

    start = time.time()

    results: dict[str, AnalysisResult] = {}
    for provider in [provider_a, provider_b]:
        if verbose:
            logger.info(f"Running analysis with provider: {provider}")
        if provider == "groq":
            if not groq_api_key:
                raise AnalysisError("GROQ_API_KEY required for groq provider")
            results[provider] = _run_groq_analysis(
                transcript, video_info, analysis_type, prompts_config, groq_api_key, verbose
            )
        elif provider == "claude":
            results[provider] = _run_claude_analysis(
                transcript, video_info, analysis_type, prompts_config, claude_model, verbose
            )
        else:
            raise AnalysisError(f"Unknown provider: {provider}")

    result_a = results[provider_a]
    result_b = results[provider_b]

    if verbose:
        logger.info(f"Judging with {resolved_judge}...")

    score_a, score_b, verdict, reasoning = _judge_summaries(
        result_a.analysis_text,
        result_b.analysis_text,
        title=video_info.title or "Unknown",
        judge_model=resolved_judge,
    )

    duration = time.time() - start

    return ComparisonResult(
        transcript_id=video_info.webpage_url or "unknown",
        analysis_type=analysis_type,
        run_id=run_id,
        timestamp=timestamp,
        provider_a=provider_a,
        provider_b=provider_b,
        result_a=result_a,
        result_b=result_b,
        score_a=score_a,
        score_b=score_b,
        verdict=verdict,
        reasoning=reasoning,
        judge_model=resolved_judge,
        duration_total=duration,
    )


def save_comparison_report(result: ComparisonResult, output_dir: Path) -> Path:
    """Save comparison result as a markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{result.run_id}.md"

    def score_row(label: str, sa: int, sb: int) -> str:
        winner = " ✓" if sa > sb else (" ✓" if sb > sa else "")
        return f"| {label} | {sa}{winner if sa > sb else ''} | {sb}{winner if sb > sa else ''} |"

    sa = result.score_a
    sb = result.score_b

    lines = [
        f"# Provider Comparison: {result.provider_a} vs {result.provider_b}",
        "",
        f"**Run ID:** `{result.run_id}`  ",
        f"**Date:** {result.timestamp}  ",
        f"**Analysis type:** {result.analysis_type}  ",
        f"**Judge:** {result.judge_model}  ",
        f"**Duration:** {result.duration_total:.1f}s",
        "",
        "## Scores",
        "",
        f"| Criterion | {result.provider_a.upper()} (A) | {result.provider_b.upper()} (B) |",
        "|-----------|------|------|",
    ]

    if sa and sb:
        lines += [
            score_row("Accuracy", sa.accuracy, sb.accuracy),
            score_row("Completeness", sa.completeness, sb.completeness),
            score_row("Clarity", sa.clarity, sb.clarity),
            score_row("Conciseness", sa.conciseness, sb.conciseness),
            f"| **Total** | **{sa.total}** | **{sb.total}** |",
            f"| **Average** | **{sa.average:.1f}** | **{sb.average:.1f}** |",
        ]

    lines += [
        "",
        f"**Verdict:** {result.verdict}  ",
        f"**Reasoning:** {result.reasoning}",
        "",
        "---",
        "",
        f"## Summary A ({result.provider_a})",
        "",
        result.result_a.analysis_text,
        "",
        f"## Summary B ({result.provider_b})",
        "",
        result.result_b.analysis_text,
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
