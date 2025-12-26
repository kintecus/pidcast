"""LLM-based transcript analysis using Groq API."""

import datetime
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .config import (
    ANALYSIS_TIMEOUT,
    CHAR_TO_TOKEN_RATIO,
    DEFAULT_ANALYSIS_PROMPTS_FILE,
    DEFAULT_GROQ_MODEL,
    DEFAULT_STATS_FILE,
    GROQ_PRICING,
    GROQ_RATE_LIMITS,
    MAX_TRANSCRIPT_LENGTH,
    TOKEN_PRICING_DENOMINATOR,
    TRANSCRIPT_TRUNCATION_BUFFER,
    TRANSCRIPT_TRUNCATION_MIN_RATIO,
    AnalysisResult,
    PromptsConfig,
    VideoInfo,
    get_fallback_chain,
)
from .exceptions import AnalysisError
from .utils import load_json_file, log_error, log_success, log_warning, save_json_file

logger = logging.getLogger(__name__)


# ============================================================================
# RATE LIMIT VIOLATION DATA
# ============================================================================


@dataclass
class RateLimitViolation:
    """Structured rate limit violation information."""

    is_violated: bool
    violation_types: list[str]  # ["TPM", "RPM", "RPD", "TPD"]
    has_tpm_violation: bool
    error_message: str | None = None
    current_usage: dict[str, int] | None = None  # rpm, tpm, rpd, tpd usage


# ============================================================================
# PROMPT MANAGEMENT
# ============================================================================


def create_default_prompts_file(prompts_file: Path, verbose: bool = False) -> bool:
    """Create a default analysis prompts configuration file.

    Args:
        prompts_file: Path where to create the file
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    default_prompts = {
        "prompts": {
            "summary": {
                "name": "Summary",
                "description": "Generate a concise summary of the transcript",
                "system_prompt": "You are an expert at summarizing video transcripts. Create clear, concise summaries that capture the main points and key insights.",
                "user_prompt": "Please create a comprehensive summary of the following transcript.\n\nTitle: {title}\nChannel: {channel}\nDuration: {duration}\n\n# Transcript\n\n{transcript}\n\n# Instructions\n\nProvide a summary with:\n1. A one-paragraph overview (2-3 sentences)\n2. Key points (3-7 bullet points)\n3. Main takeaways or conclusions\n\nBe specific and use concrete details from the transcript.",
                "max_output_tokens": 2000,
            },
            "key_points": {
                "name": "Key Points Extraction",
                "description": "Extract main ideas and important points",
                "system_prompt": "You are an expert at analyzing content and extracting key insights. Focus on identifying the most important ideas, facts, and actionable takeaways.",
                "user_prompt": "Extract the key points from this transcript.\n\nTitle: {title}\nChannel: {channel}\n\n# Transcript\n\n{transcript}\n\n# Instructions\n\nProvide:\n1. Main Topics: List 3-5 main topics discussed\n2. Key Points: 8-12 bullet points of the most important ideas\n3. Notable Quotes: 2-3 memorable or impactful quotes\n4. Actionable Insights: Any practical advice or takeaways",
                "max_output_tokens": 2500,
            },
            "action_items": {
                "name": "Action Items & Recommendations",
                "description": "Extract actionable recommendations and next steps",
                "system_prompt": "You are an expert at identifying actionable advice and practical recommendations from content. Focus on what viewers can do with this information.",
                "user_prompt": "Identify action items and recommendations from this transcript.\n\nTitle: {title}\n\n# Transcript\n\n{transcript}\n\n# Instructions\n\nProvide:\n1. Direct Action Items: Specific steps or actions mentioned\n2. Recommended Resources: Tools, books, links, or resources referenced\n3. Best Practices: Any guidelines or principles discussed\n4. Implementation Tips: How to apply these insights",
                "max_output_tokens": 2000,
            },
        }
    }

    if save_json_file(prompts_file, default_prompts, verbose=verbose):
        if verbose:
            logger.info("  Available analysis types: summary, key_points, action_items")
        return True
    return False


def load_analysis_prompts(prompts_file: str | Path, verbose: bool = False) -> PromptsConfig | None:
    """Load and validate analysis prompt templates from JSON file.

    Args:
        prompts_file: Path to prompts JSON configuration
        verbose: Enable verbose output

    Returns:
        PromptsConfig containing prompts, or None on error
    """
    prompts_file = Path(prompts_file)

    # If file doesn't exist and it's the default path, create it
    if not prompts_file.exists():
        if prompts_file == DEFAULT_ANALYSIS_PROMPTS_FILE:
            if verbose:
                logger.info(f"Prompts file not found at {prompts_file}")
                logger.info("Creating default prompts file...")
            if not create_default_prompts_file(prompts_file, verbose):
                return None
        else:
            log_error(f"Prompts file not found: {prompts_file}")
            return None

    # Load the file
    config = load_json_file(prompts_file)
    if config is None:
        return None

    # Validate structure
    if "prompts" not in config:
        log_error("Invalid prompts file: missing 'prompts' key")
        return None

    # Validate each prompt has required fields
    required_fields = ["name", "description", "system_prompt", "user_prompt"]
    for prompt_id, prompt_data in config["prompts"].items():
        missing_fields = [field for field in required_fields if field not in prompt_data]
        if missing_fields:
            log_error(f"Prompt '{prompt_id}' missing required fields: {', '.join(missing_fields)}")
            return None

    if verbose:
        log_success(f"Loaded {len(config['prompts'])} analysis prompt(s) from {prompts_file}")

    return PromptsConfig.from_dict(config)


def substitute_prompt_variables(prompt_template: str, variables: dict[str, str]) -> str:
    """Replace variables in prompt template with actual values.

    Args:
        prompt_template: Template string with {variable} placeholders
        variables: Dict mapping variable names to values

    Returns:
        Prompt with variables substituted
    """
    result = prompt_template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


# ============================================================================
# TRANSCRIPT PROCESSING
# ============================================================================


def truncate_transcript(
    transcript: str,
    max_length: int = MAX_TRANSCRIPT_LENGTH,
    verbose: bool = False,
) -> tuple[str, bool]:
    """Truncate transcript if it exceeds maximum length.

    Args:
        transcript: Full transcript text
        max_length: Maximum character length
        verbose: Enable verbose output

    Returns:
        Tuple of (truncated_transcript, was_truncated)
    """
    if len(transcript) <= max_length:
        return transcript, False

    # Try to find last sentence boundary before max_length
    truncate_point = max_length - TRANSCRIPT_TRUNCATION_BUFFER
    last_period = transcript.rfind(". ", 0, truncate_point)

    if last_period > max_length * TRANSCRIPT_TRUNCATION_MIN_RATIO:
        truncate_point = last_period + 1

    truncated = transcript[:truncate_point] + "\n\n[... transcript truncated for length ...]"

    if verbose:
        log_warning(f"Transcript truncated from {len(transcript)} to {len(truncated)} characters")

    return truncated, True


# ============================================================================
# COST ESTIMATION
# ============================================================================


def estimate_analysis_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
    """Estimate cost for Groq API call based on token counts.

    Args:
        prompt_tokens: Estimated input tokens
        completion_tokens: Estimated output tokens
        model: Model name

    Returns:
        Estimated cost in USD, or None if pricing unknown
    """
    if model not in GROQ_PRICING:
        return None

    pricing = GROQ_PRICING[model]
    input_cost = (prompt_tokens / TOKEN_PRICING_DENOMINATOR) * pricing["input"]
    output_cost = (completion_tokens / TOKEN_PRICING_DENOMINATOR) * pricing["output"]
    return input_cost + output_cost


def validate_rate_limits(
    estimated_tokens: int,
    model: str,
    stats_file_path: Path | str,
    verbose: bool = False,
) -> RateLimitViolation:
    """Validate if request would exceed Groq API rate limits.

    Checks against free tier limits for TPM, TPD, RPM, RPD.

    Args:
        estimated_tokens: Estimated total tokens for this request
        model: Model name to validate against
        stats_file_path: Path to transcription_stats.json file
        verbose: Enable verbose output

    Returns:
        RateLimitViolation with structured violation data
    """
    # Validate model is in rate limits config
    if model not in GROQ_RATE_LIMITS:
        if verbose:
            logger.debug(f"Model '{model}' not in GROQ_RATE_LIMITS, skipping validation")
        return RateLimitViolation(is_violated=False, violation_types=[], has_tpm_violation=False)

    limits = GROQ_RATE_LIMITS[model]

    # Load historical statistics
    stats_file = Path(stats_file_path)
    if not stats_file.exists():
        if verbose:
            logger.debug("Stats file doesn't exist yet, allowing first request")
        return RateLimitViolation(is_violated=False, violation_types=[], has_tpm_violation=False)

    try:
        stats_data = load_json_file(stats_file, default=[])
    except Exception as e:
        if verbose:
            logger.warning(f"Could not load stats file for rate limit checking: {e}")
        return RateLimitViolation(is_violated=False, violation_types=[], has_tpm_violation=False)

    if not isinstance(stats_data, list) or len(stats_data) == 0:
        return RateLimitViolation(is_violated=False, violation_types=[], has_tpm_violation=False)

    # Current time for window calculations
    now = datetime.datetime.now(datetime.timezone.utc)
    one_minute_ago = now - datetime.timedelta(minutes=1)
    one_day_ago = now - datetime.timedelta(days=1)

    # Filter stats to only include analysis runs
    analysis_runs = [
        stat
        for stat in stats_data
        if isinstance(stat, dict) and stat.get("analysis_performed", False)
    ]

    # Calculate current usage in different time windows
    rpm_usage = 0
    rpd_usage = 0
    tpm_usage = 0
    tpd_usage = 0

    for stat in analysis_runs:
        try:
            run_timestamp_str = stat.get("run_timestamp")
            if not run_timestamp_str:
                continue

            # Parse ISO format timestamp (handle both with/without timezone)
            if "+" in run_timestamp_str or run_timestamp_str.endswith("Z"):
                run_time = datetime.datetime.fromisoformat(run_timestamp_str.replace("Z", "+00:00"))
            else:
                run_time = datetime.datetime.fromisoformat(run_timestamp_str).replace(
                    tzinfo=datetime.timezone.utc
                )

            tokens = stat.get("analysis_tokens", 0) or 0

            # Check minute window
            if run_time >= one_minute_ago:
                rpm_usage += 1
                tpm_usage += tokens

            # Check day window
            if run_time >= one_day_ago:
                rpd_usage += 1
                tpd_usage += tokens
        except (ValueError, KeyError, TypeError) as e:
            if verbose:
                logger.debug(f"Error parsing stat entry: {e}")
            continue

    # Check each limit and track violation types
    violation_types = []
    violation_messages = []

    if rpm_usage + 1 > limits["rpm"]:
        violation_types.append("RPM")
        violation_messages.append(
            f"RPM limit exceeded: {rpm_usage} requests in last minute + 1 new request > {limits['rpm']} limit"
        )

    if tpm_usage + estimated_tokens > limits["tpm"]:
        violation_types.append("TPM")
        violation_messages.append(
            f"TPM limit exceeded: {tpm_usage} tokens in last minute + {estimated_tokens} estimated tokens > {limits['tpm']} limit"
        )

    if rpd_usage + 1 > limits["rpd"]:
        violation_types.append("RPD")
        violation_messages.append(
            f"RPD limit exceeded: {rpd_usage} requests in last 24h + 1 new request > {limits['rpd']} limit"
        )

    # Only check TPD if limit is set (not 0)
    if limits["tpd"] > 0 and tpd_usage + estimated_tokens > limits["tpd"]:
        violation_types.append("TPD")
        violation_messages.append(
            f"TPD limit exceeded: {tpd_usage} tokens in last 24h + {estimated_tokens} estimated tokens > {limits['tpd']} limit"
        )

    # If any violations, build error message and return structured data
    if violation_types:
        error_msg = _build_rate_limit_error_message(
            violation_messages,
            model,
            rpm_usage,
            tpm_usage,
            rpd_usage,
            tpd_usage,
            estimated_tokens,
            limits,
        )
        return RateLimitViolation(
            is_violated=True,
            violation_types=violation_types,
            has_tpm_violation="TPM" in violation_types,
            error_message=error_msg,
            current_usage={"rpm": rpm_usage, "tpm": tpm_usage, "rpd": rpd_usage, "tpd": tpd_usage},
        )

    return RateLimitViolation(
        is_violated=False,
        violation_types=[],
        has_tpm_violation=False,
    )


def _build_rate_limit_error_message(
    violations: list[str],
    model: str,
    rpm_usage: int,
    tpm_usage: int,
    rpd_usage: int,
    tpd_usage: int,
    estimated_tokens: int,
    limits: dict[str, int],
) -> str:
    """Build a detailed, user-friendly rate limit error message."""
    msg = "Groq API rate limit would be exceeded:\n\n"

    for violation in violations:
        msg += f"  - {violation}\n"

    msg += f"\nCurrent usage for model '{model}':\n"
    msg += f"  Last minute:   {rpm_usage} requests, {tpm_usage} tokens (limits: RPM={limits['rpm']}, TPM={limits['tpm']})\n"
    msg += f"  Last 24 hours: {rpd_usage} requests, {tpd_usage} tokens (limits: RPD={limits['rpd']}, TPD={limits['tpd']})\n"
    msg += f"\nThis request would add: 1 request, {estimated_tokens} tokens\n\n"
    msg += "Options:\n"
    msg += "  1. Wait for rate limits to reset (1-24 hours depending on which limit)\n"
    msg += "  2. Use --skip_analysis_on_error to skip analysis for this run\n"
    msg += "  3. Upgrade to a paid Groq plan for higher limits\n"

    return msg


def _get_next_fallback_model(
    tried_models: set[str],
    original_model: str | None = None,
) -> str | None:
    """Get next model to try in fallback chain.

    Args:
        tried_models: Set of models already attempted
        original_model: Original model user requested

    Returns:
        Next model to try, or None if all exhausted
    """
    fallback_chain = get_fallback_chain(original_model)
    for model in fallback_chain:
        if model not in tried_models:
            return model
    return None


# ============================================================================
# LLM ANALYSIS
# ============================================================================


def analyze_transcript_with_llm(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str = DEFAULT_GROQ_MODEL,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript using Groq LLM API with automatic TPM fallback.

    Automatically falls back to models with higher TPM capacity if the requested
    model would exceed its tokens-per-minute limit. Fallback chain is ordered by
    TPM capacity (highest first): groq/compound, mixtral, etc.

    Args:
        transcript: Full transcript text
        video_info: Video metadata
        analysis_type: Which prompt template to use
        prompts_config: Loaded prompts configuration
        api_key: Groq API key
        model: Model name to use (preferred, may fallback)
        verbose: Enable verbose output

    Returns:
        AnalysisResult with analysis data (model field shows actual model used)

    Raises:
        AnalysisError: If analysis fails (all fallback models exhausted or non-TPM error)
    """
    # Validate analysis_type exists
    if analysis_type not in prompts_config.prompts:
        available_types = ", ".join(prompts_config.prompts.keys())
        raise AnalysisError(
            f"Analysis type '{analysis_type}' not found. Available types: {available_types}"
        )

    # Get template
    template = prompts_config.prompts[analysis_type]

    # Prepare variables
    transcript_text, was_truncated = truncate_transcript(transcript, verbose=verbose)
    variables = {
        "transcript": transcript_text,
        "title": video_info.title,
        "channel": video_info.channel,
        "duration": video_info.duration_string,
        "url": video_info.webpage_url,
    }

    # Substitute variables
    system_prompt = substitute_prompt_variables(template.system_prompt, variables)
    user_prompt = substitute_prompt_variables(template.user_prompt, variables)

    # Estimate tokens
    estimated_input_tokens = (len(system_prompt) + len(user_prompt)) // CHAR_TO_TOKEN_RATIO
    estimated_output_tokens = template.max_output_tokens
    estimated_total_tokens = estimated_input_tokens + estimated_output_tokens

    # Track fallback attempts
    tried_models: set[str] = set()
    current_model = model
    original_requested_model = model

    # Log initial estimation
    estimated_cost = estimate_analysis_cost(
        estimated_input_tokens, estimated_output_tokens, current_model
    )
    if verbose or estimated_cost:
        logger.info("\nAnalysis Estimation:")
        logger.info(f"  Type: {template.name}")
        logger.info(f"  Model: {current_model}")
        logger.info(f"  Estimated input tokens: ~{estimated_input_tokens}")
        if estimated_cost:
            logger.info(f"  Estimated cost: ${estimated_cost:.4f}")

    # FALLBACK RETRY LOOP
    while True:
        tried_models.add(current_model)

        # Validate rate limits
        if verbose:
            logger.info(f"Checking Groq API rate limits for {current_model}...")

        violation = validate_rate_limits(
            estimated_total_tokens,
            current_model,
            DEFAULT_STATS_FILE,
            verbose,
        )

        if violation.is_violated:
            # Check if TPM-only violation (fallback eligible)
            if violation.has_tpm_violation and len(violation.violation_types) == 1:
                # TPM-only violation - try fallback
                logger.warning(
                    f"TPM limit would be exceeded with {current_model} "
                    f"({violation.current_usage['tpm']} + {estimated_total_tokens} > "
                    f"{GROQ_RATE_LIMITS[current_model]['tpm']})"
                )

                # Get next fallback model
                next_model = _get_next_fallback_model(tried_models, original_requested_model)
                if next_model and next_model != current_model:
                    logger.info(f"Falling back to {next_model}...")
                    current_model = next_model
                    continue  # Retry with fallback model
                else:
                    # No more fallback models available
                    all_models_tpm = ", ".join(
                        f"{m} ({GROQ_RATE_LIMITS[m]['tpm']} TPM)" for m in tried_models
                    )
                    raise AnalysisError(
                        f"TPM limit exceeded and all fallback models exhausted:\n"
                        f"  Attempted: {all_models_tpm}\n"
                        f"  Request tokens: {estimated_total_tokens}\n\n"
                        f"  Options:\n"
                        f"    1. Wait for rate limits to reset (1-24 hours)\n"
                        f"    2. Use --skip_analysis_on_error to skip analysis\n"
                        f"    3. Upgrade to a paid Groq plan for higher limits"
                    )
            else:
                # Non-TPM violation or multiple violations (hard failure)
                raise AnalysisError(violation.error_message)

        # Rate limits OK, proceed with API call
        if verbose:
            logger.info(f"✓ Rate limits OK for {current_model}")

        break  # Exit retry loop

    # Call API with selected model
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        start_time = time.time()

        response = client.chat.completions.create(
            model=current_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=template.max_output_tokens,
            temperature=0.3,
            timeout=ANALYSIS_TIMEOUT,
        )

        duration = time.time() - start_time

        # Extract results
        analysis_text = response.choices[0].message.content
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens
        tokens_total = response.usage.total_tokens
        actual_cost = estimate_analysis_cost(tokens_input, tokens_output, current_model)

        # Log fallback if it occurred
        if current_model != original_requested_model:
            logger.info(
                f"Analysis completed with fallback model {current_model} "
                f"(requested: {original_requested_model})"
            )

        return AnalysisResult(
            analysis_text=analysis_text,
            analysis_type=analysis_type,
            analysis_name=template.name,
            model=current_model,  # Track actual model used
            provider="groq",
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_total,
            estimated_cost=actual_cost,
            duration=duration,
            truncated=was_truncated,
        )

    except ImportError as e:
        raise AnalysisError("'groq' package not installed. Run: pip install groq") from e
    except Exception as e:
        if verbose:
            traceback.print_exc()
        raise AnalysisError(f"Error during LLM analysis: {type(e).__name__}: {e}") from e


# ============================================================================
# TERMINAL RENDERING
# ============================================================================


def render_analysis_to_terminal(analysis_file: Path, verbose: bool = False) -> bool:
    """Render analysis markdown file to terminal with Rich formatting.

    Args:
        analysis_file: Path to analysis markdown file
        verbose: Enable verbose output (shows extended metadata)

    Returns:
        True if successful, False on error
    """
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        if verbose:
            log_warning("'rich' package not installed. Install with: uv add rich")
        return False

    # Read analysis file
    try:
        with open(analysis_file, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        if verbose:
            log_error(f"Error reading analysis file: {e}")
        return False

    # Split YAML front matter from markdown content
    parts = content.split("---", 2)
    if len(parts) < 3:
        if verbose:
            log_error("Invalid analysis file format (missing YAML front matter)")
        return False

    yaml_content = parts[1].strip()
    markdown_content = parts[2].strip()

    # Parse YAML front matter (simple key-value extraction)
    metadata = {}
    for line in yaml_content.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")

    # Create Rich console
    console = Console()

    # Build metadata table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan bold", width=20)
    table.add_column(style="white")

    # Add key metadata rows
    table.add_row("Title", metadata.get("title", "Unknown"))
    table.add_row(
        "Analysis Type",
        metadata.get("analysis_name", metadata.get("analysis_type", "Unknown")),
    )
    table.add_row("Model", metadata.get("analysis_model", "Unknown"))
    table.add_row("Tokens", metadata.get("tokens_total", "Unknown"))

    # Show cost if available
    cost = metadata.get("estimated_cost")
    if cost and cost != "None":
        try:
            table.add_row("Cost", f"${float(cost):.4f}")
        except (ValueError, TypeError):
            table.add_row("Cost", cost)

    # Show duration
    duration = metadata.get("analysis_duration")
    if duration:
        try:
            table.add_row("Duration", f"{float(duration):.2f}s")
        except (ValueError, TypeError):
            table.add_row("Duration", duration)

    # Add verbose metadata if requested
    if verbose:
        table.add_row("", "")  # Blank row separator
        table.add_row("Source", metadata.get("source_transcript", "Unknown"))
        table.add_row("Channel", metadata.get("channel", "Unknown"))
        table.add_row("Video Duration", metadata.get("duration", "Unknown"))
        table.add_row(
            "Tokens (In/Out)",
            f"{metadata.get('tokens_input', '?')}/{metadata.get('tokens_output', '?')}",
        )
        truncated = metadata.get("transcript_truncated", "False")
        table.add_row("Truncated", "Yes" if truncated == "True" else "No")

    # Render metadata panel
    console.print(
        Panel(table, title="[bold cyan]Analysis Metadata[/bold cyan]", border_style="cyan")
    )
    console.print()  # Blank line

    # Render markdown content
    md = Markdown(markdown_content)
    console.print(md)

    return True
