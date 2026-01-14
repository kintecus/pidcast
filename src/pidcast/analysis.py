"""LLM-based transcript analysis using Groq API."""

import logging
import time
import traceback
from pathlib import Path

import yaml

from .config import (
    ANALYSIS_TIMEOUT,
    CHAR_TO_TOKEN_RATIO,
    DEFAULT_MODELS_FILE,
    DEFAULT_PROMPTS_FILE,
    MAX_TRANSCRIPT_LENGTH,
    TRANSCRIPT_TRUNCATION_BUFFER,
    TRANSCRIPT_TRUNCATION_MIN_RATIO,
    AnalysisResult,
    PromptsConfig,
    VideoInfo,
)
from .exceptions import AnalysisError
from .model_selector import (
    ModelSelector,
    is_rate_limit_error,
    load_models_config,
    with_retry,
)
from .utils import log_error, log_success, log_warning

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPT MANAGEMENT
# ============================================================================


def load_analysis_prompts(
    prompts_file: str | Path | None = None, verbose: bool = False
) -> PromptsConfig:
    """Load and validate analysis prompt templates from YAML file.

    Args:
        prompts_file: Path to prompts YAML configuration (default: config/prompts.yaml)
        verbose: Enable verbose output

    Returns:
        PromptsConfig containing prompts

    Raises:
        AnalysisError: If prompts file is missing or invalid
    """
    if prompts_file is None:
        prompts_file = DEFAULT_PROMPTS_FILE
    prompts_file = Path(prompts_file)

    # Fail clearly if file doesn't exist (no auto-generation)
    if not prompts_file.exists():
        raise AnalysisError(
            f"Prompts file not found: {prompts_file}\n"
            f"Expected: config/prompts.yaml\n"
            f"Please ensure the prompts.yaml file exists in the config directory."
        )

    # Load YAML file
    try:
        with open(prompts_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise AnalysisError(f"Invalid YAML in prompts file: {e}") from e

    if config is None:
        raise AnalysisError(f"Prompts file is empty: {prompts_file}")

    # Validate structure
    if "prompts" not in config:
        raise AnalysisError("Invalid prompts file: missing 'prompts' key")

    # Validate each prompt has required fields
    required_fields = ["name", "description", "system_prompt", "user_prompt"]
    for prompt_id, prompt_data in config["prompts"].items():
        missing_fields = [field for field in required_fields if field not in prompt_data]
        if missing_fields:
            raise AnalysisError(
                f"Prompt '{prompt_id}' missing required fields: {', '.join(missing_fields)}"
            )

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
# LLM ANALYSIS
# ============================================================================


def analyze_transcript_with_llm(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str | None = None,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript using Groq LLM API with automatic model fallback.

    Uses a quality-prioritized fallback chain when rate limits are hit.
    The chain is defined in config/models.yaml.

    Args:
        transcript: Full transcript text
        video_info: Video metadata
        analysis_type: Which prompt template to use
        prompts_config: Loaded prompts configuration
        api_key: Groq API key
        model: Model name to use (if None, uses default from models.yaml)
        verbose: Enable verbose output

    Returns:
        AnalysisResult with analysis data (model field shows actual model used)

    Raises:
        AnalysisError: If analysis fails (all fallback models exhausted or non-retryable error)
    """
    # Load model configuration
    models_config = load_models_config(DEFAULT_MODELS_FILE)
    selector = ModelSelector(models_config)

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

    # Select best model for this request (pre-check TPM limits)
    preferred_model = model or selector.get_default_model()
    current_model, is_preselected_fallback = selector.select_model_for_tokens(
        estimated_total_tokens, preferred_model
    )

    # Log initial estimation
    estimated_cost = selector.estimate_cost(
        current_model, estimated_input_tokens, estimated_output_tokens
    )
    logger.info("\nAnalysis Estimation:")
    logger.info(f"  Type: {template.name}")
    logger.info(f"  Model: {selector.get_display_name(current_model)}")
    logger.info(f"  Estimated input tokens: ~{estimated_input_tokens}")
    if estimated_cost:
        logger.info(f"  Estimated cost: ${estimated_cost:.4f}")

    # Define the API call function with retry decorator
    @with_retry(max_retries=3, base_delay=2.0)
    def _call_api(client, model_name: str):
        return client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=template.max_output_tokens,
            temperature=0.3,
            timeout=ANALYSIS_TIMEOUT,
        )

    # Initialize Groq client
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
    except ImportError as e:
        raise AnalysisError("'groq' package not installed. Run: pip install groq") from e

    # Fallback loop - try models until success or exhaustion
    original_requested_model = preferred_model
    start_time = time.time()

    while current_model:
        selector.mark_tried(current_model)

        try:
            if verbose:
                logger.info(f"Calling {selector.get_display_name(current_model)}...")

            response = _call_api(client, current_model)
            duration = time.time() - start_time

            # Extract results
            analysis_text = response.choices[0].message.content
            tokens_input = response.usage.prompt_tokens
            tokens_output = response.usage.completion_tokens
            tokens_total = response.usage.total_tokens
            actual_cost = selector.estimate_cost(current_model, tokens_input, tokens_output)

            # Log fallback if it occurred
            if current_model != original_requested_model:
                logger.info(
                    f"Analysis completed with fallback model {selector.get_display_name(current_model)} "
                    f"(requested: {selector.get_display_name(original_requested_model)})"
                )

            return AnalysisResult(
                analysis_text=analysis_text,
                analysis_type=analysis_type,
                analysis_name=template.name,
                model=current_model,
                provider="groq",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                estimated_cost=actual_cost,
                duration=duration,
                truncated=was_truncated,
            )

        except Exception as e:
            # Check if this is a rate limit error that warrants fallback
            if is_rate_limit_error(e):
                next_model = selector.handle_rate_limit(current_model)
                if next_model:
                    current_model = next_model
                    continue
                else:
                    # All models exhausted
                    tried = selector.get_tried_models()
                    raise AnalysisError(
                        f"Rate limit exceeded on all fallback models.\n"
                        f"Tried: {', '.join(sorted(tried))}\n\n"
                        f"Options:\n"
                        f"  1. Wait for rate limits to reset\n"
                        f"  2. Use --skip_analysis_on_error to skip analysis"
                    ) from e
            else:
                # Non-rate-limit error - don't fallback
                if verbose:
                    traceback.print_exc()
                raise AnalysisError(f"Error during LLM analysis: {type(e).__name__}: {e}") from e

    # Should never reach here, but just in case
    raise AnalysisError("No models available for analysis")


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
