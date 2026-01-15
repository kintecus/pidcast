"""LLM-based transcript analysis using Groq API."""

import json
import logging
import time
import traceback
from pathlib import Path

import yaml

from .chunking import (
    chunk_transcript,
    estimate_tokens,
    format_chunk_for_analysis,
    format_chunks_for_synthesis,
)
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


def _create_groq_client(api_key: str):
    """Create and return a Groq client."""
    try:
        from groq import Groq

        return Groq(api_key=api_key)
    except ImportError as e:
        raise AnalysisError("'groq' package not installed. Run: pip install groq") from e


def _call_llm_with_fallback(
    client,
    selector: ModelSelector,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    preferred_model: str,
    verbose: bool = False,
    use_json_mode: bool = False,
) -> tuple[str, str, int, int, int, float]:
    """Call LLM API with automatic model fallback on rate limits.

    Args:
        use_json_mode: If True, request JSON response format from the model

    Returns:
        Tuple of (response_text, model_used, tokens_in, tokens_out, tokens_total, duration)
    """

    @with_retry(max_retries=3, base_delay=2.0)
    def _call_api(model_name: str):
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "timeout": ANALYSIS_TIMEOUT,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    current_model = preferred_model
    start_time = time.time()

    while current_model:
        selector.mark_tried(current_model)

        try:
            if verbose:
                logger.info(f"Calling {selector.get_display_name(current_model)}...")

            response = _call_api(current_model)
            duration = time.time() - start_time

            return (
                response.choices[0].message.content,
                current_model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                duration,
            )

        except Exception as e:
            error_msg = str(e)

            # Check for JSON validation error from Groq
            if "json_validate_failed" in error_msg or "Failed to validate JSON" in error_msg:
                # Try next model in fallback chain for JSON issues
                log_warning(f"{selector.get_display_name(current_model)} failed JSON validation")
                next_model = selector.handle_rate_limit(current_model)
                if next_model:
                    logger.info(f"Retrying with {selector.get_display_name(next_model)}...")
                    current_model = next_model
                    continue
                else:
                    tried = selector.get_tried_models()
                    raise AnalysisError(
                        f"JSON validation failed on all models.\n"
                        f"Tried: {', '.join(sorted(tried))}\n\n"
                        f"This can happen with smaller models or rate-limited requests.\n"
                        f"Options:\n"
                        f"  1. Try again (rate limits may have reset)\n"
                        f"  2. Use --skip_analysis_on_error to save transcript without analysis\n"
                        f"  3. Check if your transcript is extremely short (models need enough content)"
                    ) from e
            elif is_rate_limit_error(e):
                next_model = selector.handle_rate_limit(current_model)
                if next_model:
                    current_model = next_model
                    continue
                else:
                    tried = selector.get_tried_models()
                    raise AnalysisError(
                        f"Rate limit exceeded on all fallback models.\n"
                        f"Tried: {', '.join(sorted(tried))}\n\n"
                        f"Options:\n"
                        f"  1. Wait for rate limits to reset\n"
                        f"  2. Use --skip_analysis_on_error to skip analysis"
                    ) from e
            else:
                if verbose:
                    traceback.print_exc()
                raise AnalysisError(f"Error during LLM analysis: {type(e).__name__}: {e}") from e

    raise AnalysisError("No models available for analysis")


def _analyze_single(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    client,
    selector: ModelSelector,
    preferred_model: str,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript in a single API call (for shorter content).

    Note: Truncation is NOT applied here. The calling function has already
    verified this transcript fits within the model's context window. Long
    transcripts are handled by _analyze_with_chunking instead.
    """
    template = prompts_config.prompts[analysis_type]

    # Prepare variables (no truncation - chunking handles long content)
    variables = {
        "transcript": transcript,
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

    # Select best model for this request
    current_model, _ = selector.select_model_for_tokens(estimated_total_tokens, preferred_model)

    # Log estimation
    estimated_cost = selector.estimate_cost(
        current_model, estimated_input_tokens, estimated_output_tokens
    )
    logger.info("\nAnalysis Estimation:")
    logger.info(f"  Type: {template.name}")
    logger.info(f"  Model: {selector.get_display_name(current_model)}")
    logger.info(f"  Estimated input tokens: ~{estimated_input_tokens}")
    if estimated_cost:
        logger.info(f"  Estimated cost: ${estimated_cost:.4f}")

    # Make the API call
    selector.reset()  # Reset for fresh fallback tracking
    response_text, model_used, tokens_in, tokens_out, tokens_total, duration = (
        _call_llm_with_fallback(
            client,
            selector,
            system_prompt,
            user_prompt,
            template.max_output_tokens,
            current_model,
            verbose,
            use_json_mode=True,  # Enable JSON mode for tag generation
        )
    )

    # Log if fallback occurred
    if model_used != preferred_model:
        logger.info(
            f"Analysis completed with fallback model {selector.get_display_name(model_used)} "
            f"(requested: {selector.get_display_name(preferred_model)})"
        )

    actual_cost = selector.estimate_cost(model_used, tokens_in, tokens_out)

    # Parse JSON response to extract analysis and tags
    # TODO: Strengthen JSON parsing to handle malformed responses gracefully:
    #   - Extract analysis from partial JSON (e.g., if tags are malformed but analysis is valid)
    #   - Strip markdown code fences if present (```json ... ```)
    #   - Try to fix common JSON errors (trailing commas, unescaped quotes)
    #   - Consider using a more lenient JSON parser or regex extraction as fallback
    analysis_text = response_text
    contextual_tags = []
    try:
        result_json = json.loads(response_text)
        # Get analysis text
        analysis_text = result_json.get("analysis", response_text)
        # Some LLMs may include literal \n instead of actual newlines - decode them
        analysis_text = analysis_text.replace("\\n", "\n").replace("\\t", "\t")
        contextual_tags = result_json.get("contextual_tags", [])
        if verbose and contextual_tags:
            logger.info(f"Generated tags: {contextual_tags}")
    except json.JSONDecodeError:
        # Fallback: treat entire response as analysis text
        logger.warning("Failed to parse JSON response from LLM, using plain text (no tags)")
        analysis_text = response_text
        contextual_tags = []

    return AnalysisResult(
        analysis_text=analysis_text,
        analysis_type=analysis_type,
        analysis_name=template.name,
        model=model_used,
        provider="groq",
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        tokens_total=tokens_total,
        estimated_cost=actual_cost,
        duration=duration,
        truncated=False,  # No truncation in single analysis; chunking handles long content
        contextual_tags=contextual_tags,
    )


def _analyze_with_chunking(
    transcript: str,
    video_info: VideoInfo,
    prompts_config: PromptsConfig,
    client,
    selector: ModelSelector,
    preferred_model: str,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze long transcript using chunking and synthesis."""
    try:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    except ImportError:
        # Fallback without progress display
        return _analyze_with_chunking_simple(
            transcript, video_info, prompts_config, client, selector, preferred_model, verbose
        )

    # Get prompts
    chunk_template = prompts_config.prompts["chunk_analysis"]
    synthesis_template = prompts_config.prompts["synthesis"]

    # Determine chunk size based on effective limit (min of context and TPM)
    effective_limit = selector.get_effective_token_limit(preferred_model)
    # Leave room for prompt overhead (~1500 tokens) and output (~1500 tokens)
    chunk_target_tokens = max(effective_limit - 3000, 2000)

    # Split transcript into chunks
    chunks = chunk_transcript(transcript, target_tokens=chunk_target_tokens)
    logger.info(f"\nSplitting transcript into {len(chunks)} chunks for analysis...")

    # Analyze each chunk
    chunk_results = []
    total_tokens = 0
    total_cost = 0.0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(complete_style="green"),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Analyzing chunks...", total=len(chunks))

        for chunk in chunks:
            progress.update(task, description=f"Chunk {chunk.index + 1}/{chunk.total_chunks}")

            # Prepare chunk variables
            variables = format_chunk_for_analysis(chunk, video_info.title)
            system_prompt = substitute_prompt_variables(chunk_template.system_prompt, variables)
            user_prompt = substitute_prompt_variables(chunk_template.user_prompt, variables)

            # Make API call
            selector.reset()
            response_text, model_used, tokens_in, tokens_out, tokens_total_chunk, _ = (
                _call_llm_with_fallback(
                    client,
                    selector,
                    system_prompt,
                    user_prompt,
                    chunk_template.max_output_tokens,
                    preferred_model,
                    verbose,
                )
            )

            chunk_results.append(response_text)
            total_tokens += tokens_total_chunk
            chunk_cost = selector.estimate_cost(model_used, tokens_in, tokens_out)
            total_cost += chunk_cost or 0

            progress.advance(task)

        # Synthesis step
        progress.update(task, description="Synthesizing results...")

    logger.info("\nSynthesizing chunk analyses...")

    # Prepare synthesis input
    synthesis_input = format_chunks_for_synthesis(
        chunk_results, video_info.title, video_info.duration_string
    )
    synthesis_variables = {"chunk_analyses": synthesis_input}
    synthesis_system = substitute_prompt_variables(
        synthesis_template.system_prompt, synthesis_variables
    )
    synthesis_user = substitute_prompt_variables(
        synthesis_template.user_prompt, synthesis_variables
    )

    # Make synthesis call
    selector.reset()
    synthesis_text, synthesis_model, syn_in, syn_out, syn_total, _ = _call_llm_with_fallback(
        client,
        selector,
        synthesis_system,
        synthesis_user,
        synthesis_template.max_output_tokens,
        preferred_model,
        verbose,
        use_json_mode=True,  # Enable JSON mode for tag generation
    )

    total_tokens += syn_total
    syn_cost = selector.estimate_cost(synthesis_model, syn_in, syn_out)
    total_cost += syn_cost or 0

    duration = time.time() - start_time

    # Parse JSON response to extract analysis and tags
    final_analysis = synthesis_text
    contextual_tags = []
    try:
        synthesis_json = json.loads(synthesis_text)
        final_analysis = synthesis_json.get("analysis", synthesis_text)
        # Some LLMs may include literal \n instead of actual newlines - decode them
        final_analysis = final_analysis.replace("\\n", "\n").replace("\\t", "\t")
        contextual_tags = synthesis_json.get("contextual_tags", [])
        if verbose and contextual_tags:
            logger.info(f"Generated tags from synthesis: {contextual_tags}")
    except json.JSONDecodeError:
        # Fallback: treat entire response as analysis text
        logger.warning("Failed to parse synthesis JSON response, using plain text (no tags)")
        final_analysis = synthesis_text
        contextual_tags = []

    return AnalysisResult(
        analysis_text=final_analysis,
        analysis_type="executive_summary",
        analysis_name=f"Executive Summary (Chunked: {len(chunks)} parts)",
        model=synthesis_model,
        provider="groq",
        tokens_input=total_tokens,
        tokens_output=0,
        tokens_total=total_tokens,
        estimated_cost=total_cost,
        duration=duration,
        truncated=False,
        contextual_tags=contextual_tags,
    )


def _analyze_with_chunking_simple(
    transcript: str,
    video_info: VideoInfo,
    prompts_config: PromptsConfig,
    client,
    selector: ModelSelector,
    preferred_model: str,
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze long transcript using chunking (without rich progress display)."""
    chunk_template = prompts_config.prompts["chunk_analysis"]
    synthesis_template = prompts_config.prompts["synthesis"]

    effective_limit = selector.get_effective_token_limit(preferred_model)
    chunk_target_tokens = max(effective_limit - 3000, 2000)

    chunks = chunk_transcript(transcript, target_tokens=chunk_target_tokens)
    logger.info(f"\nSplitting transcript into {len(chunks)} chunks for analysis...")

    chunk_results = []
    total_tokens = 0
    total_cost = 0.0
    start_time = time.time()

    for chunk in chunks:
        logger.info(f"Analyzing chunk {chunk.index + 1}/{chunk.total_chunks}...")

        variables = format_chunk_for_analysis(chunk, video_info.title)
        system_prompt = substitute_prompt_variables(chunk_template.system_prompt, variables)
        user_prompt = substitute_prompt_variables(chunk_template.user_prompt, variables)

        selector.reset()
        response_text, model_used, tokens_in, tokens_out, tokens_total_chunk, _ = (
            _call_llm_with_fallback(
                client,
                selector,
                system_prompt,
                user_prompt,
                chunk_template.max_output_tokens,
                preferred_model,
                verbose,
            )
        )

        chunk_results.append(response_text)
        total_tokens += tokens_total_chunk
        chunk_cost = selector.estimate_cost(model_used, tokens_in, tokens_out)
        total_cost += chunk_cost or 0

    logger.info("\nSynthesizing chunk analyses...")

    synthesis_input = format_chunks_for_synthesis(
        chunk_results, video_info.title, video_info.duration_string
    )
    synthesis_variables = {"chunk_analyses": synthesis_input}
    synthesis_system = substitute_prompt_variables(
        synthesis_template.system_prompt, synthesis_variables
    )
    synthesis_user = substitute_prompt_variables(
        synthesis_template.user_prompt, synthesis_variables
    )

    selector.reset()
    synthesis_text, synthesis_model, syn_in, syn_out, syn_total, _ = _call_llm_with_fallback(
        client,
        selector,
        synthesis_system,
        synthesis_user,
        synthesis_template.max_output_tokens,
        preferred_model,
        verbose,
        use_json_mode=True,  # Enable JSON mode for tag generation
    )

    total_tokens += syn_total
    syn_cost = selector.estimate_cost(synthesis_model, syn_in, syn_out)
    total_cost += syn_cost or 0

    duration = time.time() - start_time

    # Parse JSON response to extract analysis and tags
    final_analysis = synthesis_text
    contextual_tags = []
    try:
        synthesis_json = json.loads(synthesis_text)
        final_analysis = synthesis_json.get("analysis", synthesis_text)
        # Some LLMs may include literal \n instead of actual newlines - decode them
        final_analysis = final_analysis.replace("\\n", "\n").replace("\\t", "\t")
        contextual_tags = synthesis_json.get("contextual_tags", [])
        if verbose and contextual_tags:
            logger.info(f"Generated tags from synthesis: {contextual_tags}")
    except json.JSONDecodeError:
        # Fallback: treat entire response as analysis text
        logger.warning("Failed to parse synthesis JSON response, using plain text (no tags)")
        final_analysis = synthesis_text
        contextual_tags = []

    return AnalysisResult(
        analysis_text=final_analysis,
        analysis_type="executive_summary",
        analysis_name=f"Executive Summary (Chunked: {len(chunks)} parts)",
        model=synthesis_model,
        provider="groq",
        tokens_input=total_tokens,
        tokens_output=0,
        tokens_total=total_tokens,
        estimated_cost=total_cost,
        duration=duration,
        truncated=False,
        contextual_tags=contextual_tags,
    )


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

    For long transcripts that exceed model context windows, automatically
    uses chunking with synthesis to produce coherent results.

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

    # Create Groq client
    client = _create_groq_client(api_key)

    # Determine preferred model
    preferred_model = model or selector.get_default_model()

    # Estimate transcript tokens (rough estimate for chunking decision)
    transcript_tokens = estimate_tokens(transcript)
    # Use effective limit (considers both context window AND TPM rate limits)
    effective_limit = selector.get_effective_token_limit(preferred_model)

    # Check if chunking is needed (transcript + prompt overhead > effective limit)
    prompt_overhead = 2000  # Estimated tokens for system + user prompt template
    if transcript_tokens + prompt_overhead > effective_limit:
        # Verify we have chunking prompts
        if "chunk_analysis" not in prompts_config.prompts:
            raise AnalysisError(
                "Transcript too long for single analysis and chunk_analysis prompt not found. "
                "Add chunk_analysis and synthesis prompts to config/prompts.yaml"
            )
        if "synthesis" not in prompts_config.prompts:
            raise AnalysisError(
                "Transcript too long for single analysis and synthesis prompt not found. "
                "Add synthesis prompt to config/prompts.yaml"
            )

        logger.info(
            f"\nTranscript (~{transcript_tokens} tokens) exceeds effective limit "
            f"(~{effective_limit} tokens). Using chunked analysis..."
        )
        return _analyze_with_chunking(
            transcript,
            video_info,
            prompts_config,
            client,
            selector,
            preferred_model,
            verbose,
        )

    # Standard single-call analysis
    return _analyze_single(
        transcript,
        video_info,
        analysis_type,
        prompts_config,
        client,
        selector,
        preferred_model,
        verbose,
    )


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
