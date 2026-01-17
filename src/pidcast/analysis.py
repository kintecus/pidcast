"""LLM-based transcript analysis using Groq API."""

import json
import logging
import re
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
# CUSTOM EXCEPTIONS
# ============================================================================


class PlainTextFallbackSuccess(Exception):
    """Special exception to signal successful plain text fallback with extracted tags.

    This is used to propagate the results from _analyze_with_plain_text_fallback
    through _call_llm_with_fallback to the caller, since the return signatures differ.
    """

    def __init__(self, analysis_text: str, tags: list[str], model: str,
                 tokens_in: int, tokens_out: int, tokens_total: int, duration: float):
        self.analysis_text = analysis_text
        self.tags = tags
        self.model = model
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.tokens_total = tokens_total
        self.duration = duration
        super().__init__("Plain text fallback succeeded")


# ============================================================================
# JSON PARSING UTILITIES
# ============================================================================


def parse_llm_json_response(
    response_text: str, verbose: bool = False
) -> tuple[str, list[str]]:
    """Parse JSON response from LLM with robust error handling.

    Attempts multiple strategies to extract analysis and tags:
    1. Direct JSON parsing
    2. Strip markdown code fences (```json...```)
    3. Fix common JSON errors (trailing commas)
    4. Extract analysis field via regex if JSON fails

    Args:
        response_text: Raw response from LLM
        verbose: Enable verbose logging

    Returns:
        Tuple of (analysis_text, contextual_tags)
        Falls back to (response_text, []) if all parsing fails
    """
    analysis_text = response_text
    contextual_tags = []

    # Strategy 1: Try direct JSON parsing
    try:
        result_json = json.loads(response_text)
        analysis_text = result_json.get("analysis", response_text)
        # Decode literal \n and \t that some LLMs include
        if isinstance(analysis_text, str):
            analysis_text = analysis_text.replace("\\n", "\n").replace("\\t", "\t")

        # Validate contextual_tags is a list of strings
        tags = result_json.get("contextual_tags", [])
        if isinstance(tags, list):
            contextual_tags = [str(t) for t in tags if isinstance(t, (str, int, float))]
        else:
            if verbose:
                logger.warning(f"contextual_tags is not a list (got {type(tags).__name__}), ignoring")

        if verbose and contextual_tags:
            logger.info(f"Generated tags: {contextual_tags}")

        return analysis_text, contextual_tags

    except json.JSONDecodeError:
        pass  # Try other strategies

    # Strategy 2: Strip markdown code fences
    stripped = re.sub(r'^```(?:json)?\s*', '', response_text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r'\s*```$', '', stripped.strip(), flags=re.MULTILINE)

    if stripped != response_text:
        try:
            result_json = json.loads(stripped)
            analysis_text = result_json.get("analysis", response_text)
            if isinstance(analysis_text, str):
                analysis_text = analysis_text.replace("\\n", "\n").replace("\\t", "\t")

            tags = result_json.get("contextual_tags", [])
            if isinstance(tags, list):
                contextual_tags = [str(t) for t in tags if isinstance(t, (str, int, float))]

            if verbose:
                logger.info("Successfully parsed JSON after stripping markdown fences")
                if contextual_tags:
                    logger.info(f"Generated tags: {contextual_tags}")

            return analysis_text, contextual_tags

        except json.JSONDecodeError:
            pass  # Try next strategy

    # Strategy 3: Fix common JSON errors (trailing commas)
    fixed = re.sub(r',\s*([}\]])', r'\1', stripped)

    if fixed != stripped:
        try:
            result_json = json.loads(fixed)
            analysis_text = result_json.get("analysis", response_text)
            if isinstance(analysis_text, str):
                analysis_text = analysis_text.replace("\\n", "\n").replace("\\t", "\t")

            tags = result_json.get("contextual_tags", [])
            if isinstance(tags, list):
                contextual_tags = [str(t) for t in tags if isinstance(t, (str, int, float))]

            if verbose:
                logger.info("Successfully parsed JSON after fixing trailing commas")
                if contextual_tags:
                    logger.info(f"Generated tags: {contextual_tags}")

            return analysis_text, contextual_tags

        except json.JSONDecodeError:
            pass  # Try final fallback

    # Strategy 4: Regex extraction fallback - try to extract analysis field
    analysis_match = re.search(
        r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)"',
        response_text,
        re.DOTALL
    )

    if analysis_match:
        analysis_text = analysis_match.group(1)
        # Unescape JSON string escapes
        analysis_text = analysis_text.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        logger.warning("Failed to parse full JSON, extracted analysis field via regex (no tags)")
        return analysis_text, []

    # Final fallback: return original text with no tags
    logger.warning("Failed to parse JSON response from LLM, using plain text (no tags)")
    return response_text, []


# JSON validation error patterns from Groq and other providers
JSON_VALIDATION_ERROR_PATTERNS = [
    "json_validate_failed",
    "Failed to validate JSON",
    "Invalid JSON",
    "JSON parsing error",
    "Expected JSON",
    "json.decoder.JSONDecodeError",
    "json_schema_error",
]


def is_json_validation_error(error: Exception) -> bool:
    """Check if error is a JSON validation error from the model."""
    error_str = str(error).lower()
    return any(pattern.lower() in error_str for pattern in JSON_VALIDATION_ERROR_PATTERNS)


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


def strip_json_formatting_rules(prompt: str) -> str:
    """Remove JSON formatting instructions from a prompt for plain text mode.

    Removes sections like "CRITICAL FORMATTING RULES" or "CRITICAL:" that
    instruct the model to return JSON. Adds plain markdown instruction instead.

    Args:
        prompt: Original prompt with JSON formatting rules

    Returns:
        Modified prompt for plain text responses
    """
    lines = prompt.split("\n")
    filtered_lines = []
    skip_section = False

    for line in lines:
        # Detect start of JSON formatting section
        if any(marker in line for marker in ["CRITICAL FORMATTING RULES", "CRITICAL:", "JSON FORMAT"]):
            skip_section = True
            continue

        # Skip lines that are part of the formatting rules list
        if skip_section:
            stripped = line.strip()
            # Continue skipping if line is a list item or part of JSON example
            if (stripped.startswith(("1.", "2.", "3.", "4.", "5.", "-", "*")) or
                stripped.startswith(("{", "}", '"')) or
                "json" in stripped.lower() or
                "contextual_tags" in stripped):
                continue
            # Empty line might signal end of section
            if stripped == "":
                skip_section = False
                continue
            # Non-empty, non-list line - section ended
            skip_section = False

        if not skip_section:
            filtered_lines.append(line)

    result = "\n".join(filtered_lines)
    # Add instruction for plain text format
    result += "\n\nProvide your analysis in clear, well-structured markdown format."
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
            # Check for JSON validation error from Groq or other providers
            if is_json_validation_error(e):
                from .utils import log_error_to_file

                # Log the JSON validation error
                log_error_to_file(
                    "json_validation_error",
                    {
                        "model": current_model,
                        "error_message": str(e),
                        "system_prompt_preview": system_prompt[:200],
                        "user_prompt_preview": user_prompt[:200],
                        "use_json_mode": use_json_mode,
                    }
                )

                # Try next model in fallback chain for JSON issues
                log_warning(f"{selector.get_display_name(current_model)} failed JSON validation")
                # Mark this model as tried and get next fallback
                selector.mark_tried(current_model)
                next_model = selector.get_next_fallback()
                if next_model:
                    logger.info(f"Retrying with {selector.get_display_name(next_model)}...")
                    current_model = next_model
                    continue
                else:
                    # All models exhausted with JSON validation errors
                    # If we were using JSON mode, try plain text fallback
                    if use_json_mode:
                        from .utils import log_error_to_file

                        tried = selector.get_tried_models()
                        log_warning(
                            f"JSON mode failed on all models. "
                            f"Falling back to plain text analysis with separate tag extraction..."
                        )

                        # Log the fallback event
                        log_error_to_file(
                            "json_mode_exhausted_fallback",
                            {
                                "models_tried": list(sorted(tried)),
                                "falling_back_to": "plain_text_two_call",
                            }
                        )

                        # Use two-call strategy - this will raise its own exception if it fails
                        # Note: We need to return the expanded tuple here
                        analysis_text, tags, model_used, tokens_in, tokens_out, tokens_total, duration = (
                            _analyze_with_plain_text_fallback(
                                client, selector, system_prompt, user_prompt,
                                max_tokens, preferred_model, verbose
                            )
                        )
                        # Since parse_llm_json_response expects the full response, we need to
                        # format the tags back into the response. But since we're returning from
                        # _call_llm_with_fallback, we just return the analysis text.
                        # The caller will need to handle tags separately or we store them differently.
                        # Actually, looking at the signature, _call_llm_with_fallback returns a 6-tuple,
                        # but _analyze_with_plain_text_fallback returns a 7-tuple with tags.
                        # We need to handle this mismatch. Let's raise a special exception with the data.
                        raise PlainTextFallbackSuccess(
                            analysis_text, tags, model_used, tokens_in, tokens_out, tokens_total, duration
                        )
                    else:
                        # Not using JSON mode, so we can't fall back
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


def _analyze_with_plain_text_fallback(
    client,
    selector: ModelSelector,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    preferred_model: str,
    verbose: bool,
) -> tuple[str, list[str], str, int, int, int, float]:
    """
    Two-call strategy for when JSON mode fails:
    1. Get analysis as plain text
    2. Extract tags from analysis text

    Args:
        client: Groq API client
        selector: ModelSelector instance
        system_prompt: System prompt (will be stripped of JSON rules)
        user_prompt: User prompt (will be stripped of JSON rules)
        max_output_tokens: Max tokens for analysis output
        preferred_model: Preferred model to use
        verbose: Enable verbose logging

    Returns:
        Tuple of (analysis_text, tags, model_used, tokens_in, tokens_out, tokens_total, duration)
    """
    from .utils import log_error_to_file

    logger.info("Using plain text fallback strategy (2 API calls)...")

    # Strip JSON formatting rules from prompts
    plain_system = strip_json_formatting_rules(system_prompt)
    plain_user = strip_json_formatting_rules(user_prompt)

    # Call 1: Get analysis as plain text
    try:
        analysis_text, model_used, tokens_in_1, tokens_out_1, tokens_total_1, duration_1 = (
            _call_llm_with_fallback(
                client,
                selector,
                plain_system,
                plain_user,
                max_output_tokens,
                preferred_model,
                verbose,
                use_json_mode=False,  # Plain text
            )
        )
    except Exception as e:
        # Log the failure
        log_error_to_file(
            "plain_text_fallback_analysis_failed",
            {
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise

    if verbose:
        logger.info(f"Plain text analysis completed ({tokens_out_1} tokens)")

    # Call 2: Extract tags from analysis
    tags = []
    tokens_in_2 = 0
    tokens_out_2 = 0
    tokens_total_2 = 0
    duration_2 = 0.0

    tag_system = "You are a metadata extraction assistant. Extract 3-5 concise, relevant contextual tags from the analysis."
    tag_user = f"Extract 3-5 tags from this analysis. Return only the tags, comma-separated:\n\n{analysis_text[:2000]}"

    try:
        # Reset selector for tag extraction attempt
        selector.reset()
        tag_response, _, tokens_in_2, tokens_out_2, tokens_total_2, duration_2 = (
            _call_llm_with_fallback(
                client,
                selector,
                tag_system,
                tag_user,
                200,  # Short response
                preferred_model,
                verbose,
                use_json_mode=False,
            )
        )

        # Parse tags from response (comma-separated or newlines)
        tag_response = tag_response.strip()
        # Remove common prefixes like "Tags:", "Here are the tags:", etc.
        tag_response = re.sub(r'^(tags?|here are the tags?):?\s*', '', tag_response, flags=re.IGNORECASE)
        # Split by comma or newline
        raw_tags = [t.strip() for t in tag_response.replace("\n", ",").split(",") if t.strip()]
        # Clean up tags (remove quotes, bullets, numbers)
        tags = []
        for tag in raw_tags[:5]:  # Limit to 5
            tag = re.sub(r'^[\d\.\-\*\)]+\s*', '', tag)  # Remove list markers
            tag = tag.strip('"\'`')  # Remove quotes
            if tag and len(tag) > 2:  # Skip very short tags
                tags.append(tag)

        if verbose and tags:
            logger.info(f"Extracted tags: {tags}")

    except Exception as e:
        logger.warning(f"Tag extraction failed: {e}")
        # Don't fail the whole analysis if tag extraction fails
        log_error_to_file(
            "plain_text_tag_extraction_failed",
            {
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )

    # Combine metrics
    total_tokens_in = tokens_in_1 + tokens_in_2
    total_tokens_out = tokens_out_1 + tokens_out_2
    total_tokens = tokens_total_1 + tokens_total_2
    total_duration = duration_1 + duration_2

    return analysis_text, tags, model_used, total_tokens_in, total_tokens_out, total_tokens, total_duration


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

    if current_model is None:
        raise AnalysisError(
            f"No model can handle {estimated_total_tokens} tokens. "
            f"This should not happen - chunking should have been used instead."
        )

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
    try:
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
        analysis_text, contextual_tags = parse_llm_json_response(response_text, verbose)

    except PlainTextFallbackSuccess as e:
        # Plain text fallback succeeded - use the extracted data
        logger.info("Plain text fallback completed successfully")
        analysis_text = e.analysis_text
        contextual_tags = e.tags
        model_used = e.model
        tokens_in = e.tokens_in
        tokens_out = e.tokens_out
        tokens_total = e.tokens_total
        duration = e.duration
        actual_cost = selector.estimate_cost(model_used, tokens_in, tokens_out)

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

    # Reset selector once before processing all chunks to preserve cross-chunk failure memory
    selector.reset()

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

            # Make API call (don't reset selector to preserve failure memory across chunks)
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

    # Validate chunk results before synthesis
    valid_results = [r for r in chunk_results if r and len(r.strip()) > 50]
    if len(valid_results) < len(chunk_results):
        logger.warning(
            f"Only {len(valid_results)}/{len(chunk_results)} chunks produced valid results. "
            f"Some chunk analyses may have failed or returned minimal content."
        )
    if not valid_results:
        raise AnalysisError(
            "All chunk analyses failed or returned empty results - cannot synthesize. "
            "This may indicate API issues or model failures across all chunks."
        )

    # Prepare synthesis input
    synthesis_input = format_chunks_for_synthesis(
        valid_results, video_info.title, video_info.duration_string
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
    try:
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
        final_analysis, contextual_tags = parse_llm_json_response(synthesis_text, verbose)

    except PlainTextFallbackSuccess as e:
        # Plain text fallback succeeded for synthesis
        logger.info("Synthesis completed with plain text fallback")
        final_analysis = e.analysis_text
        contextual_tags = e.tags
        synthesis_model = e.model
        syn_in = e.tokens_in
        syn_out = e.tokens_out
        syn_total = e.tokens_total
        total_tokens += syn_total
        syn_cost = selector.estimate_cost(synthesis_model, syn_in, syn_out)
        total_cost += syn_cost or 0
        duration = time.time() - start_time

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

    # Reset selector once before processing all chunks to preserve cross-chunk failure memory
    selector.reset()

    for chunk in chunks:
        logger.info(f"Analyzing chunk {chunk.index + 1}/{chunk.total_chunks}...")

        variables = format_chunk_for_analysis(chunk, video_info.title)
        system_prompt = substitute_prompt_variables(chunk_template.system_prompt, variables)
        user_prompt = substitute_prompt_variables(chunk_template.user_prompt, variables)

        # Don't reset selector to preserve failure memory across chunks
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

    # Validate chunk results before synthesis
    valid_results = [r for r in chunk_results if r and len(r.strip()) > 50]
    if len(valid_results) < len(chunk_results):
        logger.warning(
            f"Only {len(valid_results)}/{len(chunk_results)} chunks produced valid results. "
            f"Some chunk analyses may have failed or returned minimal content."
        )
    if not valid_results:
        raise AnalysisError(
            "All chunk analyses failed or returned empty results - cannot synthesize. "
            "This may indicate API issues or model failures across all chunks."
        )

    synthesis_input = format_chunks_for_synthesis(
        valid_results, video_info.title, video_info.duration_string
    )
    synthesis_variables = {"chunk_analyses": synthesis_input}
    synthesis_system = substitute_prompt_variables(
        synthesis_template.system_prompt, synthesis_variables
    )
    synthesis_user = substitute_prompt_variables(
        synthesis_template.user_prompt, synthesis_variables
    )

    selector.reset()
    try:
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
        final_analysis, contextual_tags = parse_llm_json_response(synthesis_text, verbose)

    except PlainTextFallbackSuccess as e:
        # Plain text fallback succeeded for synthesis
        logger.info("Synthesis completed with plain text fallback")
        final_analysis = e.analysis_text
        contextual_tags = e.tags
        synthesis_model = e.model
        syn_in = e.tokens_in
        syn_out = e.tokens_out
        syn_total = e.tokens_total
        total_tokens += syn_total
        syn_cost = selector.estimate_cost(synthesis_model, syn_in, syn_out)
        total_cost += syn_cost or 0
        duration = time.time() - start_time

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
