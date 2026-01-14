"""Model selection and fallback logic for LLM analysis."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import yaml

from .exceptions import AnalysisError

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    display_name: str
    provider: str
    context_window: int
    pricing_input: float
    pricing_output: float
    rpm: int
    rpd: int
    tpm: int
    tpd: int

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        input_cost = (input_tokens / 1_000_000) * self.pricing_input
        output_cost = (output_tokens / 1_000_000) * self.pricing_output
        return input_cost + output_cost


@dataclass
class ModelsConfig:
    """Full models configuration."""

    default_model: str
    fallback_chain: list[str]
    models: dict[str, ModelConfig]

    def get_model(self, name: str) -> ModelConfig | None:
        """Get model config by name."""
        return self.models.get(name)

    def get_default(self) -> ModelConfig | None:
        """Get the default model config."""
        return self.models.get(self.default_model)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================


def load_models_config(config_path: Path) -> ModelsConfig:
    """Load models configuration from YAML.

    Args:
        config_path: Path to models.yaml

    Returns:
        ModelsConfig with all model configurations

    Raises:
        AnalysisError: If config file is missing or invalid
    """
    if not config_path.exists():
        raise AnalysisError(f"Models config not found: {config_path}\nExpected: config/models.yaml")

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise AnalysisError(f"Invalid YAML in models config: {e}") from e

    if data is None:
        raise AnalysisError(f"Models config is empty: {config_path}")

    models = {}
    for name, cfg in data.get("models", {}).items():
        pricing = cfg.get("pricing", {})
        limits = cfg.get("limits", {})
        models[name] = ModelConfig(
            name=name,
            display_name=cfg.get("display_name", name),
            provider=cfg.get("provider", "groq"),
            context_window=cfg.get("context_window", 32768),
            pricing_input=pricing.get("input", 0),
            pricing_output=pricing.get("output", 0),
            rpm=limits.get("rpm", 30),
            rpd=limits.get("rpd", 1000),
            tpm=limits.get("tpm", 10000),
            tpd=limits.get("tpd", 0),
        )

    return ModelsConfig(
        default_model=data.get("default_model", "llama-3.3-70b-versatile"),
        fallback_chain=data.get("fallback_chain", list(models.keys())),
        models=models,
    )


# ============================================================================
# RETRY LOGIC
# ============================================================================


# Error types that should trigger retry
RETRYABLE_ERROR_MESSAGES = [
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "timeout",
    "connection",
    "temporarily unavailable",
]


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient)."""
    error_str = str(error).lower()
    return any(msg in error_str for msg in RETRYABLE_ERROR_MESSAGES)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is specifically a rate limit error."""
    error_str = str(error).lower()
    return "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str


def with_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    exponential_base: float = 2.0,
) -> Callable:
    """Decorator to retry function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        exponential_base: Multiplier for each retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            retry_count = 0

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Attach retry count to result if it's a dict or has attributes
                    if hasattr(result, "__dict__"):
                        result._retry_count = retry_count
                    return result
                except Exception as e:
                    last_error = e
                    if not is_retryable_error(e) or attempt >= max_retries:
                        raise

                    retry_count = attempt + 1
                    delay = base_delay * (exponential_base**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            raise last_error  # type: ignore

        return wrapper

    return decorator


# ============================================================================
# MODEL SELECTOR
# ============================================================================


class ModelSelector:
    """Handles model selection with fallback support."""

    def __init__(self, config: ModelsConfig):
        self.config = config
        self._tried_models: set[str] = set()

    def reset(self) -> None:
        """Reset the tried models set for a new request."""
        self._tried_models = set()

    def get_default_model(self) -> str:
        """Get the default (best) model name."""
        return self.config.default_model

    def get_model_config(self, model: str) -> ModelConfig | None:
        """Get configuration for a specific model."""
        return self.config.get_model(model)

    def get_display_name(self, model: str) -> str:
        """Get human-readable display name for a model."""
        cfg = self.get_model_config(model)
        return cfg.display_name if cfg else model

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Estimate cost for a request."""
        cfg = self.get_model_config(model)
        if cfg:
            return cfg.estimate_cost(input_tokens, output_tokens)
        return None

    def tokens_within_context(self, model: str, estimated_tokens: int) -> bool:
        """Check if estimated tokens fit within model's context window."""
        cfg = self.get_model_config(model)
        if not cfg:
            return True  # Unknown model, let API decide
        # Leave 10% buffer for safety
        max_tokens = int(cfg.context_window * 0.9)
        return estimated_tokens <= max_tokens

    def get_max_context_tokens(self, model: str | None = None) -> int:
        """Get maximum tokens for a model's context window (with safety buffer).

        Args:
            model: Model name (uses default if None)

        Returns:
            Max tokens (90% of context window for safety)
        """
        model = model or self.config.default_model
        cfg = self.get_model_config(model)
        if not cfg:
            return 100000  # Conservative default
        return int(cfg.context_window * 0.9)

    def get_effective_token_limit(self, model: str | None = None) -> int:
        """Get the effective token limit considering both context window and TPM.

        The effective limit is the minimum of context window and TPM,
        since both are constraints on a single request.

        Args:
            model: Model name (uses default if None)

        Returns:
            Effective token limit for a single request
        """
        model = model or self.config.default_model
        cfg = self.get_model_config(model)
        if not cfg:
            return 8000  # Conservative default

        # TPM is the rate limit per minute, but it's also effectively
        # the max tokens for a single request (can't exceed minute quota)
        context_limit = int(cfg.context_window * 0.9)
        tpm_limit = cfg.tpm

        # Use the more restrictive limit
        return min(context_limit, tpm_limit)

    def needs_chunking(self, estimated_tokens: int, model: str | None = None) -> bool:
        """Check if the request needs to be chunked.

        Args:
            estimated_tokens: Estimated total tokens
            model: Model to check against (uses default if None)

        Returns:
            True if tokens exceed all models' context windows
        """
        # Check if any model in the chain can handle it
        for m in self.config.fallback_chain:
            if self.tokens_within_context(m, estimated_tokens):
                return False
        return True

    def get_next_fallback(self) -> str | None:
        """Get next model in fallback chain that hasn't been tried.

        Returns:
            Next model name, or None if all exhausted
        """
        for model in self.config.fallback_chain:
            if model not in self._tried_models:
                return model
        return None

    def mark_tried(self, model: str) -> None:
        """Mark a model as tried."""
        self._tried_models.add(model)

    def get_tried_models(self) -> set[str]:
        """Get set of models that have been tried."""
        return self._tried_models.copy()

    def select_model_for_tokens(
        self,
        estimated_tokens: int,
        preferred_model: str | None = None,
    ) -> tuple[str, bool]:
        """Select best model that can handle the token count.

        Pre-checks context window to avoid wasting API calls on models
        that can't handle the request size.

        Note: This does NOT check TPM (rate limits). TPM errors are handled
        dynamically via handle_rate_limit() when the API returns 429.

        Args:
            estimated_tokens: Estimated total tokens for the request
            preferred_model: User's preferred model (if specified)

        Returns:
            Tuple of (model_name, is_fallback)
        """
        # Start with preferred or default
        start_model = preferred_model or self.config.default_model
        is_fallback = False

        # If preferred model can handle tokens within context window, use it
        if self.tokens_within_context(start_model, estimated_tokens):
            return start_model, False

        # Otherwise, find first model in fallback chain that can handle it
        logger.info(
            f"Estimated {estimated_tokens} tokens exceeds {start_model} context window, "
            f"selecting fallback..."
        )

        for model in self.config.fallback_chain:
            if self.tokens_within_context(model, estimated_tokens):
                if model != start_model:
                    is_fallback = True
                    logger.info(f"Pre-selected fallback model: {model}")
                return model, is_fallback

        # No model can handle it - this request is too large
        logger.warning(
            f"No model in fallback chain can handle {estimated_tokens} tokens. "
            f"Request may need to be chunked."
        )
        return self.config.default_model, False

    def handle_rate_limit(self, current_model: str) -> str | None:
        """Handle a rate limit error by selecting next fallback.

        Args:
            current_model: The model that hit the rate limit

        Returns:
            Next model to try, or None if all exhausted
        """
        self.mark_tried(current_model)
        next_model = self.get_next_fallback()

        if next_model:
            logger.info(f"Rate limit on {current_model}, falling back to {next_model}")
            return next_model

        logger.error(
            f"Rate limit on {current_model} and no more fallback models. "
            f"Tried: {', '.join(sorted(self._tried_models))}"
        )
        return None
