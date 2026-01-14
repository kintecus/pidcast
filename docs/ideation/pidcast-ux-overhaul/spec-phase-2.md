# Implementation Spec: Pidcast UX Overhaul - Phase 2

**PRD**: ./prd-phase-2.md
**Estimated Effort**: M (Medium)

## Technical Approach

This phase implements a quality-prioritized model fallback chain that handles rate limits gracefully. The key insight is that we want to try the best model first (gptoss120b), then progressively fall back to smaller/faster models only when necessary.

The current codebase already has TPM-based fallback logic in `analysis.py`. We're extending this to:
1. Use a configurable, quality-ordered chain instead of TPM-ordered
2. Add proper retry with exponential backoff for transient errors
3. Pre-check token estimates before wasting API calls

We'll reuse the retry logic pattern from `src/pidcast/evals/retry.py`.

## File Changes

### New Files

| File Path | Purpose |
|-----------|---------|
| `config/models.yaml` | Model configuration with fallback chain and limits |
| `src/pidcast/model_selector.py` | Model selection and fallback logic |

### Modified Files

| File Path | Changes |
|-----------|---------|
| `src/pidcast/config.py` | Load models.yaml, remove hardcoded GROQ_RATE_LIMITS |
| `src/pidcast/analysis.py` | Use ModelSelector, add retry decorator, show model used |
| `src/pidcast/cli.py` | Pass through model selection, display model in output |

## Implementation Details

### 1. Models Configuration (YAML)

**Overview**: Centralize model configuration with quality-ordered fallback chain.

```yaml
# config/models.yaml
default_model: openai/gpt-oss-120b

# Quality-prioritized fallback chain (best to acceptable)
fallback_chain:
  - openai/gpt-oss-120b
  - groq/compound
  - openai/gpt-oss-20b
  - llama-3.3-70b-versatile
  - llama-3.1-8b-instant

models:
  openai/gpt-oss-120b:
    display_name: GPT-OSS 120B
    provider: groq
    pricing:
      input: 0.15   # per 1M tokens
      output: 0.60
    limits:
      rpm: 30
      rpd: 1000
      tpm: 8000
      tpd: 200000

  groq/compound:
    display_name: Groq Compound
    provider: groq
    pricing:
      input: 0.30
      output: 0.40
    limits:
      rpm: 30
      rpd: 250
      tpm: 70000
      tpd: 0  # No daily limit

  openai/gpt-oss-20b:
    display_name: GPT-OSS 20B
    provider: groq
    pricing:
      input: 0.075
      output: 0.30
    limits:
      rpm: 30
      rpd: 1000
      tpm: 8000
      tpd: 200000

  llama-3.3-70b-versatile:
    display_name: Llama 3.3 70B
    provider: groq
    pricing:
      input: 0.59
      output: 0.79
    limits:
      rpm: 30
      rpd: 1000
      tpm: 12000
      tpd: 100000

  llama-3.1-8b-instant:
    display_name: Llama 3.1 8B
    provider: groq
    pricing:
      input: 0.05
      output: 0.08
    limits:
      rpm: 30
      rpd: 14400
      tpm: 6000
      tpd: 500000
```

**Implementation steps**:
1. Create `config/models.yaml` with above content
2. Add YAML loading to `config.py`
3. Create dataclasses for ModelConfig, ModelsConfig

### 2. Model Selector Module

**Pattern to follow**: `src/pidcast/evals/retry.py` for retry logic

**Overview**: Encapsulate model selection, validation, and fallback logic.

```python
# src/pidcast/model_selector.py
"""Model selection and fallback logic for LLM analysis."""

from dataclasses import dataclass
from pathlib import Path
import logging
import yaml

from .exceptions import AnalysisError

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    display_name: str
    provider: str
    pricing_input: float
    pricing_output: float
    rpm: int
    rpd: int
    tpm: int
    tpd: int


@dataclass
class ModelsConfig:
    """Full models configuration."""
    default_model: str
    fallback_chain: list[str]
    models: dict[str, ModelConfig]


def load_models_config(config_path: Path) -> ModelsConfig:
    """Load models configuration from YAML."""
    if not config_path.exists():
        raise AnalysisError(f"Models config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    models = {}
    for name, cfg in data.get("models", {}).items():
        models[name] = ModelConfig(
            name=name,
            display_name=cfg.get("display_name", name),
            provider=cfg.get("provider", "groq"),
            pricing_input=cfg.get("pricing", {}).get("input", 0),
            pricing_output=cfg.get("pricing", {}).get("output", 0),
            rpm=cfg.get("limits", {}).get("rpm", 30),
            rpd=cfg.get("limits", {}).get("rpd", 1000),
            tpm=cfg.get("limits", {}).get("tpm", 10000),
            tpd=cfg.get("limits", {}).get("tpd", 0),
        )

    return ModelsConfig(
        default_model=data.get("default_model", "llama-3.3-70b-versatile"),
        fallback_chain=data.get("fallback_chain", list(models.keys())),
        models=models,
    )


class ModelSelector:
    """Handles model selection with fallback support."""

    def __init__(self, config: ModelsConfig):
        self.config = config
        self._tried_models: set[str] = set()

    def get_default_model(self) -> str:
        """Get the default (best) model."""
        return self.config.default_model

    def get_model_config(self, model: str) -> ModelConfig | None:
        """Get configuration for a specific model."""
        return self.config.models.get(model)

    def estimate_tokens_ok(self, model: str, estimated_tokens: int) -> bool:
        """Check if estimated tokens would fit within model's TPM limit."""
        cfg = self.get_model_config(model)
        if not cfg:
            return True  # Unknown model, let API decide
        return estimated_tokens <= cfg.tpm

    def get_next_fallback(self, exclude: set[str] | None = None) -> str | None:
        """Get next model in fallback chain that hasn't been tried."""
        exclude = exclude or set()
        for model in self.config.fallback_chain:
            if model not in exclude:
                return model
        return None

    def select_model_for_tokens(
        self,
        estimated_tokens: int,
        preferred_model: str | None = None,
    ) -> tuple[str, bool]:
        """Select best model that can handle the token count.

        Returns:
            Tuple of (model_name, is_fallback)
        """
        # Start with preferred or default
        model = preferred_model or self.config.default_model
        tried = set()

        while model:
            if self.estimate_tokens_ok(model, estimated_tokens):
                is_fallback = model != (preferred_model or self.config.default_model)
                return model, is_fallback

            logger.info(
                f"Skipping {model}: estimated {estimated_tokens} tokens exceeds TPM limit"
            )
            tried.add(model)
            model = self.get_next_fallback(tried)

        # No model can handle it - return default and let API fail
        return self.config.default_model, False
```

**Implementation steps**:
1. Create `src/pidcast/model_selector.py` with above code
2. Add `ModelsConfig` loading to `config.py`
3. Update `analysis.py` to use `ModelSelector`

### 3. Retry Logic Integration

**Pattern to follow**: `src/pidcast/evals/retry.py`

**Overview**: Apply retry decorator to API calls for transient errors.

```python
# In analysis.py - use existing retry pattern from evals

from functools import wraps
from groq import RateLimitError, APIConnectionError, APITimeoutError

RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


def with_retry(max_retries: int = 3, base_delay: float = 2.0):
    """Decorator for retrying on transient errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


# Apply to API call
@with_retry(max_retries=3, base_delay=2.0)
def _call_groq_api(client, model: str, messages: list, max_tokens: int, timeout: int):
    """Call Groq API with retry logic."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
        timeout=timeout,
    )
```

**Implementation steps**:
1. Add retry decorator to `analysis.py`
2. Extract API call into decorated function
3. Update fallback loop to handle exhausted retries

### 4. Analysis Module Updates

**Overview**: Integrate ModelSelector and show which model was used.

```python
# Updates to analyze_transcript_with_llm()

def analyze_transcript_with_llm(
    transcript: str,
    video_info: VideoInfo,
    analysis_type: str,
    prompts_config: PromptsConfig,
    api_key: str,
    model: str | None = None,  # Now optional, uses config default
    verbose: bool = False,
) -> AnalysisResult:
    """Analyze transcript with automatic model fallback."""

    # Load model config
    models_config = load_models_config(DEFAULT_MODELS_FILE)
    selector = ModelSelector(models_config)

    # Estimate tokens
    estimated_tokens = estimate_total_tokens(system_prompt, user_prompt, max_output)

    # Select best model for this request
    selected_model, is_fallback = selector.select_model_for_tokens(
        estimated_tokens,
        preferred_model=model,
    )

    if is_fallback:
        logger.info(f"Using fallback model: {selected_model}")

    # Try selected model, fall back on rate limit
    tried_models = set()
    current_model = selected_model

    while current_model:
        tried_models.add(current_model)

        try:
            result = _call_groq_api(...)
            # Success - add model info to result
            logger.info(f"Generated with: {current_model}")
            return result

        except RateLimitError as e:
            logger.warning(f"Rate limit on {current_model}: {e}")
            current_model = selector.get_next_fallback(tried_models)
            if not current_model:
                raise AnalysisError(
                    f"All models exhausted. Tried: {', '.join(tried_models)}"
                )

    raise AnalysisError("No models available")
```

**Implementation steps**:
1. Load models config at start of analysis
2. Use ModelSelector to pick initial model
3. Wrap API call in retry logic
4. Fall back on RateLimitError after retries exhausted
5. Log which model produced the result

### 5. Output Updates

**Overview**: Show model information in terminal output.

**Implementation steps**:
1. Add "Model" row to metadata panel in `render_analysis_to_terminal()`
2. Include model in YAML front matter when saving
3. Log "Generated with: {model_name}" after successful analysis

## Error Handling

| Error Scenario | Handling Strategy |
|----------------|-------------------|
| All models rate-limited | Raise `AnalysisError` listing all tried models |
| Model not in config | Use it anyway, let API validate |
| Transient API error | Retry 3x with exponential backoff (2s, 4s, 8s) |
| `models.yaml` missing | Raise `AnalysisError` with clear path |

## Validation Commands

```bash
# Lint
uv run ruff check src/pidcast/
uv run ruff format src/pidcast/

# Test with default model (gpt-oss-120b)
uv run pidcast "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Test explicit model override
uv run pidcast "VIDEO_URL" --model llama-3.3-70b-versatile

# Test fallback (use a transcript that exceeds TPM)
uv run pidcast --analyze_existing data/transcripts/long_video.md

# Verify model shown in output
# Should see "Generated with: {model}" in logs
```

---

*This spec is ready for implementation. Follow the patterns and validate at each step.*
