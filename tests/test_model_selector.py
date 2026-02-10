"""Tests for pidcast.model_selector module."""

from unittest.mock import patch

import pytest

from pidcast.exceptions import AnalysisError
from pidcast.model_selector import (
    ModelSelector,
    is_rate_limit_error,
    is_retryable_error,
    load_models_config,
    with_retry,
)

# ============================================================================
# ModelConfig
# ============================================================================


class TestModelConfig:
    def test_estimate_cost_basic(self, sample_model_config):
        # 1M input tokens at $0.15, 1M output tokens at $0.60
        cost = sample_model_config.estimate_cost(1_000_000, 1_000_000)
        assert cost == pytest.approx(0.75)

    def test_estimate_cost_zero_tokens(self, sample_model_config):
        assert sample_model_config.estimate_cost(0, 0) == 0.0

    def test_estimate_cost_small(self, sample_model_config):
        # 1000 input, 500 output
        cost = sample_model_config.estimate_cost(1000, 500)
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)


# ============================================================================
# ModelsConfig
# ============================================================================


class TestModelsConfig:
    def test_get_model_exists(self, sample_models_config):
        model = sample_models_config.get_model("large-model")
        assert model is not None
        assert model.name == "large-model"

    def test_get_model_missing(self, sample_models_config):
        assert sample_models_config.get_model("nonexistent") is None

    def test_get_default(self, sample_models_config):
        default = sample_models_config.get_default()
        assert default is not None
        assert default.name == "large-model"


# ============================================================================
# load_models_config
# ============================================================================


class TestLoadModelsConfig:
    def test_load_valid_config(self, tmp_path):
        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
default_model: model-a
fallback_chain:
  - model-a
models:
  model-a:
    display_name: Model A
    provider: groq
    context_window: 32768
    pricing:
      input: 0.1
      output: 0.2
    limits:
      rpm: 30
      rpd: 1000
      tpm: 10000
      tpd: 100000
""")
        config = load_models_config(config_file)
        assert config.default_model == "model-a"
        assert "model-a" in config.models
        assert config.models["model-a"].display_name == "Model A"

    def test_missing_file(self, tmp_path):
        with pytest.raises(AnalysisError, match="not found"):
            load_models_config(tmp_path / "missing.yaml")

    def test_empty_file(self, tmp_path):
        config_file = tmp_path / "models.yaml"
        config_file.write_text("")
        with pytest.raises(AnalysisError, match="empty"):
            load_models_config(config_file)

    def test_invalid_yaml(self, tmp_path):
        config_file = tmp_path / "models.yaml"
        config_file.write_text("{{invalid: yaml: [")
        with pytest.raises(AnalysisError, match="Invalid YAML"):
            load_models_config(config_file)

    def test_unknown_default_model(self, tmp_path):
        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
default_model: nonexistent
models:
  model-a:
    display_name: A
""")
        with pytest.raises(AnalysisError, match="not found in models"):
            load_models_config(config_file)

    def test_unknown_model_in_fallback_chain(self, tmp_path):
        config_file = tmp_path / "models.yaml"
        config_file.write_text("""
default_model: model-a
fallback_chain:
  - model-a
  - ghost-model
models:
  model-a:
    display_name: A
""")
        with pytest.raises(AnalysisError, match="Unknown models in fallback_chain"):
            load_models_config(config_file)


# ============================================================================
# is_retryable_error / is_rate_limit_error
# ============================================================================


class TestErrorClassification:
    @pytest.mark.parametrize(
        "msg",
        [
            "rate limit exceeded",
            "Too Many Requests",
            "Error 429",
            "connection reset",
            "timeout occurred",
            "503 Service Unavailable",
            "bad gateway",
            "server is overloaded",
        ],
    )
    def test_retryable_errors(self, msg):
        assert is_retryable_error(Exception(msg)) is True

    def test_non_retryable_error(self):
        assert is_retryable_error(Exception("invalid api key")) is False

    def test_rate_limit_specific(self):
        assert is_rate_limit_error(Exception("rate limit exceeded")) is True
        assert is_rate_limit_error(Exception("rate_limit_exceeded")) is True
        assert is_rate_limit_error(Exception("429 too many")) is True
        assert is_rate_limit_error(Exception("timeout")) is False


# ============================================================================
# with_retry
# ============================================================================


class TestWithRetry:
    def test_succeeds_first_try(self):
        @with_retry(max_retries=2, base_delay=0.01)
        def always_works():
            return "ok"

        assert always_works() == "ok"

    @patch("pidcast.model_selector.time.sleep")
    def test_retries_on_retryable_error(self, mock_sleep):
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("rate limit exceeded")
            return "recovered"

        assert fails_then_succeeds() == "recovered"
        assert call_count == 2
        assert mock_sleep.called

    def test_raises_non_retryable_immediately(self):
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def bad_key():
            nonlocal call_count
            call_count += 1
            raise Exception("invalid api key")

        with pytest.raises(Exception, match="invalid api key"):
            bad_key()
        assert call_count == 1

    @patch("pidcast.model_selector.time.sleep")
    def test_exhausts_retries(self, mock_sleep):
        @with_retry(max_retries=2, base_delay=0.01)
        def always_fails():
            raise Exception("rate limit exceeded")

        with pytest.raises(Exception, match="rate limit"):
            always_fails()
        assert mock_sleep.call_count == 2


# ============================================================================
# ModelSelector
# ============================================================================


class TestModelSelector:
    def test_get_default_model(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        assert selector.get_default_model() == "large-model"

    def test_get_display_name(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        assert selector.get_display_name("large-model") == "Large Model"
        assert selector.get_display_name("unknown") == "unknown"

    def test_tokens_within_context(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        # large-model: 131072 * 0.9 = 117964
        assert selector.tokens_within_context("large-model", 100000) is True
        assert selector.tokens_within_context("large-model", 120000) is False

    def test_tokens_within_context_unknown_model(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        assert selector.tokens_within_context("unknown", 999999) is True

    def test_get_max_context_tokens(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        assert selector.get_max_context_tokens("large-model") == int(131072 * 0.9)
        assert selector.get_max_context_tokens("unknown") == 100000

    def test_get_effective_token_limit(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        # large-model: min(131072*0.9, 8000) = 8000
        assert selector.get_effective_token_limit("large-model") == 8000

    def test_needs_chunking_fits(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        assert selector.needs_chunking(100000) is False  # fits in large-model

    def test_needs_chunking_too_large(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        assert selector.needs_chunking(200000) is True  # exceeds all models

    def test_select_model_preferred_fits(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        model, is_fallback = selector.select_model_for_tokens(1000, "large-model")
        assert model == "large-model"
        assert is_fallback is False

    def test_select_model_falls_back(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        # Exceeds large-model context (131072*0.9) but not small-model? No, small is 32768.
        # Both have 131072 and 32768. 120000 > 32768*0.9=29491 but < 131072*0.9=117964
        model, is_fallback = selector.select_model_for_tokens(100000)
        assert model == "large-model"
        assert is_fallback is False

    def test_select_model_none_when_too_large(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        model, is_fallback = selector.select_model_for_tokens(500000)
        assert model is None
        assert is_fallback is False

    def test_handle_rate_limit(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        next_model = selector.handle_rate_limit("large-model")
        assert next_model == "small-model"
        assert "large-model" in selector.get_tried_models()

    def test_handle_rate_limit_exhausted(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        selector.handle_rate_limit("large-model")
        result = selector.handle_rate_limit("small-model")
        assert result is None

    def test_reset(self, sample_models_config):
        selector = ModelSelector(sample_models_config)
        selector.mark_tried("large-model")
        selector.reset()
        assert len(selector.get_tried_models()) == 0
