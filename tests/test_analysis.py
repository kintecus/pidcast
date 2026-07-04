"""Tests for pidcast.analysis's Groq JSON-mode request building.

Covers the fix for GPT-OSS 120B/20B frequently failing Groq's loose
`json_object` JSON validation (HTTP 400, json_validate_failed) by using
Groq's strict `json_schema` structured-outputs mode on models with confirmed
support (ModelConfig.supports_json_schema), while models without that support
keep the existing json_object mode.
"""

import json
from unittest.mock import MagicMock

from pidcast.analysis import ANALYSIS_JSON_SCHEMA, _call_llm_with_fallback
from pidcast.model_selector import ModelConfig, ModelsConfig, ModelSelector


def _make_model(name: str, supports_json_schema: bool) -> ModelConfig:
    return ModelConfig(
        name=name,
        display_name=name,
        provider="groq",
        context_window=131072,
        pricing_input=0.1,
        pricing_output=0.1,
        rpm=30,
        rpd=1000,
        tpm=8000,
        tpd=200000,
        supports_json_schema=supports_json_schema,
    )


def _schema_and_object_config() -> ModelsConfig:
    schema_model = _make_model("schema-model", supports_json_schema=True)
    object_model = _make_model("object-model", supports_json_schema=False)
    return ModelsConfig(
        default_model="schema-model",
        fallback_chain=["schema-model", "object-model"],
        models={"schema-model": schema_model, "object-model": object_model},
    )


def _fake_groq_response(analysis: str = "some analysis", tags: list | None = None):
    """A response object shaped like the real Groq SDK's return value.

    Only a bare Mock() would let LLMCallResult's int fields (tokens_in/out/
    total) through as Mock objects without failing a broken assertion, so
    every field _call_llm_with_fallback actually reads is set explicitly.
    """
    tags = tags if tags is not None else ["tag-one", "tag-two", "tag-three"]
    content = json.dumps({"analysis": analysis, "contextual_tags": tags})

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    return response


class TestAnalysisJsonSchema:
    def test_schema_has_additional_properties_false_and_matches_required(self):
        schema = ANALYSIS_JSON_SCHEMA["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())
        assert ANALYSIS_JSON_SCHEMA["strict"] is True

    def test_contextual_tags_requires_exactly_three(self):
        tags_schema = ANALYSIS_JSON_SCHEMA["schema"]["properties"]["contextual_tags"]
        assert tags_schema["minItems"] == 3
        assert tags_schema["maxItems"] == 3


class TestCallLlmWithFallbackRequestFormat:
    def test_schema_capable_model_uses_json_schema_mode(self):
        config = _schema_and_object_config()
        selector = ModelSelector(config)
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_groq_response()

        _call_llm_with_fallback(
            client,
            selector,
            system_prompt="system",
            user_prompt="user",
            max_tokens=500,
            preferred_model="schema-model",
            use_json_mode=True,
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": ANALYSIS_JSON_SCHEMA,
        }

    def test_non_schema_capable_model_keeps_json_object_mode(self):
        config = _schema_and_object_config()
        selector = ModelSelector(config)
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_groq_response()

        _call_llm_with_fallback(
            client,
            selector,
            system_prompt="system",
            user_prompt="user",
            max_tokens=500,
            preferred_model="object-model",
            use_json_mode=True,
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_json_mode_false_sends_no_response_format(self):
        config = _schema_and_object_config()
        selector = ModelSelector(config)
        client = MagicMock()
        client.chat.completions.create.return_value = _fake_groq_response()

        _call_llm_with_fallback(
            client,
            selector,
            system_prompt="system",
            user_prompt="user",
            max_tokens=500,
            preferred_model="schema-model",
            use_json_mode=False,
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs


class TestJsonValidationFallbackTransition:
    def test_schema_model_json_validate_failure_falls_back_to_json_object_model(self):
        """A json_validate_failed error on the schema-capable model must
        advance to the next model in the chain, which then uses json_object
        mode (not the schema) - the real failure-recovery path being fixed."""
        config = _schema_and_object_config()
        selector = ModelSelector(config)
        client = MagicMock()

        json_validation_error = Exception(
            "Error code: 400 - {'error': {'message': \"Failed to validate JSON. "
            "Please adjust your prompt.\", 'code': 'json_validate_failed'}}"
        )
        client.chat.completions.create.side_effect = [
            json_validation_error,
            _fake_groq_response(),
        ]

        result = _call_llm_with_fallback(
            client,
            selector,
            system_prompt="system",
            user_prompt="user",
            max_tokens=500,
            preferred_model="schema-model",
            use_json_mode=True,
        )

        assert result.model == "object-model"
        assert client.chat.completions.create.call_count == 2

        first_call_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        assert first_call_kwargs["response_format"]["type"] == "json_schema"
        assert second_call_kwargs["response_format"] == {"type": "json_object"}
