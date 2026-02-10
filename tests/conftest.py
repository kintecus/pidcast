"""Shared fixtures for pidcast tests."""

import pytest

from pidcast.config import VideoInfo
from pidcast.model_selector import ModelConfig, ModelsConfig


@pytest.fixture
def sample_video_info():
    """Reusable VideoInfo for tests."""
    return VideoInfo(
        title="How to Build Great Software",
        webpage_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        channel="Tech Channel",
        uploader="Tech Channel",
        duration=3600,
        duration_string="1:00:00",
        view_count=100000,
        upload_date="20240101",
        description="A talk about software engineering.",
    )


@pytest.fixture
def sample_model_config():
    """Single ModelConfig for tests."""
    return ModelConfig(
        name="test-model",
        display_name="Test Model",
        provider="groq",
        context_window=131072,
        pricing_input=0.15,
        pricing_output=0.60,
        rpm=30,
        rpd=1000,
        tpm=8000,
        tpd=200000,
    )


@pytest.fixture
def sample_models_config():
    """Full ModelsConfig with fallback chain."""
    large = ModelConfig(
        name="large-model",
        display_name="Large Model",
        provider="groq",
        context_window=131072,
        pricing_input=0.15,
        pricing_output=0.60,
        rpm=30,
        rpd=1000,
        tpm=8000,
        tpd=200000,
    )
    small = ModelConfig(
        name="small-model",
        display_name="Small Model",
        provider="groq",
        context_window=32768,
        pricing_input=0.05,
        pricing_output=0.08,
        rpm=30,
        rpd=14400,
        tpm=6000,
        tpd=500000,
    )
    return ModelsConfig(
        default_model="large-model",
        fallback_chain=["large-model", "small-model"],
        models={"large-model": large, "small-model": small},
    )
