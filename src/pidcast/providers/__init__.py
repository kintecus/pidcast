"""LLM provider implementations."""

from __future__ import annotations

from typing import Protocol

from pidcast.config import AnalysisResult, PromptsConfig, VideoInfo


class AnalysisProvider(Protocol):
    """Protocol that all LLM analysis providers must satisfy."""

    def analyze(
        self,
        transcript: str,
        video_info: VideoInfo,
        analysis_type: str,
        prompts_config: PromptsConfig,
        verbose: bool = False,
    ) -> AnalysisResult: ...
