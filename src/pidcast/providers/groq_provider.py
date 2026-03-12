"""Groq API provider for pidcast analysis.

Wraps the existing analysis.py Groq integration as an AnalysisProvider.
"""

from __future__ import annotations

from pidcast.analysis import analyze_transcript_with_llm
from pidcast.config import AnalysisResult, PromptsConfig, VideoInfo


class GroqProvider:
    """Analysis provider using the Groq API."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self._api_key = api_key
        self._model = model

    def analyze(
        self,
        transcript: str,
        video_info: VideoInfo,
        analysis_type: str,
        prompts_config: PromptsConfig,
        verbose: bool = False,
    ) -> AnalysisResult:
        """Analyze transcript using the Groq API."""
        return analyze_transcript_with_llm(
            transcript,
            video_info,
            analysis_type,
            prompts_config,
            self._api_key,
            self._model,
            verbose,
        )
