"""LLM and transcription provider implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

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


@dataclass
class TranscriptionResult:
    """Result from a transcription provider."""

    text: str
    speaker_count: int | None
    duration: float
    provider: str
    language: str | None
    diarized: bool


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Protocol that all transcription providers must satisfy."""

    def transcribe(
        self,
        audio_file: str | Path,
        language: str | None = None,
        diarize: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult: ...
