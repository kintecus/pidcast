"""Pre-flight validation for eval configurations."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pidcast.config import GROQ_PRICING

from .prompt_manager import PromptManager
from .reference_transcripts import ReferenceTranscriptManager

if TYPE_CHECKING:
    from .batch_runner import BatchConfig


@dataclass
class ValidationResult:
    """Result of pre-flight validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class EvalValidator:
    """Validates eval configuration and environment."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        transcript_manager: ReferenceTranscriptManager,
    ):
        """
        Initialize validator.

        Args:
            prompt_manager: PromptManager instance
            transcript_manager: ReferenceTranscriptManager instance
        """
        self.prompt_manager = prompt_manager
        self.transcript_manager = transcript_manager

    def validate_batch_config(
        self, config: "BatchConfig", groq_api_key: str | None
    ) -> ValidationResult:
        """
        Validate batch configuration before starting.

        Args:
            config: BatchConfig to validate
            groq_api_key: Groq API key (or None)

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Check API key
        if not groq_api_key:
            errors.append("GROQ_API_KEY not configured. Set environment variable or use --groq_api_key")

        # Check prompt versions exist
        for version in config.prompt_versions:
            version_found = False
            for ptype in self.prompt_manager.list_prompt_types():
                if version in self.prompt_manager.list_versions(ptype):
                    version_found = True
                    break

            if not version_found:
                errors.append(
                    f"Prompt version '{version}' not found in any prompt type. "
                    f"Available versions: {self._get_all_versions()}"
                )

        # Check models are valid
        for model in config.models:
            if model not in GROQ_PRICING:
                errors.append(
                    f"Unknown model: '{model}'. Model not in GROQ_PRICING. "
                    f"Available models: {', '.join(GROQ_PRICING.keys())}"
                )

        # Check transcript IDs exist
        available_transcripts = set(self.transcript_manager.list_transcript_ids())
        for tid in config.transcript_ids:
            if tid not in available_transcripts:
                errors.append(
                    f"Reference transcript '{tid}' not found. "
                    f"Available: {', '.join(sorted(available_transcripts))}"
                )

        # Check disk space
        available_space = self._check_disk_space()
        min_space_mb = 100
        if available_space < min_space_mb * 1_000_000:  # 100MB
            warnings.append(
                f"Low disk space: {available_space / 1e6:.1f}MB available. "
                f"Recommended: at least {min_space_mb}MB"
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def _get_all_versions(self) -> str:
        """Get comma-separated list of all available versions."""
        all_versions = set()
        for ptype in self.prompt_manager.list_prompt_types():
            all_versions.update(self.prompt_manager.list_versions(ptype))
        return ", ".join(sorted(all_versions))

    def _check_disk_space(self) -> int:
        """
        Check available disk space in bytes.

        Returns:
            Available disk space in bytes
        """
        try:
            stat = shutil.disk_usage(Path.cwd())
            return stat.free
        except Exception:
            # If we can't check, return a large number to avoid false warnings
            return 1_000_000_000  # 1GB
