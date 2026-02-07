"""Prompt versioning and management for evals."""

import json
from dataclasses import dataclass
from pathlib import Path

from pidcast.exceptions import ConfigurationError


@dataclass
class EvalPrompt:
    """Represents a versioned eval prompt."""

    prompt_type: str  # e.g., "summary", "key_points"
    version: str  # e.g., "v1", "v2"
    name: str
    description: str
    system_prompt: str
    user_prompt: str
    max_output_tokens: int


class PromptManager:
    """Manages versioned eval prompts."""

    def __init__(self, prompts_file: Path):
        """
        Initialize prompt manager.

        Args:
            prompts_file: Path to eval_prompts.json
        """
        self.prompts_file = prompts_file
        self._prompts: dict[str, list[EvalPrompt]] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompts from JSON, validate structure."""
        if not self.prompts_file.exists():
            raise ConfigurationError(
                f"Prompts file not found: {self.prompts_file}. "
                "Create config/eval_prompts.json with prompt definitions."
            )

        try:
            with open(self.prompts_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {self.prompts_file}: {e}") from e

        if "prompts" not in data:
            raise ConfigurationError(f"Missing 'prompts' key in {self.prompts_file}")

        # Parse prompts by type
        for prompt_type, versions in data["prompts"].items():
            if not isinstance(versions, list):
                raise ConfigurationError(f"Prompt type '{prompt_type}' must be a list of versions")

            self._prompts[prompt_type] = []
            for prompt_data in versions:
                try:
                    prompt = EvalPrompt(
                        prompt_type=prompt_type,
                        version=prompt_data["version"],
                        name=prompt_data["name"],
                        description=prompt_data["description"],
                        system_prompt=prompt_data["system_prompt"],
                        user_prompt=prompt_data["user_prompt"],
                        max_output_tokens=prompt_data["max_output_tokens"],
                    )
                    self._prompts[prompt_type].append(prompt)
                except KeyError as e:
                    raise ConfigurationError(
                        f"Missing required field {e} in prompt "
                        f"'{prompt_type}' version '{prompt_data.get('version', '?')}'"
                    ) from e

    def get_prompt(self, prompt_type: str, version: str) -> EvalPrompt:
        """
        Retrieve a specific prompt by type and version.

        Args:
            prompt_type: Type of prompt (e.g., "summary")
            version: Version string (e.g., "v1")

        Returns:
            EvalPrompt instance

        Raises:
            ConfigurationError: If prompt type or version not found
        """
        if prompt_type not in self._prompts:
            available = ", ".join(sorted(self._prompts.keys()))
            raise ConfigurationError(
                f"Prompt type '{prompt_type}' not found. Available types: {available}"
            )

        for prompt in self._prompts[prompt_type]:
            if prompt.version == version:
                return prompt

        available_versions = [p.version for p in self._prompts[prompt_type]]
        raise ConfigurationError(
            f"Prompt version '{version}' not found for type '{prompt_type}'. "
            f"Available versions: {', '.join(sorted(available_versions))}"
        )

    def list_versions(self, prompt_type: str) -> list[str]:
        """
        List all versions for a given prompt type.

        Args:
            prompt_type: Type of prompt

        Returns:
            List of version strings
        """
        if prompt_type not in self._prompts:
            return []
        return [p.version for p in self._prompts[prompt_type]]

    def list_prompt_types(self) -> list[str]:
        """
        List all available prompt types.

        Returns:
            List of prompt type strings
        """
        return sorted(self._prompts.keys())

    def get_all_prompts(self) -> list[EvalPrompt]:
        """
        Get all prompts across all types and versions.

        Returns:
            List of all EvalPrompt instances
        """
        all_prompts = []
        for prompts in self._prompts.values():
            all_prompts.extend(prompts)
        return all_prompts
