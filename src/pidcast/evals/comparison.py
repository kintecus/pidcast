"""Generate side-by-side comparison markdown files."""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from .results import EvalResult


class ComparisonType(Enum):
    """Type of comparison being generated."""

    PROMPT = "prompt_comparison"
    MODEL = "model_comparison"


@dataclass
class ComparisonConfig:
    """Configuration for generating a comparison."""

    comparison_type: ComparisonType
    eval_results: list[EvalResult]
    output_dir: Path


class ComparisonGenerator:
    """Generates side-by-side comparison markdown files."""

    def generate_comparisons_from_batch(
        self, eval_results: list[EvalResult], output_dir: Path
    ) -> list[Path]:
        """
        Auto-generate logical comparisons from batch results.

        Groups results by:
        - (transcript, model) -> prompt comparison
        - (transcript, prompt) -> model comparison

        Args:
            eval_results: List of EvalResult instances from batch
            output_dir: Base comparisons directory

        Returns:
            List of generated comparison file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # Group results for prompt comparisons (same transcript + model, different prompts)
        prompt_groups = defaultdict(list)
        for result in eval_results:
            if result.success:
                key = (result.transcript_id, result.model)
                prompt_groups[key].append(result)

        # Generate prompt comparisons
        for (transcript_id, model), results in prompt_groups.items():
            if len(results) > 1:  # Only compare if we have multiple prompt versions
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_id = f"{timestamp}_prompt_{transcript_id}_{model.split('-')[0]}"
                comparison_dir = output_dir / comparison_id
                comparison_dir.mkdir(parents=True, exist_ok=True)

                config = ComparisonConfig(
                    comparison_type=ComparisonType.PROMPT,
                    eval_results=sorted(results, key=lambda r: r.prompt_version),
                    output_dir=comparison_dir,
                )
                file_path = self.generate_comparison(config)
                generated_files.append(file_path)

        # Group results for model comparisons (same transcript + prompt, different models)
        model_groups = defaultdict(list)
        for result in eval_results:
            if result.success:
                key = (result.transcript_id, result.prompt_version)
                model_groups[key].append(result)

        # Generate model comparisons
        for (transcript_id, prompt_version), results in model_groups.items():
            if len(results) > 1:  # Only compare if we have multiple models
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_id = f"{timestamp}_model_{transcript_id}_{prompt_version}"
                comparison_dir = output_dir / comparison_id
                comparison_dir.mkdir(parents=True, exist_ok=True)

                config = ComparisonConfig(
                    comparison_type=ComparisonType.MODEL,
                    eval_results=sorted(results, key=lambda r: r.model),
                    output_dir=comparison_dir,
                )
                file_path = self.generate_comparison(config)
                generated_files.append(file_path)

        # Generate comparison index
        if generated_files:
            self._create_comparison_index(generated_files, output_dir / "index.md")

        return generated_files

    def generate_comparison(self, config: ComparisonConfig) -> Path:
        """
        Generate a single comparison markdown file.

        Args:
            config: ComparisonConfig with comparison parameters

        Returns:
            Path to generated comparison file
        """
        # Generate markdown content
        markdown_content = self._create_comparison_markdown(
            config.comparison_type, config.eval_results
        )

        # Save to file
        comparison_file = config.output_dir / "comparison.md"
        comparison_file.write_text(markdown_content)

        return comparison_file

    def _create_comparison_markdown(
        self, comparison_type: ComparisonType, results: list[EvalResult]
    ) -> str:
        """
        Generate comparison markdown content.

        Args:
            comparison_type: Type of comparison
            results: List of EvalResult instances to compare

        Returns:
            Markdown content string
        """
        if not results:
            return "# No results to compare\n"

        # Get common metadata
        first = results[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create YAML front matter
        yaml_lines = [
            "---",
            f"comparison_type: {comparison_type.value}",
            f"timestamp: {timestamp}",
        ]

        if comparison_type == ComparisonType.PROMPT:
            yaml_lines.append(f"transcript: {first.transcript_id}")
            yaml_lines.append(f"model: {first.model}")
            yaml_lines.append(
                f"prompts_compared: [{', '.join(r.prompt_version for r in results)}]"
            )
        else:  # MODEL comparison
            yaml_lines.append(f"transcript: {first.transcript_id}")
            yaml_lines.append(f"prompt_version: {first.prompt_version}")
            yaml_lines.append(f"models_compared: [{', '.join(r.model for r in results)}]")

        yaml_lines.append("---")
        yaml_lines.append("")

        # Create header
        if comparison_type == ComparisonType.PROMPT:
            title = f"# Prompt Comparison: {first.transcript_id}"
            meta_lines = [
                f"**Transcript**: {first.transcript_id}",
                f"**Model**: {first.model}",
                f"**Compared**: {', '.join(r.prompt_version for r in results)}",
            ]
        else:
            title = f"# Model Comparison: {first.transcript_id}"
            meta_lines = [
                f"**Transcript**: {first.transcript_id}",
                f"**Prompt**: {first.prompt_version} ({first.prompt_type})",
                f"**Compared**: {', '.join(r.model for r in results)}",
            ]

        content_lines = [title, ""] + meta_lines + ["", "---", ""]

        # Add each result as a section
        for result in results:
            if comparison_type == ComparisonType.PROMPT:
                section_header = f"## Version {result.prompt_version} ({result.prompt_type})"
            else:
                section_header = f"## Model: {result.model}"

            section_lines = [
                section_header,
                "",
                f"**Tokens**: {result.tokens_input:,} input / {result.tokens_output:,} output",
                f"**Cost**: ${result.estimated_cost:.4f}",
                f"**Duration**: {result.duration_seconds:.2f}s",
                "",
                result.output_text,
                "",
                "---",
                "",
            ]

            content_lines.extend(section_lines)

        return "\n".join(yaml_lines + content_lines)

    def _create_comparison_index(
        self, comparison_files: list[Path], index_file: Path
    ) -> None:
        """
        Generate index.md listing all comparisons.

        Args:
            comparison_files: List of comparison file paths
            index_file: Path to index.md
        """
        # Group by type
        prompt_comparisons = []
        model_comparisons = []

        for file_path in comparison_files:
            # Parse comparison type from parent directory name
            parent_name = file_path.parent.name
            if "_prompt_" in parent_name:
                prompt_comparisons.append(file_path)
            elif "_model_" in parent_name:
                model_comparisons.append(file_path)

        # Generate markdown
        lines = [
            "# Eval Comparisons",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if prompt_comparisons:
            lines.extend(["## Prompt Comparisons", ""])
            for file_path in sorted(prompt_comparisons):
                # Create relative path
                rel_path = file_path.relative_to(index_file.parent)
                # Extract info from directory name
                dir_name = file_path.parent.name
                parts = dir_name.split("_")
                transcript_id = parts[2] if len(parts) > 2 else "unknown"
                model_hint = parts[3] if len(parts) > 3 else "unknown"

                lines.append(f"- [{transcript_id} ({model_hint})]({rel_path})")
            lines.append("")

        if model_comparisons:
            lines.extend(["## Model Comparisons", ""])
            for file_path in sorted(model_comparisons):
                rel_path = file_path.relative_to(index_file.parent)
                dir_name = file_path.parent.name
                parts = dir_name.split("_")
                transcript_id = parts[2] if len(parts) > 2 else "unknown"
                prompt_version = parts[3] if len(parts) > 3 else "unknown"

                lines.append(f"- [{transcript_id} ({prompt_version})]({rel_path})")
            lines.append("")

        index_file.write_text("\n".join(lines))
