"""Results storage and metadata management for evals."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pidcast.exceptions import FileProcessingError


@dataclass
class EvalResult:
    """Result of an eval run."""

    run_id: str
    timestamp: datetime
    prompt_type: str
    prompt_version: str
    model: str
    transcript_id: str
    output_text: str
    tokens_input: int
    tokens_output: int
    estimated_cost: float
    duration_seconds: float
    success: bool
    error_message: str | None = None
    retry_count: int = 0  # Number of retries that occurred
    error_type: str | None = None  # Type of error (e.g., "RateLimitError")


def save_eval_result(result: EvalResult, output_dir: Path) -> None:
    """
    Save eval result as markdown + metadata JSON.

    Args:
        result: EvalResult to save
        output_dir: Base output directory (e.g., data/evals/runs/)

    Raises:
        FileProcessingError: If files cannot be written
    """
    run_dir = output_dir / result.run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise FileProcessingError(f"Failed to create run directory {run_dir}: {e}")

    # Save result markdown
    try:
        markdown_content = create_result_markdown(result)
        result_file = run_dir / "result.md"
        result_file.write_text(markdown_content)
    except Exception as e:
        raise FileProcessingError(f"Failed to write result.md: {e}")

    # Save metadata JSON
    try:
        metadata = create_metadata_json(result)
        metadata_file = run_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
    except Exception as e:
        raise FileProcessingError(f"Failed to write metadata.json: {e}")

    # If failed, also save error.json with detailed error info
    if not result.success:
        try:
            error_data = {
                "run_id": result.run_id,
                "timestamp": result.timestamp.isoformat(),
                "error_type": result.error_type or "Unknown",
                "error_message": result.error_message or "No error message",
                "retry_count": result.retry_count,
                "config": {
                    "prompt_type": result.prompt_type,
                    "prompt_version": result.prompt_version,
                    "model": result.model,
                    "transcript_id": result.transcript_id,
                },
            }
            error_file = run_dir / "error.json"
            error_file.write_text(json.dumps(error_data, indent=2))
        except Exception as e:
            # Don't fail the whole save if error.json fails
            pass


def create_result_markdown(result: EvalResult) -> str:
    """
    Generate markdown with YAML front matter.

    Args:
        result: EvalResult to format

    Returns:
        Markdown content string
    """
    # Format timestamp for YAML
    timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    # Create YAML front matter
    yaml_lines = [
        "---",
        f"run_id: {result.run_id}",
        f"timestamp: {timestamp_str}",
        f"prompt_type: {result.prompt_type}",
        f"prompt_version: {result.prompt_version}",
        f"model: {result.model}",
        f"transcript_id: {result.transcript_id}",
        f"tokens_input: {result.tokens_input}",
        f"tokens_output: {result.tokens_output}",
        f"tokens_total: {result.tokens_input + result.tokens_output}",
        f"estimated_cost: ${result.estimated_cost:.4f}",
        f"duration_seconds: {result.duration_seconds:.2f}",
        f"success: {result.success}",
    ]

    if result.error_message:
        yaml_lines.append(f"error: {result.error_message}")

    yaml_lines.append("---")
    yaml_lines.append("")

    # Add title and metadata header
    content_lines = [
        f"# Eval Result: {result.prompt_type} {result.prompt_version}",
        "",
        f"**Transcript**: {result.transcript_id}",
        f"**Model**: {result.model}",
        f"**Tokens**: {result.tokens_input:,} input / {result.tokens_output:,} output",
        f"**Cost**: ${result.estimated_cost:.4f}",
        f"**Duration**: {result.duration_seconds:.2f}s",
        "",
        "---",
        "",
    ]

    # Add LLM output
    content_lines.append(result.output_text)

    # Combine YAML front matter and content
    return "\n".join(yaml_lines + content_lines)


def create_metadata_json(result: EvalResult) -> dict:
    """
    Convert EvalResult to JSON-serializable dict.

    Args:
        result: EvalResult to convert

    Returns:
        Dictionary with metadata
    """
    metadata = asdict(result)

    # Convert datetime to ISO string
    metadata["timestamp"] = result.timestamp.isoformat()

    # Add computed fields
    metadata["tokens_total"] = result.tokens_input + result.tokens_output

    # Remove output_text from metadata (it's in result.md)
    del metadata["output_text"]

    return metadata
