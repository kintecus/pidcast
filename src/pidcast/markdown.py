"""Markdown file creation with YAML front matter."""

import datetime
import logging
from pathlib import Path
from typing import Any

from .config import VideoInfo
from .utils import get_unique_filename, log_error, log_success

logger = logging.getLogger(__name__)


# ============================================================================
# YAML FRONT MATTER
# ============================================================================


def format_yaml_front_matter(front_matter: dict[str, Any]) -> str:
    """Format dictionary as YAML front matter for Markdown files.

    Args:
        front_matter: Dict with metadata key-value pairs

    Returns:
        YAML front matter string with --- delimiters
    """
    lines = ["---"]
    for key, value in front_matter.items():
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(repr(v) for v in value)}]")
        elif isinstance(value, str):
            if ":" in value or "#" in value or value.startswith(("*", "-", "[")):
                lines.append(f'{key}: "{value}"')
            else:
                lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


# ============================================================================
# MARKDOWN FILE CREATION
# ============================================================================


def create_markdown_file(
    markdown_file: str | Path,
    transcript_file: str | Path,
    video_info: VideoInfo,
    front_matter: dict[str, Any] | None = None,
    verbose: bool = False,
) -> bool:
    """Create a Markdown file with front matter and transcript.

    Args:
        markdown_file: Path to output markdown file
        transcript_file: Path to transcript text file
        video_info: Video metadata
        front_matter: Additional front matter fields
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    markdown_file = Path(markdown_file)
    transcript_file = Path(transcript_file)
    front_matter = front_matter or {}

    try:
        if not transcript_file.exists():
            log_error(f"Transcript file not found: {transcript_file}")
            return False

        with open(transcript_file, encoding="utf-8") as f:
            transcript = f.read()

        obsidian_front_matter = {
            "title": video_info.title,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "transcribed": datetime.datetime.now().isoformat(),
            "url": video_info.webpage_url,
            "duration": video_info.duration_string,
            "channel": video_info.channel,
            "tags": ["podcast", "youtube", "transcription"],
        }

        obsidian_front_matter.update(front_matter)

        front_matter_str = format_yaml_front_matter(obsidian_front_matter)

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(front_matter_str)
            f.write("\n\n")
            f.write(transcript)

        if verbose:
            log_success(f"Markdown file created: {markdown_file}")

        return True

    except Exception as e:
        log_error(f"Error creating Markdown file: {e}")
        return False


def create_analysis_markdown_file(
    analysis_results: dict[str, Any],
    source_markdown_file: str | Path,
    video_info: VideoInfo,
    output_dir: str | Path,
    verbose: bool = False,
) -> Path | None:
    """Create markdown file for LLM analysis results.

    Args:
        analysis_results: Results dict from analyze_transcript_with_llm()
        source_markdown_file: Path to original transcript markdown
        video_info: Video metadata
        output_dir: Directory for output file
        verbose: Enable verbose output

    Returns:
        Path to created file, or None on error
    """
    source_markdown_file = Path(source_markdown_file)
    output_dir = Path(output_dir)

    try:
        # Get base filename
        base_name = source_markdown_file.stem
        analysis_suffix = f"_analysis_{analysis_results['analysis_type']}"
        analysis_filename = f"{base_name}{analysis_suffix}"

        # Get unique path
        analysis_file = get_unique_filename(output_dir, analysis_filename, ".md")

        # Build front matter
        # Build tags list: static + contextual
        static_tags = [
            "analysis",
            "ai-generated",
            analysis_results["analysis_type"],
            "youtube",
        ]
        contextual_tags = analysis_results.get("contextual_tags", [])

        # Combine and deduplicate
        all_tags = static_tags + [tag for tag in contextual_tags if tag not in static_tags]

        front_matter = {
            "title": f"[Analysis] {video_info.title}",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "analyzed_at": datetime.datetime.now().isoformat(),
            "analysis_type": analysis_results["analysis_type"],
            "analysis_name": analysis_results["analysis_name"],
            "source_transcript": source_markdown_file.name,
            "source_url": video_info.webpage_url,
            "channel": video_info.channel,
            "duration": video_info.duration_string,
            "analysis_provider": analysis_results["provider"],
            "analysis_model": analysis_results["model"],
            "tokens_input": analysis_results["tokens_input"],
            "tokens_output": analysis_results["tokens_output"],
            "tokens_total": analysis_results["tokens_total"],
            "estimated_cost": analysis_results.get("estimated_cost"),
            "analysis_duration": round(analysis_results["duration"], 2),
            "transcript_truncated": analysis_results["truncated"],
            "tags": all_tags,
        }

        # Format front matter
        front_matter_str = format_yaml_front_matter(front_matter)

        # Write file
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(front_matter_str)
            f.write("\n\n")
            f.write(analysis_results["analysis_text"])

        if verbose:
            log_success(f"Analysis file created: {analysis_file}")

        return analysis_file

    except Exception as e:
        log_error(f"Error creating analysis file: {e}")
        return None
