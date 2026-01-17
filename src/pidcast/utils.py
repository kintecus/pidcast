"""Utility functions for logging, filenames, and duration formatting."""

import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR LOGGING
# ============================================================================


def log_error_to_file(
    error_type: str,
    error_details: dict,
    log_dir: Path | str | None = None,
) -> None:
    """Log errors to persistent JSON Lines file.

    Creates a structured error log entry with timestamp and details.
    Each line is a separate JSON object for easy parsing and rotation.

    Args:
        error_type: Category of error (e.g., "json_validation_error", "rate_limit_error")
        error_details: Dictionary of error-specific details to log
        log_dir: Directory for log files (defaults to data/logs/)
    """
    from .config import PROJECT_ROOT

    if log_dir is None:
        log_dir = PROJECT_ROOT / "data" / "logs"
    else:
        log_dir = Path(log_dir)

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "errors.jsonl"

    # Create log entry
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "error_type": error_type,
        **error_details
    }

    # Append to log file (JSONL format - one JSON object per line)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Don't fail the main operation if logging fails
        logger.debug(f"Failed to write to error log: {e}")


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO level.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(message)s",
        level=level,
    )


def log_success(message: str) -> None:
    """Log success message with checkmark."""
    logger.info(f"✓ {message}")


def log_error(message: str) -> None:
    """Log error message with X mark."""
    logger.error(f"✗ {message}")


def log_warning(message: str) -> None:
    """Log warning message."""
    logger.warning(f"⚠ {message}")


def log_section(title: str, width: int = 60) -> None:
    """Log section header with separator line."""
    logger.info(f"\n{'=' * width}")
    logger.info(title)
    logger.info(f"{'=' * width}")


# ============================================================================
# FILENAME UTILITIES
# ============================================================================


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove invalid characters."""
    return re.sub(r"[^\w\s-]", "", filename).strip()


# Low priority words for filename generation
LOW_PRIORITY_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "how",
    "what",
    "why",
    "when",
    "where",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
}

# Filler patterns to strip from titles
TITLE_FILLER_PATTERNS = [
    r"^Episode\s+\d+[:\s-]*",
    r"^EP\.?\s*\d+[:\s-]*",
    r"^\d+[:\s-]+",
    r"\s*[-–—]\s*Keynote Speakers?\s*",
    r"\s*[-–—]\s*Interview\s*",
    r"\s*\|\s*",
    r"\s*[-–—]\s*Part\s+\d+",
]


def create_smart_filename(title: str, max_length: int = 60, include_date: bool = True) -> str:
    """Create a smart, shortened filename from video title.

    Args:
        title: Original video title
        max_length: Maximum filename length (excluding date prefix)
        include_date: Whether to prepend YYYY-MM-DD date

    Returns:
        Sanitized filename string
    """
    cleaned_title = title
    for pattern in TITLE_FILLER_PATTERNS:
        cleaned_title = re.sub(pattern, "", cleaned_title, flags=re.IGNORECASE)

    cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip()
    words = cleaned_title.split()

    important_words = []
    regular_words = []

    for word in words:
        word_lower = word.lower()
        if word[0].isupper() or word.isupper() or word_lower not in LOW_PRIORITY_WORDS:
            important_words.append(word)
        else:
            regular_words.append(word)

    result_words = []
    current_length = 0

    for word in important_words:
        word_len = len(word) + 1
        if current_length + word_len <= max_length:
            result_words.append(word)
            current_length += word_len
        else:
            break

    for word in regular_words:
        word_len = len(word) + 1
        if current_length + word_len <= max_length:
            result_words.append(word)
            current_length += word_len
        else:
            break

    filename = "_".join(result_words)
    filename = re.sub(r"[^\w\s-]", "", filename)
    filename = re.sub(r"[-\s]+", "_", filename)

    if include_date:
        date_prefix = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_prefix}_{filename}"

    return filename


def get_unique_filename(directory: str | Path, base_filename: str, extension: str = ".md") -> Path:
    """Get a unique filename by adding version suffix if file exists.

    Args:
        directory: Directory path
        base_filename: Base filename without extension
        extension: File extension (including dot)

    Returns:
        Path to unique file
    """
    directory = Path(directory)
    filepath = directory / f"{base_filename}{extension}"

    if not filepath.exists():
        return filepath

    version = 2
    while True:
        versioned_filename = f"{base_filename}_v{version}{extension}"
        filepath = directory / versioned_filename
        if not filepath.exists():
            return filepath
        version += 1


# ============================================================================
# DURATION FORMATTING
# ============================================================================


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1m 30s" or "45.23 seconds"
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = int(seconds // 60)
    rem_seconds = int(seconds % 60)
    return f"{minutes}m {rem_seconds}s"


# ============================================================================
# JSON FILE I/O HELPERS
# ============================================================================


def load_json_file(filepath: str | Path, default: Any = None) -> Any:
    """Safely load JSON file with default fallback.

    Args:
        filepath: Path to JSON file
        default: Value to return if file doesn't exist

    Returns:
        Parsed JSON data or default value
    """
    filepath = Path(filepath)
    try:
        if not filepath.exists():
            return default
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON in {filepath}: {e}")
        return default
    except Exception as e:
        log_error(f"Error reading {filepath}: {e}")
        return default


def save_json_file(filepath: str | Path, data: Any, indent: int = 2, verbose: bool = False) -> bool:
    """Safely save data to JSON file.

    Args:
        filepath: Path to JSON file
        data: Data to serialize
        indent: JSON indentation level
        verbose: Whether to print success message

    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

        if verbose:
            log_success(f"Saved: {filepath}")
        return True
    except Exception as e:
        log_error(f"Error saving {filepath}: {e}")
        return False


# ============================================================================
# INPUT VALIDATION
# ============================================================================

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac"}
YOUTUBE_URL_PATTERN = re.compile(r"https?://(www\.)?(youtube\.com|youtu\.be)/")
YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def validate_input_source(source: str) -> tuple[str, bool]:
    """Validate and classify input source.

    Args:
        source: Input source (URL or file path)

    Returns:
        Tuple of (source, is_local_file)

    Raises:
        ValueError: If source is invalid
    """
    # Check if it's a local file
    if os.path.exists(source):
        ext = os.path.splitext(source)[1].lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {ext}. "
                f"Supported: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
            )
        return source, True

    # Check if it's a YouTube URL
    if not YOUTUBE_URL_PATTERN.match(source):
        raise ValueError(
            f"Invalid input: '{source}' is neither a local file nor a valid YouTube URL"
        )

    return source, False


def extract_youtube_video_id(url: str) -> str | None:
    """Extract normalized video ID from various YouTube URL formats.

    Handles:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://www.youtube.com/watch?v=VIDEO_ID&feature=shared
    - https://youtu.be/VIDEO_ID
    - https://youtu.be/VIDEO_ID?si=TRACKING_PARAM
    - https://www.youtube.com/live/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID

    Args:
        url: YouTube URL in any supported format

    Returns:
        11-character video ID or None if not a valid YouTube URL
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().replace("www.", "").replace("m.", "")

        if host == "youtu.be":
            # Short URL: https://youtu.be/VIDEO_ID
            video_id = parsed.path.lstrip("/").split("/")[0]
        elif host == "youtube.com":
            if parsed.path.startswith("/watch"):
                # Standard: /watch?v=VIDEO_ID
                params = parse_qs(parsed.query)
                video_id = params.get("v", [None])[0]
            elif parsed.path.startswith("/live/"):
                # Live: /live/VIDEO_ID
                video_id = parsed.path.replace("/live/", "").split("/")[0]
            elif parsed.path.startswith("/embed/"):
                # Embed: /embed/VIDEO_ID
                video_id = parsed.path.replace("/embed/", "").split("/")[0]
            elif parsed.path.startswith("/v/"):
                # Old format: /v/VIDEO_ID
                video_id = parsed.path.replace("/v/", "").split("/")[0]
            else:
                return None
        else:
            return None

        # Validate video ID format (11 chars, alphanumeric + _ -)
        if video_id and YOUTUBE_VIDEO_ID_PATTERN.match(video_id):
            return video_id
        return None
    except Exception:
        return None


def find_existing_transcription(
    stats_file: Path,
    input_source: str,
    output_dir: Path,
) -> "PreviousTranscription | None":
    """Search stats file for a previous transcription of the same video.

    Args:
        stats_file: Path to transcription_stats.json
        input_source: Current input (URL or file path)
        output_dir: Current output directory (to locate transcript file)

    Returns:
        PreviousTranscription if found, None otherwise
    """
    from .config import PreviousTranscription

    # Load stats
    stats = load_json_file(stats_file, default=[])
    if not stats:
        return None

    # Determine what to match on
    if os.path.exists(input_source):
        # Local file: match by absolute path
        abs_path = os.path.abspath(input_source)
        for entry in reversed(stats):  # Most recent first
            entry_url = entry.get("video_url", "")
            if entry.get("is_local_file") and os.path.abspath(entry_url) == abs_path:
                if entry.get("success") and entry.get("smart_filename"):
                    return PreviousTranscription(
                        video_id="",
                        video_title=entry.get("video_title", "Unknown"),
                        video_url=entry_url,
                        run_timestamp=entry.get("run_timestamp", ""),
                        smart_filename=entry.get("smart_filename", ""),
                        output_dir=output_dir,
                        analysis_performed=entry.get("analysis_performed", False),
                        analysis_type=entry.get("analysis_type"),
                    )
    else:
        # YouTube URL: match by normalized video ID
        current_video_id = extract_youtube_video_id(input_source)
        if not current_video_id:
            return None

        for entry in reversed(stats):  # Most recent first
            entry_url = entry.get("video_url", "")
            entry_video_id = extract_youtube_video_id(entry_url)

            if entry_video_id == current_video_id:
                if entry.get("success") and entry.get("smart_filename"):
                    return PreviousTranscription(
                        video_id=current_video_id,
                        video_title=entry.get("video_title", "Unknown"),
                        video_url=entry_url,
                        run_timestamp=entry.get("run_timestamp", ""),
                        smart_filename=entry.get("smart_filename", ""),
                        output_dir=output_dir,
                        analysis_performed=entry.get("analysis_performed", False),
                        analysis_type=entry.get("analysis_type"),
                    )

    return None


def is_interactive() -> bool:
    """Check if running in interactive mode (TTY)."""
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


# ============================================================================
# CLEANUP UTILITIES
# ============================================================================


def cleanup_temp_files(audio_file: str | Path, verbose: bool = False) -> None:
    """Clean up temporary audio files.

    Args:
        audio_file: Base audio file path
        verbose: Enable verbose logging
    """
    audio_file = str(audio_file)
    for ext in [".wav", ".webm", ".m4a", ".mp3"]:
        temp_file = audio_file.replace(".wav", ext)
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                if verbose:
                    logger.debug(f"Cleaned up: {temp_file}")
            except Exception as e:
                if verbose:
                    log_warning(f"Could not remove {temp_file}: {e}")
