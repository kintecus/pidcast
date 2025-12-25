"""Utility functions for logging, filenames, and duration formatting."""

import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# Configure module logger
logger = logging.getLogger(__name__)


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
