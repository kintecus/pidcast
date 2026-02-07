"""Utility functions for logging, filenames, and duration formatting."""

import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from .config import PreviousTranscription, TranscriptionStats

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

    log_dir = PROJECT_ROOT / "data" / "logs" if log_dir is None else Path(log_dir)

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "errors.jsonl"

    # Create log entry
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "error_type": error_type,
        **error_details,
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
            if (
                entry.get("is_local_file")
                and os.path.abspath(entry_url) == abs_path
                and entry.get("success")
                and entry.get("smart_filename")
            ):
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

            if (
                entry_video_id == current_video_id
                and entry.get("success")
                and entry.get("smart_filename")
            ):
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


# ============================================================================
# STATISTICS
# ============================================================================


def save_statistics(stats_file: Path, stats: "TranscriptionStats", verbose: bool = False) -> bool:
    """Save transcription statistics to a JSON file.

    Args:
        stats_file: Path to stats file
        stats: Statistics to save (TranscriptionStats object)
        verbose: Enable verbose output

    Returns:
        True if successful
    """
    from .config import TranscriptionStats  # noqa: F401

    existing_stats = load_json_file(stats_file, default=[])
    existing_stats.append(stats.to_dict())
    return save_json_file(stats_file, existing_stats, verbose=verbose)


# ============================================================================
# FUZZY MATCHING
# ============================================================================


def fuzzy_match_key(
    input_str: str, available_keys: list[str], normalize: bool = False
) -> str | None:
    """Fuzzy match input string against available keys.

    Matching rules (in priority order):
    1. Exact match (case-insensitive)
    2. Key starts with input (case-insensitive)
    3. Key contains input (case-insensitive)
    4. If normalize=True, also match against normalized versions (removes /, -, _)

    Args:
        input_str: User input string
        available_keys: List of valid keys to match against
        normalize: If True, also try matching with special chars removed

    Returns:
        Matched key or None if no match found
    """
    input_lower = input_str.lower()

    # Exact match
    for key in available_keys:
        if key.lower() == input_lower:
            return key

    # Starts with
    matches = [key for key in available_keys if key.lower().startswith(input_lower)]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Multiple matches - check if one is an exact prefix match
        exact_prefix = [m for m in matches if m.lower() == input_lower]
        if exact_prefix:
            return exact_prefix[0]
        # Return shortest match (most specific)
        return min(matches, key=len)

    # Contains
    matches = [key for key in available_keys if input_lower in key.lower()]
    if len(matches) == 1:
        return matches[0]

    # Normalized matching (for model names with slashes/hyphens)
    if normalize:

        def normalize_str(s: str) -> str:
            """Remove special chars for fuzzy matching."""
            return re.sub(r"[/\-_.]", "", s.lower())

        input_norm = normalize_str(input_str)

        # Try normalized exact match
        for key in available_keys:
            if normalize_str(key) == input_norm:
                return key

        # Try normalized contains
        matches = [key for key in available_keys if input_norm in normalize_str(key)]
        if len(matches) == 1:
            return matches[0]

    return None


def suggest_closest_match(
    input_str: str, available_keys: list[str], threshold: int = 3
) -> str | None:
    """Suggest closest match using edit distance.

    Args:
        input_str: User input string
        available_keys: List of valid keys
        threshold: Maximum edit distance to suggest

    Returns:
        Suggested key or None
    """
    import difflib

    # Use difflib for similarity matching
    matches = difflib.get_close_matches(input_str, available_keys, n=1, cutoff=0.6)
    return matches[0] if matches else None


# ============================================================================
# DISCOVERABILITY - LIST AVAILABLE OPTIONS
# ============================================================================


def resolve_analysis_type(user_input: str, prompts_file: Path | None = None) -> str:
    """Resolve user input to valid analysis type with fuzzy matching.

    Args:
        user_input: User's input for analysis type
        prompts_file: Path to prompts YAML file (uses default if None)

    Returns:
        Resolved analysis type key

    Raises:
        ValueError: If no match found
    """
    import yaml

    from .config import DEFAULT_PROMPTS_FILE

    prompts_file = prompts_file or DEFAULT_PROMPTS_FILE

    try:
        with open(prompts_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        available_keys = list(config.get("prompts", {}).keys())

        # Try fuzzy match
        matched = fuzzy_match_key(user_input, available_keys)
        if matched:
            return matched

        # No match - suggest alternatives
        suggestion = suggest_closest_match(user_input, available_keys)
        if suggestion:
            raise ValueError(
                f"Unknown analysis type: '{user_input}'. Did you mean '{suggestion}'?\n"
                f"Use -L to list all available types."
            )
        else:
            raise ValueError(
                f"Unknown analysis type: '{user_input}'.\nUse -L to list all available types."
            )

    except FileNotFoundError:
        log_warning(f"Prompts file not found: {prompts_file}. Using input as-is.")
        return user_input
    except yaml.YAMLError:
        log_warning("Invalid YAML in prompts file. Using input as-is.")
        return user_input


def resolve_model_name(user_input: str, models_file: Path | None = None) -> str:
    """Resolve user input to valid model name with fuzzy matching.

    Supports common aliases:
    - llama33, llama3.3 -> llama-3.3-70b-versatile
    - llama31, llama8 -> llama-3.1-8b-instant
    - gpt120 -> openai/gpt-oss-120b
    - gpt20 -> openai/gpt-oss-20b
    - compound -> groq/compound

    Args:
        user_input: User's input for model name
        models_file: Path to models YAML file (uses default if None)

    Returns:
        Resolved model name

    Raises:
        ValueError: If no match found
    """
    import yaml

    from .config import DEFAULT_MODELS_FILE

    models_file = models_file or DEFAULT_MODELS_FILE

    try:
        with open(models_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        available_keys = list(config.get("models", {}).keys())

        # Common aliases for easier typing
        aliases = {
            "llama33": "llama-3.3-70b-versatile",
            "llama3.3": "llama-3.3-70b-versatile",
            "llama31": "llama-3.1-8b-instant",
            "llama8": "llama-3.1-8b-instant",
            "llama3.1": "llama-3.1-8b-instant",
            "gpt120": "openai/gpt-oss-120b",
            "gpt20": "openai/gpt-oss-20b",
            "compound": "groq/compound",
        }

        input_lower = user_input.lower()
        if input_lower in aliases:
            return aliases[input_lower]

        # Try fuzzy match with normalization (handles slashes/hyphens)
        matched = fuzzy_match_key(user_input, available_keys, normalize=True)
        if matched:
            return matched

        # No match - suggest alternatives
        suggestion = suggest_closest_match(user_input, available_keys)
        if suggestion:
            raise ValueError(
                f"Unknown model: '{user_input}'. Did you mean '{suggestion}'?\n"
                f"Use -M to list all available models."
            )
        else:
            raise ValueError(
                f"Unknown model: '{user_input}'.\nUse -M to list all available models."
            )

    except FileNotFoundError:
        log_warning(f"Models file not found: {models_file}. Using input as-is.")
        return user_input
    except yaml.YAMLError:
        log_warning("Invalid YAML in models file. Using input as-is.")
        return user_input


def list_available_analyses(prompts_file: Path | None = None) -> None:
    """Display available analysis types from prompts.yaml.

    Args:
        prompts_file: Path to prompts YAML file (uses default if None)
    """
    import yaml

    from .config import DEFAULT_PROMPTS_FILE

    prompts_file = prompts_file or DEFAULT_PROMPTS_FILE

    try:
        with open(prompts_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        prompts = config.get("prompts", {})

        if not prompts:
            log_error(f"No prompts found in {prompts_file}")
            return

        print("\nAvailable Analysis Types:")
        print("=" * 70)

        for key, data in prompts.items():
            name = data.get("name", key)
            desc = data.get("description", "No description")
            print(f"  {key:20s} {name}")
            print(f"  {' ' * 20} {desc}")
            print()

        print("Usage: pidcast URL -a TYPE")
        print("Example: pidcast URL -a executive_summary")
        print()

    except FileNotFoundError:
        log_error(f"Prompts file not found: {prompts_file}")
    except yaml.YAMLError as e:
        log_error(f"Invalid YAML in {prompts_file}: {e}")
    except Exception as e:
        log_error(f"Error reading prompts: {e}")


def list_available_models(models_file: Path | None = None) -> None:
    """Display available Groq models from models.yaml.

    Args:
        models_file: Path to models YAML file (uses default if None)
    """
    import yaml

    from .config import DEFAULT_MODELS_FILE

    models_file = models_file or DEFAULT_MODELS_FILE

    try:
        with open(models_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        fallback_chain = config.get("fallback_chain", [])
        models = config.get("models", {})

        if not models:
            log_error(f"No models found in {models_file}")
            return

        print("\nAvailable Models (Quality Order):")
        print("=" * 80)

        for model_id in fallback_chain:
            if model_id not in models:
                continue

            model = models[model_id]
            display_name = model.get("display_name", model_id)
            tpm = model.get("limits", {}).get("tpm", 0)
            tpd = model.get("limits", {}).get("tpd", 0)

            tpm_str = f"{tpm:,}" if tpm else "N/A"
            tpd_str = f"{tpd:,}" if tpd else "unlimited"

            print(f"  {model_id}")
            print(f"    Name: {display_name}")
            print(f"    Limits: {tpm_str} tokens/min, {tpd_str} tokens/day")
            print()

        print("Usage: pidcast URL -m MODEL")
        print(f"Example: pidcast URL -m {fallback_chain[0] if fallback_chain else 'MODEL_ID'}")
        print()

    except FileNotFoundError:
        log_error(f"Models file not found: {models_file}")
    except yaml.YAMLError as e:
        log_error(f"Invalid YAML in {models_file}: {e}")
    except Exception as e:
        log_error(f"Error reading models: {e}")
