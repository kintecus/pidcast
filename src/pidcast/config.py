"""Configuration constants and loading for pidcast."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


# ============================================================================
# PATH RESOLUTION
# ============================================================================


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml is)."""
    # Start from this file's directory and go up until we find pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to parent of src
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()


# ============================================================================
# EXTERNAL TOOLS CONFIGURATION
# ============================================================================

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
WHISPER_CPP_PATH = os.environ.get("WHISPER_CPP_PATH")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL")
OBSIDIAN_PATH = os.environ.get("OBSIDIAN_VAULT_PATH")


# ============================================================================
# PATHS
# ============================================================================

DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
DEFAULT_STATS_FILE = DEFAULT_TRANSCRIPTS_DIR / "transcription_stats.json"
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "config" / "prompts.yaml"
DEFAULT_MODELS_FILE = PROJECT_ROOT / "config" / "models.yaml"
DEFAULT_DIGESTS_DIR = DEFAULT_TRANSCRIPTS_DIR  # Digests saved alongside transcripts

# Deprecated: kept for backward compatibility
DEFAULT_ANALYSIS_PROMPTS_FILE = DEFAULT_PROMPTS_FILE


def get_digest_output_path(date: datetime | None = None) -> Path:
    """Get output path for digest file.

    Args:
        date: Date for digest (defaults to today)

    Returns:
        Path to digest markdown file
    """
    if date is None:
        date = datetime.now()

    filename = f"{date.strftime('%Y-%m-%d')}_podcast-digest.md"
    return DEFAULT_DIGESTS_DIR / filename


# ============================================================================
# LIBRARY CONFIGURATION
# ============================================================================


def get_config_dir() -> Path:
    """Get config directory (XDG compliant)."""
    if os.name == "posix":
        # Use XDG_CONFIG_HOME if set, otherwise use ~/.config
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / "pidcast"
        return Path.home() / ".config" / "pidcast"
    else:  # Windows
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "pidcast"
        return Path.home() / "AppData" / "Roaming" / "pidcast"


CONFIG_DIR = get_config_dir()
LIBRARY_FILE = CONFIG_DIR / "library.yaml"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
HISTORY_FILE = CONFIG_DIR / "history.json"
SYNC_LOGS_DIR = CONFIG_DIR / "logs"

# Library defaults
DEFAULT_BACKFILL_LIMIT = 5
DEFAULT_FEED_CACHE_HOURS = 1

# Sync defaults
DEFAULT_MAX_CONCURRENT_FEEDS = 5


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

AUDIO_SAMPLE_RATE = "16000"
AUDIO_CHANNELS = "1"
AUDIO_CODEC = "pcm_s16le"
AUDIO_QUALITY = "192"


# ============================================================================
# DOWNLOAD RETRY CONFIGURATION
# ============================================================================

MAX_DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 10

# Download strategy configurations
DOWNLOAD_STRATEGY_CONFIGS = {
    "android": {
        "socket_timeout": 45,
        "retries": 15,
        "fragment_retries": 15,
        "http_chunk_size": 10 * 1024 * 1024,  # 10MB
    },
    "web": {
        "socket_timeout": 60,
        "retries": 20,
        "fragment_retries": 20,
        "http_chunk_size": 5 * 1024 * 1024,  # 5MB
    },
    "mixed": {
        "socket_timeout": 60,
        "retries": 20,
        "fragment_retries": 20,
        "http_chunk_size": 10 * 1024 * 1024,  # 10MB
    },
    "ios": {
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "http_chunk_size": 10 * 1024 * 1024,  # 10MB
    },
}


# ============================================================================
# LLM ANALYSIS CONFIGURATION
# ============================================================================

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"

# Groq pricing (per 1M tokens) for cost estimation
GROQ_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "groq/compound": {"input": 0.30, "output": 0.40},  # Estimated (uses multiple models)
    "moonshotai/kimi-k2-instruct-0905": {"input": 1.00, "output": 3.00},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}

# Groq rate limits (free tier) for validation
GROQ_RATE_LIMITS = {
    "llama-3.3-70b-versatile": {
        "rpm": 30,  # Requests per minute
        "rpd": 1000,  # Requests per day
        "tpm": 12000,  # Tokens per minute
        "tpd": 100000,  # Tokens per day
    },
    "llama-3.1-70b-versatile": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 12000,
        "tpd": 100000,
    },
    "llama-3.1-8b-instant": {
        "rpm": 30,
        "rpd": 14400,
        "tpm": 6000,
        "tpd": 500000,
    },
    "mixtral-8x7b-32768": {
        "rpm": 30,
        "rpd": 14400,
        "tpm": 40000,
        "tpd": 1000000,
    },
    "groq/compound": {
        "rpm": 30,
        "rpd": 250,
        "tpm": 70000,
        "tpd": 0,  # No daily token limit specified
    },
    "moonshotai/kimi-k2-instruct-0905": {
        "rpm": 60,
        "rpd": 1000,
        "tpm": 10000,
        "tpd": 300000,
    },
    "openai/gpt-oss-120b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
    },
    "openai/gpt-oss-20b": {
        "rpm": 30,
        "rpd": 1000,
        "tpm": 8000,
        "tpd": 200000,
    },
}


# ============================================================================
# MODEL FALLBACK HELPERS
# ============================================================================


def get_fallback_models_by_tpm() -> list[str]:
    """Get list of models ordered by TPM (tokens per minute) in descending order.

    Returns:
        List of model names ordered highest to lowest TPM
    """
    models_with_tpm = [(model, limits["tpm"]) for model, limits in GROQ_RATE_LIMITS.items()]
    sorted_models = sorted(models_with_tpm, key=lambda x: x[1], reverse=True)
    return [model for model, _ in sorted_models]


def get_fallback_chain(requested_model: str | None = None) -> list[str]:
    """Get ordered list of models to try for fallback on TPM violation.

    Args:
        requested_model: Model user explicitly requested (if any)

    Returns:
        List of models in fallback order
    """
    all_models = get_fallback_models_by_tpm()

    if not requested_model or requested_model not in GROQ_RATE_LIMITS:
        return all_models

    # Return only models with higher or equal TPM than requested
    requested_tpm = GROQ_RATE_LIMITS[requested_model]["tpm"]
    return [m for m in all_models if GROQ_RATE_LIMITS[m]["tpm"] >= requested_tpm]


def get_model_tpm_limit(model: str) -> int:
    """Get TPM limit for a specific model.

    Args:
        model: Model name

    Returns:
        TPM limit, or 0 if model not found
    """
    return GROQ_RATE_LIMITS.get(model, {}).get("tpm", 0)


MAX_TRANSCRIPT_LENGTH = 120000  # Characters, roughly 30k tokens for safety
ANALYSIS_TIMEOUT = 300  # 5 minutes max for API call

# Transcript processing constants
TRANSCRIPT_TRUNCATION_BUFFER = 100
TRANSCRIPT_TRUNCATION_MIN_RATIO = 0.8
CHAR_TO_TOKEN_RATIO = 4
TOKEN_PRICING_DENOMINATOR = 1_000_000


# ============================================================================
# DATACLASSES FOR STRUCTURED DATA
# ============================================================================


@dataclass
class VideoInfo:
    """Metadata about a video or audio source."""

    title: str
    webpage_url: str = ""
    channel: str = ""
    uploader: str = ""
    duration: float = 0
    duration_string: str = ""
    view_count: int = 0
    upload_date: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoInfo":
        """Create VideoInfo from a dictionary (e.g., yt-dlp info_dict)."""
        return cls(
            title=data.get("title", "Unknown"),
            webpage_url=data.get("webpage_url", data.get("url", "")),
            channel=data.get("channel", data.get("uploader", "")),
            uploader=data.get("uploader", ""),
            duration=data.get("duration", 0),
            duration_string=data.get("duration_string", ""),
            view_count=data.get("view_count", 0),
            upload_date=data.get("upload_date", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "title": self.title,
            "webpage_url": self.webpage_url,
            "channel": self.channel,
            "uploader": self.uploader,
            "duration": self.duration,
            "duration_string": self.duration_string,
            "view_count": self.view_count,
            "upload_date": self.upload_date,
        }


@dataclass
class TranscriptionStats:
    """Statistics for a transcription run."""

    run_uid: str
    run_timestamp: str
    video_title: str
    smart_filename: str
    video_url: str
    run_duration: float
    transcription_duration: float
    audio_duration: float
    success: bool
    saved_to_obsidian: bool = False
    is_local_file: bool = False
    analysis_only: bool = False
    # Analysis metadata
    analysis_performed: bool = False
    analysis_type: str | None = None
    analysis_name: str | None = None
    analysis_duration: float = 0
    analysis_model: str | None = None
    analysis_tokens: int = 0
    analysis_cost: float = 0
    analysis_truncated: bool = False
    analysis_file: str | None = None
    # Fallback tracking
    json_mode_failed: bool = False
    used_plain_text_fallback: bool = False
    tag_extraction_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_uid": self.run_uid,
            "run_timestamp": self.run_timestamp,
            "video_title": self.video_title,
            "smart_filename": self.smart_filename,
            "video_url": self.video_url,
            "run_duration": self.run_duration,
            "transcription_duration": self.transcription_duration,
            "audio_duration": self.audio_duration,
            "success": self.success,
            "saved_to_obsidian": self.saved_to_obsidian,
            "is_local_file": self.is_local_file,
            "analysis_only": self.analysis_only,
            "analysis_performed": self.analysis_performed,
            "analysis_type": self.analysis_type,
            "analysis_name": self.analysis_name,
            "analysis_duration": self.analysis_duration,
            "analysis_model": self.analysis_model,
            "analysis_tokens": self.analysis_tokens,
            "analysis_cost": self.analysis_cost,
            "analysis_truncated": self.analysis_truncated,
            "analysis_file": self.analysis_file,
            "json_mode_failed": self.json_mode_failed,
            "used_plain_text_fallback": self.used_plain_text_fallback,
            "tag_extraction_failed": self.tag_extraction_failed,
        }


@dataclass
class PreviousTranscription:
    """Information about a previously completed transcription."""

    video_id: str
    video_title: str
    video_url: str
    run_timestamp: str
    smart_filename: str
    output_dir: Path
    analysis_performed: bool = False
    analysis_type: str | None = None

    @property
    def transcript_path(self) -> Path:
        """Full path to the transcript file."""
        return self.output_dir / self.smart_filename

    @property
    def formatted_date(self) -> str:
        """Human-readable date string."""
        try:
            import datetime

            dt = datetime.datetime.fromisoformat(self.run_timestamp)
            return dt.strftime("%B %d, %Y at %H:%M")
        except Exception:
            return self.run_timestamp


@dataclass
class AnalysisResult:
    """Result from LLM analysis."""

    analysis_text: str
    analysis_type: str
    analysis_name: str
    model: str
    provider: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    estimated_cost: float | None
    duration: float
    truncated: bool
    contextual_tags: list[str] = field(default_factory=list)


@dataclass
class PromptTemplate:
    """A prompt template for LLM analysis."""

    name: str
    description: str
    system_prompt: str
    user_prompt: str
    max_output_tokens: int = 2000


@dataclass
class PromptsConfig:
    """Configuration for analysis prompts."""

    prompts: dict[str, PromptTemplate] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptsConfig":
        """Create PromptsConfig from dictionary."""
        prompts = {}
        for key, value in data.get("prompts", {}).items():
            prompts[key] = PromptTemplate(
                name=value["name"],
                description=value["description"],
                system_prompt=value["system_prompt"],
                user_prompt=value["user_prompt"],
                max_output_tokens=value.get("max_output_tokens", 2000),
            )
        return cls(prompts=prompts)
