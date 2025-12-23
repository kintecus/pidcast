import argparse
import datetime
import json
import os
import re
import subprocess
import time
import uuid
from typing import Any

import yt_dlp
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ============================================================================
# EXTERNAL TOOLS CONFIGURATION
# ============================================================================
FFMPEG_PATH = "ffmpeg"  # Ensure ffmpeg is installed and in PATH
WHISPER_CPP_PATH = "/Users/ostaps/Code/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/Users/ostaps/Code/whisper.cpp/models/ggml-base.en.bin"
OBSIDIAN_PATH = "/Users/ostaps/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/03 - RESOURCES/Podcasts"

# ============================================================================
# PATHS
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRANSCRIPTS_DIR = os.path.join(SCRIPT_DIR, "transcripts")
DEFAULT_STATS_FILE = os.path.join(DEFAULT_TRANSCRIPTS_DIR, "transcription_stats.json")
DEFAULT_ANALYSIS_PROMPTS_FILE = os.path.join(SCRIPT_DIR, "analysis_prompts.json")

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
}

MAX_TRANSCRIPT_LENGTH = 120000  # Characters, roughly 30k tokens for safety
ANALYSIS_TIMEOUT = 300  # 5 minutes max for API call

# Transcript processing constants
TRANSCRIPT_TRUNCATION_BUFFER = 100
TRANSCRIPT_TRUNCATION_MIN_RATIO = 0.8
CHAR_TO_TOKEN_RATIO = 4
TOKEN_PRICING_DENOMINATOR = 1_000_000


# ============================================================================
# LOGGING HELPERS
# ============================================================================


def log_verbose(message: str, verbose: bool = False) -> None:
    """Log message if verbose mode enabled."""
    if verbose:
        print(message)


def log_success(message: str, verbose: bool = False) -> None:
    """Log success message with checkmark."""
    if verbose:
        print(f"✓ {message}")


def log_error(message: str) -> None:
    """Log error message with X mark."""
    print(f"✗ {message}")


def log_section(title: str, width: int = 60) -> None:
    """Log section header with separator line."""
    print(f"\n{'=' * width}")
    print(title)
    print(f"{'=' * width}")


# ============================================================================
# FILENAME UTILITIES
# ============================================================================


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove invalid characters."""
    return re.sub(r"[^\w\s-]", "", filename).strip()


def create_smart_filename(title: str, max_length: int = 60, include_date: bool = True) -> str:
    """Create a smart, shortened filename from video title."""
    filler_patterns = [
        r"^Episode\s+\d+[:\s-]*",
        r"^EP\.?\s*\d+[:\s-]*",
        r"^\d+[:\s-]+",
        r"\s*[-–—]\s*Keynote Speakers?\s*",
        r"\s*[-–—]\s*Interview\s*",
        r"\s*\|\s*",
        r"\s*[-–—]\s*Part\s+\d+",
    ]

    cleaned_title = title
    for pattern in filler_patterns:
        cleaned_title = re.sub(pattern, "", cleaned_title, flags=re.IGNORECASE)

    cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip()
    words = cleaned_title.split()

    important_words = []
    regular_words = []
    low_priority = {
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

    for word in words:
        word_lower = word.lower()
        if word[0].isupper() or word.isupper() or word_lower not in low_priority:
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


def get_unique_filename(directory: str, base_filename: str, extension: str = ".md") -> str:
    """Get a unique filename by adding version suffix if file exists."""
    filepath = os.path.join(directory, f"{base_filename}{extension}")

    if not os.path.exists(filepath):
        return filepath

    version = 2
    while True:
        versioned_filename = f"{base_filename}_v{version}{extension}"
        filepath = os.path.join(directory, versioned_filename)
        if not os.path.exists(filepath):
            return filepath
        version += 1


# ============================================================================
# AUDIO PROCESSING
# ============================================================================


def build_ffmpeg_audio_conversion_command(
    input_file: str, output_file: str, overwrite: bool = False
) -> list[str]:
    """Build FFmpeg command for audio conversion to 16kHz mono WAV.

    Args:
        input_file: Path to input audio file
        output_file: Path to output WAV file
        overwrite: Whether to overwrite existing output file

    Returns:
        FFmpeg command as list of arguments
    """
    command = [FFMPEG_PATH]
    if overwrite:
        command.append("-y")
    command.extend(
        [
            "-i",
            input_file,
            "-ar",
            AUDIO_SAMPLE_RATE,
            "-ac",
            AUDIO_CHANNELS,
            "-c:a",
            AUDIO_CODEC,
            output_file,
        ]
    )
    return command


def ensure_wav_file(input_file: str, verbose: bool = False) -> bool:
    """Ensure the audio file exists and is in WAV format."""
    if os.path.exists(input_file):
        return True

    # Check if we have a webm file instead
    webm_file = input_file.replace(".wav", ".webm")
    if os.path.exists(webm_file):
        if verbose:
            print(f"Converting {webm_file} to WAV format...")
        try:
            command = build_ffmpeg_audio_conversion_command(webm_file, input_file)
            subprocess.run(command, check=True, capture_output=True)
            os.remove(webm_file)  # Clean up the webm file
            if verbose:
                print("Conversion successful.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting audio: {e}")
            return False

    return False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    minutes = int(seconds // 60)
    rem_seconds = int(seconds % 60)
    return f"{minutes}m {rem_seconds}s"


# ============================================================================
# DOWNLOAD MANAGEMENT
# ============================================================================


def build_ytdlp_audio_postprocessor_config() -> dict[str, Any]:
    """Build yt-dlp postprocessor configuration for audio extraction.

    Returns:
        Dict with postprocessors and postprocessor_args for yt-dlp options
    """
    return {
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": AUDIO_QUALITY,
            }
        ],
        "postprocessor_args": [
            "-ar",
            AUDIO_SAMPLE_RATE,
            "-ac",
            AUDIO_CHANNELS,
            "-c:a",
            AUDIO_CODEC,
        ],
    }


def download_audio_with_retry(
    video_url: str, output_template: str, verbose: bool = False, po_token: str | None = None
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Download audio from YouTube with retry logic and multiple fallback strategies.

    Args:
        video_url: YouTube video URL
        output_template: Output file template
        verbose: Enable verbose output
        po_token: Optional PO Token for bypassing restrictions (format: "client.type+TOKEN")

    Returns:
        tuple: (audio_file_path, video_info_dict) or (None, None) on failure
    """

    # Strategy 1: Android client (most reliable without PO token)
    # Strategy 2: Web client with aggressive settings
    # Strategy 3: iOS client (requires PO token for some videos)
    config_android = DOWNLOAD_STRATEGY_CONFIGS["android"]
    strategies = [
        {
            "name": "Android client (recommended)",
            "opts": {
                "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                "outtmpl": output_template,
                "extractor_args": {"youtube": {"player_client": ["android"]}},
                **build_ytdlp_audio_postprocessor_config(),
                "socket_timeout": config_android["socket_timeout"],
                "retries": config_android["retries"],
                "fragment_retries": config_android["fragment_retries"],
                "http_chunk_size": config_android["http_chunk_size"],
                "quiet": not verbose,
                "no_warnings": not verbose,
            },
        },
        {
            "name": "Web client with retry",
            "opts": {
                "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/worst",
                "outtmpl": output_template,
                "extractor_args": {"youtube": {"player_client": ["web"]}},
                **build_ytdlp_audio_postprocessor_config(),
                "socket_timeout": DOWNLOAD_STRATEGY_CONFIGS["web"]["socket_timeout"],
                "retries": DOWNLOAD_STRATEGY_CONFIGS["web"]["retries"],
                "fragment_retries": DOWNLOAD_STRATEGY_CONFIGS["web"]["fragment_retries"],
                "http_chunk_size": DOWNLOAD_STRATEGY_CONFIGS["web"]["http_chunk_size"],
                "quiet": not verbose,
                "no_warnings": not verbose,
            },
        },
        {
            "name": "Mixed clients (Android + Web)",
            "opts": {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
                **build_ytdlp_audio_postprocessor_config(),
                "socket_timeout": DOWNLOAD_STRATEGY_CONFIGS["mixed"]["socket_timeout"],
                "retries": DOWNLOAD_STRATEGY_CONFIGS["mixed"]["retries"],
                "fragment_retries": DOWNLOAD_STRATEGY_CONFIGS["mixed"]["fragment_retries"],
                "http_chunk_size": DOWNLOAD_STRATEGY_CONFIGS["mixed"]["http_chunk_size"],
                "quiet": not verbose,
                "no_warnings": not verbose,
            },
        },
    ]

    # Add iOS strategy if PO token is provided
    if po_token:
        config_ios = DOWNLOAD_STRATEGY_CONFIGS["ios"]
        ios_strategy = {
            "name": "iOS client with PO Token",
            "opts": {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": output_template,
                "extractor_args": {"youtube": {"player_client": ["ios"], "po_token": po_token}},
                **build_ytdlp_audio_postprocessor_config(),
                "socket_timeout": config_ios["socket_timeout"],
                "retries": config_ios["retries"],
                "fragment_retries": config_ios["fragment_retries"],
                "http_chunk_size": config_ios["http_chunk_size"],
                "quiet": not verbose,
                "no_warnings": not verbose,
            },
        }
        # Insert iOS as first strategy if we have a token
        strategies.insert(0, ios_strategy)

    for strategy_idx, strategy in enumerate(strategies, 1):
        if verbose:
            print(
                f"\n=== Attempting Strategy {strategy_idx}/{len(strategies)}: {strategy['name']} ==="
            )

        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                if verbose:
                    print(f"Attempt {attempt}/{MAX_DOWNLOAD_RETRIES}...")

                with yt_dlp.YoutubeDL(strategy["opts"]) as ydl:
                    info_dict = ydl.extract_info(video_url, download=True)

                    # Check if audio file was created
                    audio_file = "temp_audio.wav"
                    if ensure_wav_file(audio_file, verbose):
                        if verbose:
                            print(f"✓ Download successful with {strategy['name']}!")
                        return audio_file, info_dict
                    else:
                        if verbose:
                            print("Audio file not found after download, retrying...")

            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                if verbose:
                    print(f"✗ Download error: {error_msg}")

                # Check for specific errors that indicate we should try next strategy
                if "Operation timed out" in error_msg or "SABR" in error_msg:
                    if attempt < MAX_DOWNLOAD_RETRIES:
                        if verbose:
                            print(f"Retrying in {RETRY_SLEEP_SECONDS} seconds...")
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    else:
                        if verbose:
                            print(
                                f"Max retries reached for {strategy['name']}, trying next strategy..."
                            )
                        break
                else:
                    # For other errors, try next strategy immediately
                    if verbose:
                        print("Trying next strategy...")
                    break

            except Exception as e:
                if verbose:
                    print(f"✗ Unexpected error: {type(e).__name__}: {e}")

                if attempt < MAX_DOWNLOAD_RETRIES:
                    if verbose:
                        print(f"Retrying in {RETRY_SLEEP_SECONDS} seconds...")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    break

    # All strategies failed
    return None, None


# ============================================================================
# TRANSCRIPTION
# ============================================================================


def run_whisper_transcription(
    audio_file: str, whisper_model: str, output_format: str, output_file: str, verbose: bool = False
) -> bool:
    """
    Runs whisper transcription.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Build base command
        command = [WHISPER_CPP_PATH, "-f", audio_file, "-m", whisper_model, "-t", "8"]

        # Add output format flag
        if output_format == "txt":
            command.append("--output-txt")
        elif output_format == "vtt":
            command.append("--output-vtt")
        elif output_format == "srt":
            command.append("--output-srt")
        elif output_format == "json":
            command.append("--output-json")

        # Add output file path
        if output_file:
            command.extend(["-of", output_file])

        if verbose:
            print("Running Whisper command:", " ".join(command))

        # Don't use capture_output to avoid buffer truncation on long transcriptions
        # Let output go directly to stdout/stderr or suppress entirely
        if verbose:
            # Show all output in verbose mode
            subprocess.run(command, check=True)
        else:
            # Suppress output in non-verbose mode but don't capture (avoids truncation)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if verbose:
            print("✓ Transcription completed successfully.")

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Whisper transcription failed with exit code {e.returncode}")
        if hasattr(e, "stderr") and e.stderr:
            print(f"Error output: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print(f"✗ Whisper binary not found at: {WHISPER_CPP_PATH}")
        print("Please check the WHISPER_CPP_PATH configuration.")
        return False


# ============================================================================
# MARKDOWN FILE CREATION
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


def create_markdown_file(
    markdown_file: str,
    transcript_file: str,
    video_info: dict[str, Any],
    front_matter: dict[str, Any],
    verbose: bool = False,
) -> bool:
    """Create a Markdown file with front matter and transcript."""
    try:
        if not os.path.exists(transcript_file):
            print(f"✗ Transcript file not found: {transcript_file}")
            return False

        with open(transcript_file, encoding="utf-8") as f:
            transcript = f.read()

        obsidian_front_matter = {
            "title": video_info.get("title", "Untitled"),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "transcribed": datetime.datetime.now().isoformat(),
            "url": video_info.get("webpage_url", video_info.get("url", "")),
            "duration": video_info.get("duration_string", ""),
            "channel": video_info.get("channel", video_info.get("uploader", "")),
            "tags": ["podcast", "youtube", "transcription"],
        }

        obsidian_front_matter.update(front_matter)

        front_matter_str = format_yaml_front_matter(obsidian_front_matter)

        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(front_matter_str)
            f.write("\n\n")
            f.write(transcript)

        if verbose:
            print(f"✓ Markdown file created: {markdown_file}")

        return True

    except Exception as e:
        print(f"✗ Error creating Markdown file: {e}")
        return False


def estimate_transcription_time(
    stats_file: str, audio_duration: float, max_records: int = 100
) -> float | None:
    """Estimate transcription time based on historical data."""
    try:
        if not os.path.exists(stats_file):
            return None

        with open(stats_file, encoding="utf-8") as f:
            existing_stats = json.load(f)

        successful_runs = [
            s
            for s in existing_stats
            if s.get("success") and "transcription_duration" in s and "audio_duration" in s
        ]

        if not successful_runs:
            return None

        recent_runs = successful_runs[-max_records:]

        ratios = []
        for run in recent_runs:
            trans_duration = run["transcription_duration"]
            audio_dur = run["audio_duration"]
            if audio_dur > 0:
                ratios.append(trans_duration / audio_dur)

        if not ratios:
            return None

        avg_ratio = sum(ratios) / len(ratios)
        estimated_time = audio_duration * avg_ratio

        return estimated_time

    except Exception:
        return None


# ============================================================================
# JSON FILE I/O HELPERS
# ============================================================================


def load_json_file(filepath: str, default: Any = None) -> Any:
    """Safely load JSON file with default fallback.

    Args:
        filepath: Path to JSON file
        default: Value to return if file doesn't exist

    Returns:
        Parsed JSON data or default value
    """
    try:
        if not os.path.exists(filepath):
            return default
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in {filepath}: {e}")
        return default
    except Exception as e:
        print(f"✗ Error reading {filepath}: {e}")
        return default


def save_json_file(filepath: str, data: Any, indent: int = 2, verbose: bool = False) -> bool:
    """Safely save data to JSON file.

    Args:
        filepath: Path to JSON file
        data: Data to serialize
        indent: JSON indentation level
        verbose: Whether to print success message

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

        if verbose:
            print(f"✓ Saved: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error saving {filepath}: {e}")
        return False


# ============================================================================
# STATISTICS & FILE MANAGEMENT
# ============================================================================


def save_statistics(stats_file: str, stats: dict[str, Any], verbose: bool = False) -> bool:
    """Save transcription statistics to a JSON file."""
    existing_stats = load_json_file(stats_file, default=[])
    existing_stats.append(stats)
    return save_json_file(stats_file, existing_stats, verbose=verbose)


def cleanup_temp_files(audio_file: str, verbose: bool = False):
    """Clean up temporary audio files."""
    for ext in [".wav", ".webm", ".m4a", ".mp3"]:
        temp_file = audio_file.replace(".wav", ext)
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                if verbose:
                    print(f"Cleaned up: {temp_file}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not remove {temp_file}: {e}")


# ============================================================================
# LLM ANALYSIS
# ============================================================================


def create_default_prompts_file(prompts_file: str, verbose: bool = False) -> bool:
    """
    Create a default analysis prompts configuration file.

    Args:
        prompts_file: Path where to create the file
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    default_prompts = {
        "prompts": {
            "summary": {
                "name": "Summary",
                "description": "Generate a concise summary of the transcript",
                "system_prompt": "You are an expert at summarizing video transcripts. Create clear, concise summaries that capture the main points and key insights.",
                "user_prompt": "Please create a comprehensive summary of the following transcript.\n\nTitle: {title}\nChannel: {channel}\nDuration: {duration}\n\n# Transcript\n\n{transcript}\n\n# Instructions\n\nProvide a summary with:\n1. A one-paragraph overview (2-3 sentences)\n2. Key points (3-7 bullet points)\n3. Main takeaways or conclusions\n\nBe specific and use concrete details from the transcript.",
                "max_output_tokens": 2000,
            },
            "key_points": {
                "name": "Key Points Extraction",
                "description": "Extract main ideas and important points",
                "system_prompt": "You are an expert at analyzing content and extracting key insights. Focus on identifying the most important ideas, facts, and actionable takeaways.",
                "user_prompt": "Extract the key points from this transcript.\n\nTitle: {title}\nChannel: {channel}\n\n# Transcript\n\n{transcript}\n\n# Instructions\n\nProvide:\n1. Main Topics: List 3-5 main topics discussed\n2. Key Points: 8-12 bullet points of the most important ideas\n3. Notable Quotes: 2-3 memorable or impactful quotes\n4. Actionable Insights: Any practical advice or takeaways",
                "max_output_tokens": 2500,
            },
            "action_items": {
                "name": "Action Items & Recommendations",
                "description": "Extract actionable recommendations and next steps",
                "system_prompt": "You are an expert at identifying actionable advice and practical recommendations from content. Focus on what viewers can do with this information.",
                "user_prompt": "Identify action items and recommendations from this transcript.\n\nTitle: {title}\n\n# Transcript\n\n{transcript}\n\n# Instructions\n\nProvide:\n1. Direct Action Items: Specific steps or actions mentioned\n2. Recommended Resources: Tools, books, links, or resources referenced\n3. Best Practices: Any guidelines or principles discussed\n4. Implementation Tips: How to apply these insights",
                "max_output_tokens": 2000,
            },
        }
    }

    if save_json_file(prompts_file, default_prompts, verbose=verbose):
        if verbose:
            print("  Available analysis types: summary, key_points, action_items")
        return True
    return False


def load_analysis_prompts(prompts_file: str, verbose: bool = False) -> dict[str, Any] | None:
    """
    Load and validate analysis prompt templates from JSON file.

    Args:
        prompts_file: Path to prompts JSON configuration
        verbose: Enable verbose output

    Returns:
        Dict containing prompts configuration, or None on error
    """
    # If file doesn't exist and it's the default path, create it
    if not os.path.exists(prompts_file):
        if prompts_file == DEFAULT_ANALYSIS_PROMPTS_FILE:
            if verbose:
                print(f"Prompts file not found at {prompts_file}")
                print("Creating default prompts file...")
            if not create_default_prompts_file(prompts_file, verbose):
                return None
        else:
            print(f"✗ Prompts file not found: {prompts_file}")
            return None

    # Load the file
    config = load_json_file(prompts_file)
    if config is None:
        return None

    # Validate structure
    if "prompts" not in config:
        print("✗ Invalid prompts file: missing 'prompts' key")
        return None

    # Validate each prompt has required fields
    required_fields = ["name", "description", "system_prompt", "user_prompt"]
    for prompt_id, prompt_data in config["prompts"].items():
        missing_fields = [field for field in required_fields if field not in prompt_data]
        if missing_fields:
            print(f"✗ Prompt '{prompt_id}' missing required fields: {', '.join(missing_fields)}")
            return None

    if verbose:
        print(f"✓ Loaded {len(config['prompts'])} analysis prompt(s) from {prompts_file}")

    return config


def substitute_prompt_variables(prompt_template: str, variables: dict[str, str]) -> str:
    """
    Replace variables in prompt template with actual values.

    Args:
        prompt_template: Template string with {variable} placeholders
        variables: Dict mapping variable names to values

    Returns:
        Prompt with variables substituted
    """
    result = prompt_template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def truncate_transcript(
    transcript: str, max_length: int = MAX_TRANSCRIPT_LENGTH, verbose: bool = False
) -> tuple[str, bool]:
    """
    Truncate transcript if it exceeds maximum length.

    Args:
        transcript: Full transcript text
        max_length: Maximum character length
        verbose: Enable verbose output

    Returns:
        tuple: (truncated_transcript, was_truncated)
    """
    if len(transcript) <= max_length:
        return transcript, False

    # Try to find last sentence boundary before max_length
    truncate_point = max_length - TRANSCRIPT_TRUNCATION_BUFFER  # Leave room for indicator
    last_period = transcript.rfind(". ", 0, truncate_point)

    if last_period > max_length * TRANSCRIPT_TRUNCATION_MIN_RATIO:  # At least 80% of target
        truncate_point = last_period + 1

    truncated = transcript[:truncate_point] + "\n\n[... transcript truncated for length ...]"

    if verbose:
        print(f"⚠ Transcript truncated from {len(transcript)} to {len(truncated)} characters")

    return truncated, True


def estimate_analysis_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float | None:
    """
    Estimate cost for Groq API call based on token counts.

    Args:
        prompt_tokens: Estimated input tokens
        completion_tokens: Estimated output tokens
        model: Model name

    Returns:
        Estimated cost in USD, or None if pricing unknown
    """
    if model not in GROQ_PRICING:
        return None

    pricing = GROQ_PRICING[model]
    input_cost = (prompt_tokens / TOKEN_PRICING_DENOMINATOR) * pricing["input"]
    output_cost = (completion_tokens / TOKEN_PRICING_DENOMINATOR) * pricing["output"]
    return input_cost + output_cost


def analyze_transcript_with_llm(
    transcript: str,
    video_info: dict[str, Any],
    analysis_type: str,
    prompts_config: dict[str, Any],
    api_key: str,
    model: str = DEFAULT_GROQ_MODEL,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """
    Analyze transcript using Groq LLM API.

    Args:
        transcript: Full transcript text
        video_info: Video metadata dict
        analysis_type: Which prompt template to use
        prompts_config: Loaded prompts configuration
        api_key: Groq API key
        model: Model name to use
        verbose: Enable verbose output

    Returns:
        Dict with analysis results and metadata, or None on error:
        {
            'analysis_text': str,
            'analysis_type': str,
            'analysis_name': str,
            'model': str,
            'provider': 'groq',
            'tokens_input': int,
            'tokens_output': int,
            'tokens_total': int,
            'estimated_cost': float,
            'duration': float,
            'truncated': bool
        }
    """
    # Validate analysis_type exists in prompts_config
    if analysis_type not in prompts_config["prompts"]:
        available_types = ", ".join(prompts_config["prompts"].keys())
        print(f"✗ Analysis type '{analysis_type}' not found")
        print(f"  Available types: {available_types}")
        return None

    # Get template
    template = prompts_config["prompts"][analysis_type]

    # Prepare variables
    transcript_text, was_truncated = truncate_transcript(transcript, verbose=verbose)
    variables = {
        "transcript": transcript_text,
        "title": video_info.get("title", "Unknown"),
        "channel": video_info.get("channel", video_info.get("uploader", "Unknown")),
        "duration": video_info.get("duration_string", "Unknown"),
        "url": video_info.get("webpage_url", video_info.get("url", "")),
    }

    # Substitute variables
    system_prompt = substitute_prompt_variables(template["system_prompt"], variables)
    user_prompt = substitute_prompt_variables(template["user_prompt"], variables)

    # Estimate tokens (rough: 4 chars = 1 token)
    estimated_input_tokens = (len(system_prompt) + len(user_prompt)) // CHAR_TO_TOKEN_RATIO
    estimated_output_tokens = template.get("max_output_tokens", 2000)
    estimated_cost = estimate_analysis_cost(estimated_input_tokens, estimated_output_tokens, model)

    if verbose or estimated_cost:
        print("\nAnalysis Estimation:")
        print(f"  Type: {template['name']}")
        print(f"  Model: {model}")
        print(f"  Estimated input tokens: ~{estimated_input_tokens}")
        if estimated_cost:
            print(f"  Estimated cost: ${estimated_cost:.4f}")

    # Call API
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=template.get("max_output_tokens", 2000),
            temperature=0.3,  # Fairly deterministic for analysis
            timeout=ANALYSIS_TIMEOUT,
        )

        duration = time.time() - start_time

        # Extract results
        analysis_text = response.choices[0].message.content
        tokens_input = response.usage.prompt_tokens
        tokens_output = response.usage.completion_tokens
        tokens_total = response.usage.total_tokens
        actual_cost = estimate_analysis_cost(tokens_input, tokens_output, model)

        return {
            "analysis_text": analysis_text,
            "analysis_type": analysis_type,
            "analysis_name": template["name"],
            "model": model,
            "provider": "groq",
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "tokens_total": tokens_total,
            "estimated_cost": actual_cost,
            "duration": duration,
            "truncated": was_truncated,
        }

    except ImportError:
        print("✗ 'groq' package not installed. Run: pip install groq")
        return None
    except Exception as e:
        print(f"✗ Error during LLM analysis: {type(e).__name__}: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return None


def create_analysis_markdown_file(
    analysis_results: dict[str, Any],
    source_markdown_file: str,
    video_info: dict[str, Any],
    output_dir: str,
    verbose: bool = False,
) -> str | None:
    """
    Create markdown file for LLM analysis results.

    Args:
        analysis_results: Results dict from analyze_transcript_with_llm()
        source_markdown_file: Path to original transcript markdown
        video_info: Video metadata
        output_dir: Directory for output file
        verbose: Enable verbose output

    Returns:
        Path to created file, or None on error
    """
    try:
        # Get base filename
        base_name = os.path.basename(source_markdown_file).replace(".md", "")
        analysis_suffix = f"_analysis_{analysis_results['analysis_type']}"
        analysis_filename = f"{base_name}{analysis_suffix}"

        # Get unique path
        analysis_file = get_unique_filename(output_dir, analysis_filename, ".md")

        # Build front matter
        front_matter = {
            "title": f"[Analysis] {video_info.get('title', 'Unknown')}",
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "analyzed_at": datetime.datetime.now().isoformat(),
            "analysis_type": analysis_results["analysis_type"],
            "analysis_name": analysis_results["analysis_name"],
            "source_transcript": os.path.basename(source_markdown_file),
            "source_url": video_info.get("webpage_url", video_info.get("url", "")),
            "channel": video_info.get("channel", video_info.get("uploader", "")),
            "duration": video_info.get("duration_string", ""),
            "analysis_provider": analysis_results["provider"],
            "analysis_model": analysis_results["model"],
            "tokens_input": analysis_results["tokens_input"],
            "tokens_output": analysis_results["tokens_output"],
            "tokens_total": analysis_results["tokens_total"],
            "estimated_cost": analysis_results.get("estimated_cost"),
            "analysis_duration": round(analysis_results["duration"], 2),
            "transcript_truncated": analysis_results["truncated"],
            "tags": ["analysis", "ai-generated", analysis_results["analysis_type"], "youtube"],
        }

        # Format front matter
        front_matter_str = format_yaml_front_matter(front_matter)

        # Write file
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(front_matter_str)
            f.write("\n\n")
            f.write(analysis_results["analysis_text"])

        if verbose:
            print(f"✓ Analysis file created: {analysis_file}")

        return analysis_file

    except Exception as e:
        print(f"✗ Error creating analysis file: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return None


# ============================================================================
# LOCAL FILE PROCESSING
# ============================================================================


def process_local_file(
    input_file: str, output_dir: str, verbose: bool = False
) -> tuple[str | None, dict[str, Any] | None]:
    """
    Process a local audio file for transcription.

    Args:
        input_file: Path to local audio file
        output_dir: Directory to store processed files
        verbose: Enable verbose output

    Returns:
        tuple: (audio_file_path, info_dict) or (None, None) on failure
    """
    if not os.path.exists(input_file):
        print(f"✗ File not found: {input_file}")
        return None, None

    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)

    # Create info dict similar to yt-dlp output
    info_dict = {
        "title": name.replace("_", " "),
        "webpage_url": f"file://{os.path.abspath(input_file)}",
        "channel": "Local File",
        "uploader": "User",
        "duration": 0,  # Will try to get actual duration
        "duration_string": "Unknown",
        "view_count": 0,
        "upload_date": datetime.datetime.fromtimestamp(os.path.getmtime(input_file)).strftime(
            "%Y%m%d"
        ),
    }

    # Try to get duration using ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            info_dict["duration"] = duration
            info_dict["duration_string"] = format_duration(duration)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not determine duration: {e}")

    # Convert to 16kHz mono WAV if needed
    output_wav = os.path.join(output_dir, f"{name}_16k.wav")

    if verbose:
        print(f"Converting {filename} to 16kHz mono WAV...")

    try:
        command = build_ffmpeg_audio_conversion_command(input_file, output_wav, overwrite=True)

        if verbose:
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        return output_wav, info_dict

    except subprocess.CalledProcessError as e:
        print(f"✗ Error converting audio: {e}")
        return None, None


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Automate audio transcription with Whisper (YouTube URL or local file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"
        %(prog)s "/path/to/audio/file.mp3"
        %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" --output_dir ./transcripts --verbose
        """,
    )
    parser.add_argument("input_source", help="YouTube video URL or path to local audio file")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Output directory for Markdown files (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--save_to_obsidian",
        action="store_true",
        help=f"Save to Obsidian vault at: {OBSIDIAN_PATH}",
    )
    parser.add_argument("--whisper_model", default=WHISPER_MODEL, help="Path to Whisper model file")
    parser.add_argument(
        "--output_format",
        default="otxt",
        help="Whisper output format (txt, vtt, srt, json). Prefix with 'o' for original filename.",
    )
    parser.add_argument(
        "--front_matter", default="{}", help="JSON string for Markdown front matter"
    )
    parser.add_argument(
        "--stats_file",
        default=None,
        help=f"File to store statistics (default: {DEFAULT_STATS_FILE})",
    )
    parser.add_argument(
        "--keep_transcript",
        action="store_true",
        help="Keep the .txt transcript file alongside the .md file",
    )
    parser.add_argument(
        "--po_token",
        default=None,
        help="PO Token for bypassing YouTube restrictions (format: 'client.type+TOKEN'). See https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # LLM Analysis arguments
    analysis_group = parser.add_argument_group("LLM Analysis Options")
    analysis_group.add_argument(
        "--analyze", action="store_true", help="Enable LLM-based transcript analysis"
    )
    analysis_group.add_argument(
        "--analysis_type",
        default="summary",
        help="Analysis type/prompt template to use (default: summary)",
    )
    analysis_group.add_argument(
        "--analysis_prompts_file",
        default=None,
        help=f"Path to analysis prompts JSON file (default: {DEFAULT_ANALYSIS_PROMPTS_FILE})",
    )
    analysis_group.add_argument(
        "--groq_api_key",
        default=None,
        help="Groq API key (default: GROQ_API_KEY environment variable)",
    )
    analysis_group.add_argument(
        "--groq_model",
        default=DEFAULT_GROQ_MODEL,
        help=f"Groq model to use for analysis (default: {DEFAULT_GROQ_MODEL})",
    )
    analysis_group.add_argument(
        "--skip_analysis_on_error",
        action="store_true",
        help="Continue if analysis fails instead of aborting",
    )

    args = parser.parse_args()

    # Set defaults for output_dir and stats_file
    if args.output_dir is None:
        args.output_dir = DEFAULT_TRANSCRIPTS_DIR
    if args.stats_file is None:
        args.stats_file = DEFAULT_STATS_FILE
    if args.analysis_prompts_file is None:
        args.analysis_prompts_file = DEFAULT_ANALYSIS_PROMPTS_FILE

    # Initialize tracking variables
    run_uid = str(uuid.uuid4())
    run_timestamp = datetime.datetime.now().isoformat()
    start_time = time.time()
    audio_file = None
    success = False
    is_local_file = os.path.exists(args.input_source)

    if args.verbose:
        print(f"\n{'=' * 60}")
        print("Transcription Tool")
        print(f"{'=' * 60}")
        print(f"Input: {args.input_source}")
        print(f"Type: {'Local File' if is_local_file else 'YouTube URL'}")
        print(f"Run ID: {run_uid}")
        print(f"Timestamp: {run_timestamp}")
        print(f"{'=' * 60}\n")

    try:
        if args.save_to_obsidian:
            output_dir = OBSIDIAN_PATH
            if args.verbose:
                print(f"Saving to Obsidian vault: {output_dir}")
        else:
            output_dir = args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        if is_local_file:
            print("Processing local file...")
            audio_file, info_dict = process_local_file(
                args.input_source,
                output_dir,  # Use output_dir for temporary converted file
                args.verbose,
            )
        else:
            # Download audio with retry logic
            print("Downloading audio from YouTube...")
            audio_file, info_dict = download_audio_with_retry(
                args.input_source, "temp_audio.%(ext)s", args.verbose, args.po_token
            )

        if audio_file is None or info_dict is None:
            print("\n✗ Failed to process input.")
            return

        video_title = info_dict.get("title", "unknown_video")
        smart_filename = create_smart_filename(video_title, max_length=60, include_date=True)

        if not is_local_file:
            print("\n✓ Audio downloaded successfully!")
        else:
            print("\n✓ Local file processed successfully!")

        print(f"Title: {video_title}")
        if args.verbose:
            print(f"Smart filename: {smart_filename}")

        # Verify audio file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Estimate transcription time
        audio_duration = info_dict.get("duration", 0)
        estimated_time = estimate_transcription_time(args.stats_file, audio_duration)

        # Run Whisper transcription
        print("\nTranscribing audio with Whisper...")
        if estimated_time:
            print(
                f"Estimated transcription time: ~{format_duration(estimated_time)} (based on historical data)"
            )
        else:
            if audio_duration > 0:
                print(f"Audio duration: {format_duration(audio_duration)}")

        transcription_start = time.time()
        temp_whisper_output = os.path.join(output_dir, f"temp_transcript_{uuid.uuid4().hex[:8]}")
        output_format = args.output_format.replace("o", "")

        if not run_whisper_transcription(
            audio_file, args.whisper_model, output_format, temp_whisper_output, args.verbose
        ):
            print("\n✗ Transcription failed.")
            return

        transcription_duration = time.time() - transcription_start

        # Create Markdown file
        print("\nCreating Markdown file...")
        markdown_file = get_unique_filename(output_dir, smart_filename, ".md")
        transcript_file = f"{temp_whisper_output}.txt"

        try:
            front_matter = json.loads(args.front_matter)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in front_matter, using empty dict: {e}")
            front_matter = {}

        if not create_markdown_file(
            markdown_file, transcript_file, info_dict, front_matter, args.verbose
        ):
            print("\n✗ Failed to create Markdown file.")
            return

        # LLM Analysis
        analysis_file = None
        analysis_duration = 0
        analysis_performed = False
        analysis_metadata = {}

        if args.analyze:
            print("\n" + "=" * 60)
            print("Starting LLM Analysis")
            print("=" * 60)

            # Get API key with fallback
            groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY")

            if not groq_api_key:
                error_msg = "✗ Groq API key not found. Set GROQ_API_KEY environment variable or use --groq_api_key"
                if args.skip_analysis_on_error:
                    print(error_msg)
                    print("Skipping analysis (--skip_analysis_on_error enabled)")
                else:
                    print(error_msg)
                    print("\nTo skip analysis on errors, use --skip_analysis_on_error flag")
                    return
            else:
                # Load prompts
                prompts_config = load_analysis_prompts(args.analysis_prompts_file, args.verbose)

                if not prompts_config:
                    error_msg = (
                        f"✗ Failed to load analysis prompts from: {args.analysis_prompts_file}"
                    )
                    if args.skip_analysis_on_error:
                        print(error_msg)
                        print("Skipping analysis")
                    else:
                        print(error_msg)
                        return
                else:
                    # Read transcript from markdown file
                    try:
                        with open(markdown_file, encoding="utf-8") as f:
                            content = f.read()
                        # Extract transcript (everything after front matter)
                        parts = content.split("---", 2)
                        transcript_text = parts[2].strip() if len(parts) >= 3 else content

                        # Analyze
                        analysis_start = time.time()
                        analysis_results = analyze_transcript_with_llm(
                            transcript_text,
                            info_dict,
                            args.analysis_type,
                            prompts_config,
                            groq_api_key,
                            args.groq_model,
                            args.verbose,
                        )

                        if analysis_results:
                            # Create analysis markdown
                            analysis_file = create_analysis_markdown_file(
                                analysis_results, markdown_file, info_dict, output_dir, args.verbose
                            )

                            if analysis_file:
                                analysis_performed = True
                                analysis_duration = time.time() - analysis_start
                                analysis_metadata = {
                                    "analysis_type": analysis_results["analysis_type"],
                                    "analysis_name": analysis_results["analysis_name"],
                                    "model": analysis_results["model"],
                                    "tokens_total": analysis_results["tokens_total"],
                                    "estimated_cost": analysis_results.get("estimated_cost"),
                                    "truncated": analysis_results["truncated"],
                                }

                                print(
                                    f"\n✓ Analysis completed in {format_duration(analysis_duration)}"
                                )
                                print(f"✓ Analysis file: {analysis_file}")
                                if analysis_results.get("estimated_cost"):
                                    print(f"✓ Cost: ${analysis_results['estimated_cost']:.4f}")
                            else:
                                error_msg = "✗ Failed to create analysis file"
                                if not args.skip_analysis_on_error:
                                    print(error_msg)
                                    return
                                print(error_msg + " (continuing)")
                        else:
                            error_msg = "✗ Analysis failed"
                            if not args.skip_analysis_on_error:
                                print(error_msg)
                                return
                            print(error_msg + " (continuing)")

                    except Exception as e:
                        error_msg = f"✗ Error during analysis: {e}"
                        if not args.skip_analysis_on_error:
                            print(error_msg)
                            if args.verbose:
                                import traceback

                                traceback.print_exc()
                            return
                        print(error_msg + " (continuing)")

        # Optionally keep transcript file
        if not args.keep_transcript and os.path.exists(transcript_file):
            os.remove(transcript_file)
            if args.verbose:
                print(f"✓ Cleaned up temporary transcript file: {transcript_file}")
        elif args.keep_transcript and os.path.exists(transcript_file):
            if args.verbose:
                print(f"✓ Kept transcript file: {transcript_file}")

        # Store statistics
        end_time = time.time()
        duration = end_time - start_time
        stats = {
            "run_uid": run_uid,
            "run_timestamp": run_timestamp,
            "video_title": video_title,
            "smart_filename": os.path.basename(markdown_file),
            "video_url": args.input_source,
            "run_duration": duration,
            "transcription_duration": transcription_duration,
            "audio_duration": audio_duration,
            "success": True,
            "saved_to_obsidian": args.save_to_obsidian,
            "is_local_file": is_local_file,
            # Analysis metadata
            "analysis_performed": analysis_performed,
            "analysis_type": analysis_metadata.get("analysis_type") if analysis_performed else None,
            "analysis_name": analysis_metadata.get("analysis_name") if analysis_performed else None,
            "analysis_duration": analysis_duration if analysis_performed else 0,
            "analysis_model": analysis_metadata.get("model") if analysis_performed else None,
            "analysis_tokens": analysis_metadata.get("tokens_total") if analysis_performed else 0,
            "analysis_cost": analysis_metadata.get("estimated_cost") if analysis_performed else 0,
            "analysis_truncated": analysis_metadata.get("truncated")
            if analysis_performed
            else False,
            "analysis_file": os.path.basename(analysis_file) if analysis_file else None,
        }

        save_statistics(args.stats_file, stats, args.verbose)

        success = True
        print(f"\n{'=' * 60}")
        print("✓ Transcription completed successfully!")
        print(f"{'=' * 60}")
        print(f"Markdown file: {markdown_file}")
        print(f"Total duration: {format_duration(duration)}")
        print(f"Transcription duration: {format_duration(transcription_duration)}")

        if estimated_time:
            diff = transcription_duration - estimated_time
            diff_abs = abs(diff)
            percentage_diff = (diff_abs / estimated_time) * 100
            direction = "slower" if diff > 0 else "faster"
            print(
                f"Estimation accuracy: {percentage_diff:.1f}% ({format_duration(diff_abs)}) {direction} than estimated"
            )
        if args.save_to_obsidian:
            print("✓ Saved to Obsidian vault")
        print(f"{'=' * 60}\n")

    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user.")

    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {type(e).__name__}: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

    finally:
        # Clean up temporary audio files
        if audio_file:
            # For local files, we created a temporary 16k wav that needs cleanup
            # For YouTube, we have the downloaded file
            if is_local_file:
                if os.path.exists(audio_file) and "_16k.wav" in audio_file:
                    try:
                        os.remove(audio_file)
                        if args.verbose:
                            print(f"Cleaned up temporary file: {audio_file}")
                    except Exception as e:
                        if args.verbose:
                            print(f"Warning: Could not remove {audio_file}: {e}")
            else:
                cleanup_temp_files(audio_file, args.verbose)

        # Save failure statistics if needed
        if not success:
            end_time = time.time()
            duration = end_time - start_time
            stats = {
                "run_uid": run_uid,
                "run_timestamp": run_timestamp,
                "video_url": args.input_source,
                "run_duration": duration,
                "success": False,
                "is_local_file": is_local_file,
            }
            save_statistics(args.stats_file, stats, False)


if __name__ == "__main__":
    main()
