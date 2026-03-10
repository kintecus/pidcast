"""YouTube audio download functionality."""

import logging
import time
from typing import Any

import yt_dlp

from .config import (
    AUDIO_CHANNELS,
    AUDIO_CODEC,
    AUDIO_QUALITY,
    AUDIO_SAMPLE_RATE,
    DOWNLOAD_STRATEGY_CONFIGS,
    MAX_DOWNLOAD_RETRIES,
    RETRY_SLEEP_SECONDS,
    VideoInfo,
)
from .exceptions import DownloadError
from .utils import log_success

logger = logging.getLogger(__name__)


# ============================================================================
# DOWNLOAD STRATEGY BUILDING
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


def build_download_strategy(
    name: str,
    config_key: str,
    format_str: str,
    player_clients: list[str],
    output_template: str,
    verbose: bool,
    po_token: str | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
) -> dict[str, Any]:
    """Build a single download strategy configuration.

    Args:
        name: Human-readable strategy name
        config_key: Key in DOWNLOAD_STRATEGY_CONFIGS
        format_str: yt-dlp format string
        player_clients: List of player clients to use
        output_template: Output file template
        verbose: Enable verbose output
        po_token: Optional PO token for restricted videos
        cookies_from_browser: Optional browser name to extract cookies from
        cookies: Optional path to Netscape format cookies file

    Returns:
        Strategy configuration dict
    """
    config = DOWNLOAD_STRATEGY_CONFIGS[config_key]
    extractor_args = {"youtube": {"player_client": player_clients}}
    if po_token:
        extractor_args["youtube"]["po_token"] = po_token

    opts = {
        "format": format_str,
        "outtmpl": output_template,
        "extractor_args": extractor_args,
        **build_ytdlp_audio_postprocessor_config(),
        "socket_timeout": config["socket_timeout"],
        "retries": config["retries"],
        "fragment_retries": config["fragment_retries"],
        "http_chunk_size": config["http_chunk_size"],
        "quiet": not verbose,
        "no_warnings": not verbose,
    }

    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies:
        opts["cookiefile"] = cookies
    if cookies_from_browser or cookies:
        opts["remote_components"] = ["ejs:github"]

    return {
        "name": name,
        "opts": opts,
    }


def build_download_strategies(
    output_template: str,
    verbose: bool,
    po_token: str | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
) -> list[dict[str, Any]]:
    """Build list of download strategies in priority order.

    Args:
        output_template: Output file template
        verbose: Enable verbose output
        po_token: Optional PO token
        cookies_from_browser: Optional browser name to extract cookies from
        cookies: Optional path to Netscape format cookies file

    Returns:
        List of strategy configurations
    """
    has_cookies = bool(cookies_from_browser or cookies)
    cookie_kwargs = {
        "cookies_from_browser": cookies_from_browser,
        "cookies": cookies,
    }
    strategies = []

    if po_token:
        strategies.append(
            build_download_strategy(
                "iOS client with PO Token",
                "ios",
                "bestaudio[ext=m4a]/bestaudio/best",
                ["ios"],
                output_template,
                verbose,
                po_token,
                **cookie_kwargs,
            )
        )

    if has_cookies:
        # When cookies are provided, prioritize web client since
        # android/ios clients don't support cookies
        strategies.extend(
            [
                build_download_strategy(
                    "Web client with cookies",
                    "web",
                    "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                    ["web"],
                    output_template,
                    verbose,
                    **cookie_kwargs,
                ),
                build_download_strategy(
                    "Android client (no cookies)",
                    "android",
                    "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                    ["android"],
                    output_template,
                    verbose,
                ),
            ]
        )
    else:
        strategies.extend(
            [
                build_download_strategy(
                    "Android client (recommended)",
                    "android",
                    "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                    ["android"],
                    output_template,
                    verbose,
                ),
                build_download_strategy(
                    "Web client with retry",
                    "web",
                    "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/worst",
                    ["web"],
                    output_template,
                    verbose,
                ),
                build_download_strategy(
                    "Mixed clients (Android + Web)",
                    "mixed",
                    "bestaudio/best",
                    ["android", "web"],
                    output_template,
                    verbose,
                ),
            ]
        )

    return strategies


# ============================================================================
# DOWNLOAD EXECUTION
# ============================================================================


def ensure_wav_file(input_file: str, verbose: bool = False) -> bool:
    """Ensure the audio file exists and is in WAV format.

    Args:
        input_file: Expected WAV file path
        verbose: Enable verbose logging

    Returns:
        True if file exists or was converted successfully
    """
    import os
    import subprocess

    from .config import FFMPEG_PATH

    if os.path.exists(input_file):
        return True

    # Check if we have a webm file instead
    webm_file = input_file.replace(".wav", ".webm")
    if os.path.exists(webm_file):
        if verbose:
            logger.info(f"Converting {webm_file} to WAV format...")
        try:
            command = [
                FFMPEG_PATH,
                "-i",
                webm_file,
                "-ar",
                AUDIO_SAMPLE_RATE,
                "-ac",
                AUDIO_CHANNELS,
                "-c:a",
                AUDIO_CODEC,
                input_file,
            ]
            subprocess.run(command, check=True, capture_output=True)
            os.remove(webm_file)  # Clean up the webm file
            if verbose:
                logger.info("Conversion successful.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting audio: {e}")
            return False

    return False


def is_retryable_error(error_msg: str) -> bool:
    """Check if the download error warrants a retry.

    Args:
        error_msg: Error message string

    Returns:
        True if error is retryable
    """
    return "Operation timed out" in error_msg or "SABR" in error_msg


def try_single_download(
    video_url: str, strategy: dict[str, Any], verbose: bool
) -> tuple[str | None, dict[str, Any] | None]:
    """Attempt a single download with the given strategy.

    Args:
        video_url: YouTube video URL
        strategy: Download strategy configuration
        verbose: Enable verbose output

    Returns:
        Tuple of (audio_file_path, info_dict) or (None, None)
    """
    with yt_dlp.YoutubeDL(strategy["opts"]) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        audio_file = "temp_audio.wav"
        if ensure_wav_file(audio_file, verbose):
            return audio_file, info_dict
    return None, None


def download_audio(
    video_url: str,
    output_template: str = "temp_audio.%(ext)s",
    verbose: bool = False,
    po_token: str | None = None,
    cookies_from_browser: str | None = None,
    cookies: str | None = None,
    chrome_profile: str | None = None,
) -> tuple[str, VideoInfo]:
    """Download audio from YouTube with auto-escalation to cookies on auth errors.

    Escalation flow (when no explicit cookies are provided):
    1. Try without cookies (fast path for public videos)
    2. On auth error, retry with cached cookies if available
    3. If cache is stale or missing, extract fresh cookies from browser
    4. Retry with fresh cookies

    When --cookies or --cookies-from-browser is explicitly passed, skips
    the no-cookie attempt and uses cookies from the start.

    Args:
        video_url: YouTube video URL
        output_template: Output file template
        verbose: Enable verbose output
        po_token: Optional PO Token for bypassing restrictions
        cookies_from_browser: Optional browser name to extract cookies from
        cookies: Optional path to Netscape format cookies file
        chrome_profile: Optional Chrome profile (directory or display name)

    Returns:
        Tuple of (audio_file_path, VideoInfo)

    Raises:
        DownloadError: If all download strategies fail
    """
    from .cookies import (
        extract_and_cache_cookies,
        get_cached_cookies,
        invalidate_cookie_cache,
        is_auth_error,
        resolve_chrome_profile,
    )

    explicit_cookies = bool(cookies_from_browser or cookies)

    # Resolve cookie file from browser if explicitly requested
    if cookies_from_browser and not cookies:
        browser, profile = _parse_browser_spec(cookies_from_browser)
        profile = profile or resolve_chrome_profile(chrome_profile)
        cookies = str(extract_and_cache_cookies(browser, profile))
        cookies_from_browser = None

    # Phase 1: Try with whatever cookies we have (or none)
    strategies = build_download_strategies(
        output_template, verbose, po_token, cookies_from_browser=None, cookies=cookies
    )
    try:
        return _run_download_strategies(video_url, strategies, verbose)
    except DownloadError as e:
        if explicit_cookies or not is_auth_error(str(e)):
            raise

    # Phase 2: Auth error without explicit cookies - try cached cookies
    logger.info("Authentication required. Checking for cached cookies...")
    browser = "chrome"
    profile = resolve_chrome_profile(chrome_profile)

    cached = get_cached_cookies(browser, profile)
    if cached:
        strategies = build_download_strategies(
            output_template, verbose, po_token, cookies=str(cached)
        )
        try:
            return _run_download_strategies(video_url, strategies, verbose)
        except DownloadError as e:
            if not is_auth_error(str(e)):
                raise
            logger.info("Cached cookies failed. Re-extracting fresh cookies...")
            invalidate_cookie_cache(browser, profile)

    # Phase 3: Extract fresh cookies and retry
    cookie_file = extract_and_cache_cookies(browser, profile)
    strategies = build_download_strategies(
        output_template, verbose, po_token, cookies=str(cookie_file)
    )
    return _run_download_strategies(video_url, strategies, verbose)


def _parse_browser_spec(spec: str) -> tuple[str, str | None]:
    """Parse a browser spec like 'chrome:Profile 1' into (browser, profile).

    Args:
        spec: Browser specification string

    Returns:
        Tuple of (browser_name, profile_or_none)
    """
    if ":" in spec:
        browser, profile = spec.split(":", 1)
        return browser, profile
    return spec, None


def _run_download_strategies(
    video_url: str, strategies: list[dict[str, Any]], verbose: bool
) -> tuple[str, VideoInfo]:
    """Execute download strategies with retry logic.

    Args:
        video_url: YouTube video URL
        strategies: List of strategy configurations
        verbose: Enable verbose output

    Returns:
        Tuple of (audio_file_path, VideoInfo)

    Raises:
        DownloadError: If all strategies fail
    """
    last_error = ""
    for strategy_idx, strategy in enumerate(strategies, 1):
        logger.debug(
            f"\n=== Attempting Strategy {strategy_idx}/{len(strategies)}: {strategy['name']} ==="
        )

        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                logger.debug(f"Attempt {attempt}/{MAX_DOWNLOAD_RETRIES}...")

                audio_file, info_dict = try_single_download(video_url, strategy, verbose)
                if audio_file and info_dict:
                    log_success(f"Download successful with {strategy['name']}!")
                    return audio_file, VideoInfo.from_dict(info_dict)

                logger.debug("Audio file not found after download, retrying...")

            except yt_dlp.utils.DownloadError as e:
                last_error = str(e)
                logger.debug(f"Download error: {last_error}")

                if is_retryable_error(last_error):
                    if attempt < MAX_DOWNLOAD_RETRIES:
                        logger.debug(f"Retrying in {RETRY_SLEEP_SECONDS} seconds...")
                        time.sleep(RETRY_SLEEP_SECONDS)
                        continue
                    logger.debug(
                        f"Max retries reached for {strategy['name']}, trying next strategy..."
                    )
                    break

                logger.debug("Trying next strategy...")
                break

            except Exception as e:
                last_error = str(e)
                logger.debug(f"Unexpected error: {type(e).__name__}: {e}")

                if attempt < MAX_DOWNLOAD_RETRIES:
                    logger.debug(f"Retrying in {RETRY_SLEEP_SECONDS} seconds...")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    break

    raise DownloadError(
        f"Failed to download audio from {video_url} after trying all strategies: {last_error}"
    )
