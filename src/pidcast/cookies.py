"""Cookie management for YouTube authentication.

Handles cookie extraction from browsers, caching to avoid repeated
keychain prompts, and staleness detection for automatic refresh.
"""

import json
import logging
import os
import platform
import time
from pathlib import Path

from .config import COOKIE_CACHE_DIR, COOKIE_CACHE_MAX_AGE_HOURS

logger = logging.getLogger(__name__)

# Chrome Local State file location per platform
_CHROME_LOCAL_STATE_PATHS = {
    "Darwin": Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Local State",
    "Linux": Path.home() / ".config" / "google-chrome" / "Local State",
    "Windows": Path(os.environ.get("LOCALAPPDATA", ""))
    / "Google"
    / "Chrome"
    / "User Data"
    / "Local State",
}


def _cache_file_for_browser(browser: str, profile: str | None = None) -> Path:
    """Get the cache file path for a given browser/profile combo."""
    suffix = f"_{profile}" if profile else ""
    safe_name = f"{browser}{suffix}".replace(" ", "_").replace("/", "_")
    return COOKIE_CACHE_DIR / f"cookies_{safe_name}.txt"


def get_cached_cookies(browser: str, profile: str | None = None) -> Path | None:
    """Return cached cookie file path if it exists and is fresh.

    Args:
        browser: Browser name (e.g., 'chrome')
        profile: Optional browser profile directory name

    Returns:
        Path to cached cookie file, or None if missing/stale
    """
    cache_path = _cache_file_for_browser(browser, profile)
    if not cache_path.exists():
        return None

    age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
    if age_hours > COOKIE_CACHE_MAX_AGE_HOURS:
        logger.info(
            f"Cookie cache expired ({age_hours:.1f}h old, max {COOKIE_CACHE_MAX_AGE_HOURS}h)"
        )
        return None

    logger.info(f"Using cached cookies ({age_hours:.1f}h old)")
    return cache_path


def extract_and_cache_cookies(browser: str, profile: str | None = None) -> Path:
    """Extract cookies from browser and save to cache.

    Args:
        browser: Browser name (e.g., 'chrome')
        profile: Optional browser profile directory name

    Returns:
        Path to the cached cookie file

    Raises:
        RuntimeError: If cookie extraction fails
    """
    from yt_dlp.cookies import extract_cookies_from_browser

    if platform.system() == "Darwin":
        logger.info(
            f"Extracting cookies from {browser}"
            + (f" (profile: {profile})" if profile else "")
            + ". macOS may prompt for Keychain access - please allow it."
        )
    else:
        logger.info(
            f"Extracting cookies from {browser}"
            + (f" (profile: {profile})" if profile else "")
            + "..."
        )

    try:
        jar = extract_cookies_from_browser(browser, profile=profile)
    except Exception as e:
        raise RuntimeError(f"Failed to extract cookies from {browser}: {e}") from e

    COOKIE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file_for_browser(browser, profile)
    jar.save(str(cache_path))
    logger.info(f"Cached {len(jar)} cookies to {cache_path}")
    return cache_path


def invalidate_cookie_cache(browser: str, profile: str | None = None) -> None:
    """Remove cached cookie file to force re-extraction on next use."""
    cache_path = _cache_file_for_browser(browser, profile)
    if cache_path.exists():
        cache_path.unlink()
        logger.debug(f"Invalidated cookie cache: {cache_path}")


def is_auth_error(error_msg: str) -> bool:
    """Check if a download error is an authentication/bot-check error."""
    return "Sign in to confirm" in error_msg or "confirm you're not a bot" in error_msg


def list_chrome_profiles() -> dict[str, dict[str, str]]:
    """List available Chrome profiles with display names.

    Returns:
        Dict mapping directory names to profile metadata:
        {
            "Default": {"display_name": "Personal", "yt_dlp_value": "Default"},
            "Profile 1": {"display_name": "Work", "yt_dlp_value": "Profile 1"},
        }
    """
    local_state_path = _CHROME_LOCAL_STATE_PATHS.get(platform.system())
    if not local_state_path or not local_state_path.exists():
        return {}

    try:
        with open(local_state_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read Chrome Local State: {e}")
        return {}

    info_cache = data.get("profile", {}).get("info_cache", {})
    profiles = {}
    for dir_name, profile_data in info_cache.items():
        profiles[dir_name] = {
            "display_name": profile_data.get("name", dir_name),
            "yt_dlp_value": dir_name,
        }
    return profiles


def resolve_chrome_profile(profile_input: str | None) -> str | None:
    """Resolve a profile input (display name or dir name) to a directory name.

    Accepts either the Chrome directory name ("Profile 1") or the display
    name ("Work"). Returns the directory name for yt-dlp.

    Args:
        profile_input: User-provided profile identifier, or None

    Returns:
        Chrome profile directory name, or None if input is None
    """
    if not profile_input:
        return None

    profiles = list_chrome_profiles()

    # Direct match on directory name
    if profile_input in profiles:
        return profile_input

    # Match on display name (case-insensitive)
    for dir_name, meta in profiles.items():
        if meta["display_name"].lower() == profile_input.lower():
            return dir_name

    # Fuzzy substring match on display name
    for dir_name, meta in profiles.items():
        if profile_input.lower() in meta["display_name"].lower():
            return dir_name

    available = ", ".join("{} ({})".format(m["display_name"], d) for d, m in profiles.items())
    logger.warning(f"Chrome profile '{profile_input}' not found. Available: {available}")
    return profile_input  # Pass through as-is, let yt-dlp handle it
