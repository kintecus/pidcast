"""Resolve Apple Podcasts episode URLs to direct audio URLs."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

import feedparser

from .config import VideoInfo
from .exceptions import ApplePodcastsResolutionError, FeedFetchError, FeedParseError

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Podcast episode parsed from an RSS feed."""

    guid: str
    title: str
    description: str
    pub_date: datetime
    duration: int | None
    audio_url: str


def _parse_pub_date(entry) -> datetime:
    for attr in ("published_parsed", "updated_parsed"):
        value = getattr(entry, attr, None) or entry.get(attr)
        if value:
            try:
                return datetime(*value[:6])
            except Exception:
                pass
    return datetime.now()


def _parse_duration(entry) -> int | None:
    raw = entry.get("itunes_duration") or entry.get("duration")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    try:
        parts = str(raw).split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(raw)
    except Exception:
        return None


def _extract_audio_url(entry) -> str:
    for enclosure in entry.get("enclosures", []) or []:
        mime = enclosure.get("type", "")
        if mime.startswith("audio/") or mime.startswith("video/"):
            return enclosure.get("href") or enclosure.get("url", "")
    enclosures = entry.get("enclosures", []) or []
    if enclosures:
        return enclosures[0].get("href") or enclosures[0].get("url", "")
    media = getattr(entry, "media_content", None)
    if isinstance(media, list) and media:
        return media[0].get("url", "")
    return ""


def _parse_feed_episodes(feed_url: str, verbose: bool = False) -> list[Episode]:
    """Fetch an RSS feed and return parsed episodes.

    feedparser fetches the URL itself; we don't need a separate HTTP client.
    """
    if verbose:
        logger.info(f"Fetching RSS feed: {feed_url}")

    feed = feedparser.parse(feed_url)

    if feed.bozo and hasattr(feed, "bozo_exception"):
        msg = str(feed.bozo_exception).lower()
        if "not well-formed" in msg or "no element found" in msg:
            raise FeedParseError(f"Invalid feed format: {feed.bozo_exception}")

    if not getattr(feed, "feed", None):
        raise FeedFetchError(f"Feed fetch returned no data: {feed_url}")

    episodes: list[Episode] = []
    for entry in feed.entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        guid = entry.get("id") or entry.get("guid") or entry.get("link") or ""
        audio_url = _extract_audio_url(entry)
        if not audio_url:
            continue
        episodes.append(
            Episode(
                guid=guid,
                title=title,
                description=entry.get("summary")
                or entry.get("description")
                or entry.get("subtitle", ""),
                pub_date=_parse_pub_date(entry),
                duration=_parse_duration(entry),
                audio_url=audio_url,
            )
        )

    if not episodes:
        raise FeedParseError(f"Feed has no valid episodes: {feed_url}")

    return episodes


ITUNES_LOOKUP_URL = "https://itunes.apple.com/lookup"


def is_apple_podcasts_url(url: str) -> bool:
    """Check whether *url* points to Apple Podcasts."""
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return False
    return host in ("podcasts.apple.com", "itunes.apple.com")


def parse_apple_podcasts_url(url: str) -> tuple[str, str | None]:
    """Extract collection ID and optional track (episode) ID from an Apple Podcasts URL.

    Expected path patterns:
        /us/podcast/<slug>/id<collectionId>?i=<trackId>
        /podcast/id<collectionId>?i=<trackId>

    Returns:
        (collection_id, track_id) where track_id may be None for show-level URLs.

    Raises:
        ApplePodcastsResolutionError: If the URL cannot be parsed.
    """
    parsed = urllib.parse.urlparse(url)

    # Extract collection ID from path segment like "id1234567890"
    match = re.search(r"/id(\d+)", parsed.path)
    if not match:
        raise ApplePodcastsResolutionError(f"Could not extract podcast ID from URL: {url}")
    collection_id = match.group(1)

    # Extract track (episode) ID from query param ?i=
    query = urllib.parse.parse_qs(parsed.query)
    track_id = query.get("i", [None])[0]

    return collection_id, track_id


def _itunes_fetch(params: dict) -> list[dict]:
    """Make a single iTunes Lookup API request and return the results list."""
    qs = urllib.parse.urlencode(params)
    url = f"{ITUNES_LOOKUP_URL}?{qs}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pidcast/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        raise ApplePodcastsResolutionError(f"iTunes Lookup API request failed: {e}") from e

    return data.get("results", [])


def lookup_itunes(collection_id: str, track_id: str | None) -> dict:
    """Call the iTunes Lookup API and return show + episode metadata.

    Always looks up the collection (show) first to get the feedUrl, since the
    iTunes Lookup API does not support looking up episode-level ``?i=`` IDs
    directly.  If a *track_id* is provided, a second request attempts to
    fetch episode-specific metadata (trackName, releaseDate) which is used
    later for matching inside the RSS feed.

    Returns:
        Dict with keys like feedUrl, collectionName, trackName, releaseDate, etc.

    Raises:
        ApplePodcastsResolutionError: On API failure or missing data.
    """
    # Step 1: always look up the collection to get feedUrl
    collection_results = _itunes_fetch({"id": collection_id})
    if not collection_results:
        raise ApplePodcastsResolutionError(
            "iTunes Lookup API returned no results for this podcast."
        )

    collection_info = collection_results[0]

    if not collection_info.get("feedUrl"):
        raise ApplePodcastsResolutionError(
            "No RSS feed URL found for this podcast. The podcast may not have a public feed."
        )

    # Step 2: optionally look up episode metadata by collection+track ID.
    # iTunes Lookup defaults to ~50 results; pass the documented max so older
    # episodes on long-running shows are still discoverable.
    episode_info: dict = {}
    if track_id:
        ep_results = _itunes_fetch({"id": collection_id, "entity": "podcastEpisode", "limit": 200})
        for r in ep_results:
            if r.get("kind") == "podcast-episode" and str(r.get("trackId", "")) == track_id:
                episode_info = r
                break

    # Only carry forward the show-level fields we actually need. Merging the
    # full collection dict leaks the show's own ``releaseDate`` (= newest
    # episode) into matching, which silently selects the wrong episode when
    # the per-episode lookup misses.
    merged = {
        "feedUrl": collection_info["feedUrl"],
        "collectionName": collection_info.get("collectionName", ""),
        "artistName": collection_info.get("artistName", ""),
        **episode_info,
    }

    return merged


def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric chars for fuzzy title matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _strip_query(url: str) -> str:
    """Drop query string and fragment so audio URLs compare cleanly across CDNs."""
    parsed = urllib.parse.urlparse(url)
    return parsed._replace(query="", fragment="").geturl()


def find_episode_in_feed(feed_url: str, itunes_meta: dict, verbose: bool = False) -> Episode:
    """Parse the podcast RSS feed and find the episode matching *itunes_meta*.

    Matching strategy (most → least reliable):
    1. Enclosure (audio) URL — iTunes ``episodeUrl`` vs RSS ``audio_url``
    2. Normalized title
    3. Unique same-day publication date (only if exactly one feed entry matches)

    Raises:
        ApplePodcastsResolutionError: If the episode cannot be found.
    """
    try:
        episodes = _parse_feed_episodes(feed_url, verbose=verbose)
    except Exception as e:
        raise ApplePodcastsResolutionError(f"Failed to fetch/parse podcast RSS feed: {e}") from e

    # Primary: match by enclosure URL (ignoring query strings, which CDNs mutate)
    target_audio = itunes_meta.get("episodeUrl", "")
    if target_audio:
        norm_audio = _strip_query(target_audio)
        for ep in episodes:
            if _strip_query(ep.audio_url) == norm_audio:
                return ep

    # Secondary: match by normalized title
    target_title = itunes_meta.get("trackName", "")
    if target_title:
        norm_target = _normalize(target_title)
        for ep in episodes:
            if _normalize(ep.title) == norm_target:
                return ep

    # Tertiary: same-day release date, but only when the date uniquely
    # identifies one episode. The previous unconditional fallback silently
    # returned a wrong episode whenever the iTunes lookup missed the target.
    release_date_str = itunes_meta.get("releaseDate", "")
    if release_date_str:
        try:
            release_date = datetime.fromisoformat(release_date_str.replace("Z", "+00:00")).date()
            same_day = [ep for ep in episodes if ep.pub_date.date() == release_date]
            if len(same_day) == 1:
                return same_day[0]
        except (ValueError, AttributeError):
            pass

    raise ApplePodcastsResolutionError(
        f"Could not find episode '{target_title or itunes_meta.get('trackId', '?')}' "
        f"in the podcast feed. The feed has {len(episodes)} episodes. "
        "The iTunes Lookup API may not have returned this episode "
        "(common for very old episodes on long-running shows)."
    )


def resolve_apple_podcasts_url(url: str, verbose: bool = False) -> tuple[str, VideoInfo]:
    """Resolve an Apple Podcasts URL to (audio_url, VideoInfo).

    Orchestrates: parse URL -> iTunes Lookup -> RSS parse -> episode match.

    Args:
        url: Apple Podcasts episode URL.
        verbose: Enable verbose logging.

    Returns:
        Tuple of (direct audio URL, VideoInfo with episode metadata).

    Raises:
        ApplePodcastsResolutionError: If resolution fails at any step.
    """
    collection_id, track_id = parse_apple_podcasts_url(url)

    if not track_id:
        raise ApplePodcastsResolutionError(
            "This is a show-level URL (no episode specified). "
            "Please provide a URL for a specific episode "
            "(the URL should contain '?i=' with an episode ID)."
        )

    if verbose:
        logger.info(f"Parsed Apple Podcasts URL: collection={collection_id}, track={track_id}")

    itunes_meta = lookup_itunes(collection_id, track_id)
    feed_url = itunes_meta.get("feedUrl", "")

    if verbose:
        logger.info(
            f"iTunes lookup: {itunes_meta.get('collectionName', 'Unknown')} - {itunes_meta.get('trackName', 'Unknown')}"
        )
        logger.info(f"Feed URL: {feed_url}")

    episode = find_episode_in_feed(feed_url, itunes_meta, verbose=verbose)

    if verbose:
        logger.info(f"Matched episode: {episode.title}")
        logger.info(f"Audio URL: {episode.audio_url}")

    # Build VideoInfo from combined iTunes + RSS metadata
    duration = episode.duration or 0
    hours, remainder = divmod(int(duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        duration_string = f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        duration_string = f"{minutes}:{seconds:02d}"

    video_info = VideoInfo(
        title=episode.title,
        webpage_url=url,
        channel=itunes_meta.get("collectionName", ""),
        uploader=itunes_meta.get("artistName", ""),
        duration=float(duration),
        duration_string=duration_string,
        upload_date=episode.pub_date.strftime("%Y%m%d"),
        description=episode.description[:500] if episode.description else "",
    )

    return episode.audio_url, video_info
