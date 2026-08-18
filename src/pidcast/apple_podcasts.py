"""Resolve Apple Podcasts episode URLs to direct audio URLs."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from datetime import datetime

from .config import VideoInfo
from .exceptions import ApplePodcastsResolutionError
from .rss import Episode, RSSParser

logger = logging.getLogger(__name__)

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


def parse_apple_podcasts_storefront(url: str) -> str | None:
    """Extract the two-letter storefront code from an Apple Podcasts URL.

    Apple Podcasts URLs carry the storefront as the first path segment
    (``/pl/podcast/...``).  Each storefront has its own episode index, and a
    freshly published episode can be present in one and missing from another,
    so the storefront must be carried into the iTunes Lookup call.

    Returns:
        Upper-case country code, or None for storefront-less URLs.
    """
    path = urllib.parse.urlparse(url).path
    match = re.match(r"^/([a-zA-Z]{2})(?:/|$)", path)
    return match.group(1).upper() if match else None


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


def _with_country(params: dict, country: str | None) -> dict:
    """Return *params* with a storefront ``country`` added when one is known."""
    return {**params, "country": country} if country else dict(params)


def lookup_itunes(collection_id: str, track_id: str | None, storefront: str | None = None) -> dict:
    """Call the iTunes Lookup API and return show + episode metadata.

    Always looks up the collection (show) first to get the feedUrl, since the
    iTunes Lookup API does not support looking up episode-level ``?i=`` IDs
    directly.  If a *track_id* is provided, a second request attempts to
    fetch episode-specific metadata (trackName, releaseDate) which is used
    later for matching inside the RSS feed.

    Each storefront keeps its own episode index, so the lookup is tried in the
    URL's *storefront* first and then in the API default (US); a just-published
    episode is often visible in one and not yet in the other.

    Returns:
        Dict with keys like feedUrl, collectionName, trackName, releaseDate, etc.

    Raises:
        ApplePodcastsResolutionError: On API failure or missing data.
    """
    storefronts: list[str | None] = [storefront, None] if storefront else [None]

    # Step 1: always look up the collection to get feedUrl
    collection_info: dict = {}
    for country in storefronts:
        collection_results = _itunes_fetch(_with_country({"id": collection_id}, country))
        if collection_results:
            collection_info = collection_results[0]
            break

    if not collection_info:
        raise ApplePodcastsResolutionError(
            "iTunes Lookup API returned no results for this podcast."
        )

    if not collection_info.get("feedUrl"):
        raise ApplePodcastsResolutionError(
            "No RSS feed URL found for this podcast. The podcast may not have a public feed."
        )

    # Step 2: optionally look up episode metadata by collection+track ID.
    # iTunes Lookup defaults to ~50 results; pass the documented max so older
    # episodes on long-running shows are still discoverable.
    episode_info: dict = {}
    if track_id:
        ep_params = {"id": collection_id, "entity": "podcastEpisode", "limit": 200}
        for country in storefronts:
            ep_results = _itunes_fetch(_with_country(ep_params, country))
            for r in ep_results:
                if r.get("kind") == "podcast-episode" and str(r.get("trackId", "")) == track_id:
                    episode_info = r
                    break
            if episode_info:
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
        # Kept separate from ``trackId`` so error messages can name the episode
        # the user asked for even when the lookup returned nothing for it.
        "requestedTrackId": track_id or "",
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
        _, episodes = RSSParser.parse_feed(feed_url, verbose=verbose)
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

    identifier = target_title or itunes_meta.get("requestedTrackId") or itunes_meta.get("trackId")
    raise ApplePodcastsResolutionError(
        f"Could not find episode '{identifier or 'unknown'}' "
        f"in the podcast feed. The feed has {len(episodes)} episodes. "
        "The iTunes Lookup API did not return this episode - it may be older "
        "than the API's per-show window, or too new to be indexed yet. "
        "Try again shortly, or pass the episode's direct audio URL."
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
    storefront = parse_apple_podcasts_storefront(url)

    if not track_id:
        raise ApplePodcastsResolutionError(
            "This is a show-level URL (no episode specified). "
            "Please provide a URL for a specific episode "
            "(the URL should contain '?i=' with an episode ID)."
        )

    if verbose:
        logger.info(
            f"Parsed Apple Podcasts URL: collection={collection_id}, "
            f"track={track_id}, storefront={storefront or 'default'}"
        )

    itunes_meta = lookup_itunes(collection_id, track_id, storefront=storefront)
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
