"""Podcast discovery by name via Apple Podcasts local DB and iTunes Search API."""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

APPLE_PODCASTS_DB = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "243LU875E5.groups.com.apple.podcasts"
    / "Documents"
    / "MTLibrary.sqlite"
)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def search_apple_podcasts_local(query: str) -> list[dict]:
    """Search local Apple Podcasts SQLite DB for matching shows.

    Returns an empty list if the DB is not found (non-macOS or Podcasts not installed).
    """
    if not APPLE_PODCASTS_DB.exists():
        return []

    results = []
    try:
        with sqlite3.connect(APPLE_PODCASTS_DB) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT ZTITLE, ZFEEDURL, ZAUTHOR, ZITEMDESCRIPTION
                FROM ZMTPODCAST
                WHERE ZTITLE LIKE ? AND ZFEEDURL IS NOT NULL AND ZFEEDURL != ''
                ORDER BY ZTITLE
                LIMIT 10
                """,
                (f"%{query}%",),
            )
            for row in cur.fetchall():
                results.append(
                    {
                        "title": row["ZTITLE"] or "",
                        "feed_url": row["ZFEEDURL"] or "",
                        "author": row["ZAUTHOR"] or "",
                        "description": (row["ZITEMDESCRIPTION"] or "")[:200],
                        "source": "local",
                    }
                )
    except sqlite3.Error as e:
        logger.debug("Apple Podcasts DB query failed: %s", e)

    return results


def search_itunes_api(query: str, limit: int = 10) -> list[dict]:
    """Search the iTunes Search API for podcasts matching query."""
    params = urllib.parse.urlencode(
        {
            "term": query,
            "media": "podcast",
            "entity": "podcast",
            "limit": limit,
        }
    )
    url = f"{ITUNES_SEARCH_URL}?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pidcast/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.debug("iTunes Search API failed: %s", e)
        return []

    results = []
    for item in data.get("results", []):
        feed_url = item.get("feedUrl", "")
        if not feed_url:
            continue
        results.append(
            {
                "title": item.get("collectionName", ""),
                "feed_url": feed_url,
                "author": item.get("artistName", ""),
                "description": item.get("description", "")[:200],
                "source": "itunes",
            }
        )
    return results


def discover_podcast(query: str) -> list[dict]:
    """Search for podcasts by name.

    Tries the local Apple Podcasts DB first (macOS only), then falls back to
    the iTunes Search API. Results are deduped by feed URL, with local results
    taking precedence.
    """
    local = search_apple_podcasts_local(query)
    # Skip network call if local DB already returned a full page of results
    remote = [] if len(local) >= 10 else search_itunes_api(query)

    seen_urls: set[str] = set()
    merged: list[dict] = []

    for item in local + remote:
        url = item["feed_url"].rstrip("/")
        if url not in seen_urls:
            seen_urls.add(url)
            merged.append(item)

    return merged


def prompt_user_selection(results: list[dict]) -> dict | None:
    """Present a numbered list and prompt the user to pick one.

    Returns the selected result dict, or None if cancelled.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Title", style="bold white")
        table.add_column("Author", style="yellow")
        table.add_column("Source", style="dim")

        for i, r in enumerate(results, 1):
            table.add_row(str(i), r["title"], r["author"], r["source"])

        console.print("\n[bold]Matching podcasts:[/bold]")
        console.print(table)
    except ImportError:
        print("\nMatching podcasts:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['title']} — {r['author']} [{r['source']}]")

    while True:
        raw = input("\nSelect number (or 'q' to cancel): ").strip()
        if raw.lower() in ("q", "quit", ""):
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(results):
                return results[idx - 1]
        print(f"Enter a number between 1 and {len(results)}, or 'q' to cancel.")
