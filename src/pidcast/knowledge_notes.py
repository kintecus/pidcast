"""Knowledge notes: map a long conversation's themes, then write Obsidian notes focused on picked ones.

Same two-pass shape as ``discord_recap``: a small model maps the numbered
transcript into themes and the user picks the focus. Unlike Discord, the writer
gets the full transcript, because interviews circle back to themes and the
executive summary has to cover the whole episode. Picked themes get a detailed
section; the rest get one line. Styles live as YAML presets in ``config/notes/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import NOTES_PRESETS_DIR, VideoInfo
from .discord_recap import Story, parse_stories, segment
from .markdown import format_yaml_front_matter
from .providers.claude_provider import run_claude

# Bigger than Discord's segments: a 90-minute episode stays ~250 numbered lines
TOPIC_SEGMENT_CHARS = 400
# Opus writing several thousand words can outlast run_claude's 300s default
WRITER_TIMEOUT = 900

# parse_stories files any kind other than ad/intro/outro as a pickable story,
# so "topic" lands there without a separate parser.
TOPICS_SYSTEM = """You map the themes of a long-form podcast conversation (interview, panel, deep dive).
Lines are prefixed with [n]. Return ONLY JSON:
{"stories": [{"kind": "topic", "headline": "...", "gist": "...", "start": 0, "end": 12}]}
Rules:
- Cover every line exactly once with contiguous, ordered, non-overlapping [start, end] ranges (inclusive).
- kind is one of: topic, ad, intro, outro. Sponsor reads and promos are "ad". Greetings and guest bios are "intro".
- A topic is a theme discussed for several minutes. Return 8-15 topics for an hour or more, never more than 15 and never one for the whole episode. Merge adjacent lines that serve one theme (e.g. several systems in the same platform layer) and fold short tangents into the topic they serve.
- headline: max 10 words, English, specific (systems, names, numbers). gist: one English sentence with the main claim or insight, not "they discuss X"."""

_FENCE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", re.DOTALL)


@dataclass
class Notes:
    text: str
    model: str
    cost_usd: float | None


def list_styles() -> list[str]:
    return sorted(p.stem for p in Path(NOTES_PRESETS_DIR).glob("*.yaml"))


def load_preset(style: str) -> dict[str, Any]:
    path = Path(NOTES_PRESETS_DIR) / f"{style}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(list_styles())}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def topic_segments(body: str) -> list[str]:
    return segment(body, TOPIC_SEGMENT_CHARS)


def find_topics(segments: list[str], title: str, model: str) -> tuple[list[Story], float | None]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(segments))
    run = run_claude(f"Title: {title}\n\n{numbered}", model, system_prompt=TOPICS_SYSTEM)
    return parse_stories(run.text, len(segments)), run.cost_usd


def build_notes_prompt(
    body: str,
    video_info: VideoInfo,
    topics: list[Story],
    focus: list[int],
    steer: str | None,
    preset: dict[str, Any],
) -> tuple[str, str]:
    """Render the writer prompt: full transcript plus each theme tagged FOCUS or BRIEF.

    Focus themes are listed first in pick order, so the user's ordering drives the
    section order; brief themes follow in episode order.
    """
    by_id = {t.id: t for t in topics if t.pickable}
    ordered = [by_id[i] for i in focus] + [t for t in by_id.values() if t.id not in focus]
    lines = [f"- [{'FOCUS' if t.id in focus else 'BRIEF'}] {t.headline}: {t.gist}" for t in ordered]
    steer_block = (
        f"\nReader's focus (weight the notes toward this): {steer.strip()}\n" if steer else ""
    )
    channel = video_info.channel if video_info.channel not in ("", "Unknown") else "unknown"
    user = preset["user_prompt"].format(
        title=video_info.title,
        channel=channel,
        duration=video_info.duration_string or "unknown",
        topics_block="\n".join(lines),
        steer_block=steer_block,
        transcript=body,
    )
    return preset["system_prompt"], user


def clean_output(text: str) -> str:
    """Drop a code fence wrapped around the whole answer, which models add now and then."""
    match = _FENCE.match(text.strip())
    return (match.group(1) if match else text).strip()


def write_notes(system: str, user: str, model: str) -> Notes:
    run = run_claude(user, model, system_prompt=system, timeout=WRITER_TIMEOUT)
    return Notes(text=clean_output(run.text), model=run.model, cost_usd=run.cost_usd)


def render_document(
    notes: Notes,
    video_info: VideoInfo,
    transcript_path: Path,
    focus_topics: list[Story],
    style: str,
    tags: list[str] | None = None,
) -> str:
    """Front matter + H1 + the model's body, ready to drop into an Obsidian vault."""
    date = re.match(r"\d{4}-\d{2}-\d{2}", transcript_path.stem)
    front: dict[str, Any] = {"title": video_info.title}
    if date:
        front["date"] = date.group(0)
    if video_info.webpage_url:
        front["url"] = video_info.webpage_url
    if video_info.channel not in ("", "Unknown"):
        front["channel"] = video_info.channel
    if video_info.duration_string not in ("", "Unknown"):
        front["duration"] = video_info.duration_string
    front["transcript"] = str(transcript_path)
    front["focus"] = [t.headline for t in focus_topics]
    front["model"] = notes.model
    front["tags"] = ["podcast-notes", f"notes-{style}", *(tags or [])]
    return f"{format_yaml_front_matter(front)}\n\n# {video_info.title}\n\n{notes.text}\n"
