"""Discord recaps: list the stories in a transcript, then write a Ukrainian recap of picked ones.

Two passes keep the recap focused and cheap: a small model splits the numbered
transcript into stories, the user picks some, and only those excerpts reach the
writer model. Styles live as YAML presets in ``config/discord/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import DISCORD_PRESETS_DIR
from .exceptions import AnalysisError
from .providers.claude_provider import run_claude

SEGMENT_CHARS = 150
STORY_KINDS = ("story", "ad", "intro", "outro")
# The presets ask for <= budget - 50 so the user has room to add a line of their own
SHRINK_MARGIN = 50

STORIES_SYSTEM = """You segment a podcast transcript into its distinct stories.
Lines are prefixed with [n]. Return ONLY JSON:
{"stories": [{"kind": "story", "headline": "...", "gist": "...", "start": 0, "end": 12}]}
Rules:
- Cover every line exactly once with contiguous, ordered, non-overlapping [start, end] ranges (inclusive).
- kind is one of: story, ad, intro, outro. Sponsor reads and promos are "ad".
- headline: max 10 words, English, specific (names, numbers). gist: one English sentence with the key fact.
- Split back-to-back news items into separate stories even when the host transitions quickly.
- A single-topic episode (interview, deep dive) is one story."""

SHRINK_INSTRUCTION = """Your draft below is {length} characters; the hard limit is {limit}.
Rewrite it to at most {limit} characters. Keep every fact, name, number, and the required form.
Cut adjectives, filler, and the weakest line first. Output only the rewritten message.

Draft:
{draft}"""


@dataclass
class Story:
    # Numbered 1..n across pickable stories only; ads/intro/outro carry None
    id: int | None
    kind: str
    headline: str
    gist: str
    start: int
    end: int

    @property
    def pickable(self) -> bool:
        return self.kind == "story"


@dataclass
class Recap:
    text: str
    model: str
    budget: int
    cost_usd: float | None
    shortened: bool

    @property
    def fits(self) -> bool:
        return len(self.text) <= self.budget


def list_styles() -> list[str]:
    return sorted(p.stem for p in Path(DISCORD_PRESETS_DIR).glob("*.yaml"))


def load_preset(style: str) -> dict[str, Any]:
    path = Path(DISCORD_PRESETS_DIR) / f"{style}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(list_styles())}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def segment(body: str, target: int = SEGMENT_CHARS) -> list[str]:
    """Group transcript sentences into ~target-char segments for line-numbered prompting."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body) if s.strip()]
    segments: list[str] = []
    buf = ""
    for sentence in sentences:
        buf = f"{buf} {sentence}".strip()
        if len(buf) >= target:
            segments.append(buf)
            buf = ""
    if buf:
        segments.append(buf)
    return segments


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise AnalysisError(f"Story pass returned no JSON: {text[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise AnalysisError(f"Story pass returned invalid JSON: {e}") from e


def parse_stories(raw: str, n_segments: int) -> list[Story]:
    """Parse the story-pass JSON, numbering pickable stories 1..n and clamping ranges."""
    items = _extract_json(raw).get("stories") or []
    if not items:
        raise AnalysisError("Story pass found no stories")
    last = n_segments - 1
    stories = []
    next_id = 1
    for item in items:
        start = min(max(int(item.get("start", 0)), 0), last)
        end = min(max(int(item.get("end", start)), start), last)
        kind = item.get("kind") if item.get("kind") in STORY_KINDS else "story"
        story_id = None
        if kind == "story":
            story_id, next_id = next_id, next_id + 1
        stories.append(
            Story(
                id=story_id,
                kind=kind,
                headline=str(item.get("headline", "")).strip(),
                gist=str(item.get("gist", "")).strip(),
                start=start,
                end=end,
            )
        )
    return stories


def find_stories(segments: list[str], title: str, model: str) -> tuple[list[Story], float | None]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(segments))
    run = run_claude(f"Title: {title}\n\n{numbered}", model, system_prompt=STORIES_SYSTEM)
    return parse_stories(run.text, len(segments)), run.cost_usd


def parse_pick(raw: str, stories: list[Story]) -> list[int]:
    """Parse "1,3" into story ids, in the given order. Empty input means every pickable story."""
    pickable = [s.id for s in stories if s.pickable]
    if not raw.strip():
        return pickable
    picks: list[int] = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token.isdigit() or int(token) not in pickable:
            raise ValueError(f"'{token}' is not a story number. Pick from: {pickable}")
        if int(token) not in picks:
            picks.append(int(token))
    return picks


def build_recap_prompt(
    stories: list[Story],
    picks: list[int],
    segments: list[str],
    source_line: str,
    preset: dict[str, Any],
) -> tuple[str, str]:
    """Render the writer prompt from only the picked stories' excerpts.

    The episode title is deliberately left out: it names every story, so including
    it lets skipped stories leak back into the recap.
    """
    by_id = {s.id: s for s in stories if s.pickable}
    blocks = []
    for n, pid in enumerate(picks, 1):
        story = by_id[pid]
        text = " ".join(segments[story.start : story.end + 1])
        blocks.append(f"### Story {n}: {story.headline}\n{text}")
    user = preset["user_prompt"].format(
        source_line=source_line, n_stories=len(picks), stories_block="\n\n".join(blocks)
    )
    return preset["system_prompt"], user


def write_recap(system: str, user: str, model: str, budget: int) -> Recap:
    """Write the recap; if it overshoots the budget, ask once for a tighter rewrite."""
    run = run_claude(user, model, system_prompt=system)
    text, cost, shortened = run.text, run.cost_usd, False
    limit = budget - SHRINK_MARGIN
    if len(text) > budget:
        retry = run_claude(
            f"{user}\n\n{SHRINK_INSTRUCTION.format(length=len(text), limit=limit, draft=text)}",
            model,
            system_prompt=system,
        )
        text, shortened = retry.text, True
        if cost is not None and retry.cost_usd is not None:
            cost += retry.cost_usd
    return Recap(text=text, model=run.model, budget=budget, cost_usd=cost, shortened=shortened)


def source_line(channel: str | None, transcript_path: Path) -> str:
    """'Channel | YYYY-MM-DD', using the date prefix pidcast puts on transcript filenames."""
    date = re.match(r"\d{4}-\d{2}-\d{2}", transcript_path.stem)
    parts = [p for p in (channel if channel != "Unknown" else None, date and date.group(0)) if p]
    return " | ".join(parts) or transcript_path.stem


def skipped_summary(stories: list[Story]) -> str:
    """'3 ads, intro, outro' for the non-story segments, in order of first appearance."""
    counts: dict[str, int] = {}
    for s in stories:
        if not s.pickable:
            counts[s.kind] = counts.get(s.kind, 0) + 1
    return ", ".join(kind if n == 1 else f"{n} {kind}s" for kind, n in counts.items())
