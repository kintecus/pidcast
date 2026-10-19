"""Handler for ``pidcast discord <transcript>`` (story picker + Ukrainian Discord recap)."""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

from .. import ui
from ..discord_recap import (
    Story,
    build_recap_prompt,
    find_stories,
    load_preset,
    parse_pick,
    segment,
    skipped_summary,
    source_line,
    write_recap,
)
from ..exceptions import PidcastError
from ..utils import log_error, log_success, log_warning
from ..workflow import parse_transcript_file

logger = logging.getLogger(__name__)


def _share(story: Story, n_segments: int) -> str:
    pct = (story.end - story.start + 1) * 100 / n_segments
    return "<1%" if pct < 1 else f"~{pct:.0f}%"


def _choose_picks(args: argparse.Namespace, stories: list[Story], n_segments: int) -> list[int]:
    """Show the stories and return picked ids; an empty list means the user quit."""
    pickable = [s for s in stories if s.pickable]
    items = [ui.ChecklistItem(s.headline, s.gist, _share(s, n_segments)) for s in pickable]
    skipped = skipped_summary(stories)
    notes = [f"Skipped: {skipped}"] if skipped else []

    if args.pick is None and len(pickable) > 1 and sys.stdin.isatty():
        picked = ui.select_many(items, notes)
        return [pickable[i].id for i in picked] if picked is not None else []

    Console().print(ui.checklist_view(items, notes=notes))
    sys.stdout.flush()
    if args.pick is not None:
        return parse_pick(args.pick, stories)
    if len(pickable) <= 1:
        return [s.id for s in pickable]
    log_warning("Not a TTY and no --pick given: using every story")
    return parse_pick("", stories)


def _copy_to_clipboard(text: str) -> bool:
    pbcopy = shutil.which("pbcopy")
    if not pbcopy:
        return False
    subprocess.run([pbcopy], input=text, text=True, check=False)
    return True


def cmd_discord(args: argparse.Namespace) -> None:
    """List the transcript's stories, let the user pick, and write a Discord recap."""
    transcript_path = Path(args.transcript).expanduser()
    try:
        preset = load_preset(args.style)
        body, video_info = parse_transcript_file(str(transcript_path))
        segments = segment(body)

        with ui.spinner(f"Finding stories ({args.stories_model})"):
            stories, stories_cost = find_stories(segments, video_info.title, args.stories_model)
        if not any(s.pickable for s in stories):
            log_error("No stories found in this transcript (only ads/intro/outro)")
            return
        channel = video_info.channel if video_info.channel != "Unknown" else ""
        header = " · ".join(p for p in (video_info.title, channel) if p)
        Console().print(header, style="bold", markup=False, highlight=False)
        picks = _choose_picks(args, stories, len(segments))
        if not picks:
            logger.info("Cancelled")
            return

        system, user = build_recap_prompt(
            stories, picks, segments, source_line(video_info.channel, transcript_path), preset
        )
        noun = "story" if len(picks) == 1 else "stories"
        with ui.spinner(f"Writing {args.style} from {len(picks)} {noun} ({args.model})"):
            recap = write_recap(system, user, args.model, preset["budget_chars"])
    except (PidcastError, ValueError) as e:
        log_error(str(e))
        return

    print(f"\n{recap.text}\n")

    out_path = transcript_path.with_name(f"{transcript_path.stem}.discord-{args.style}.md")
    out_path.write_text(recap.text + "\n", encoding="utf-8")

    status = [f"{len(recap.text)} / {recap.budget} chars"]
    if recap.shortened:
        status.append("shortened once")
    costs = [c for c in (stories_cost, recap.cost_usd) if c is not None]
    if costs:
        status.append(f"~${sum(costs):.3f} of plan usage")
    status.append(f"saved: {out_path}")
    if args.copy and _copy_to_clipboard(recap.text):
        status.append("copied to clipboard")

    if recap.fits:
        log_success(" | ".join(status))
    else:
        log_warning(
            "Still over the Discord budget after one rewrite - trim by hand. " + " | ".join(status)
        )
