"""Handler for ``pidcast discord <transcript>`` (story picker + Ukrainian Discord recap)."""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from ..discord_recap import (
    Story,
    build_recap_prompt,
    find_stories,
    load_preset,
    parse_pick,
    segment,
    source_line,
    write_recap,
)
from ..exceptions import PidcastError
from ..utils import log_error, log_success, log_warning
from ..workflow import parse_transcript_file

logger = logging.getLogger(__name__)


def _print_stories(stories: list[Story]) -> None:
    for s in stories:
        if s.pickable:
            print(f"  {s.id:>2}. {s.headline}\n      {s.gist}")
        else:
            print(f"   -  {s.headline}  [{s.kind}, skipped]")
    # Status lines go to stderr via logging; flush so the menu always precedes them
    sys.stdout.flush()


def _choose_picks(args: argparse.Namespace, stories: list[Story]) -> list[int]:
    pickable = [s for s in stories if s.pickable]
    if args.pick is not None:
        return parse_pick(args.pick, stories)
    if len(pickable) <= 1:
        return [s.id for s in pickable]
    if not sys.stdin.isatty():
        log_warning("Not a TTY and no --pick given: using every story")
        return parse_pick("", stories)
    while True:
        try:
            return parse_pick(input("Stories to include (e.g. 1,3) [Enter = all]: "), stories)
        except ValueError as e:
            print(e)


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

        logger.info(f"Finding stories (claude {args.stories_model})...")
        stories, stories_cost = find_stories(segments, video_info.title, args.stories_model)
        if not any(s.pickable for s in stories):
            log_error("No stories found in this transcript (only ads/intro/outro)")
            return
        _print_stories(stories)
        picks = _choose_picks(args, stories)

        system, user = build_recap_prompt(
            stories, picks, segments, source_line(video_info.channel, transcript_path), preset
        )
        logger.info(f"Writing {args.style} for stories {picks} (claude {args.model})...")
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
