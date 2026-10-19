"""Handler for ``pidcast notes <transcript>`` (theme picker + long-form knowledge notes)."""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from .. import ui
from ..config import OBSIDIAN_PATH
from ..discord_recap import Story, parse_pick, skipped_summary
from ..exceptions import PidcastError
from ..knowledge_notes import (
    build_notes_prompt,
    find_topics,
    load_preset,
    render_document,
    topic_segments,
    write_notes,
)
from ..utils import get_unique_filename, log_error, log_success, log_warning
from ..workflow import parse_transcript_file

logger = logging.getLogger(__name__)

FOCUS_NOTE = "Checked = detailed section, unchecked = one line under 'Also discussed'"


def _share(topic: Story, n_segments: int) -> str:
    pct = (topic.end - topic.start + 1) * 100 / n_segments
    return "<1%" if pct < 1 else f"~{pct:.0f}%"


def _choose_focus(args: argparse.Namespace, topics: list[Story], n_segments: int) -> list[int]:
    """Show the themes and return focus ids; an empty list means the user quit."""
    pickable = [t for t in topics if t.pickable]
    items = [ui.ChecklistItem(t.headline, t.gist, _share(t, n_segments)) for t in pickable]
    skipped = skipped_summary(topics)
    notes = [FOCUS_NOTE] + ([f"Skipped: {skipped}"] if skipped else [])

    if args.pick is None and len(pickable) > 1 and sys.stdin.isatty():
        picked = ui.select_many(items, notes)
        return [pickable[i].id for i in picked] if picked is not None else []

    focus = parse_pick(args.pick or "", topics)
    # Show the checkboxes so a --pick subset is visible; cursor -1 hides the pointer
    state = ui.Checklist(selected=[t.id in focus for t in pickable], cursor=-1)
    Console().print(ui.checklist_view(items, state, notes))
    sys.stdout.flush()
    if args.pick is None and len(pickable) > 1:
        log_warning("Not a TTY and no --pick given: every theme gets a detailed section")
    return focus


def _output_dir(args: argparse.Namespace, transcript_path: Path) -> Path:
    if not args.save_to_obsidian:
        return transcript_path.parent
    if not OBSIDIAN_PATH:
        raise ValueError("-o needs OBSIDIAN_VAULT_PATH set (see .env.example)")
    return Path(OBSIDIAN_PATH).expanduser()


def cmd_notes(args: argparse.Namespace) -> None:
    """Map the transcript's themes, let the user pick the focus, and write knowledge notes."""
    transcript_path = Path(args.transcript).expanduser().resolve()
    try:
        preset = load_preset(args.style)
        out_dir = _output_dir(args, transcript_path)
        body, video_info = parse_transcript_file(str(transcript_path))
        segments = topic_segments(body)

        with ui.spinner(f"Mapping themes ({args.topics_model})"):
            topics, topics_cost = find_topics(segments, video_info.title, args.topics_model)
        if not any(t.pickable for t in topics):
            log_error("No themes found in this transcript (only ads/intro/outro)")
            return
        channel = video_info.channel if video_info.channel != "Unknown" else ""
        header = " · ".join(p for p in (video_info.title, channel) if p)
        Console().print(header, style="bold", markup=False, highlight=False)
        focus = _choose_focus(args, topics, len(segments))
        if not focus:
            logger.info("Cancelled")
            return

        system, user = build_notes_prompt(body, video_info, topics, focus, args.focus, preset)
        n_brief = sum(1 for t in topics if t.pickable) - len(focus)
        label = f"Writing {args.style} notes: {len(focus)} focus, {n_brief} brief ({args.model})"
        with ui.spinner(label):
            notes = write_notes(system, user, args.model)
    except (PidcastError, ValueError) as e:
        log_error(str(e))
        return

    by_id = {t.id: t for t in topics if t.pickable}
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    doc = render_document(
        notes, video_info, transcript_path, [by_id[i] for i in focus], args.style, tags
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = get_unique_filename(out_dir, f"{transcript_path.stem}.notes-{args.style}")
    out_path.write_text(doc, encoding="utf-8")

    status = [f"{len(notes.text.split())} words"]
    costs = [c for c in (topics_cost, notes.cost_usd) if c is not None]
    if costs:
        status.append(f"~${sum(costs):.3f} of plan usage")
    status.append(f"saved: {out_path}")
    log_success(" | ".join(status))
