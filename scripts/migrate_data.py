#!/usr/bin/env python3
"""One-shot migration of legacy in-repo pidcast data to the XDG data dir.

This is a one-time upgrade helper, not part of the CLI. It relocates the
existing in-repo corpus (transcripts, whisper sidecars, kept audio, error log)
to the XDG data dir and merges the two legacy history stores
(``history.json`` + ``transcription_stats.json``) into the unified
``runs.json``, rewriting absolute paths that pointed into the old repo location.

Usage (from the repo root):

    uv run scripts/migrate_data.py --dry-run    # preview, move nothing
    uv run scripts/migrate_data.py              # apply

Honors PIDCAST_DATA_DIR / XDG_DATA_HOME for the destination (same as the CLI).
The legacy history.json is left in place; remove it manually once satisfied.
"""

import argparse
import sys

from pidcast.config import DATA_DIR, HISTORY_FILE, PROJECT_ROOT
from pidcast.migrate import migrate_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would move without moving or writing anything.",
    )
    args = parser.parse_args()

    legacy_transcripts = PROJECT_ROOT / "data" / "transcripts"
    print("\npidcast data migration" + (" (dry run)" if args.dry_run else ""))
    print("=" * 40)
    print(f"  from: {PROJECT_ROOT / 'data'} (+ stray transcripts at repo root)")
    print(f"  to:   {DATA_DIR}")

    if not legacy_transcripts.is_dir() and not HISTORY_FILE.exists():
        print("\n  Nothing to migrate (no legacy data/transcripts or history.json found).")
        return 0

    report = migrate_data(
        legacy_repo_root=PROJECT_ROOT,
        data_dir=DATA_DIR,
        legacy_history_file=HISTORY_FILE,
        dry_run=args.dry_run,
    )

    print()
    print(report.summary())
    for note in report.notes:
        print(f"  ! {note}")

    if args.dry_run:
        print("\n  Dry run - nothing was moved. Re-run without --dry-run to apply.")
    else:
        print("\n  Done. Run 'pidcast paths' to confirm, 'pidcast doctor' to verify integrity.")
        print("  The legacy history.json is left in place; remove it manually once happy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
