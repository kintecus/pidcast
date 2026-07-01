"""Handler for ``pidcast info`` — resolved data/config paths (folds in `paths`)."""

import argparse
import logging

logger = logging.getLogger(__name__)


def cmd_info(args: argparse.Namespace) -> None:
    """Print resolved data/config directories."""
    from ..config import (
        AUDIO_DIR,
        CONFIG_DIR,
        DATA_DIR,
        LOGS_DIR,
        RUNS_FILE,
        STATE_DIR,
        TRANSCRIPTS_DIR,
        ensure_data_dirs,
    )

    ensure_data_dirs()
    print("pidcast info")
    print(f"  data dir:     {DATA_DIR}")
    print(f"  transcripts:  {TRANSCRIPTS_DIR}")
    print(f"  audio:        {AUDIO_DIR}")
    print(f"  logs:         {LOGS_DIR}")
    print(f"  state:        {STATE_DIR}")
    print(f"  run history:  {RUNS_FILE}")
    print(f"  config dir:   {CONFIG_DIR}")
    print("\n  Override the data dir with PIDCAST_DATA_DIR or XDG_DATA_HOME.")
