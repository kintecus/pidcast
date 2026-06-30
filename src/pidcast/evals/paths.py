"""Path resolution for the eval subsystem.

Read-only fixtures (prompts, reference registry + transcripts) ship inside the
package under ``pidcast/evals/data/`` and resolve via importlib.resources, so
``pidcast-eval`` works from an installed wheel as well as a source checkout.

Generated output (runs, batches, comparisons, cost tracking) is written under
the XDG data dir (``DATA_DIR/evals/``), keeping the repo free of runtime output.
"""

from importlib import resources
from pathlib import Path

from ..config import DATA_DIR


def fixtures_dir() -> Path:
    """Directory holding the shipped eval fixtures (prompts, references)."""
    return Path(str(resources.files("pidcast.evals") / "data"))


def prompts_file() -> Path:
    return fixtures_dir() / "eval_prompts.json"


def registry_file() -> Path:
    return fixtures_dir() / "reference_transcripts.json"


def evals_output_dir() -> Path:
    """Writable root for generated eval output (under the XDG data dir)."""
    return DATA_DIR / "evals"


def runs_dir() -> Path:
    return evals_output_dir() / "runs"


def batches_dir() -> Path:
    return evals_output_dir() / "batches"


def comparisons_dir() -> Path:
    return evals_output_dir() / "comparisons"


def cost_tracking_file() -> Path:
    return evals_output_dir() / "cost_tracking.json"
