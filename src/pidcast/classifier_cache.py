"""On-disk cache for the small-model classifier pass behind ``discord`` and ``notes``.

Keyed on model + system prompt + user prompt, so editing the transcript, the title,
the segmentation, or the prompt text invalidates the entry on its own. Only
responses that parsed are stored, so a bad reply is never replayed.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from .config import STATE_DIR
from .providers.claude_provider import run_claude

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_DIR = STATE_DIR / "cache" / "classifier"


def _key(model: str, system_prompt: str, prompt: str) -> str:
    payload = json.dumps([model, system_prompt, prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classify_cached(
    prompt: str,
    model: str,
    system_prompt: str,
    parse: Callable[[str], T],
    refresh: bool = False,
) -> tuple[T, float | None]:
    """Run the classifier (or replay its cached reply) and return ``(parse(text), cost)``.

    A cache hit costs 0.0. ``refresh`` skips the lookup and overwrites the entry.
    """
    path = CACHE_DIR / f"{_key(model, system_prompt, prompt)}.json"
    if not refresh and path.is_file():
        try:
            entry: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            parsed = parse(entry["text"])
        except Exception:  # corrupt or stale entry: fall through and re-ask
            logger.debug("Ignoring unreadable classifier cache entry %s", path)
        else:
            logger.info("Reusing cached %s pass (--refresh to redo)", model)
            return parsed, 0.0

    run = run_claude(prompt, model, system_prompt=system_prompt)
    parsed = parse(run.text)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": run.text}), encoding="utf-8")
    except OSError as e:
        logger.debug("Could not write classifier cache: %s", e)
    return parsed, run.cost_usd
