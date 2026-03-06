# CLI presets implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add user-defined CLI presets to pidcast so common flag combinations can be invoked with `-p <name>`.

**Architecture:** Presets are stored in `~/.config/pidcast/config.yaml` under a `presets:` key. A new `apply_preset()` function loads the preset and sets arg defaults before the main workflow runs. Explicit CLI flags always override preset values.

**Tech Stack:** Python argparse, ruamel.yaml (already a dependency), existing ConfigManager.

---

### Task 1: Add preset loading to ConfigManager

**Files:**
- Modify: `src/pidcast/config_manager.py`
- Test: `tests/test_presets.py`

**Step 1: Write the failing test**

Create `tests/test_presets.py`:

```python
"""Tests for CLI preset loading."""

import pytest
from unittest.mock import patch
from pidcast.config_manager import ConfigManager


class TestLoadPreset:
    def test_load_existing_preset(self, tmp_path):
        config = {
            "backfill_limit": 5,
            "presets": {
                "daily": {
                    "whisper_model": "large-v3",
                    "language": "uk",
                    "diarize": True,
                    "no_analyze": True,
                }
            },
        }
        with patch.object(ConfigManager, "load_config", return_value=config):
            result = ConfigManager.load_preset("daily")
        assert result == {
            "whisper_model": "large-v3",
            "language": "uk",
            "diarize": True,
            "no_analyze": True,
        }

    def test_load_nonexistent_preset_raises(self):
        config = {"presets": {"daily": {"diarize": True}}}
        with patch.object(ConfigManager, "load_config", return_value=config):
            with pytest.raises(ValueError, match="Unknown preset 'nope'"):
                ConfigManager.load_preset("nope")

    def test_load_preset_no_presets_section(self):
        config = {"backfill_limit": 5}
        with patch.object(ConfigManager, "load_config", return_value=config):
            with pytest.raises(ValueError, match="No presets defined"):
                ConfigManager.load_preset("daily")

    def test_list_presets_returns_names_and_flags(self):
        config = {
            "presets": {
                "daily": {"whisper_model": "large-v3", "diarize": True},
                "meeting": {"whisper_model": "small"},
            }
        }
        with patch.object(ConfigManager, "load_config", return_value=config):
            result = ConfigManager.list_presets()
        assert result == {
            "daily": {"whisper_model": "large-v3", "diarize": True},
            "meeting": {"whisper_model": "small"},
        }

    def test_list_presets_empty(self):
        config = {"backfill_limit": 5}
        with patch.object(ConfigManager, "load_config", return_value=config):
            result = ConfigManager.list_presets()
        assert result == {}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_presets.py -v`
Expected: FAIL - `ConfigManager` has no `load_preset` or `list_presets` methods.

**Step 3: Write minimal implementation**

Add two static methods to `ConfigManager` in `src/pidcast/config_manager.py`:

```python
@staticmethod
def load_preset(name: str) -> dict[str, Any]:
    """Load a named preset from config.

    Args:
        name: Preset name

    Returns:
        Dictionary of flag overrides

    Raises:
        ValueError: If preset not found or no presets defined
    """
    config = ConfigManager.load_config()
    presets = config.get("presets")
    if not presets:
        raise ValueError(
            "No presets defined in config. "
            f"Add presets to {CONFIG_FILE}"
        )
    if name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"Unknown preset '{name}'. Available: {available}"
        )
    return dict(presets[name])

@staticmethod
def list_presets() -> dict[str, dict[str, Any]]:
    """List all defined presets.

    Returns:
        Dictionary of preset names to their flag overrides
    """
    config = ConfigManager.load_config()
    presets = config.get("presets")
    if not presets:
        return {}
    return {name: dict(flags) for name, flags in presets.items()}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_presets.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add tests/test_presets.py src/pidcast/config_manager.py
git commit -m "feat: add preset loading to ConfigManager"
```

---

### Task 2: Add -p/--preset and -P/--list-presets CLI flags

**Files:**
- Modify: `src/pidcast/cli.py` (in `parse_arguments()`)

**Step 1: Add the flags to the argument parser**

In `parse_arguments()`, add after the discovery group (around line 521):

```python
# Preset options
preset_group = parser.add_argument_group("Preset Options")
preset_group.add_argument(
    "-p",
    "--preset",
    default=None,
    help="Use a named preset from config.yaml (e.g., 'daily', 'meeting'). "
    "Explicit flags override preset values. Use -P to list presets.",
)
preset_group.add_argument(
    "-P",
    "--list-presets",
    action="store_true",
    dest="list_presets",
    help="List available presets and exit",
)
```

**Step 2: Commit**

```bash
git add src/pidcast/cli.py
git commit -m "feat: add -p/--preset and -P/--list-presets CLI flags"
```

---

### Task 3: Add apply_preset() and wire it into main()

**Files:**
- Modify: `src/pidcast/cli.py` (add `apply_preset()`, modify `main()`)
- Test: `tests/test_presets.py` (add integration tests)

**Step 1: Write failing tests for apply_preset**

Add to `tests/test_presets.py`:

```python
import argparse
from pidcast.cli import apply_preset


class TestApplyPreset:
    def test_preset_sets_unset_args(self):
        args = argparse.Namespace(
            preset="daily",
            whisper_model=None,
            language=None,
            diarize=False,
            no_analyze=False,
            verbose=False,
        )
        preset_values = {
            "whisper_model": "large-v3",
            "language": "uk",
            "diarize": True,
            "no_analyze": True,
        }
        with patch(
            "pidcast.cli.ConfigManager.load_preset", return_value=preset_values
        ):
            apply_preset(args)

        assert args.whisper_model == "large-v3"
        assert args.language == "uk"
        assert args.diarize is True
        assert args.no_analyze is True

    def test_explicit_cli_flag_overrides_preset(self):
        args = argparse.Namespace(
            preset="daily",
            whisper_model="medium",  # explicitly set by user
            language=None,
            diarize=False,
            no_analyze=False,
            verbose=False,
        )
        preset_values = {
            "whisper_model": "large-v3",
            "language": "uk",
        }
        # Simulate that whisper_model was explicitly provided on CLI
        explicitly_set = {"whisper_model"}
        with patch(
            "pidcast.cli.ConfigManager.load_preset", return_value=preset_values
        ):
            apply_preset(args, explicitly_set=explicitly_set)

        assert args.whisper_model == "medium"  # NOT overridden
        assert args.language == "uk"  # set from preset

    def test_unknown_preset_key_warns(self, caplog):
        import logging

        args = argparse.Namespace(
            preset="daily",
            verbose=False,
        )
        preset_values = {"bogus_flag": "value"}
        with patch(
            "pidcast.cli.ConfigManager.load_preset", return_value=preset_values
        ):
            with caplog.at_level(logging.WARNING):
                apply_preset(args)
        assert "Unknown preset key 'bogus_flag'" in caplog.text
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_presets.py::TestApplyPreset -v`
Expected: FAIL - `apply_preset` does not exist.

**Step 3: Write apply_preset() implementation**

Add to `src/pidcast/cli.py` after the imports section:

```python
def apply_preset(
    args: argparse.Namespace,
    explicitly_set: set[str] | None = None,
) -> None:
    """Apply a named preset to args, without overriding explicitly set flags.

    Args:
        args: Parsed arguments namespace
        explicitly_set: Set of arg names that were explicitly provided on CLI
    """
    from .config_manager import ConfigManager

    if explicitly_set is None:
        explicitly_set = set()

    preset_values = ConfigManager.load_preset(args.preset)

    for key, value in preset_values.items():
        if not hasattr(args, key):
            logger.warning(f"Unknown preset key '{key}', skipping")
            continue
        if key in explicitly_set:
            continue
        setattr(args, key, value)

    if getattr(args, "verbose", False):
        logger.info(f"Applied preset '{args.preset}': {preset_values}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_presets.py::TestApplyPreset -v`
Expected: PASS (3 tests)

**Step 5: Wire into main()**

In `main()`, after `args = parse_arguments()` and `setup_logging()` (line 1127-1130), add the list-presets handler and preset application:

```python
# Handle --list-presets
if getattr(args, "list_presets", False):
    from .config_manager import ConfigManager

    presets = ConfigManager.list_presets()
    if not presets:
        print("No presets defined. Add presets to ~/.config/pidcast/config.yaml")
        print("\nExample:")
        print("  presets:")
        print("    daily:")
        print("      whisper_model: large-v3")
        print("      language: uk")
        print("      diarize: true")
        print("      no_analyze: true")
    else:
        print("Available presets:\n")
        for name, flags in presets.items():
            flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
            print(f"  {name}: {flag_str}")
        print(f"\nUsage: pidcast <input> -p <preset>")
    return

# Apply preset if specified
if getattr(args, "preset", None):
    import sys

    # Determine which args were explicitly set on CLI
    # by comparing against parser defaults
    explicitly_set = set()
    parser_defaults = parse_arguments.__defaults_cache__ if hasattr(parse_arguments, "__defaults_cache__") else {}
    for arg in vars(args):
        # Check if this arg appears as a CLI flag in sys.argv
        # Simple heuristic: if --flag or -flag appears in argv
        if any(
            a in (f"--{arg}", f"-{arg}", f"--{arg.replace('_', '-')}")
            for a in sys.argv[1:]
        ):
            explicitly_set.add(arg)
    try:
        apply_preset(args, explicitly_set=explicitly_set)
    except ValueError as e:
        log_error(str(e))
        return
```

**Step 6: Run all tests**

Run: `uv run pytest tests/test_presets.py -v`
Expected: PASS (8 tests)

**Step 7: Commit**

```bash
git add src/pidcast/cli.py tests/test_presets.py
git commit -m "feat: wire preset loading into main CLI flow"
```

---

### Task 4: Manual smoke test

**Step 1: Add a test preset to config**

Create or edit `~/.config/pidcast/config.yaml` and add:

```yaml
presets:
  daily:
    whisper_model: large-v3
    language: uk
    diarize: true
    no_analyze: true
  meeting:
    whisper_model: small
    diarize: true
    no_analyze: true
```

**Step 2: Test list-presets**

Run: `uv run pidcast -P`
Expected: Lists both presets with their flags.

**Step 3: Test preset with --help to verify flag is registered**

Run: `uv run pidcast --help`
Expected: Shows `-p`/`--preset` and `-P`/`--list-presets` in output.

**Step 4: Test dry run with verbose to see preset application**

Run: `uv run pidcast test.wav -p daily -v` (or any input)
Expected: Log shows "Applied preset 'daily': {...}" and uses large-v3 model.

---

### Notes

- The `explicitly_set` detection uses `sys.argv` scanning. This is a pragmatic approach - it handles `--whisper_model medium`, `--diarize`, `-l uk` etc. It won't catch edge cases like positional args, but those aren't preset-overridable anyway.
- Boolean flags like `diarize` default to `False` in argparse. If the preset sets `diarize: true`, and the user doesn't pass `--diarize`, the preset value wins. If the user passes `--diarize`, the explicit flag wins (both are `True` so it doesn't matter).
- No preset inheritance or composition - keep it simple.
