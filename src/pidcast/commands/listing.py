"""Handler for ``pidcast list <thing>`` — consolidated discovery commands.

Replaces the old cryptic ``-L/-M/-W/-P`` flags and ``--list-chrome-profiles``
with one verb whose ``thing`` positional selects what to list.
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace) -> None:
    """Dispatch ``pidcast list <thing>`` to the matching discovery routine."""
    thing = args.thing
    dispatch = {
        "analyses": _list_analyses,
        "models": _list_models,
        "whisper-models": _list_whisper_models,
        "presets": _list_presets,
        "profiles": _list_profiles,
    }
    handler = dispatch.get(thing)
    if handler is None:  # argparse choices guard this, but be defensive
        print(f"Unknown list target: {thing}")
        return
    handler()


def _list_analyses() -> None:
    from ..utils import list_available_analyses

    list_available_analyses()


def _list_models() -> None:
    from ..utils import list_available_models

    list_available_models()


def _list_whisper_models() -> None:
    from ..transcription import list_whisper_models

    models = list_whisper_models()
    if not models:
        print("No whisper models found. Set WHISPER_MODELS_DIR or WHISPER_MODEL env var.")
    else:
        print("Available Whisper models:\n")
        for m in models:
            print(f"  {m['name']:<25} {m['size']:>10}")
        print(f"\nUsage: pidcast transcribe <input> --whisper-model {models[0]['name']}")


def _list_presets() -> None:
    from ..config_manager import ConfigManager

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
        print("      vad: true            # strip silence (anti-hallucination)")
        print("      vad_threshold: 0.5")
    else:
        print("Available presets:\n")
        for name, flags in presets.items():
            flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
            print(f"  {name}: {flag_str}")
        print("\nUsage: pidcast transcribe <input> -p <preset>")


def _list_profiles() -> None:
    from ..cookies import list_chrome_profiles

    profiles = list_chrome_profiles()
    if not profiles:
        print("No Chrome profiles found.")
    else:
        print("Available Chrome profiles:\n")
        print(f"  {'Display Name':<25} {'Directory':<20} {'Config Value'}")
        print(f"  {'-' * 25} {'-' * 20} {'-' * 30}")
        for dir_name, meta in profiles.items():
            print(f"  {meta['display_name']:<25} {dir_name:<20} {dir_name}")
        print(f'\nUsage: pidcast transcribe <input> --chrome-profile "{list(profiles.keys())[0]}"')
        print("   Or: Set 'chrome_profile' in ~/.config/pidcast/config.yaml")
