"""Setup wizard and health check for pidcast."""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


@dataclass
class CheckResult:
    """Result of a single dependency/config check."""

    name: str
    ok: bool
    detail: str
    hint: str = ""
    required: bool = True


def check_ffmpeg() -> CheckResult:
    """Check if ffmpeg is available."""
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")
    path = shutil.which(ffmpeg_path)
    if path:
        try:
            result = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
            version = result.stdout.split("\n")[0].split(" ")[2] if result.returncode == 0 else "?"
            return CheckResult("ffmpeg", True, f"ok ({version}, {path})")
        except Exception:
            return CheckResult("ffmpeg", True, f"found ({path})")
    return CheckResult(
        "ffmpeg",
        False,
        "not found",
        hint="Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)",
    )


def check_whisper() -> CheckResult:
    """Check if whisper.cpp is configured and available."""
    whisper_path = os.environ.get("WHISPER_CPP_PATH")
    if not whisper_path:
        return CheckResult(
            "whisper.cpp",
            False,
            "not configured",
            hint="Set WHISPER_CPP_PATH in .env, or use ElevenLabs instead",
            required=False,
        )
    path = shutil.which(whisper_path) or (
        Path(whisper_path) if Path(whisper_path).exists() else None
    )
    if path:
        return CheckResult("whisper.cpp", True, f"ok ({whisper_path})")
    return CheckResult(
        "whisper.cpp",
        False,
        f"not found at: {whisper_path}",
        hint="Check WHISPER_CPP_PATH in .env points to the whisper-cli binary",
        required=False,
    )


def check_whisper_model() -> CheckResult:
    """Check if a whisper model is available."""
    model = os.environ.get("WHISPER_MODEL")
    models_dir = os.environ.get("WHISPER_MODELS_DIR")
    if model and Path(model).exists():
        size_mb = Path(model).stat().st_size / (1024 * 1024)
        return CheckResult("whisper model", True, f"ok ({Path(model).name}, {size_mb:.0f} MB)")
    if models_dir and Path(models_dir).is_dir():
        models = list(Path(models_dir).glob("ggml-*.bin"))
        if models:
            return CheckResult(
                "whisper model", True, f"ok ({len(models)} model(s) in {models_dir})"
            )
    if not model and not models_dir:
        return CheckResult(
            "whisper model",
            False,
            "not configured",
            hint="Set WHISPER_MODEL in .env",
            required=False,
        )
    return CheckResult(
        "whisper model",
        False,
        "not found",
        hint="Download a model: cd whisper.cpp && bash models/download-ggml-model.sh base.en",
        required=False,
    )


def check_vad_model() -> CheckResult:
    """Check if a Silero VAD model is available (optional, enables --vad)."""
    from .transcription import resolve_vad_model

    vad_model = resolve_vad_model()
    if vad_model:
        size_mb = Path(vad_model).stat().st_size / (1024 * 1024)
        return CheckResult("VAD model", True, f"ok ({Path(vad_model).name}, {size_mb:.0f} MB)")
    return CheckResult(
        "VAD model",
        False,
        "not found",
        hint="Optional, enables --vad anti-hallucination. Download: "
        "cd whisper.cpp && bash models/download-vad-model.sh silero-v5.1.2 "
        "(then set WHISPER_VAD_MODEL in .env)",
        required=False,
    )


def check_env_var(name: str, display: str, required: bool = True, hint: str = "") -> CheckResult:
    """Check if an environment variable is set."""
    value = os.environ.get(name)
    if value:
        # Mask the value for display
        masked = value[:4] + "..." if len(value) > 4 else "***"
        return CheckResult(display, True, f"set ({masked})")
    return CheckResult(display, False, "not set", hint=hint, required=required)


def check_env_file() -> CheckResult:
    """Check if .env file exists."""
    if ENV_FILE.exists():
        return CheckResult(".env file", True, f"found ({ENV_FILE})")
    return CheckResult(
        ".env file",
        False,
        "not found",
        hint=f"Create one: cp {ENV_EXAMPLE} {ENV_FILE}"
        if ENV_EXAMPLE.exists()
        else "Create .env in project root",
    )


def check_data_dirs() -> CheckResult:
    """Report where pidcast stores generated data (and create the tree)."""
    from .config import DATA_DIR, ensure_data_dirs

    try:
        ensure_data_dirs()
    except Exception as e:
        return CheckResult("data dir", False, f"could not create {DATA_DIR}: {e}", required=False)
    return CheckResult(
        "data dir",
        True,
        f"ok ({DATA_DIR})",
        hint="Override with PIDCAST_DATA_DIR or XDG_DATA_HOME. See 'pidcast info'.",
        required=False,
    )


def check_stats_integrity() -> CheckResult:
    """Check for phantom run entries (success recorded, transcript file gone)."""
    from .config import RUNS_FILE
    from .utils import find_phantom_stats

    try:
        phantoms = find_phantom_stats(RUNS_FILE)
    except Exception as e:
        return CheckResult("stats integrity", True, f"skipped ({e})", required=False)
    if not phantoms:
        return CheckResult("stats integrity", True, "ok (no phantom entries)", required=False)
    return CheckResult(
        "stats integrity",
        False,
        f"{len(phantoms)} stats entr(y/ies) point at a missing transcript",
        hint="These block duplicate detection. Clean them with: pidcast doctor --prune-stats",
        required=False,
    )


def run_all_checks() -> list[CheckResult]:
    """Run all health checks and return results."""
    return [
        check_env_file(),
        check_ffmpeg(),
        check_whisper(),
        check_whisper_model(),
        check_vad_model(),
        check_data_dirs(),
        check_stats_integrity(),
        check_env_var(
            "GROQ_API_KEY",
            "GROQ_API_KEY",
            required=False,
            hint="Get a free key at: https://console.groq.com/",
        ),
        check_env_var(
            "ELEVENLABS_API_KEY",
            "ELEVENLABS_API_KEY",
            required=False,
            hint="Get a key at: https://elevenlabs.io/",
        ),
        check_env_var(
            "HUGGINGFACE_TOKEN",
            "HUGGINGFACE_TOKEN",
            required=False,
            hint="Needed only for --diarize. Get at: https://huggingface.co/settings/tokens",
        ),
    ]


def determine_status(checks: list[CheckResult]) -> tuple[str, str]:
    """Determine overall status and a tip based on check results.

    Returns:
        Tuple of (status_message, tip_message)
    """
    has_whisper = any(c.name == "whisper.cpp" and c.ok for c in checks)
    has_elevenlabs = any(c.name == "ELEVENLABS_API_KEY" and c.ok for c in checks)
    has_ffmpeg = any(c.name == "ffmpeg" and c.ok for c in checks)
    has_groq = any(c.name == "GROQ_API_KEY" and c.ok for c in checks)

    if not has_ffmpeg:
        return "not ready", "Install ffmpeg first: brew install ffmpeg"

    if not has_whisper and not has_elevenlabs:
        return "not ready", "Run 'pidcast setup' to configure a transcription provider"

    provider = "whisper (local)" if has_whisper else "ElevenLabs (cloud)"
    analysis = " + Groq analysis" if has_groq else " (transcription only, no analysis)"

    tip = ""
    if not has_groq:
        tip = "Get a free Groq key for AI analysis: https://console.groq.com/"
    elif not has_whisper and has_elevenlabs:
        tip = "For local/private transcription, run 'pidcast setup' to configure whisper.cpp"

    return f"ready ({provider}{analysis})", tip


def read_env_file() -> dict[str, str]:
    """Read key-value pairs from .env file."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


def write_env_var(key: str, value: str) -> None:
    """Write or update a key in the .env file."""
    if not ENV_FILE.exists():
        # Copy from example if available, else create empty
        if ENV_EXAMPLE.exists():
            import shutil as _shutil

            _shutil.copy2(ENV_EXAMPLE, ENV_FILE)
        else:
            ENV_FILE.touch()

    content = ENV_FILE.read_text()
    lines = content.splitlines()

    # Find and replace existing key, or append
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            found = True
            break

    if not found:
        lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(lines) + "\n")


# ============================================================================
# CLI HANDLERS (pidcast doctor / pidcast setup)
# ============================================================================


def run_doctor(args=None) -> None:
    """Run health checks and display configuration status (``pidcast doctor``)."""
    from .config import RUNS_FILE
    from .utils import prune_phantom_stats

    prune_stats = bool(getattr(args, "prune_stats", False))

    print("\npidcast doctor")
    print("=" * 40)

    if prune_stats:
        removed = prune_phantom_stats(RUNS_FILE)
        print(f"  Pruned {removed} phantom stats entr(y/ies) with a missing transcript.\n")

    checks = run_all_checks()
    max_name = max(len(c.name) for c in checks)

    for check in checks:
        dots = "." * (max_name + 6 - len(check.name))
        if check.ok:
            print(f"  {check.name} {dots} {check.detail}")
        else:
            label = "MISSING" if check.required else "not set"
            msg = check.detail if check.detail != "not set" else label
            print(f"  {check.name} {dots} {msg}")
            if check.hint:
                print(f"    -> {check.hint}")

    status, tip = determine_status(checks)
    print(f"\n  Status: {status}")
    if tip:
        print(f"  Tip: {tip}")
    print()


def run_setup(args=None) -> None:
    """Interactive setup wizard for first-time users (``pidcast setup``)."""
    print("\npidcast setup")
    print("=" * 40)

    # Step 1: System dependencies
    print("\nStep 1/3: Checking system dependencies...\n")

    ffmpeg = check_ffmpeg()
    if ffmpeg.ok:
        print(f"  ffmpeg: {ffmpeg.detail}")
    else:
        print("  ffmpeg: not found")
        print("    Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        print("    Then re-run: pidcast setup")
        return

    whisper = check_whisper()
    whisper_model = check_whisper_model()
    use_elevenlabs = False

    if whisper.ok and whisper_model.ok:
        print(f"  whisper.cpp: {whisper.detail}")
        print(f"  whisper model: {whisper_model.detail}")
        vad = check_vad_model()
        if vad.ok:
            print(f"  VAD model: {vad.detail}")
        else:
            print("  VAD model: not found (optional, enables --vad anti-hallucination)")
            print("    Download: cd whisper.cpp && bash models/download-vad-model.sh silero-v5.1.2")
            print("    Then set in .env: WHISPER_VAD_MODEL=/path/to/ggml-silero-v5.1.2.bin")
    else:
        print("  whisper.cpp: not configured")
        print("\n  whisper.cpp is needed for local (private) transcription.")
        print("  You can skip it and use ElevenLabs (cloud) instead.\n")
        print("  [1] I'll set up whisper.cpp myself (see README for instructions)")
        print("  [2] Skip - use ElevenLabs cloud transcription")
        try:
            choice = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return
        if choice == "2":
            use_elevenlabs = True
        else:
            print("\n  To set up whisper.cpp:")
            print("    1. git clone https://github.com/ggerganov/whisper.cpp.git")
            print("    2. cd whisper.cpp && make")
            print("    3. bash models/download-ggml-model.sh base.en")
            print("    4. Set in .env: WHISPER_CPP_PATH=/path/to/whisper.cpp/build/bin/whisper-cli")
            print("    5. Set in .env: WHISPER_MODEL=/path/to/whisper.cpp/models/ggml-base.en.bin")
            print("    6. Re-run: pidcast setup")
            return

    # Step 2: API keys
    print("\nStep 2/3: API keys\n")

    # Groq (for analysis)
    groq = check_env_var("GROQ_API_KEY", "GROQ_API_KEY")
    if groq.ok:
        print(f"  GROQ_API_KEY: {groq.detail}")
    else:
        print("  GROQ_API_KEY: not set (needed for AI analysis)")
        print("  Get a free key at: https://console.groq.com/")
        try:
            key = input("  Paste your key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
        if key:
            write_env_var("GROQ_API_KEY", key)
            print("  Saved to .env")
        else:
            print("  Skipped - transcription will work, but AI analysis will be disabled")

    # ElevenLabs (if chosen or if no whisper)
    if use_elevenlabs:
        el = check_env_var("ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY")
        if el.ok:
            print(f"  ELEVENLABS_API_KEY: {el.detail}")
        else:
            print("\n  ELEVENLABS_API_KEY: not set (needed for cloud transcription)")
            print("  Get a key at: https://elevenlabs.io/")
            try:
                key = input("  Paste your key: ").strip()
            except (EOFError, KeyboardInterrupt):
                key = ""
            if key:
                write_env_var("ELEVENLABS_API_KEY", key)
                print("  Saved to .env")
            else:
                print("  Without this key, transcription won't work.")
                print("  Add it to .env later: ELEVENLABS_API_KEY=your_key")
                return

    # Step 3: Summary
    print("\nStep 3/3: Ready!\n")

    if not ENV_FILE.exists():
        print(f"  Note: .env file created at {ENV_FILE}")

    if use_elevenlabs:
        print("  Provider: ElevenLabs (cloud)")
        print("\n  Try it:")
        print('    pidcast "https://youtube.com/watch?v=VIDEO_ID"')
    else:
        print("  Provider: whisper.cpp (local)")
        print("\n  Try it:")
        print('    pidcast "https://youtube.com/watch?v=VIDEO_ID"')

    print()
