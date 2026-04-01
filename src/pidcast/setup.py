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


def run_all_checks() -> list[CheckResult]:
    """Run all health checks and return results."""
    return [
        check_env_file(),
        check_ffmpeg(),
        check_whisper(),
        check_whisper_model(),
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
