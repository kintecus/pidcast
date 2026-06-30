import os
from pathlib import Path

import pytest

import pidcast.config as config
from pidcast.config import PROJECT_ROOT, get_data_dir, get_project_root


def test_project_root_exists():
    """Test that project root is correctly resolved and exists."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_get_project_root():
    """Test get_project_root function."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "src").exists()


# ============================================================================
# get_data_dir() - XDG data dir resolution
#
# These test the FUNCTION, not the module-level DATA_DIR constant, which is
# frozen at import time. PIDCAST_DATA_DIR must be set before import to affect
# the constant; the function re-reads the environment on every call.
# ============================================================================


def test_data_dir_pidcast_override_wins(monkeypatch):
    """PIDCAST_DATA_DIR takes precedence over everything, on all platforms."""
    monkeypatch.setenv("PIDCAST_DATA_DIR", "/custom/pidcast/data")
    monkeypatch.setenv("XDG_DATA_HOME", "/should/be/ignored")
    assert get_data_dir() == Path("/custom/pidcast/data")


def test_data_dir_override_expands_user(monkeypatch):
    """A ~ in PIDCAST_DATA_DIR is expanded."""
    monkeypatch.setenv("PIDCAST_DATA_DIR", "~/pidcast-data")
    assert get_data_dir() == Path.home() / "pidcast-data"


def test_data_dir_honors_xdg_data_home(monkeypatch):
    """On POSIX, XDG_DATA_HOME/pidcast is used when no override is set."""
    monkeypatch.delenv("PIDCAST_DATA_DIR", raising=False)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
    assert get_data_dir() == Path("/xdg/data/pidcast")


def test_data_dir_posix_default(monkeypatch):
    """On POSIX with no env set, defaults to ~/.local/share/pidcast."""
    monkeypatch.delenv("PIDCAST_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("os.name", "posix")
    assert get_data_dir() == Path.home() / ".local" / "share" / "pidcast"


def test_data_dir_override_beats_xdg(monkeypatch):
    """PIDCAST_DATA_DIR beats XDG_DATA_HOME on POSIX."""
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
    monkeypatch.setenv("PIDCAST_DATA_DIR", "/override")
    assert get_data_dir() == Path("/override")


# pathlib hard-binds concrete Path classes to the host OS: Path.home() and
# Path(r"C:\...") cannot instantiate a WindowsPath on a POSIX host. So the
# Windows branch can only be exercised meaningfully when actually on Windows.
@pytest.mark.skipif(os.name != "nt", reason="WindowsPath not instantiable on POSIX")
def test_data_dir_windows_localappdata(monkeypatch):
    """On Windows, %LOCALAPPDATA%/pidcast is used when set."""
    monkeypatch.delenv("PIDCAST_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\me\AppData\Local")
    assert get_data_dir() == Path(r"C:\Users\me\AppData\Local") / "pidcast"


@pytest.mark.skipif(os.name != "nt", reason="WindowsPath not instantiable on POSIX")
def test_data_dir_windows_default(monkeypatch):
    """On Windows with no LOCALAPPDATA, defaults to ~/AppData/Local/pidcast."""
    monkeypatch.delenv("PIDCAST_DATA_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert get_data_dir() == Path.home() / "AppData" / "Local" / "pidcast"


# ============================================================================
# Derived constants and back-compat aliases
# ============================================================================


def test_data_dir_constants_derive_from_data_dir():
    """The path constants hang off DATA_DIR with the documented layout."""
    assert config.TRANSCRIPTS_DIR == config.DATA_DIR / "transcripts"
    assert config.AUDIO_DIR == config.DATA_DIR / "audio"
    assert config.LOGS_DIR == config.DATA_DIR / "logs"
    assert config.STATE_DIR == config.DATA_DIR / "state"
    assert config.RUNS_FILE == config.STATE_DIR / "runs.json"
    assert config.ERROR_LOG_FILE == config.LOGS_DIR / "errors.jsonl"
    assert config.LOG_FILE == config.LOGS_DIR / "pidcast.log"


def test_backcompat_aliases_identity():
    """Deprecated names alias the new constants so old imports keep working."""
    assert config.DEFAULT_TRANSCRIPTS_DIR is config.TRANSCRIPTS_DIR
    assert config.DEFAULT_DIGESTS_DIR is config.DIGESTS_DIR
    assert config.DEFAULT_PROMPTS_FILE is config.PROMPTS_FILE
    assert config.DEFAULT_MODELS_FILE is config.MODELS_FILE
    assert config.DEFAULT_ANALYSIS_PROMPTS_FILE is config.PROMPTS_FILE
    # The legacy stats file is absorbed by the unified run history.
    assert config.DEFAULT_STATS_FILE is config.RUNS_FILE


def test_packaged_config_files_resolve_and_exist():
    """prompts.yaml / models.yaml resolve to a real file (repo or wheel)."""
    assert config.PROMPTS_FILE.exists()
    assert config.MODELS_FILE.exists()


def test_sync_logs_dir_removed():
    """The dead SYNC_LOGS_DIR constant is gone."""
    assert not hasattr(config, "SYNC_LOGS_DIR")


def test_ensure_data_dirs_creates_tree(monkeypatch, tmp_path):
    """ensure_data_dirs() creates the four subdirs idempotently."""
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    config.ensure_data_dirs()
    config.ensure_data_dirs()  # idempotent
    assert (tmp_path / "transcripts").is_dir()
    assert (tmp_path / "audio").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "state").is_dir()
