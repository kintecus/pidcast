"""Tests for ConfigManager's default config.yaml template.

The template is hand-written line by line, so it can silently drift from
``_default_config()`` - a missing key raised KeyError inside the try/except and
the file was never written. These tests pin template/defaults agreement and the
fact that what gets written actually loads back.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from ruamel.yaml import YAML

from pidcast.config_manager import ConfigManager


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "config" / "config.yaml"
    monkeypatch.setattr("pidcast.config_manager.CONFIG_FILE", path)
    monkeypatch.setattr("pidcast.config_manager.CONFIG_DIR", path.parent)
    return path


def test_init_writes_the_file(config_file):
    assert ConfigManager.init_default_config() is True
    assert config_file.exists()


def test_init_is_a_no_op_when_the_file_exists(config_file):
    config_file.parent.mkdir(parents=True)
    config_file.write_text("backfill_limit: 99\n", encoding="utf-8")

    assert ConfigManager.init_default_config() is True
    assert "99" in config_file.read_text(encoding="utf-8")


def test_written_config_round_trips_through_load(config_file):
    ConfigManager.init_default_config()
    loaded = ConfigManager.load_config()

    defaults = ConfigManager._default_config()
    assert loaded["backfill_limit"] == defaults["backfill_limit"]
    assert loaded["feed_cache_hours"] == defaults["feed_cache_hours"]
    assert loaded["reveal_on_success"] == defaults["reveal_on_success"]
    assert loaded["chrome_profile"] is None


def test_template_covers_every_default_key(config_file):
    # Guards the drift that broke this: a key in _default_config() with no
    # corresponding template line (or vice versa) means one of the two is stale.
    ConfigManager.init_default_config()
    written = YAML().load(config_file.read_text(encoding="utf-8"))

    assert set(written) == set(ConfigManager._default_config())


def test_output_dir_is_commented_out(config_file):
    # Baking the resolved transcripts dir into config pins a path that goes stale
    # if the data dir moves; it ships as a commented hint instead.
    ConfigManager.init_default_config()
    text = config_file.read_text(encoding="utf-8")

    assert "# output_dir:" in text
    assert "output_dir" not in YAML().load(text)


def test_obsidian_vault_is_written_when_configured(config_file):
    with patch("pidcast.config_manager.OBSIDIAN_PATH", "/vault"):
        ConfigManager.init_default_config()

    assert YAML().load(config_file.read_text(encoding="utf-8"))["obsidian_vault"] == "/vault"


def test_write_failure_is_reported(config_file):
    with patch("builtins.open", side_effect=OSError("read-only fs")):
        assert ConfigManager.init_default_config() is False


def test_ensure_config_dir_creates_the_tree(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "config"
    monkeypatch.setattr("pidcast.config_manager.CONFIG_DIR", target)

    assert ConfigManager.ensure_config_dir() == Path(target)
    assert target.is_dir()
