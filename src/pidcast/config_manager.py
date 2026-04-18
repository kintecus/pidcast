"""User config management (~/.config/pidcast/config.yaml) for pidcast."""

import logging
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_TRANSCRIPTS_DIR,
    OBSIDIAN_PATH,
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage pidcast configuration files."""

    @staticmethod
    def ensure_config_dir() -> Path:
        """Ensure config directory exists."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return CONFIG_DIR

    @staticmethod
    def load_config() -> dict[str, Any]:
        """Load config.yaml with defaults."""
        if not CONFIG_FILE.exists():
            return ConfigManager._default_config()

        try:
            yaml = YAML()
            with open(CONFIG_FILE, encoding="utf-8") as f:
                config = yaml.load(f)
            return config or ConfigManager._default_config()
        except Exception as e:
            logger.warning(f"Failed to load config from {CONFIG_FILE}: {e}")
            return ConfigManager._default_config()

    @staticmethod
    def save_config(config: dict[str, Any]) -> bool:
        """Save configuration to config.yaml."""
        try:
            ConfigManager.ensure_config_dir()
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 4096

            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(config, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {CONFIG_FILE}: {e}")
            return False

    @staticmethod
    def init_default_config() -> bool:
        """Initialize default config.yaml if it doesn't exist."""
        if CONFIG_FILE.exists():
            return True

        config = ConfigManager._default_config()
        ConfigManager.ensure_config_dir()

        try:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 4096

            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write("# Pidcast user config\n\n")
                f.write("# Directory for transcript output\n")
                f.write(f'output_dir: "{config["output_dir"]}"\n\n')
                f.write("# Obsidian vault path (optional)\n")
                if config["obsidian_vault"]:
                    f.write(f'obsidian_vault: "{config["obsidian_vault"]}"\n\n')
                else:
                    f.write("obsidian_vault: null\n\n")
                f.write("# Chrome profile for cookie extraction (display name or directory name)\n")
                f.write("# Run 'pidcast --list-chrome-profiles' to see available profiles\n")
                f.write("chrome_profile: null\n")

            logger.info(f"Created default config at {CONFIG_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
            return False

    @staticmethod
    def load_preset(name: str) -> dict[str, Any]:
        """Load a named preset from config."""
        config = ConfigManager.load_config()
        presets = config.get("presets")
        if not presets:
            raise ValueError(f"No presets defined in config. Add presets to {CONFIG_FILE}")
        if name not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        return dict(presets[name])

    @staticmethod
    def list_presets() -> dict[str, dict[str, Any]]:
        """List all defined presets."""
        config = ConfigManager.load_config()
        presets = config.get("presets")
        if not presets:
            return {}
        return {name: dict(flags) for name, flags in presets.items()}

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Default configuration."""
        return {
            "output_dir": str(DEFAULT_TRANSCRIPTS_DIR),
            "obsidian_vault": OBSIDIAN_PATH,
            "chrome_profile": None,
        }
