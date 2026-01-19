"""Configuration file management for pidcast library."""

import logging
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .config import (
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_BACKFILL_LIMIT,
    DEFAULT_FEED_CACHE_HOURS,
    DEFAULT_TRANSCRIPTS_DIR,
    OBSIDIAN_PATH,
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage pidcast configuration files."""

    @staticmethod
    def ensure_config_dir() -> Path:
        """Ensure config directory exists.

        Returns:
            Path to config directory
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return CONFIG_DIR

    @staticmethod
    def load_config() -> dict[str, Any]:
        """Load config.yaml with defaults.

        Returns:
            Configuration dictionary
        """
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
        """Save configuration to config.yaml.

        Args:
            config: Configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            ConfigManager.ensure_config_dir()
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 4096  # Prevent line wrapping

            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(config, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {CONFIG_FILE}: {e}")
            return False

    @staticmethod
    def init_default_config() -> bool:
        """Initialize default config.yaml if it doesn't exist.

        Returns:
            True if created or already exists, False on error
        """
        if CONFIG_FILE.exists():
            return True

        config = ConfigManager._default_config()
        ConfigManager.ensure_config_dir()

        try:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.width = 4096

            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                # Write comments manually for better documentation
                f.write("# Pidcast Global Configuration\n\n")
                f.write("# Number of recent episodes to process when adding a new show\n")
                f.write(f"backfill_limit: {config['backfill_limit']}\n\n")
                f.write("# Directory for transcript output\n")
                f.write(f"output_dir: \"{config['output_dir']}\"\n\n")
                f.write("# Obsidian vault path (optional)\n")
                if config['obsidian_vault']:
                    f.write(f"obsidian_vault: \"{config['obsidian_vault']}\"\n\n")
                else:
                    f.write("obsidian_vault: null\n\n")
                f.write("# RSS feed cache duration (hours)\n")
                f.write(f"feed_cache_hours: {config['feed_cache_hours']}\n")

            logger.info(f"Created default config at {CONFIG_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
            return False

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "backfill_limit": DEFAULT_BACKFILL_LIMIT,
            "output_dir": str(DEFAULT_TRANSCRIPTS_DIR),
            "obsidian_vault": OBSIDIAN_PATH,
            "feed_cache_hours": DEFAULT_FEED_CACHE_HOURS,
        }
