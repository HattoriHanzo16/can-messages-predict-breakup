"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config file is invalid or empty")

    return config
