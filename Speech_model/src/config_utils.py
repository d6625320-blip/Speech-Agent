"""Helpers for loading experiment configuration files.

Relative paths in YAML files are resolved from the ``Speech_model`` directory,
so the same command works from either the repository root or ``Speech_model``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent


def resolve_existing_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, PROJECT_DIR / path, REPOSITORY_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    path = resolve_existing_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def config_value(
    config: dict[str, Any], section: str, key: str, default: Any = None
) -> Any:
    section_data = config.get(section, {})
    if isinstance(section_data, dict) and key in section_data:
        return section_data[key]
    return config.get(key, default)


def choose(cli_value: Any, config_value_: Any, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value_ is not None:
        return config_value_
    return default
