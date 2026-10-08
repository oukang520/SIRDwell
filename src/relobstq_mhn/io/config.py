"""Small configuration loader with explicit path handling."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML mapping and reject empty or non-mapping documents."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    return value
