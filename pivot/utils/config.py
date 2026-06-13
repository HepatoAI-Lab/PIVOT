from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def resolve_config_path(cfg: dict[str, Any], value: str | Path | None) -> str | None:
    """Resolve relative paths against the directory containing the YAML config."""
    if value is None or str(value) == "":
        return None
    value = Path(value)
    if value.is_absolute():
        return str(value)
    config_dir = Path(cfg.get("_config_dir", "."))
    return str((config_dir / value).resolve())
