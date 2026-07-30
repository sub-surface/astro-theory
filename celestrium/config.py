"""Tiny local instrument config — theme persistence today, one future home
for prefs/credentials (a `~/.celestrium/config.toml`-style need flagged in
the 2026-07 feature review) once more than one setting needs to survive a
restart. Deliberately minimal: a flat JSON dict under data/config.json
(already git-ignored alongside cache/manifest/candidates).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import paths

CONFIG_PATH = paths.DATA / "config.json"


def load() -> dict[str, Any]:
    """Read the config dict, or {} if it doesn't exist yet or is corrupt."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(updates: dict[str, Any]) -> None:
    """Merge `updates` into the on-disk config and write it back."""
    cfg = load()
    cfg.update(updates)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set(key: str, value: Any) -> None:
    save({key: value})
