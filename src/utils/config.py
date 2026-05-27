from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_merged_config(paths: list[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        merged = deep_update(merged, load_config(path))

    return merged


def get_by_dotted_key(config: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_by_dotted_key(config: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    current = config
    parts = key.split(".")

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str] | None = None) -> dict[str, Any]:
    result = deepcopy(config)

    if not overrides:
        return result

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")

        key, raw_value = item.split("=", 1)
        value = parse_scalar(raw_value)
        set_by_dotted_key(result, key, value)

    return result


def parse_scalar(value: str) -> Any:
    text = value.strip()

    if text.lower() in {"true", "false"}:
        return text.lower() == "true"

    if text.lower() in {"null", "none"}:
        return None

    try:
        if "." not in text:
            return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text
