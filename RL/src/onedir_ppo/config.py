from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def deep_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(config: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    cur = config
    parts = key.split('.')
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return config


def parse_value(text: str) -> Any:
    value = text.strip()
    if value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    if value.lower() in {'none', 'null'}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'Override must use key=value format: {item}')
        key, value = item.split('=', 1)
        deep_set(config, key, parse_value(value))
    return config


def resolve_device(device: str = 'auto') -> str:
    if device == 'auto':
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    return device
