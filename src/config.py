from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    config: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return deep_get(self.config, key, default)

    def require(self, key: str) -> Any:
        value = deep_get(self.config, key, None)
        if value is None:
            raise ConfigError(f"Required config key is missing: {key}")
        return value

    def path(self, key: str, default: str | Path | None = None) -> Path:
        value = deep_get(self.config, key, default)
        if value is None:
            raise ConfigError(f"Required path config key is missing: {key}")
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path

    def ensure_dir(self, key: str, default: str | Path | None = None) -> Path:
        path = self.path(key, default)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config_file(path: str | Path, required: bool = True) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")

    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config

    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


def deep_set(config: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    current = config
    parts = dotted_key.split(".")

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value
    return config


def parse_override_value(value: str) -> Any:
    text = value.strip()

    if text.lower() in {"true", "false"}:
        return text.lower() == "true"

    if text.lower() in {"none", "null"}:
        return None

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_override_value(item.strip()) for item in inner.split(",")]

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def apply_cli_overrides(
    config: dict[str, Any],
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    if not overrides:
        return config

    merged = dict(config)

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Override must use KEY=VALUE format, got: {override}")
        key, raw_value = override.split("=", 1)
        deep_set(merged, key.strip(), parse_override_value(raw_value))

    return merged


def find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    return current


def load_project_config(
    root: str | Path | None = None,
    config_files: Iterable[str | Path] | None = None,
    include_local: bool = True,
    overrides: Iterable[str] | None = None,
) -> ProjectConfig:
    repo_root = find_repo_root(root)

    if config_files is None:
        config_files = [
            "configs/default.yaml",
            "configs/data.yaml",
            "configs/model.yaml",
            "configs/train.yaml",
            "configs/eval.yaml",
        ]

    merged: dict[str, Any] = {}

    for path in config_files:
        path = Path(path)
        if not path.is_absolute():
            path = repo_root / path
        merged = deep_merge(merged, load_config_file(path, required=False))

    if include_local:
        local_path = repo_root / "configs/local.yaml"
        merged = deep_merge(merged, load_config_file(local_path, required=False))

    merged = apply_cli_overrides(merged, overrides)
    return ProjectConfig(root=repo_root, config=merged)


def resolve_device(device: str | None = None) -> str:
    requested = device or "auto"

    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    if requested not in {"cpu", "cuda"}:
        raise ConfigError(f"Unsupported device: {requested}")

    if requested == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise ConfigError("CUDA was requested but is not available.")
        except ImportError as exc:
            raise ConfigError("CUDA was requested but torch is not installed.") from exc

    return requested


def save_resolved_config(project_config: ProjectConfig, path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = project_config.root / path

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(project_config.config, f, sort_keys=False)

    return path
