from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(path: str | Path) -> Path:
    path = to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = to_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    path = to_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=indent, default=_json_default)
    return path


def load_yaml(path: str | Path) -> Any:
    path = to_path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str | Path) -> Path:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def load_pickle(path: str | Path) -> Any:
    path = to_path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str | Path) -> Path:
    path = ensure_parent(path)
    with path.open("wb") as f:
        pickle.dump(data, f)
    return path


def read_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = to_path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, **kwargs)
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path, **kwargs)

    raise ValueError(f"Unsupported dataframe input format: {path}")


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    path = ensure_parent(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df.to_parquet(path, index=index, **kwargs)
        return path
    if suffix == ".csv":
        df.to_csv(path, index=index, **kwargs)
        return path
    if suffix in {".json", ".jsonl"}:
        df.to_json(path, orient="records", lines=suffix == ".jsonl", **kwargs)
        return path
    if suffix in {".feather", ".arrow"}:
        df.to_feather(path, **kwargs)
        return path

    raise ValueError(f"Unsupported dataframe output format: {path}")


def path_exists(path: str | Path) -> bool:
    return to_path(path).exists()


def require_file(path: str | Path) -> Path:
    path = to_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected file, got directory: {path}")
    return path


def require_dir(path: str | Path) -> Path:
    path = to_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    directory = require_dir(directory)
    iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)
