from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def create_trajectory_splits(
    trajectory_ids: Sequence[int] | np.ndarray | pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0. Got {total} from "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    ids = np.array(sorted(set(int(x) for x in trajectory_ids)), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    if n_train + n_val > n:
        n_val = max(0, n - n_train)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


def save_split_ids(ids: Iterable[int], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for trajectory_id in ids:
            f.write(f"{int(trajectory_id)}\n")

    return path


def load_split_ids(path: str | Path) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(line))
    return ids


def split_dataframe_by_trajectory_id(
    df: pd.DataFrame,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    id_col: str = "trajectory_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if id_col not in df.columns:
        raise ValueError(f"DataFrame missing id column: {id_col}")

    train_set = set(int(x) for x in train_ids)
    val_set = set(int(x) for x in val_ids)
    test_set = set(int(x) for x in test_ids)

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train/val/test trajectory ID sets must be disjoint.")

    train_df = df[df[id_col].astype(int).isin(train_set)].copy()
    val_df = df[df[id_col].astype(int).isin(val_set)].copy()
    test_df = df[df[id_col].astype(int).isin(test_set)].copy()

    return train_df, val_df, test_df
