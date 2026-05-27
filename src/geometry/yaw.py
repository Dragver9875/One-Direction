from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .bearing import normalize_angle_rad


@dataclass(frozen=True)
class YawDerivationConfig:
    min_step_m: float = 0.25
    max_reasonable_speed_mps: float = 70.0
    fill_method: str = "nearest"


def fill_yaw_gaps(
    yaw: np.ndarray,
    method: str = "nearest",
    fallback: float = 0.0,
) -> np.ndarray:
    values = np.asarray(yaw, dtype=float).copy()

    if len(values) == 0:
        return values

    valid_idx = np.where(np.isfinite(values))[0]

    if len(valid_idx) == 0:
        return np.full_like(values, fill_value=float(fallback), dtype=float)

    if method == "zero":
        values[~np.isfinite(values)] = float(fallback)
        return normalize_angle_rad(values)

    if method == "nearest":
        invalid_idx = np.where(~np.isfinite(values))[0]
        for idx in invalid_idx:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
            values[idx] = values[nearest]
        return normalize_angle_rad(values)

    if method == "ffill_bfill":
        series = pd.Series(values)
        values = series.ffill().bfill().fillna(float(fallback)).to_numpy(dtype=float)
        return normalize_angle_rad(values)

    raise ValueError(f"Unknown fill method: {method}")


def derive_yaw_and_speed_for_group(
    group: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    timestamp_col: str = "timestamp",
    cfg: YawDerivationConfig = YawDerivationConfig(),
) -> pd.DataFrame:
    required = {x_col, y_col, timestamp_col}
    missing = required - set(group.columns)
    if missing:
        raise ValueError(f"group missing required columns: {sorted(missing)}")

    g = group.sort_values(timestamp_col, kind="mergesort").copy().reset_index(drop=True)

    n = len(g)
    g["t"] = np.arange(n, dtype=np.int64)

    x = g[x_col].to_numpy(dtype=float)
    y = g[y_col].to_numpy(dtype=float)
    ts = pd.to_datetime(g[timestamp_col], utc=True, errors="raise")

    dx_next = np.empty(n, dtype=float)
    dy_next = np.empty(n, dtype=float)
    dx_next[:-1] = np.diff(x)
    dy_next[:-1] = np.diff(y)
    dx_next[-1] = np.nan
    dy_next[-1] = np.nan

    dt_next = ts.shift(-1).sub(ts).dt.total_seconds().to_numpy(dtype=float)
    step_next = np.sqrt(dx_next**2 + dy_next**2)

    dx_prev = np.empty(n, dtype=float)
    dy_prev = np.empty(n, dtype=float)
    dx_prev[0] = np.nan
    dy_prev[0] = np.nan
    dx_prev[1:] = np.diff(x)
    dy_prev[1:] = np.diff(y)

    dt_prev = ts.diff().dt.total_seconds().to_numpy(dtype=float)
    step_prev = np.sqrt(dx_prev**2 + dy_prev**2)

    valid_next_yaw = (
        np.isfinite(dx_next)
        & np.isfinite(dy_next)
        & np.isfinite(dt_next)
        & (dt_next > 0)
        & (step_next >= cfg.min_step_m)
    )
    valid_prev_yaw = (
        np.isfinite(dx_prev)
        & np.isfinite(dy_prev)
        & np.isfinite(dt_prev)
        & (dt_prev > 0)
        & (step_prev >= cfg.min_step_m)
    )

    yaw_next = np.full(n, np.nan, dtype=float)
    yaw_prev = np.full(n, np.nan, dtype=float)

    yaw_next[valid_next_yaw] = np.arctan2(dy_next[valid_next_yaw], dx_next[valid_next_yaw])
    yaw_prev[valid_prev_yaw] = np.arctan2(dy_prev[valid_prev_yaw], dx_prev[valid_prev_yaw])

    yaw = np.where(np.isfinite(yaw_next), yaw_next, yaw_prev)
    yaw = fill_yaw_gaps(yaw, method=cfg.fill_method, fallback=0.0)

    valid_next_speed = np.isfinite(step_next) & np.isfinite(dt_next) & (dt_next > 0)
    valid_prev_speed = np.isfinite(step_prev) & np.isfinite(dt_prev) & (dt_prev > 0)

    speed_next = np.full(n, np.nan, dtype=float)
    speed_prev = np.full(n, np.nan, dtype=float)

    speed_next[valid_next_speed] = step_next[valid_next_speed] / dt_next[valid_next_speed]
    speed_prev[valid_prev_speed] = step_prev[valid_prev_speed] / dt_prev[valid_prev_speed]

    speed = np.where(np.isfinite(speed_next), speed_next, speed_prev)

    speed = pd.Series(speed).ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    speed = np.clip(speed, 0.0, float(cfg.max_reasonable_speed_mps))

    g["yaw"] = yaw
    g["yaw_deg"] = np.degrees(yaw)
    g["speed_mps"] = speed
    g["dt_prev_s"] = dt_prev
    g["dt_next_s"] = dt_next
    g["step_prev_m"] = step_prev
    g["step_next_m"] = step_next

    return g


def derive_yaw_and_speed(
    df: pd.DataFrame,
    trajectory_col: str = "trajectory_id",
    x_col: str = "x",
    y_col: str = "y",
    timestamp_col: str = "timestamp",
    cfg: YawDerivationConfig = YawDerivationConfig(),
) -> pd.DataFrame:
    required = {trajectory_col, x_col, y_col, timestamp_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    chunks = []
    for _, group in df.groupby(trajectory_col, sort=True):
        chunks.append(
            derive_yaw_and_speed_for_group(
                group,
                x_col=x_col,
                y_col=y_col,
                timestamp_col=timestamp_col,
                cfg=cfg,
            )
        )

    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values([trajectory_col, "t"], kind="mergesort").reset_index(drop=True)


def yaw_change_series(yaw: Iterable[float]) -> np.ndarray:
    yaw_arr = np.asarray(list(yaw), dtype=float)
    if len(yaw_arr) == 0:
        return np.asarray([], dtype=float)
    if len(yaw_arr) == 1:
        return np.asarray([0.0], dtype=float)

    diff = np.empty_like(yaw_arr, dtype=float)
    diff[0] = 0.0
    diff[1:] = normalize_angle_rad(np.diff(yaw_arr))
    return diff


__all__ = [
    "YawDerivationConfig",
    "fill_yaw_gaps",
    "derive_yaw_and_speed_for_group",
    "derive_yaw_and_speed",
    "yaw_change_series",
]
