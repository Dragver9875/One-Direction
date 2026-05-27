from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pyproj import Transformer


def normalize_angle_rad(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def project_lonlat_dataframe(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
) -> pd.DataFrame:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(df[lon_col].to_numpy(), df[lat_col].to_numpy())

    out = df.copy()
    out["x"] = np.asarray(x, dtype=float)
    out["y"] = np.asarray(y, dtype=float)
    return out


def compute_single_trajectory_kinematics(
    group: pd.DataFrame,
    min_step_m: float = 0.25,
) -> pd.DataFrame:
    g = group.sort_values("timestamp", kind="mergesort").copy()
    n = len(g)
    g["t"] = np.arange(n, dtype=np.int64)

    x = g["x"].to_numpy(dtype=float)
    y = g["y"].to_numpy(dtype=float)
    ts = pd.to_datetime(g["timestamp"], utc=True)

    x_prev = np.roll(x, 1)
    y_prev = np.roll(y, 1)
    dx_prev = x - x_prev
    dy_prev = y - y_prev
    dx_prev[0] = np.nan
    dy_prev[0] = np.nan

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    dx_next = x_next - x
    dy_next = y_next - y
    dx_next[-1] = np.nan
    dy_next[-1] = np.nan

    dt_prev = ts.diff().dt.total_seconds().to_numpy(dtype=float)
    dt_next = ts.shift(-1).sub(ts).dt.total_seconds().to_numpy(dtype=float)

    step_prev = np.sqrt(dx_prev**2 + dy_prev**2)
    step_next = np.sqrt(dx_next**2 + dy_next**2)

    valid_prev = (
        np.isfinite(dx_prev)
        & np.isfinite(dy_prev)
        & np.isfinite(dt_prev)
        & (dt_prev > 0.0)
        & (step_prev >= min_step_m)
    )
    valid_next = (
        np.isfinite(dx_next)
        & np.isfinite(dy_next)
        & np.isfinite(dt_next)
        & (dt_next > 0.0)
        & (step_next >= min_step_m)
    )

    yaw_prev = np.full(n, np.nan, dtype=float)
    yaw_next = np.full(n, np.nan, dtype=float)
    yaw_prev[valid_prev] = np.arctan2(dy_prev[valid_prev], dx_prev[valid_prev])
    yaw_next[valid_next] = np.arctan2(dy_next[valid_next], dx_next[valid_next])

    yaw = np.where(np.isfinite(yaw_next), yaw_next, yaw_prev)
    yaw = pd.Series(yaw, index=g.index).ffill().bfill().fillna(0.0).to_numpy()
    yaw = normalize_angle_rad(yaw)

    speed_prev = np.full(n, np.nan, dtype=float)
    speed_next = np.full(n, np.nan, dtype=float)

    valid_prev_speed = np.isfinite(dt_prev) & (dt_prev > 0.0) & np.isfinite(step_prev)
    valid_next_speed = np.isfinite(dt_next) & (dt_next > 0.0) & np.isfinite(step_next)

    speed_prev[valid_prev_speed] = step_prev[valid_prev_speed] / dt_prev[valid_prev_speed]
    speed_next[valid_next_speed] = step_next[valid_next_speed] / dt_next[valid_next_speed]
    speed = np.where(np.isfinite(speed_next), speed_next, speed_prev)
    speed = pd.Series(speed, index=g.index).ffill().bfill().fillna(0.0).to_numpy()

    g["dt_prev_s"] = dt_prev
    g["dt_next_s"] = dt_next
    g["step_prev_m"] = step_prev
    g["step_next_m"] = step_next
    g["yaw"] = yaw
    g["yaw_deg"] = np.degrees(yaw)
    g["speed_mps"] = speed
    return g


def preprocess_points_dataframe(
    raw_points: pd.DataFrame,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
    min_step_m: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    required = {"trajectory_id", "timestamp", "lon", "lat"}
    missing = required - set(raw_points.columns)
    if missing:
        raise ValueError(f"raw_points is missing required columns: {sorted(missing)}")

    df = raw_points.copy()
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")

    df = project_lonlat_dataframe(
        df,
        lon_col="lon",
        lat_col="lat",
        source_crs=source_crs,
        target_crs=target_crs,
    )

    df = df.sort_values(["trajectory_id", "timestamp"], kind="mergesort")
    df = df.reset_index(drop=True)

    out = (
        df.groupby("trajectory_id", group_keys=False)
        .apply(lambda g: compute_single_trajectory_kinematics(g, min_step_m=min_step_m))
        .reset_index(drop=True)
    )

    columns = [
        "trajectory_id",
        "t",
        "timestamp",
        "lon",
        "lat",
        "x",
        "y",
        "yaw",
        "yaw_deg",
        "speed_mps",
        "dt_prev_s",
        "dt_next_s",
        "step_prev_m",
        "step_next_m",
    ]
    if "source_geometry_wkt" in out.columns:
        columns.append("source_geometry_wkt")

    out = out[columns]

    counts = out.groupby("trajectory_id").size()
    report = {
        "rows": int(len(out)),
        "num_trajectories": int(out["trajectory_id"].nunique()),
        "points_per_trajectory": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
        },
        "source_crs": source_crs,
        "target_crs": target_crs,
        "min_step_m": float(min_step_m),
    }
    return out, report
