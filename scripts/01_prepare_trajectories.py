from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point

REQUIRED_COLUMNS = {"id", "time", "geometry"}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def parse_point_wkt(value: str) -> tuple[float, float]:
    geom = wkt.loads(value)
    if not isinstance(geom, Point):
        raise ValueError(f"Expected POINT geometry, got {geom.geom_type}: {value}")
    return float(geom.x), float(geom.y)


def load_points(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {input_path}. Found {list(df.columns)}")
    return df


def extract_lon_lat(df: pd.DataFrame) -> pd.DataFrame:
    lon, lat = [], []
    bad = []
    for idx, value in df["geometry"].items():
        try:
            x, y = parse_point_wkt(str(value))
            lon.append(x)
            lat.append(y)
        except Exception as exc:  # noqa: BLE001
            lon.append(np.nan)
            lat.append(np.nan)
            bad.append((idx, value, str(exc)))
    if bad:
        raise ValueError(f"Invalid POINT WKT rows. First examples: {bad[:5]}")
    out = df.copy()
    out["lon"] = lon
    out["lat"] = lat
    return out


def project_lonlat(df: pd.DataFrame, source_crs: str, target_crs: str) -> pd.DataFrame:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy(float), df["lat"].to_numpy(float))
    out = df.copy()
    out["x"] = np.asarray(x, dtype=float)
    out["y"] = np.asarray(y, dtype=float)
    return out


def compute_kinematics(group: pd.DataFrame, min_step_m: float) -> pd.DataFrame:
    g = group.sort_values("timestamp", kind="mergesort").copy()
    n = len(g)
    g["t"] = np.arange(n, dtype=np.int64)

    x = g["x"].to_numpy(float)
    y = g["y"].to_numpy(float)
    ts = g["timestamp"]

    dx_prev = x - np.roll(x, 1)
    dy_prev = y - np.roll(y, 1)
    dx_prev[0] = np.nan
    dy_prev[0] = np.nan
    dt_prev = ts.diff().dt.total_seconds().to_numpy(float)
    step_prev = np.sqrt(dx_prev**2 + dy_prev**2)

    dx_next = np.roll(x, -1) - x
    dy_next = np.roll(y, -1) - y
    dx_next[-1] = np.nan
    dy_next[-1] = np.nan
    dt_next = ts.shift(-1).sub(ts).dt.total_seconds().to_numpy(float)
    step_next = np.sqrt(dx_next**2 + dy_next**2)

    valid_prev = np.isfinite(dx_prev) & np.isfinite(dy_prev) & np.isfinite(dt_prev) & (dt_prev > 0) & (step_prev >= min_step_m)
    valid_next = np.isfinite(dx_next) & np.isfinite(dy_next) & np.isfinite(dt_next) & (dt_next > 0) & (step_next >= min_step_m)

    yaw_prev = np.full(n, np.nan)
    yaw_next = np.full(n, np.nan)
    yaw_prev[valid_prev] = np.arctan2(dy_prev[valid_prev], dx_prev[valid_prev])
    yaw_next[valid_next] = np.arctan2(dy_next[valid_next], dx_next[valid_next])
    yaw = np.where(np.isfinite(yaw_next), yaw_next, yaw_prev)
    yaw = pd.Series(yaw, index=g.index).ffill().bfill().fillna(0.0).to_numpy(float)
    yaw = normalize_angle_rad(yaw)

    speed_prev = np.full(n, np.nan)
    speed_next = np.full(n, np.nan)
    valid_prev_speed = np.isfinite(step_prev) & np.isfinite(dt_prev) & (dt_prev > 0)
    valid_next_speed = np.isfinite(step_next) & np.isfinite(dt_next) & (dt_next > 0)
    speed_prev[valid_prev_speed] = step_prev[valid_prev_speed] / dt_prev[valid_prev_speed]
    speed_next[valid_next_speed] = step_next[valid_next_speed] / dt_next[valid_next_speed]
    speed = np.where(np.isfinite(speed_next), speed_next, speed_prev)
    speed = pd.Series(speed, index=g.index).ffill().bfill().fillna(0.0).to_numpy(float)

    g["dt_prev_s"] = dt_prev
    g["dt_next_s"] = dt_next
    g["step_prev_m"] = step_prev
    g["step_next_m"] = step_next
    g["yaw"] = yaw
    g["yaw_deg"] = np.degrees(yaw)
    g["speed_mps"] = speed
    return g


def preprocess(input_path: Path, source_crs: str, target_crs: str, min_step_m: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_points(input_path)
    df = raw.rename(columns={"id": "trajectory_id", "time": "timestamp", "geometry": "source_geometry_wkt"})
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = extract_lon_lat(df.rename(columns={"source_geometry_wkt": "geometry"})).rename(columns={"geometry": "source_geometry_wkt"})
    df = project_lonlat(df, source_crs, target_crs)
    df = df.sort_values(["trajectory_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    processed = df.groupby("trajectory_id", group_keys=False).apply(lambda g: compute_kinematics(g, min_step_m)).reset_index(drop=True)

    cols = [
        "trajectory_id", "t", "timestamp", "lon", "lat", "x", "y", "yaw", "yaw_deg", "speed_mps",
        "dt_prev_s", "dt_next_s", "step_prev_m", "step_next_m", "source_geometry_wkt",
    ]
    processed = processed[cols]
    counts = processed.groupby("trajectory_id").size()
    report: dict[str, Any] = {
        "input_path": str(input_path),
        "source_crs": source_crs,
        "target_crs": target_crs,
        "rows_input": int(len(raw)),
        "rows_output": int(len(processed)),
        "num_trajectories": int(processed["trajectory_id"].nunique()),
        "points_per_trajectory": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "mean": float(counts.mean()),
            "max": int(counts.max()),
        },
        "time_range": {"min": processed["timestamp"].min().isoformat(), "max": processed["timestamp"].max().isoformat()},
        "coordinate_range_lonlat": {
            "lon_min": float(processed["lon"].min()), "lon_max": float(processed["lon"].max()),
            "lat_min": float(processed["lat"].min()), "lat_max": float(processed["lat"].max()),
        },
        "coordinate_range_projected": {
            "x_min": float(processed["x"].min()), "x_max": float(processed["x"].max()),
            "y_min": float(processed["y"].min()), "y_max": float(processed["y"].max()),
        },
        "non_positive_dt_next_count": int(((processed["dt_next_s"].notna()) & (processed["dt_next_s"] <= 0)).sum()),
        "speed_mps": {
            "mean": float(processed["speed_mps"].mean()),
            "median": float(processed["speed_mps"].median()),
            "p95": float(processed["speed_mps"].quantile(0.95)),
            "max": float(processed["speed_mps"].max()),
        },
    }
    return processed, report


def save_df(df: pd.DataFrame, path: Path) -> Path:
    ensure_parent(path)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as exc:  # noqa: BLE001
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            print(f"[WARN] Parquet failed ({exc}); wrote {fallback}")
            return fallback
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/trajectories/points.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/trajectory_clean.parquet"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/preprocessing_report.json"))
    parser.add_argument("--source-crs", default="EPSG:4326")
    parser.add_argument("--target-crs", default="EPSG:32632")
    parser.add_argument("--min-step-m", type=float, default=0.25)
    args = parser.parse_args()

    df, report = preprocess(args.input, args.source_crs, args.target_crs, args.min_step_m)
    actual_output = save_df(df, args.output)
    ensure_parent(args.report)
    report["output_path"] = str(actual_output)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Prepared trajectories: {len(df)} rows, {df['trajectory_id'].nunique()} trajectories")
    print(f"[OK] Output: {actual_output}")
    print(f"[OK] Report: {args.report}")


if __name__ == "__main__":
    main()
