from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from shapely import wkt
from shapely.geometry import Point

RAW_POINTS_REQUIRED_COLUMNS = {"id", "time", "geometry"}
CLEAN_TRAJECTORY_REQUIRED_COLUMNS = {
    "trajectory_id",
    "t",
    "timestamp",
    "lon",
    "lat",
    "x",
    "y",
    "yaw",
    "speed_mps",
}


def _check_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _require_columns(df: pd.DataFrame, required: Iterable[str], file_path: Path) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{file_path} is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


def parse_point_wkt(point_wkt: str) -> Tuple[float, float]:
    geom = wkt.loads(str(point_wkt))
    if not isinstance(geom, Point):
        raise ValueError(f"Expected POINT geometry, got {geom.geom_type}: {point_wkt}")
    return float(geom.x), float(geom.y)


def load_raw_points_csv(path: str | Path) -> pd.DataFrame:

    path = Path(path)
    _check_exists(path)

    df = pd.read_csv(path)
    _require_columns(df, RAW_POINTS_REQUIRED_COLUMNS, path)

    lon_lat = df["geometry"].map(parse_point_wkt)
    lon = [p[0] for p in lon_lat]
    lat = [p[1] for p in lon_lat]

    out = pd.DataFrame(
        {
            "trajectory_id": df["id"].astype(int),
            "timestamp": pd.to_datetime(df["time"], utc=True, errors="raise"),
            "source_geometry_wkt": df["geometry"].astype(str),
            "lon": lon,
            "lat": lat,
        }
    )

    out = out.sort_values(["trajectory_id", "timestamp"], kind="mergesort")
    out = out.reset_index(drop=True)
    return out


def load_clean_trajectories(path: str | Path) -> pd.DataFrame:

    path = Path(path)
    _check_exists(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported clean trajectory format: {path.suffix}")

    _require_columns(df, CLEAN_TRAJECTORY_REQUIRED_COLUMNS, path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df["t"] = df["t"].astype(int)

    return df.sort_values(["trajectory_id", "t"], kind="mergesort").reset_index(
        drop=True
    )
