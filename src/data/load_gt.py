from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform as shapely_transform

GT_REQUIRED_COLUMNS = {"id", "WKT"}
PROJECTED_GT_REQUIRED_COLUMNS = {
    "trajectory_id",
    "gt_wkt_lonlat",
    "gt_wkt_projected",
    "gt_length_m",
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


def parse_route_wkt(route_wkt: str) -> LineString | MultiLineString:
    geom = wkt.loads(str(route_wkt))
    if not isinstance(geom, (LineString, MultiLineString)):
        raise ValueError(
            f"Expected LINESTRING or MULTILINESTRING, got {geom.geom_type}: {route_wkt[:120]}"
        )
    return geom


def project_geometry(
    geom: LineString | MultiLineString,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
) -> LineString | MultiLineString:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def load_ground_truth_routes_csv(
    path: str | Path,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
    project: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    _check_exists(path)

    df = pd.read_csv(path)
    _require_columns(df, GT_REQUIRED_COLUMNS, path)

    route_geoms = df["WKT"].map(parse_route_wkt)

    out = pd.DataFrame(
        {
            "trajectory_id": df["id"].astype(int),
            "gt_wkt_lonlat": df["WKT"].astype(str),
            "gt_geom_lonlat": route_geoms,
        }
    )

    if project:
        projected = route_geoms.map(
            lambda geom: project_geometry(
                geom,
                source_crs=source_crs,
                target_crs=target_crs,
            )
        )
        out["gt_geom_projected"] = projected
        out["gt_wkt_projected"] = projected.map(lambda geom: geom.wkt)
        out["gt_length_m"] = projected.map(lambda geom: float(geom.length))
        out["source_crs"] = source_crs
        out["target_crs"] = target_crs

    return out.sort_values("trajectory_id").reset_index(drop=True)


def save_projected_gt_routes(df: pd.DataFrame, path: str | Path) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    for col in ["gt_geom_lonlat", "gt_geom_projected"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    if path.suffix.lower() == ".parquet":
        try:
            out.to_parquet(path, index=False)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            out.to_csv(fallback, index=False)
            return fallback

    if path.suffix.lower() == ".csv":
        out.to_csv(path, index=False)
        return path

    raise ValueError(f"Unsupported output extension: {path.suffix}")


def load_projected_gt_routes(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    _check_exists(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported GT route format: {path.suffix}")

    _require_columns(df, PROJECTED_GT_REQUIRED_COLUMNS, path)
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df["gt_geom_lonlat"] = df["gt_wkt_lonlat"].map(parse_route_wkt)
    df["gt_geom_projected"] = df["gt_wkt_projected"].map(wkt.loads)
    return df.sort_values("trajectory_id").reset_index(drop=True)
