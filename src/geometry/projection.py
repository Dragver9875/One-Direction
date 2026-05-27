from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely import wkt
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


@dataclass(frozen=True)
class CRSConfig:
    source_crs: str = "EPSG:4326"
    target_crs: str = "EPSG:32632"
    always_xy: bool = True


def build_transformer(
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
    always_xy: bool = True,
) -> Transformer:
    return Transformer.from_crs(source_crs, target_crs, always_xy=always_xy)


def detect_utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))

    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return f"EPSG:{epsg}"


def project_xy(
    lon: np.ndarray | Iterable[float],
    lat: np.ndarray | Iterable[float],
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
) -> Tuple[np.ndarray, np.ndarray]:
    transformer = build_transformer(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def project_dataframe_points(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    x_col: str = "x",
    y_col: str = "y",
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
) -> pd.DataFrame:
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(
            f"Input DataFrame must contain {lon_col!r} and {lat_col!r} columns."
        )

    out = df.copy()
    x, y = project_xy(
        out[lon_col].to_numpy(dtype=float),
        out[lat_col].to_numpy(dtype=float),
        source_crs=source_crs,
        target_crs=target_crs,
    )
    out[x_col] = x
    out[y_col] = y
    return out


def project_geometry(
    geometry: BaseGeometry,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
) -> BaseGeometry:
    if geometry is None:
        raise ValueError("geometry cannot be None.")

    transformer = build_transformer(source_crs, target_crs, always_xy=True)
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geometry)


def project_wkt_series(
    series: pd.Series,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:32632",
) -> pd.Series:
    transformer = build_transformer(source_crs, target_crs, always_xy=True)

    def _project_one(value: object) -> BaseGeometry:
        if isinstance(value, BaseGeometry):
            geom = value
        elif isinstance(value, str):
            geom = wkt.loads(value)
        else:
            raise ValueError(f"Cannot parse WKT geometry from value: {value!r}")

        return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)

    return series.apply(_project_one)


def geometry_to_wkt(geometry: BaseGeometry) -> str:
    return geometry.wkt


def crs_to_string(crs: CRS | str) -> str:
    return CRS.from_user_input(crs).to_string()


__all__ = [
    "CRSConfig",
    "build_transformer",
    "detect_utm_epsg",
    "project_xy",
    "project_dataframe_points",
    "project_geometry",
    "project_wkt_series",
    "geometry_to_wkt",
    "crs_to_string",
]
