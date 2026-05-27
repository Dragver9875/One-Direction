from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Tuple

import numpy as np
from pyproj import Geod
from shapely.geometry import LineString, Point


WGS84_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ProjectionResult:
    distance_m: float
    proj_x: float
    proj_y: float
    offset_m: float
    offset_ratio: float


def euclidean_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(math.hypot(x1 - x0, y1 - y0))


def euclidean_distance_array(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
) -> np.ndarray:
    return np.sqrt((np.asarray(x1) - np.asarray(x0)) ** 2 + (np.asarray(y1) - np.asarray(y0)) ** 2)


def path_length_xy(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    if len(x_arr) < 2:
        return 0.0

    return float(np.sum(np.sqrt(np.diff(x_arr) ** 2 + np.diff(y_arr) ** 2)))


def haversine_distance_m(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    _, _, dist = WGS84_GEOD.inv(lon0, lat0, lon1, lat1)
    return float(dist)


def point_to_linestring_distance(point: Point, line: LineString) -> float:
    if point is None or line is None:
        raise ValueError("point and line cannot be None.")
    return float(point.distance(line))


def project_point_to_linestring(point: Point, line: LineString) -> ProjectionResult:
    if point is None or line is None:
        raise ValueError("point and line cannot be None.")
    if line.is_empty or line.length <= 0:
        raise ValueError("Cannot project onto an empty or zero-length LineString.")

    offset = float(line.project(point))
    proj = line.interpolate(offset)
    distance = float(point.distance(line))
    length = float(line.length)

    return ProjectionResult(
        distance_m=distance,
        proj_x=float(proj.x),
        proj_y=float(proj.y),
        offset_m=offset,
        offset_ratio=float(offset / length) if length > 0 else 0.0,
    )


def project_xy_to_linestring(x: float, y: float, line: LineString) -> ProjectionResult:
    return project_point_to_linestring(Point(float(x), float(y)), line)


def distance_point_to_polyline_xy(
    x: float,
    y: float,
    coords: Iterable[Tuple[float, float]],
) -> float:
    line = LineString(list(coords))
    return float(Point(float(x), float(y)).distance(line))


def cumulative_distances_xy(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    if len(x_arr) == 0:
        return np.asarray([], dtype=float)
    if len(x_arr) == 1:
        return np.asarray([0.0], dtype=float)

    steps = np.sqrt(np.diff(x_arr) ** 2 + np.diff(y_arr) ** 2)
    return np.concatenate([[0.0], np.cumsum(steps)])


__all__ = [
    "ProjectionResult",
    "euclidean_distance",
    "euclidean_distance_array",
    "path_length_xy",
    "haversine_distance_m",
    "point_to_linestring_distance",
    "project_point_to_linestring",
    "project_xy_to_linestring",
    "distance_point_to_polyline_xy",
    "cumulative_distances_xy",
]
