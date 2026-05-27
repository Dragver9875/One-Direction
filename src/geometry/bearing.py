from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
from pyproj import Geod
from shapely.geometry import LineString


WGS84_GEOD = Geod(ellps="WGS84")


def normalize_angle_rad(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def angular_difference_rad(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    return np.abs(normalize_angle_rad(np.asarray(a) - np.asarray(b)))


def signed_turn_angle_rad(prev_bearing: float, curr_bearing: float) -> float:
    return float(normalize_angle_rad(curr_bearing - prev_bearing))


def bearing_from_xy(x0: float, y0: float, x1: float, y1: float) -> float:
    return float(normalize_angle_rad(math.atan2(y1 - y0, x1 - x0)))


def bearing_series_from_xy(
    x: Iterable[float],
    y: Iterable[float],
    min_step_m: float = 0.0,
) -> np.ndarray:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length.")

    n = len(x_arr)
    if n == 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([0.0], dtype=float)

    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    step = np.sqrt(dx**2 + dy**2)

    bearings = np.full(n, np.nan, dtype=float)
    valid = step >= min_step_m

    bearings[:-1][valid] = np.arctan2(dy[valid], dx[valid])

    bearings[-1] = bearings[-2]

    valid_idx = np.where(np.isfinite(bearings))[0]
    if len(valid_idx) == 0:
        return np.zeros(n, dtype=float)

    for i in range(n):
        if not np.isfinite(bearings[i]):
            nearest = valid_idx[np.argmin(np.abs(valid_idx - i))]
            bearings[i] = bearings[nearest]

    return normalize_angle_rad(bearings)


def linestring_bearing_rad(line: LineString) -> float:
    if line is None or line.is_empty:
        return 0.0

    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0

    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    return bearing_from_xy(float(x0), float(y0), float(x1), float(y1))


def local_segment_bearing_at_offset(
    line: LineString,
    offset_m: float,
    window_m: float = 5.0,
) -> float:
    if line is None or line.is_empty or line.length <= 0:
        return 0.0

    a = max(0.0, float(offset_m) - float(window_m))
    b = min(float(line.length), float(offset_m) + float(window_m))

    if abs(b - a) < 1e-6:
        return linestring_bearing_rad(line)

    p0 = line.interpolate(a)
    p1 = line.interpolate(b)
    return bearing_from_xy(float(p0.x), float(p0.y), float(p1.x), float(p1.y))


def bearing_from_lonlat(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    fwd_azimuth_deg, _, _ = WGS84_GEOD.inv(lon0, lat0, lon1, lat1)
    math_angle_deg = 90.0 - fwd_azimuth_deg
    return float(normalize_angle_rad(math.radians(math_angle_deg)))


def sin_cos_angle(angle: float | np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(angle, dtype=float)
    return np.sin(arr), np.cos(arr)


__all__ = [
    "normalize_angle_rad",
    "angular_difference_rad",
    "signed_turn_angle_rad",
    "bearing_from_xy",
    "bearing_series_from_xy",
    "linestring_bearing_rad",
    "local_segment_bearing_at_offset",
    "bearing_from_lonlat",
    "sin_cos_angle",
]
