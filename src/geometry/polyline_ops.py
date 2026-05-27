from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge


def parse_wkt_geometry(value: str | BaseGeometry) -> BaseGeometry:
    if isinstance(value, BaseGeometry):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Expected WKT string or Shapely geometry, got {type(value)}.")
    return wkt.loads(value)


def ensure_linestring(value: str | BaseGeometry) -> LineString:
    geom = parse_wkt_geometry(value)

    if isinstance(geom, LineString):
        return geom

    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        return flatten_multilinestring(geom)

    raise ValueError(f"Expected LineString or MultiLineString, got {geom.geom_type}.")


def flatten_multilinestring(geom: MultiLineString | LineString) -> LineString:
    if isinstance(geom, LineString):
        return geom

    if not isinstance(geom, MultiLineString):
        raise TypeError(f"Expected MultiLineString, got {type(geom)}.")

    coords: List[Tuple[float, float]] = []

    for part in geom.geoms:
        part_coords = list(part.coords)
        if not part_coords:
            continue
        if not coords:
            coords.extend(part_coords)
        else:
            if coords[-1] == part_coords[0]:
                coords.extend(part_coords[1:])
            else:
                coords.extend(part_coords)

    if len(coords) < 2:
        raise ValueError("Cannot flatten MultiLineString with fewer than 2 coordinates.")

    return LineString(coords)


def geometry_length_m(value: str | BaseGeometry) -> float:
    geom = parse_wkt_geometry(value)
    return float(geom.length)


def interpolate_along_linestring(line: LineString, offset_m: float) -> Point:
    if line.is_empty or line.length <= 0:
        raise ValueError("Cannot interpolate on an empty or zero-length LineString.")
    offset = min(max(float(offset_m), 0.0), float(line.length))
    return line.interpolate(offset)


def sample_linestring_by_distance(
    line: LineString,
    spacing_m: float,
    include_end: bool = True,
) -> List[Point]:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive.")
    if line.is_empty or line.length <= 0:
        return []

    distances = list(np.arange(0.0, float(line.length), float(spacing_m)))
    if include_end and (not distances or distances[-1] < float(line.length)):
        distances.append(float(line.length))

    return [line.interpolate(d) for d in distances]


def linestring_to_xy_arrays(line: LineString) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(line.coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return coords[:, 0], coords[:, 1]


def xy_arrays_to_linestring(x: Sequence[float], y: Sequence[float]) -> LineString:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("Need at least two coordinates to build a LineString.")
    return LineString(list(zip(x, y)))


def reverse_linestring(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def simplify_linestring_safe(
    line: LineString,
    tolerance_m: float,
    preserve_topology: bool = True,
) -> LineString:
    simplified = line.simplify(tolerance_m, preserve_topology=preserve_topology)
    if isinstance(simplified, LineString) and len(simplified.coords) >= 2:
        return simplified
    return line


__all__ = [
    "parse_wkt_geometry",
    "ensure_linestring",
    "flatten_multilinestring",
    "geometry_length_m",
    "interpolate_along_linestring",
    "sample_linestring_by_distance",
    "linestring_to_xy_arrays",
    "xy_arrays_to_linestring",
    "reverse_linestring",
    "simplify_linestring_safe",
]
