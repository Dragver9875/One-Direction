from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString


@dataclass(frozen=True)
class RoadGraphCleanConfig:
    min_edge_length_m: float = 0.5
    remove_invalid_geometry: bool = True
    remove_duplicate_edge_ids: bool = True
    normalize_angles: bool = True
    recompute_length: bool = True
    recompute_bearing: bool = True


def _load_geometry(value: object) -> Optional[LineString]:
    if isinstance(value, LineString):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        geom = wkt.loads(value)
        if geom.geom_type == "LineString":
            return geom
    return None


def _bearing_rad(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    return math.atan2(y1 - y0, x1 - x0)


def _normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def clean_edge_table(
    edge_table: pd.DataFrame,
    cfg: RoadGraphCleanConfig = RoadGraphCleanConfig(),
) -> pd.DataFrame:
    df = edge_table.copy()

    if "geometry" not in df.columns and "geometry_wkt" in df.columns:
        df["geometry"] = df["geometry_wkt"].apply(_load_geometry)
    elif "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(_load_geometry)

    if cfg.remove_invalid_geometry:
        df = df[df["geometry"].notna()].copy()
        df = df[df["geometry"].apply(lambda g: not g.is_empty and len(g.coords) >= 2)].copy()

    if cfg.recompute_length:
        df["length_m"] = df["geometry"].apply(lambda g: float(g.length))

    df = df[df["length_m"] >= cfg.min_edge_length_m].copy()

    if cfg.recompute_bearing:
        df["bearing_rad"] = df["geometry"].apply(_bearing_rad)

    if cfg.normalize_angles and "bearing_rad" in df.columns:
        df["bearing_rad"] = df["bearing_rad"].apply(_normalize_angle)

    if cfg.remove_duplicate_edge_ids and "edge_id" in df.columns:
        df = df.drop_duplicates(subset=["edge_id"], keep="first").copy()

    if "edge_idx" not in df.columns:
        df["edge_idx"] = np.arange(len(df), dtype=np.int64)
    else:
        df = df.sort_values("edge_idx").reset_index(drop=True)

    if "geometry_wkt" not in df.columns:
        df["geometry_wkt"] = df["geometry"].apply(lambda g: g.wkt)

    return df.reset_index(drop=True)


def clean_road_graph(
    graph: nx.MultiDiGraph,
    cfg: RoadGraphCleanConfig = RoadGraphCleanConfig(),
) -> nx.MultiDiGraph:
    cleaned = nx.MultiDiGraph()
    cleaned.graph.update(graph.graph)

    for node, attrs in graph.nodes(data=True):
        cleaned.add_node(node, **attrs)

    seen_edge_ids = set()

    for u, v, key, attrs in graph.edges(keys=True, data=True):
        edge_id = attrs.get("edge_id", str(key))
        if cfg.remove_duplicate_edge_ids and edge_id in seen_edge_ids:
            continue

        geom = _load_geometry(attrs.get("geometry", attrs.get("geometry_wkt")))
        if cfg.remove_invalid_geometry:
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                continue

        length = float(geom.length) if cfg.recompute_length and geom is not None else float(attrs.get("length_m", 0.0))
        if length < cfg.min_edge_length_m:
            continue

        out_attrs = dict(attrs)
        if geom is not None:
            out_attrs["geometry"] = geom
            out_attrs["geometry_wkt"] = geom.wkt

        out_attrs["length_m"] = length

        if cfg.recompute_bearing and geom is not None:
            out_attrs["bearing_rad"] = _bearing_rad(geom)

        if cfg.normalize_angles and "bearing_rad" in out_attrs:
            out_attrs["bearing_rad"] = _normalize_angle(float(out_attrs["bearing_rad"]))

        cleaned.add_edge(u, v, key=key, **out_attrs)
        seen_edge_ids.add(edge_id)

    isolated = list(nx.isolates(cleaned.to_undirected()))
    cleaned.remove_nodes_from(isolated)

    return cleaned


__all__ = [
    "RoadGraphCleanConfig",
    "clean_edge_table",
    "clean_road_graph",
]
