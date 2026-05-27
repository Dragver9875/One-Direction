from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString


@dataclass(frozen=True)
class RoadFeatureConfig:
    normalize_numeric: bool = True
    unknown_maxspeed_mps: float = 13.89  # 50 km/h
    unknown_lanes: float = 1.0
    road_class_column: str = "road_class"


ROAD_CLASS_ORDER = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "road",
    "unknown",
]


def road_class_mapping() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(ROAD_CLASS_ORDER)}


def _geometry_from_row(row: pd.Series) -> LineString | None:
    geom = row.get("geometry", None)
    if isinstance(geom, LineString):
        return geom
    text = row.get("geometry_wkt", None)
    if isinstance(text, str):
        parsed = wkt.loads(text)
        if isinstance(parsed, LineString):
            return parsed
    return None


def _curvature(line: LineString | None) -> float:
    if line is None or line.is_empty:
        return 0.0
    coords = list(line.coords)
    if len(coords) < 3:
        return 0.0
    chord = np.linalg.norm(np.asarray(coords[-1][:2]) - np.asarray(coords[0][:2]))
    if chord <= 1e-9:
        return 0.0
    return float(max(line.length / chord - 1.0, 0.0))


def _coerce_float(series: pd.Series, default: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.fillna(default).astype(float)


def _maxspeed_to_mps(value: object, default_mps: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default_mps
    if isinstance(value, (int, float, np.number)):
        v = float(value)
        return v / 3.6 if v > 5.0 else v

    text = str(value).strip().lower()
    if not text:
        return default_mps
    tokens = text.replace(";", " ").replace(",", " ").split()
    value_num = None
    for token in tokens:
        try:
            value_num = float(token)
            break
        except ValueError:
            continue
    if value_num is None:
        return default_mps

    if "mph" in text:
        return value_num * 0.44704
    return value_num / 3.6


def add_graph_degrees(edge_table: pd.DataFrame) -> pd.DataFrame:
    df = edge_table.copy()

    out_deg_by_u = df.groupby("u").size().to_dict()
    in_deg_by_v = df.groupby("v").size().to_dict()

    df["out_degree"] = df["v"].map(lambda node: out_deg_by_u.get(node, 0)).astype(float)
    df["in_degree"] = df["u"].map(lambda node: in_deg_by_v.get(node, 0)).astype(float)

    return df


def build_segment_feature_table(
    edge_table: pd.DataFrame,
    cfg: RoadFeatureConfig = RoadFeatureConfig(),
) -> Tuple[pd.DataFrame, List[str]]:
    df = edge_table.copy()

    if "edge_idx" not in df.columns:
        df["edge_idx"] = np.arange(len(df), dtype=np.int64)

    if "in_degree" not in df.columns or "out_degree" not in df.columns:
        df = add_graph_degrees(df)

    bearing = pd.to_numeric(df.get("bearing_rad", 0.0), errors="coerce").fillna(0.0)
    df["sin_bearing"] = np.sin(bearing)
    df["cos_bearing"] = np.cos(bearing)

    df["length_m"] = _coerce_float(df.get("length_m", pd.Series([0.0] * len(df))), 0.0)

    df["maxspeed_mps"] = df.get("maxspeed", pd.Series([np.nan] * len(df))).apply(
        lambda x: _maxspeed_to_mps(x, cfg.unknown_maxspeed_mps)
    )
    df["lanes_norm_raw"] = _coerce_float(
        df.get("lanes", pd.Series([np.nan] * len(df))),
        cfg.unknown_lanes,
    )

    if "curvature" not in df.columns:
        df["curvature"] = df.apply(lambda row: _curvature(_geometry_from_row(row)), axis=1)

    df["bridge_flag"] = _coerce_float(df.get("bridge", pd.Series([0] * len(df))), 0.0)
    df["tunnel_flag"] = _coerce_float(df.get("tunnel", pd.Series([0] * len(df))), 0.0)
    df["oneway_flag"] = df.get("oneway", pd.Series([False] * len(df))).astype(bool).astype(float)

    mapping = road_class_mapping()
    road_class = df.get(cfg.road_class_column, pd.Series(["unknown"] * len(df))).fillna("unknown")
    df["road_class_id"] = road_class.map(lambda x: mapping.get(str(x), mapping["unknown"])).astype(int)

    df["log_length_m"] = np.log1p(df["length_m"].clip(lower=0.0))

    numeric_columns = [
        "log_length_m",
        "sin_bearing",
        "cos_bearing",
        "maxspeed_mps",
        "lanes_norm_raw",
        "curvature",
        "in_degree",
        "out_degree",
        "bridge_flag",
        "tunnel_flag",
        "oneway_flag",
    ]

    if cfg.normalize_numeric:
        for col in numeric_columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            mean = values.mean()
            std = values.std(ddof=0)
            if std < 1e-8:
                std = 1.0
            df[col] = (values - mean) / std

    feature_columns = numeric_columns + ["road_class_id"]

    keep_columns = ["edge_idx", "edge_id"] + feature_columns
    feature_table = df[keep_columns].sort_values("edge_idx").reset_index(drop=True)

    return feature_table, feature_columns


__all__ = [
    "RoadFeatureConfig",
    "ROAD_CLASS_ORDER",
    "road_class_mapping",
    "add_graph_degrees",
    "build_segment_feature_table",
]
