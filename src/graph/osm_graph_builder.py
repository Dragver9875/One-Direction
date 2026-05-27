from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform


DEFAULT_DRIVE_HIGHWAYS = {
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
}


@dataclass(frozen=True)
class OSMGraphBuildConfig:
    osm_pbf: str | Path
    output_dir: str | Path = "data/processed/road_graph"
    source_crs: str = "EPSG:4326"
    target_crs: str = "EPSG:32632"

    network_type: str = "driving"
    directed: bool = True
    add_reverse_edges_for_two_way: bool = True
    respect_oneway: bool = True
    retain_all: bool = True

    min_edge_length_m: float = 0.5
    include_highways: Tuple[str, ...] = tuple(sorted(DEFAULT_DRIVE_HIGHWAYS))

    output_graph_name: str = "road_graph.pkl"
    output_edge_table_name: str = "edge_table.parquet"
    output_node_table_name: str = "node_table.parquet"
    output_edges_geojson_name: str = "edges.geojson"
    output_nodes_geojson_name: str = "nodes.geojson"
    output_report_name: str = "osm_graph_report.json"


def _require_pyrosm() -> Any:
    try:
        from pyrosm import OSM
    except ImportError as exc:
        raise ImportError(
            "pyrosm is required for reading .osm.pbf files in "
            "build_osm_road_graph(). Install with `pip install pyrosm`, "
            "or use conda/mamba if pip compilation fails."
        ) from exc
    return OSM


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _project_geometry(geom: Any, transformer: Transformer) -> Any:
    if geom is None:
        return None
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def _as_bool_oneway(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"yes", "true", "1", "-1", "roundabout"}


def _is_reverse_oneway(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() == "-1"


def _normalise_highway(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if value is None:
        return "unknown"
    return str(value)


def _parse_numeric(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float(default)
    parts = text.replace(";", " ").replace(",", " ").split()
    for token in parts:
        try:
            return float(token)
        except ValueError:
            continue
    return float(default)


def _bearing_rad_from_linestring(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    return math.atan2(y1 - y0, x1 - x0)


def _reverse_linestring(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def _safe_geometry_from_row(row: pd.Series) -> Optional[LineString]:
    geom = row.get("geometry", None)
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        return geom

    if geom.geom_type == "MultiLineString":
        coords = []
        for part in geom.geoms:
            part_coords = list(part.coords)
            if not coords:
                coords.extend(part_coords)
            else:
                coords.extend(part_coords[1:])
        if len(coords) >= 2:
            return LineString(coords)

    return None


def _build_node_table_from_edges(edge_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    node_records: Dict[str, Dict[str, Any]] = {}

    for row in edge_rows:
        for node_id_key, x_key, y_key in [
            ("u", "u_x", "u_y"),
            ("v", "v_x", "v_y"),
        ]:
            node_id = row[node_id_key]
            if node_id not in node_records:
                node_records[node_id] = {
                    "node_id": node_id,
                    "x": row[x_key],
                    "y": row[y_key],
                    "geometry": Point(row[x_key], row[y_key]),
                }

    return pd.DataFrame(list(node_records.values()))


def _extract_edges_from_pyrosm(
    osm_pbf: Path,
    cfg: OSMGraphBuildConfig,
) -> pd.DataFrame:
    OSM = _require_pyrosm()

    osm = OSM(str(osm_pbf))
    edges = osm.get_network(network_type=cfg.network_type, nodes=False)

    if edges is None or len(edges) == 0:
        raise ValueError(f"No road edges could be extracted from {osm_pbf}")

    edges = edges.copy()

    if "highway" in edges.columns and cfg.include_highways:
        keep = edges["highway"].apply(
            lambda x: _normalise_highway(x) in set(cfg.include_highways)
        )
        edges = edges[keep].copy()

    if len(edges) == 0:
        raise ValueError(
            "All extracted OSM edges were filtered out. "
            "Check include_highways or network_type."
        )

    return edges


def _make_directed_edge_rows(
    edges: pd.DataFrame,
    cfg: OSMGraphBuildConfig,
) -> List[Dict[str, Any]]:
    transformer = _build_transformer(cfg.source_crs, cfg.target_crs)
    edge_rows: List[Dict[str, Any]] = []
    edge_counter = 0

    for _, row in edges.iterrows():
        geom_lonlat = _safe_geometry_from_row(row)
        if geom_lonlat is None:
            continue

        geom = _project_geometry(geom_lonlat, transformer)
        if geom is None or geom.is_empty or geom.length < cfg.min_edge_length_m:
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        u_raw = row.get("u", None)
        v_raw = row.get("v", None)
        osm_way_id = row.get("id", row.get("osm_id", row.get("way_id", None)))

        if pd.isna(u_raw) if u_raw is not None else True:
            u_raw = f"synthetic_{edge_counter}_u"
        if pd.isna(v_raw) if v_raw is not None else True:
            v_raw = f"synthetic_{edge_counter}_v"

        highway = _normalise_highway(row.get("highway", "unknown"))
        oneway_value = row.get("oneway", None)
        is_oneway = _as_bool_oneway(oneway_value)
        is_reverse = _is_reverse_oneway(oneway_value)

        maxspeed = _parse_numeric(row.get("maxspeed", np.nan))
        lanes = _parse_numeric(row.get("lanes", np.nan))

        base = {
            "osm_way_id": None if pd.isna(osm_way_id) else str(osm_way_id),
            "road_class": highway,
            "oneway": bool(is_oneway),
            "maxspeed": maxspeed,
            "lanes": lanes,
            "bridge": int(str(row.get("bridge", "")).lower() in {"yes", "true", "1"}),
            "tunnel": int(str(row.get("tunnel", "")).lower() in {"yes", "true", "1"}),
        }

        def add_edge(
            u: str,
            v: str,
            line: LineString,
            direction: str,
            original_direction: int,
        ) -> None:
            nonlocal edge_counter

            line_coords = list(line.coords)
            u_x, u_y = line_coords[0][:2]
            v_x, v_y = line_coords[-1][:2]
            edge_id = f"e_{edge_counter:09d}"

            edge_rows.append(
                {
                    "edge_idx": edge_counter,
                    "edge_id": edge_id,
                    "u": str(u),
                    "v": str(v),
                    "u_x": float(u_x),
                    "u_y": float(u_y),
                    "v_x": float(v_x),
                    "v_y": float(v_y),
                    "geometry": line,
                    "geometry_wkt": line.wkt,
                    "length_m": float(line.length),
                    "bearing_rad": float(_bearing_rad_from_linestring(line)),
                    "direction": direction,
                    "original_direction": int(original_direction),
                    **base,
                }
            )
            edge_counter += 1

        if cfg.respect_oneway and is_oneway:
            if is_reverse:
                add_edge(
                    u=str(v_raw),
                    v=str(u_raw),
                    line=_reverse_linestring(geom),
                    direction="reverse_oneway",
                    original_direction=-1,
                )
            else:
                add_edge(
                    u=str(u_raw),
                    v=str(v_raw),
                    line=geom,
                    direction="forward_oneway",
                    original_direction=1,
                )
        else:
            add_edge(
                u=str(u_raw),
                v=str(v_raw),
                line=geom,
                direction="forward",
                original_direction=1,
            )
            if cfg.add_reverse_edges_for_two_way:
                add_edge(
                    u=str(v_raw),
                    v=str(u_raw),
                    line=_reverse_linestring(geom),
                    direction="reverse",
                    original_direction=-1,
                )

    return edge_rows


def _build_networkx_graph(
    edge_table: pd.DataFrame,
    node_table: pd.DataFrame,
    crs: str,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = crs

    for _, row in node_table.iterrows():
        graph.add_node(
            str(row["node_id"]),
            x=float(row["x"]),
            y=float(row["y"]),
            geometry=row["geometry"],
        )

    for _, row in edge_table.iterrows():
        attrs = row.to_dict()
        u = str(attrs.pop("u"))
        v = str(attrs.pop("v"))
        graph.add_edge(u, v, key=str(row["edge_id"]), **attrs)

    return graph


def _save_geojson(df: pd.DataFrame, path: Path, crs: str) -> None:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required to write GeoJSON outputs. "
            "Install with `pip install geopandas`."
        ) from exc

    gdf = gpd.GeoDataFrame(df.copy(), geometry="geometry", crs=crs)
    gdf.to_file(path, driver="GeoJSON")


def build_osm_road_graph(cfg: OSMGraphBuildConfig) -> Dict[str, Path]:
    osm_pbf = _to_path(cfg.osm_pbf)
    output_dir = _to_path(cfg.output_dir)
    _ensure_output_dir(output_dir)

    if not osm_pbf.exists():
        raise FileNotFoundError(f"OSM PBF not found: {osm_pbf}")

    raw_edges = _extract_edges_from_pyrosm(osm_pbf, cfg)
    edge_rows = _make_directed_edge_rows(raw_edges, cfg)

    if not edge_rows:
        raise ValueError("No valid directed road edges were produced.")

    edge_table = pd.DataFrame(edge_rows)
    node_table = _build_node_table_from_edges(edge_rows)
    graph = _build_networkx_graph(edge_table, node_table, crs=cfg.target_crs)

    graph_path = output_dir / cfg.output_graph_name
    edge_table_path = output_dir / cfg.output_edge_table_name
    node_table_path = output_dir / cfg.output_node_table_name
    edges_geojson_path = output_dir / cfg.output_edges_geojson_name
    nodes_geojson_path = output_dir / cfg.output_nodes_geojson_name
    report_path = output_dir / cfg.output_report_name

    with graph_path.open("wb") as f:
        pickle.dump(graph, f)

    edge_table.drop(columns=["geometry"], errors="ignore").to_parquet(
        edge_table_path,
        index=False,
    )
    node_table.drop(columns=["geometry"], errors="ignore").to_parquet(
        node_table_path,
        index=False,
    )

    _save_geojson(edge_table, edges_geojson_path, cfg.target_crs)
    _save_geojson(node_table, nodes_geojson_path, cfg.target_crs)

    report = {
        "config": asdict(cfg),
        "num_raw_edges": int(len(raw_edges)),
        "num_directed_edges": int(len(edge_table)),
        "num_nodes": int(len(node_table)),
        "length_m": {
            "min": float(edge_table["length_m"].min()),
            "mean": float(edge_table["length_m"].mean()),
            "median": float(edge_table["length_m"].median()),
            "max": float(edge_table["length_m"].max()),
        },
        "road_class_counts": edge_table["road_class"].value_counts().to_dict(),
        "outputs": {
            "graph": str(graph_path),
            "edge_table": str(edge_table_path),
            "node_table": str(node_table_path),
            "edges_geojson": str(edges_geojson_path),
            "nodes_geojson": str(nodes_geojson_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "graph": graph_path,
        "edge_table": edge_table_path,
        "node_table": node_table_path,
        "edges_geojson": edges_geojson_path,
        "nodes_geojson": nodes_geojson_path,
        "report": report_path,
    }


__all__ = ["OSMGraphBuildConfig", "build_osm_road_graph"]
