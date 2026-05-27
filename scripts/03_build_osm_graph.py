from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely import wkt

try:
    from pyrosm import OSM
except ImportError as exc:  # pragma: no cover
    raise ImportError("pyrosm is required for this script. Install with `pip install pyrosm`.") from exc

DRIVE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "service", "living_street", "road",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_bool_oneway(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    return s in {"yes", "true", "1", "-1", "reversible"}


def reverse_geometry(geom):
    if isinstance(geom, LineString):
        return LineString(list(geom.coords)[::-1])
    if isinstance(geom, MultiLineString):
        return MultiLineString([LineString(list(line.coords)[::-1]) for line in reversed(geom.geoms)])
    return geom


def first_last_xy(geom) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(geom, MultiLineString):
        parts = list(geom.geoms)
        first = list(parts[0].coords)[0]
        last = list(parts[-1].coords)[-1]
    else:
        coords = list(geom.coords)
        first, last = coords[0], coords[-1]
    return (float(first[0]), float(first[1])), (float(last[0]), float(last[1]))


def bearing_rad(geom) -> float:
    (x1, y1), (x2, y2) = first_last_xy(geom)
    return float(math.atan2(y2 - y1, x2 - x1))


def curvature_ratio(geom) -> float:
    length = float(geom.length)
    (x1, y1), (x2, y2) = first_last_xy(geom)
    chord = float(math.hypot(x2 - x1, y2 - y1))
    if chord <= 1e-9:
        return 1.0
    return float(length / chord)


def parse_numeric_first(value: Any, default: float = np.nan) -> float:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).replace(",", ";")
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return default
    return float(m.group(0))


def normalize_highway(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if pd.isna(value):
        return "unknown"
    s = str(value)
    if ";" in s:
        return s.split(";")[0]
    return s


def load_osm_network(osm_path: Path, target_crs: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    osm = OSM(str(osm_path))
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    if edges is None or len(edges) == 0:
        raise RuntimeError("pyrosm returned no driving edges.")
    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")
    if nodes is not None and nodes.crs is None:
        nodes = nodes.set_crs("EPSG:4326")
    edges = edges.to_crs(target_crs)
    nodes = nodes.to_crs(target_crs) if nodes is not None else gpd.GeoDataFrame(geometry=[], crs=target_crs)
    return nodes, edges


def edge_records_from_gdf(edges: gpd.GeoDataFrame, add_reverse: bool, min_edge_length_m: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for raw_idx, row in edges.iterrows():
        if "u" not in row or "v" not in row:
            raise ValueError("OSM edge table must contain u and v columns from pyrosm.")
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        highway = normalize_highway(row.get("highway", "unknown"))
        if highway not in DRIVE_HIGHWAYS:
            continue
        length = float(geom.length)
        if length < min_edge_length_m:
            continue
        u = int(row["u"])
        v = int(row["v"])
        osm_way_id = str(row.get("id", row.get("osm_id", raw_idx)))
        oneway = parse_bool_oneway(row.get("oneway", False))
        maxspeed = parse_numeric_first(row.get("maxspeed", np.nan))
        lanes = parse_numeric_first(row.get("lanes", np.nan))
        bridge = int(str(row.get("bridge", "")).lower() in {"yes", "true", "1"})
        tunnel = int(str(row.get("tunnel", "")).lower() in {"yes", "true", "1"})

        def add_record(src: int, dst: int, g, direction: str, is_reverse: bool) -> None:
            b = bearing_rad(g)
            records.append(
                {
                    "edge_id": f"{osm_way_id}:{src}:{dst}:{direction}",
                    "osm_way_id": osm_way_id,
                    "u": src,
                    "v": dst,
                    "direction": direction,
                    "is_reverse": int(is_reverse),
                    "geometry_wkt": g.wkt,
                    "length_m": float(g.length),
                    "bearing_rad": b,
                    "sin_bearing": float(math.sin(b)),
                    "cos_bearing": float(math.cos(b)),
                    "road_class": highway,
                    "oneway": int(oneway),
                    "maxspeed": maxspeed,
                    "lanes": lanes,
                    "bridge": bridge,
                    "tunnel": tunnel,
                    "curvature": curvature_ratio(g),
                }
            )

        add_record(u, v, geom, "fwd", False)
        if add_reverse and not oneway:
            add_record(v, u, reverse_geometry(geom), "rev", True)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No directed road edges produced.")
    class_map = {name: idx for idx, name in enumerate(sorted(df["road_class"].dropna().unique()))}
    df["road_class_id"] = df["road_class"].map(class_map).astype(int)
    return df


def build_networkx(edge_df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for _, row in edge_df.iterrows():
        attrs = row.to_dict()
        attrs["geometry"] = wkt.loads(attrs.pop("geometry_wkt"))
        g.add_edge(int(row["u"]), int(row["v"]), key=row["edge_id"], **attrs)
    return g


def save_edge_geojson(edge_df: pd.DataFrame, path: Path, target_crs: str) -> None:
    gdf = gpd.GeoDataFrame(edge_df.drop(columns=[]).copy(), geometry=edge_df["geometry_wkt"].map(wkt.loads), crs=target_crs)
    gdf.to_file(path, driver="GeoJSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm", type=Path, default=Path("data/raw/osm/oberfranken-latest.osm.pbf"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/road_graph"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/osm_graph_report.json"))
    parser.add_argument("--target-crs", default="EPSG:32632")
    parser.add_argument("--add-reverse", action="store_true", default=True)
    parser.add_argument("--min-edge-length-m", type=float, default=0.5)
    args = parser.parse_args()

    if not args.osm.exists():
        raise FileNotFoundError(args.osm)
    ensure_dir(args.output)
    ensure_dir(args.report.parent)

    nodes, edges = load_osm_network(args.osm, args.target_crs)
    edge_df = edge_records_from_gdf(edges, args.add_reverse, args.min_edge_length_m)

    out_degree = edge_df.groupby("u").size().to_dict()
    in_degree = edge_df.groupby("v").size().to_dict()
    edge_df["out_degree_u"] = edge_df["u"].map(out_degree).fillna(0).astype(int)
    edge_df["in_degree_v"] = edge_df["v"].map(in_degree).fillna(0).astype(int)

    graph = build_networkx(edge_df)
    with (args.output / "road_graph.pkl").open("wb") as f:
        pickle.dump(graph, f)
    edge_df.to_parquet(args.output / "edge_table.parquet", index=False)
    edge_df.drop(columns=["geometry_wkt"]).head(0).to_csv(args.output / "edge_table_schema.csv", index=False)
    save_edge_geojson(edge_df, args.output / "edges.geojson", args.target_crs)

    if nodes is not None and len(nodes) > 0:
        nodes_out = nodes.copy()
        nodes_out.to_file(args.output / "nodes.geojson", driver="GeoJSON")
        try:
            nodes_out.drop(columns="geometry").to_parquet(args.output / "node_table.parquet", index=False)
        except Exception:
            nodes_out.drop(columns="geometry").to_csv(args.output / "node_table.csv", index=False)

    report = {
        "osm_path": str(args.osm),
        "target_crs": args.target_crs,
        "raw_edges": int(len(edges)),
        "directed_edges": int(len(edge_df)),
        "nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "road_classes": edge_df["road_class"].value_counts().to_dict(),
        "length_m": {
            "min": float(edge_df["length_m"].min()),
            "median": float(edge_df["length_m"].median()),
            "mean": float(edge_df["length_m"].mean()),
            "max": float(edge_df["length_m"].max()),
        },
        "outputs": {
            "road_graph": str(args.output / "road_graph.pkl"),
            "edge_table": str(args.output / "edge_table.parquet"),
            "edges_geojson": str(args.output / "edges.geojson"),
        },
    }
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Built directed road graph: {len(edge_df)} directed edges")
    print(f"[OK] Output: {args.output}")
    print(f"[OK] Report: {args.report}")


if __name__ == "__main__":
    main()
