from __future__ import annotations

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point, box


def read_edges(path: Path, source_crs: str) -> gpd.GeoDataFrame:
    edges = gpd.read_file(path)
    if edges.crs is None:
        edges = edges.set_crs(source_crs)
    return edges


def read_matches(path: Path, source_crs: str, trajectory_id: int | None) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported matches format: {path}")

    if trajectory_id is not None:
        df = df[df["trajectory_id"] == trajectory_id].copy()

    required = {"pred_proj_x", "pred_proj_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Matches missing columns: {sorted(missing)}")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(float(x), float(y)) for x, y in zip(df["pred_proj_x"], df["pred_proj_y"])],
        crs=source_crs,
    )

    return gdf


def read_raw_points(path: Path, source_crs: str, trajectory_id: int | None) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None

    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

    if trajectory_id is not None:
        df = df[df["trajectory_id"] == trajectory_id].copy()

    if not {"x", "y"}.issubset(df.columns):
        return None

    return gpd.GeoDataFrame(
        df,
        geometry=[Point(float(x), float(y)) for x, y in zip(df["x"], df["y"])],
        crs=source_crs,
    )


def read_gt_from_matches(matches: gpd.GeoDataFrame, source_crs: str) -> gpd.GeoDataFrame | None:
    if not {"gt_proj_x", "gt_proj_y"}.issubset(matches.columns):
        return None

    df = pd.DataFrame(matches.drop(columns="geometry")).copy()

    return gpd.GeoDataFrame(
        df,
        geometry=[Point(float(x), float(y)) for x, y in zip(df["gt_proj_x"], df["gt_proj_y"])],
        crs=source_crs,
    )


def make_linestring_from_points(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    if points is None or len(points) < 2:
        return None

    if "t" in points.columns:
        points = points.sort_values("t")

    coords = [(geom.x, geom.y) for geom in points.geometry]
    line = LineString(coords)

    return gpd.GeoDataFrame({"name": ["path"]}, geometry=[line], crs=points.crs)


def crop_edges(edges: gpd.GeoDataFrame, focus: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    if focus.empty:
        return edges

    bounds = focus.total_bounds
    xmin, ymin, xmax, ymax = bounds

    bbox = box(
        xmin - buffer_m,
        ymin - buffer_m,
        xmax + buffer_m,
        ymax + buffer_m,
    )

    return edges[edges.intersects(bbox)].copy()


def add_geojson_layer(map_obj, gdf: gpd.GeoDataFrame, name: str, color: str, weight: float, opacity: float, dash_array: str | None = None):
    if gdf is None or gdf.empty:
        return

    gdf_4326 = gdf.to_crs("EPSG:4326")

    style = {
        "color": color,
        "weight": weight,
        "opacity": opacity,
    }

    if dash_array is not None:
        style["dashArray"] = dash_array

    folium.GeoJson(
        gdf_4326.to_json(),
        name=name,
        style_function=lambda _: style,
    ).add_to(map_obj)


def add_point_layer(map_obj, points: gpd.GeoDataFrame, name: str, color: str, radius: int):
    if points is None or points.empty:
        return

    points_4326 = points.to_crs("EPSG:4326")
    group = folium.FeatureGroup(name=name)

    for _, row in points_4326.iterrows():
        geom = row.geometry
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.8,
            weight=1,
        ).add_to(group)

    group.add_to(map_obj)


def build_map(
    edges: gpd.GeoDataFrame,
    matches: gpd.GeoDataFrame,
    raw_points: gpd.GeoDataFrame | None,
    gt_points: gpd.GeoDataFrame | None,
    output: Path,
    buffer_m: float,
    max_edges: int,
):
    focus_parts = [matches]
    if raw_points is not None:
        focus_parts.append(raw_points)
    if gt_points is not None:
        focus_parts.append(gt_points)

    focus = pd.concat(focus_parts, ignore_index=True)
    focus = gpd.GeoDataFrame(focus, geometry="geometry", crs=matches.crs)

    cropped_edges = crop_edges(edges, focus, buffer_m)

    if len(cropped_edges) > max_edges:
        cropped_edges = cropped_edges.sample(max_edges, random_state=42)

    center_geom = focus.to_crs("EPSG:4326").unary_union.centroid
    m = folium.Map(
        location=[center_geom.y, center_geom.x],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    pred_line = make_linestring_from_points(matches)
    gt_line = make_linestring_from_points(gt_points) if gt_points is not None else None

    add_geojson_layer(m, cropped_edges, "Extracted OSM road graph", "#888888", 1.0, 0.55)
    add_geojson_layer(m, pred_line, "Predicted path", "#0066cc", 5.0, 0.95)
    add_geojson_layer(m, gt_line, "GT-derived path", "#cc0000", 4.0, 0.85, dash_array="8,8")

    add_point_layer(m, raw_points, "Raw GPS points", "#000000", 3)
    add_point_layer(m, matches, "Predicted projected points", "#0066cc", 3)
    add_point_layer(m, gt_points, "GT projected points", "#cc0000", 2)

    folium.LayerControl(collapsed=False).add_to(m)

    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(output)

    print(f"[OK] Wrote OSM overlay map: {output}")
    print(f"[OK] Cropped road edges shown: {len(cropped_edges)}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edges.geojson"))
    parser.add_argument("--matches", type=Path, default=Path("outputs/matches/gnn_hmm_matches.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("data/interim/trajectory_clean.parquet"))
    parser.add_argument("--trajectory-id", type=int, default=None)
    parser.add_argument("--source-crs", type=str, default="EPSG:32632")
    parser.add_argument("--buffer-m", type=float, default=150.0)
    parser.add_argument("--max-edges", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        if args.trajectory_id is None:
            output = Path("outputs/figures/osm_overlay.html")
        else:
            output = Path(f"outputs/figures/osm_overlay_trajectory_{args.trajectory_id}.html")
    else:
        output = args.output

    edges = read_edges(args.edges, args.source_crs)
    matches = read_matches(args.matches, args.source_crs, args.trajectory_id)
    raw_points = read_raw_points(args.raw, args.source_crs, args.trajectory_id)
    gt_points = read_gt_from_matches(matches, args.source_crs)

    build_map(
        edges=edges,
        matches=matches,
        raw_points=raw_points,
        gt_points=gt_points,
        output=output,
        buffer_m=args.buffer_m,
        max_edges=args.max_edges,
    )


if __name__ == "__main__":
    main()