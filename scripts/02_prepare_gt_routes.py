from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import mapping
from shapely.ops import transform

REQUIRED_COLUMNS = {"id", "WKT"}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def project_geometry(geom, source_crs: str, target_crs: str):
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform(transformer.transform, geom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/trajectories/ground_truth.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/gt_routes_projected.parquet"))
    parser.add_argument("--geojson", type=Path, default=Path("data/interim/gt_routes_projected.geojson"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/gt_route_report.json"))
    parser.add_argument("--source-crs", default="EPSG:4326")
    parser.add_argument("--target-crs", default="EPSG:32632")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    df = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {args.input}. Found {list(df.columns)}")

    records: list[dict[str, Any]] = []
    geoms_projected = []
    for _, row in df.iterrows():
        tid = int(row["id"])
        geom_ll = wkt.loads(str(row["WKT"]))
        geom_xy = project_geometry(geom_ll, args.source_crs, args.target_crs)
        geoms_projected.append(geom_xy)
        records.append(
            {
                "trajectory_id": tid,
                "gt_wkt_lonlat": geom_ll.wkt,
                "gt_wkt_projected": geom_xy.wkt,
                "gt_geom_type": geom_ll.geom_type,
                "gt_length_m": float(geom_xy.length),
                "gt_bounds_projected": json.dumps(tuple(float(v) for v in geom_xy.bounds)),
            }
        )

    out = pd.DataFrame(records).sort_values("trajectory_id").reset_index(drop=True)
    ensure_parent(args.output)
    try:
        out.to_parquet(args.output, index=False)
        output_path = args.output
    except Exception as exc:  # noqa: BLE001
        output_path = args.output.with_suffix(".csv")
        out.to_csv(output_path, index=False)
        print(f"[WARN] Parquet failed ({exc}); wrote {output_path}")

    ensure_parent(args.geojson)
    gdf = gpd.GeoDataFrame(out[["trajectory_id", "gt_length_m"]].copy(), geometry=geoms_projected, crs=args.target_crs)
    gdf.to_file(args.geojson, driver="GeoJSON")

    report = {
        "input_path": str(args.input),
        "output_path": str(output_path),
        "geojson_path": str(args.geojson),
        "source_crs": args.source_crs,
        "target_crs": args.target_crs,
        "rows": int(len(out)),
        "num_trajectories": int(out["trajectory_id"].nunique()),
        "geometry_types": out["gt_geom_type"].value_counts().to_dict(),
        "length_m": {
            "min": float(out["gt_length_m"].min()),
            "median": float(out["gt_length_m"].median()),
            "mean": float(out["gt_length_m"].mean()),
            "max": float(out["gt_length_m"].max()),
        },
    }
    ensure_parent(args.report)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Prepared GT routes: {len(out)} trajectories")
    print(f"[OK] Output: {output_path}")
    print(f"[OK] GeoJSON: {args.geojson}")
    print(f"[OK] Report: {args.report}")


if __name__ == "__main__":
    main()
