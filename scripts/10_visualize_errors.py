from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=Path("outputs/matches/gnn_hmm_matches.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edges.geojson"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--max-trajectories", type=int, default=12)
    parser.add_argument("--crs", default="EPSG:32632")
    args = parser.parse_args()

    ensure_dir(args.output)
    pred = pd.read_parquet(args.pred)
    edges = gpd.read_file(args.edges)
    if edges.crs is None:
        edges = edges.set_crs(args.crs)

    labelled = pred[pred["gt_edge_idx"] >= 0].copy()
    labelled["edge_correct"] = labelled["pred_edge_idx"].astype(int) == labelled["gt_edge_idx"].astype(int)
    error_tids = labelled.loc[~labelled["edge_correct"], "trajectory_id"].drop_duplicates().head(args.max_trajectories).tolist()
    if not error_tids:
        error_tids = labelled["trajectory_id"].drop_duplicates().head(args.max_trajectories).tolist()

    for tid in error_tids:
        g = labelled[labelled["trajectory_id"] == tid].sort_values("t")
        if len(g) < 2:
            continue
        xs = g["pred_proj_x"].tolist()
        ys = g["pred_proj_y"].tolist()
        gt_xs = g["gt_proj_x"].tolist()
        gt_ys = g["gt_proj_y"].tolist()
        bounds = (min(xs + gt_xs) - 200, min(ys + gt_ys) - 200, max(xs + gt_xs) + 200, max(ys + gt_ys) + 200)
        xmin, ymin, xmax, ymax = bounds
        local_edges = edges.cx[xmin:xmax, ymin:ymax]

        fig, ax = plt.subplots(figsize=(9, 9))
        if len(local_edges) > 0:
            local_edges.plot(ax=ax, linewidth=0.6, alpha=0.5)
        gpd.GeoSeries([LineString(list(zip(xs, ys)))], crs=args.crs).plot(ax=ax, linewidth=2, label="pred")
        gpd.GeoSeries([LineString(list(zip(gt_xs, gt_ys)))], crs=args.crs).plot(ax=ax, linewidth=2, linestyle="--", label="gt-derived")
        err = g[~g["edge_correct"]]
        if len(err) > 0:
            gpd.GeoSeries([Point(x, y) for x, y in zip(err["pred_proj_x"], err["pred_proj_y"])], crs=args.crs).plot(ax=ax, markersize=20)
        ax.set_title(f"Trajectory {tid} | errors={(~g['edge_correct']).sum()} / {len(g)}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.legend()
        fig.tight_layout()
        out = args.output / f"trajectory_{int(tid)}_errors.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"[OK] Wrote {out}")

    pred_gdf = gpd.GeoDataFrame(
        pred.copy(),
        geometry=[Point(x, y) for x, y in zip(pred["pred_proj_x"], pred["pred_proj_y"])],
        crs=args.crs,
    )
    geojson_out = args.output / "predicted_points.geojson"
    pred_gdf.to_file(geojson_out, driver="GeoJSON")
    print(f"[OK] Wrote {geojson_out}")


if __name__ == "__main__":
    main()
