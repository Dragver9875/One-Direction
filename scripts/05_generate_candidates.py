from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2 * math.pi) - math.pi)


def strtree_query_indices(tree: STRtree, geom) -> list[int]:
    result = tree.query(geom)
    if len(result) == 0:
        return []
    first = result[0]
    if isinstance(first, (int, np.integer)):
        return [int(x) for x in result]
    raise RuntimeError("This script expects Shapely 2 STRtree returning indices. Upgrade shapely>=2.0.")


def load_gt_routes(path: Path) -> dict[int, Any]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    routes = {}
    for _, row in df.iterrows():
        routes[int(row["trajectory_id"])] = wkt.loads(str(row["gt_wkt_projected"]))
    return routes


def nearest_point_on_route(route_geom, point: Point) -> Point:
    dist_along = route_geom.project(point)
    return route_geom.interpolate(dist_along)


def choose_gt_edge_for_point(
    point: Point,
    route_geom,
    tree: STRtree,
    edge_geoms: list[Any],
    edge_df: pd.DataFrame,
    align_radius_m: float,
) -> tuple[int | None, float, float, float]:
    if route_geom is None:
        return None, np.nan, np.nan, np.nan
    gt_point = nearest_point_on_route(route_geom, point)
    idxs = strtree_query_indices(tree, gt_point.buffer(align_radius_m))
    if not idxs:
        idxs = strtree_query_indices(tree, gt_point.buffer(align_radius_m * 2.0))
    if not idxs:
        return None, float(gt_point.x), float(gt_point.y), np.nan
    best_idx = None
    best_dist = float("inf")
    for idx in idxs:
        d = float(edge_geoms[idx].distance(gt_point))
        if d < best_dist:
            best_idx = idx
            best_dist = d
    if best_idx is None:
        return None, float(gt_point.x), float(gt_point.y), np.nan
    return int(edge_df.iloc[best_idx]["edge_idx"]), float(gt_point.x), float(gt_point.y), best_dist


def project_to_edge(point: Point, geom) -> tuple[float, float, float, float]:
    offset = float(geom.project(point))
    proj = geom.interpolate(offset)
    ratio = offset / max(float(geom.length), 1e-9)
    return float(proj.x), float(proj.y), offset, ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, default=Path("data/interim/trajectory_clean.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--gt-routes", type=Path, default=Path("data/interim/gt_routes_projected.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/candidates"))
    parser.add_argument("--split-output", type=Path, default=Path("data/processed/splits"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/candidate_report.json"))
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--gt-align-radius-m", type=float, default=30.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output)
    ensure_dir(args.split_output)
    ensure_dir(args.report.parent)

    traj = pd.read_parquet(args.trajectories)
    edge_df = pd.read_parquet(args.edges).reset_index(drop=True)
    edge_df["edge_idx"] = np.arange(len(edge_df), dtype=np.int64)
    edge_geoms = [wkt.loads(s) for s in edge_df["geometry_wkt"]]
    tree = STRtree(edge_geoms)
    gt_routes = load_gt_routes(args.gt_routes)

    rows: list[dict[str, Any]] = []
    gt_records: list[dict[str, Any]] = []
    for row in traj.itertuples(index=False):
        pt = Point(float(row.x), float(row.y))
        tid = int(row.trajectory_id)
        gt_edge_idx, gt_x, gt_y, gt_edge_dist = choose_gt_edge_for_point(
            pt, gt_routes.get(tid), tree, edge_geoms, edge_df, args.gt_align_radius_m
        )
        gt_records.append({"trajectory_id": tid, "t": int(row.t), "gt_edge_idx": gt_edge_idx, "gt_proj_x": gt_x, "gt_proj_y": gt_y, "gt_edge_dist_m": gt_edge_dist})

        idxs = strtree_query_indices(tree, pt.buffer(args.radius_m))
        if not idxs:
            idxs = strtree_query_indices(tree, pt.buffer(args.radius_m * 2.0))
        cand = []
        for idx in idxs:
            erow = edge_df.iloc[idx]
            geom = edge_geoms[idx]
            dist = float(geom.distance(pt))
            proj_x, proj_y, offset, offset_ratio = project_to_edge(pt, geom)
            yaw_diff = abs(angle_diff(float(row.yaw), float(erow["bearing_rad"])))
            maxspeed = erow.get("maxspeed", np.nan)
            maxspeed = 50.0 if pd.isna(maxspeed) else float(maxspeed)
            speed_consistency = abs(float(row.speed_mps) - maxspeed / 3.6) / max(maxspeed / 3.6, 1.0)
            cand.append(
                {
                    "trajectory_id": tid,
                    "t": int(row.t),
                    "timestamp": str(row.timestamp),
                    "x": float(row.x),
                    "y": float(row.y),
                    "yaw": float(row.yaw),
                    "speed_mps": float(row.speed_mps),
                    "edge_idx": int(erow["edge_idx"]),
                    "edge_id": str(erow["edge_id"]),
                    "distance_m": dist,
                    "yaw_diff_rad": yaw_diff,
                    "proj_x": proj_x,
                    "proj_y": proj_y,
                    "offset_m": offset,
                    "offset_ratio": offset_ratio,
                    "segment_bearing_rad": float(erow["bearing_rad"]),
                    "sin_segment_bearing": float(erow["sin_bearing"]),
                    "cos_segment_bearing": float(erow["cos_bearing"]),
                    "road_class": str(erow["road_class"]),
                    "road_class_id": int(erow.get("road_class_id", 0)),
                    "speed_consistency": float(speed_consistency),
                    "gt_edge_idx": -1 if gt_edge_idx is None else int(gt_edge_idx),
                    "gt_proj_x": gt_x,
                    "gt_proj_y": gt_y,
                    "gt_edge_dist_m": gt_edge_dist,
                }
            )
        cand.sort(key=lambda r: (r["distance_m"], r["yaw_diff_rad"]))
        cand = cand[: args.max_candidates]
        for rank, r in enumerate(cand):
            r["candidate_rank"] = rank
            r["is_gt"] = int(r["edge_idx"] == r["gt_edge_idx"])
            rows.append(r)

    candidates = pd.DataFrame(rows)
    candidates.to_parquet(args.output / "candidates_all.parquet", index=False)
    pd.DataFrame(gt_records).to_parquet(args.output / "point_gt_derived.parquet", index=False)

    tids = sorted(traj["trajectory_id"].unique().tolist())
    rng = np.random.default_rng(args.seed)
    tids = np.array(tids)
    rng.shuffle(tids)
    n = len(tids)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    split_ids = {
        "train": tids[:n_train].tolist(),
        "val": tids[n_train : n_train + n_val].tolist(),
        "test": tids[n_train + n_val :].tolist(),
    }
    for split, ids in split_ids.items():
        subset = candidates[candidates["trajectory_id"].isin(ids)].copy()
        subset.to_parquet(args.output / f"candidates_{split}.parquet", index=False)
        with (args.split_output / f"{split}_ids.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(str(int(x)) for x in ids))

    point_groups = candidates.groupby(["trajectory_id", "t"])
    recall = {}
    for k in [1, 3, 5, 10]:
        hits = 0
        total = 0
        for _, g in point_groups:
            if int(g["gt_edge_idx"].iloc[0]) < 0:
                continue
            total += 1
            top = g.sort_values("candidate_rank").head(k)
            hits += int((top["edge_idx"] == int(g["gt_edge_idx"].iloc[0])).any())
        recall[f"top_{k}"] = None if total == 0 else hits / total

    report = {
        "num_candidate_rows": int(len(candidates)),
        "num_points": int(traj.groupby(["trajectory_id", "t"]).ngroups),
        "num_trajectories": int(traj["trajectory_id"].nunique()),
        "radius_m": args.radius_m,
        "max_candidates": args.max_candidates,
        "gt_align_radius_m": args.gt_align_radius_m,
        "candidate_recall": recall,
        "mean_candidates_per_point": float(candidates.groupby(["trajectory_id", "t"]).size().mean()),
        "splits": {k: len(v) for k, v in split_ids.items()},
        "outputs": {"candidates_all": str(args.output / "candidates_all.parquet")},
    }
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Generated candidates: {len(candidates)} rows")
    print(f"[OK] Recall: {recall}")
    print(f"[OK] Output: {args.output}")


if __name__ == "__main__":
    main()
