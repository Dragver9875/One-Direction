#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EMISSION_FEATURE_NAMES = [
    "distance_norm",
    "log_distance_norm",
    "yaw_diff_norm",
    "abs_yaw_diff_norm",
    "offset_ratio",
    "speed_norm",
    "sin_yaw",
    "cos_yaw",
    "sin_segment_bearing",
    "cos_segment_bearing",
    "candidate_rank_norm",
    "speed_consistency",
    "oneway",
    "road_class_id_norm",
    "edge_length_log_norm",
    "yaw_reliability",
]

TRANSITION_FEATURE_NAMES = [
    "gps_dist_norm",
    "dt_norm",
    "observed_speed_norm",
    "route_dist_norm",
    "route_minus_gps_norm",
    "route_gps_ratio_norm",
    "turn_norm",
    "yaw_change_norm",
    "speed_consistency",
    "connected",
    "legal",
    "same_edge",
    "same_osm_way",
    "same_road_class",
    "direction_change",
    "candidate_rank_delta_norm",
    "prev_distance_norm",
    "curr_distance_norm",
    "distance_delta_norm",
    "time_feasible",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2 * math.pi) - math.pi)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_transition_set(path: Path) -> set[tuple[int, int]]:
    df = pd.read_parquet(path)
    return set(zip(df["prev_edge_idx"].astype(int), df["curr_edge_idx"].astype(int)))


def get_dt_seconds(timestamps: list[str], ti: int, fallback: float = 1.0) -> float:
    if ti <= 0:
        return fallback
    try:
        t0 = pd.to_datetime(timestamps[ti - 1])
        t1 = pd.to_datetime(timestamps[ti])
        dt = (t1 - t0).total_seconds()
        if np.isfinite(dt) and dt > 0:
            return float(dt)
    except Exception:
        pass
    return fallback


def transition_allowed(
    mode: str,
    legal: int,
    same: int,
    time_feasible: float,
) -> bool:
    if mode == "all":
        return True
    if mode == "legal":
        return bool(legal or same)
    if mode == "speed":
        return bool((legal or same) and time_feasible > 0.5)
    raise ValueError(f"Unsupported transition mask mode: {mode}")


def build_one_trajectory(
    tid: int,
    candidates: pd.DataFrame,
    edge_df: pd.DataFrame,
    transition_set: set[tuple[int, int]],
    max_candidates: int,
    transition_mask_mode: str,
    speed_margin_m: float,
) -> dict:
    g = candidates[candidates["trajectory_id"] == tid].sort_values(["t", "candidate_rank"]).copy()
    timesteps = sorted(g["t"].unique().tolist())
    t_to_rows = {t: g[g["t"] == t].sort_values("candidate_rank") for t in timesteps}

    T, K = len(timesteps), max_candidates

    candidate_edge_idx = torch.full((T, K), -1, dtype=torch.long)
    candidate_mask = torch.zeros((T, K), dtype=torch.bool)
    emission_features = torch.zeros((T, K, len(EMISSION_FEATURE_NAMES)), dtype=torch.float32)
    candidate_proj_xy = torch.full((T, K, 2), float("nan"), dtype=torch.float32)

    candidate_distance = torch.full((T, K), float("nan"), dtype=torch.float32)
    candidate_rank = torch.full((T, K), float("nan"), dtype=torch.float32)

    gt_candidate_pos = torch.full((T,), -1, dtype=torch.long)
    gt_edge_idx = torch.full((T,), -1, dtype=torch.long)
    gt_proj_xy = torch.full((T, 2), float("nan"), dtype=torch.float32)

    timestamps: list[str] = []
    xy = torch.zeros((T, 2), dtype=torch.float32)
    yaw = torch.zeros((T,), dtype=torch.float32)
    speed = torch.zeros((T,), dtype=torch.float32)

    edge_lookup = edge_df.set_index("edge_idx")

    for ti, t in enumerate(timesteps):
        rows = t_to_rows[t].head(K).reset_index(drop=True)
        first = rows.iloc[0]

        timestamps.append(str(first["timestamp"]))
        xy[ti] = torch.tensor([safe_float(first["x"]), safe_float(first["y"])], dtype=torch.float32)
        yaw[ti] = safe_float(first["yaw"])
        speed[ti] = safe_float(first["speed_mps"])

        gt_edge = int(first.get("gt_edge_idx", -1))
        gt_edge_idx[ti] = gt_edge
        gt_proj_xy[ti] = torch.tensor(
            [
                safe_float(first.get("gt_proj_x", np.nan), float("nan")),
                safe_float(first.get("gt_proj_y", np.nan), float("nan")),
            ],
            dtype=torch.float32,
        )

        for ci, row in rows.iterrows():
            edge_idx = int(row["edge_idx"])
            edge_row = edge_lookup.loc[edge_idx]

            distance_m = safe_float(row["distance_m"])
            yaw_diff = safe_float(row["yaw_diff_rad"])
            speed_mps = safe_float(row["speed_mps"])
            rank = safe_float(row["candidate_rank"])

            maxspeed_kmh = safe_float(edge_row.get("maxspeed", np.nan), 50.0)
            maxspeed_mps = maxspeed_kmh / 3.6
            speed_consistency = abs(speed_mps - maxspeed_mps) / max(maxspeed_mps, 1.0)

            edge_length = safe_float(edge_row.get("length_m", 0.0))
            road_class_id = safe_float(edge_row.get("road_class_id", 0.0))
            yaw_reliability = min(max(speed_mps / 5.0, 0.0), 1.0)

            candidate_edge_idx[ti, ci] = edge_idx
            candidate_mask[ti, ci] = True
            candidate_proj_xy[ti, ci] = torch.tensor([safe_float(row["proj_x"]), safe_float(row["proj_y"])])
            candidate_distance[ti, ci] = distance_m
            candidate_rank[ti, ci] = rank

            if int(row.get("is_gt", 0)) == 1:
                gt_candidate_pos[ti] = ci

            emission_features[ti, ci] = torch.tensor(
                [
                    min(distance_m, 100.0) / 50.0,
                    math.log1p(max(distance_m, 0.0)) / math.log1p(100.0),
                    yaw_diff / math.pi,
                    abs(yaw_diff) / math.pi,
                    safe_float(row["offset_ratio"]),
                    min(speed_mps, 50.0) / 30.0,
                    math.sin(safe_float(row["yaw"])),
                    math.cos(safe_float(row["yaw"])),
                    safe_float(row["sin_segment_bearing"]),
                    safe_float(row["cos_segment_bearing"]),
                    rank / max(K - 1, 1),
                    speed_consistency,
                    safe_float(edge_row.get("oneway", 0.0)),
                    road_class_id / 20.0,
                    math.log1p(max(edge_length, 0.0)) / math.log1p(1000.0),
                    yaw_reliability,
                ],
                dtype=torch.float32,
            )

    transition_features = torch.zeros((max(T - 1, 0), K, K, len(TRANSITION_FEATURE_NAMES)), dtype=torch.float32)
    transition_mask = torch.zeros((max(T - 1, 0), K, K), dtype=torch.bool)

    for ti in range(1, T):
        gps_dist = float(torch.linalg.vector_norm(xy[ti] - xy[ti - 1]).item())
        dt_s = get_dt_seconds(timestamps, ti, fallback=1.0)
        observed_speed = gps_dist / max(dt_s, 1.0e-3)
        yaw_change = abs(angle_diff(float(yaw[ti]), float(yaw[ti - 1])))

        for pi in range(K):
            prev_idx = int(candidate_edge_idx[ti - 1, pi])
            if prev_idx < 0:
                continue
            prev_row = edge_lookup.loc[prev_idx]

            for ci in range(K):
                curr_idx = int(candidate_edge_idx[ti, ci])
                if curr_idx < 0:
                    continue

                curr_row = edge_lookup.loc[curr_idx]

                same = int(prev_idx == curr_idx)
                connected = int((prev_idx, curr_idx) in transition_set)
                legal = int(same or connected)

                prev_osm_way = str(prev_row.get("osm_way_id", ""))
                curr_osm_way = str(curr_row.get("osm_way_id", ""))
                same_osm_way = int(prev_osm_way == curr_osm_way and prev_osm_way != "")

                prev_class = str(prev_row.get("road_class", ""))
                curr_class = str(curr_row.get("road_class", ""))
                same_road_class = int(prev_class == curr_class and prev_class != "")

                turn = abs(angle_diff(safe_float(curr_row["bearing_rad"]), safe_float(prev_row["bearing_rad"])))

                if same:
                    route_dist = gps_dist
                elif connected:
                    route_dist = max(gps_dist, min(safe_float(curr_row["length_m"]), 100.0))
                else:
                    route_dist = abs(safe_float(curr_row["length_m"]) - safe_float(prev_row["length_m"]))

                route_minus_gps = abs(route_dist - gps_dist)
                route_gps_ratio = route_dist / max(gps_dist, 1.0)

                maxspeed_kmh = safe_float(curr_row.get("maxspeed", np.nan), 50.0)
                maxspeed_mps = maxspeed_kmh / 3.6
                speed_consistency = abs(float(speed[ti]) - maxspeed_mps) / max(maxspeed_mps, 1.0)

                max_feasible_dist = maxspeed_mps * dt_s + speed_margin_m
                time_feasible = float(route_dist <= max_feasible_dist or same)

                direction_change = abs(
                    safe_float(curr_row.get("is_reverse", 0.0)) - safe_float(prev_row.get("is_reverse", 0.0))
                )

                prev_dist = float(candidate_distance[ti - 1, pi].item())
                curr_dist = float(candidate_distance[ti, ci].item())
                rank_delta = abs(float(candidate_rank[ti - 1, pi].item()) - float(candidate_rank[ti, ci].item()))

                transition_features[ti - 1, pi, ci] = torch.tensor(
                    [
                        min(gps_dist, 200.0) / 100.0,
                        min(dt_s, 60.0) / 10.0,
                        min(observed_speed, 60.0) / 30.0,
                        min(route_dist, 500.0) / 100.0,
                        min(route_minus_gps, 500.0) / 100.0,
                        min(route_gps_ratio, 10.0) / 10.0,
                        turn / math.pi,
                        yaw_change / math.pi,
                        speed_consistency,
                        float(connected),
                        float(legal),
                        float(same),
                        float(same_osm_way),
                        float(same_road_class),
                        float(direction_change),
                        rank_delta / max(K - 1, 1),
                        min(prev_dist, 100.0) / 50.0,
                        min(curr_dist, 100.0) / 50.0,
                        min(abs(curr_dist - prev_dist), 100.0) / 50.0,
                        float(time_feasible),
                    ],
                    dtype=torch.float32,
                )

                transition_mask[ti - 1, pi, ci] = transition_allowed(
                    mode=transition_mask_mode,
                    legal=legal,
                    same=same,
                    time_feasible=time_feasible,
                )

    return {
        "trajectory_id": int(tid),
        "timesteps": torch.tensor(timesteps, dtype=torch.long),
        "timestamps": timestamps,
        "xy": xy,
        "yaw": yaw,
        "speed_mps": speed,
        "candidate_edge_idx": candidate_edge_idx,
        "candidate_mask": candidate_mask,
        "candidate_proj_xy": candidate_proj_xy,
        "emission_features": emission_features,
        "transition_features": transition_features,
        "transition_mask": transition_mask,
        "gt_candidate_pos": gt_candidate_pos,
        "gt_edge_idx": gt_edge_idx,
        "gt_proj_xy": gt_proj_xy,
        "emission_feature_names": EMISSION_FEATURE_NAMES,
        "transition_feature_names": TRANSITION_FEATURE_NAMES,
    }


def build_split(
    split: str,
    candidate_dir: Path,
    edge_df: pd.DataFrame,
    transition_set: set[tuple[int, int]],
    max_candidates: int,
    output_dir: Path,
    transition_mask_mode: str,
    speed_margin_m: float,
) -> dict:
    path = candidate_dir / f"candidates_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    candidates = pd.read_parquet(path)
    datasets = []

    for tid in sorted(candidates["trajectory_id"].unique().tolist()):
        datasets.append(
            build_one_trajectory(
                tid=int(tid),
                candidates=candidates,
                edge_df=edge_df,
                transition_set=transition_set,
                max_candidates=max_candidates,
                transition_mask_mode=transition_mask_mode,
                speed_margin_m=speed_margin_m,
            )
        )

    out_path = output_dir / f"{split}_dataset.pt"
    torch.save(datasets, out_path)

    labelled = sum(int((d["gt_candidate_pos"] >= 0).sum().item()) for d in datasets)
    total = sum(int(d["gt_candidate_pos"].numel()) for d in datasets)
    legal_density = float(np.mean([float(d["transition_mask"].float().mean().item()) for d in datasets if d["transition_mask"].numel() > 0]))

    return {
        "split": split,
        "trajectories": len(datasets),
        "points": total,
        "labelled_points": labelled,
        "transition_mask_density": legal_density,
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--transition-table", type=Path, default=Path("data/processed/line_graph/transition_table.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/tensors"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/tensor_report.json"))
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--transition-mask-mode", choices=["all", "legal", "speed"], default="all")
    parser.add_argument("--speed-margin-m", type=float, default=30.0)
    args = parser.parse_args()

    ensure_dir(args.output)
    ensure_dir(args.report.parent)

    edge_df = pd.read_parquet(args.edges).reset_index(drop=True)
    edge_df["edge_idx"] = np.arange(len(edge_df), dtype=np.int64)
    transition_set = load_transition_set(args.transition_table)

    reports = [
        build_split(
            split=split,
            candidate_dir=args.candidates,
            edge_df=edge_df,
            transition_set=transition_set,
            max_candidates=args.max_candidates,
            output_dir=args.output,
            transition_mask_mode=args.transition_mask_mode,
            speed_margin_m=args.speed_margin_m,
        )
        for split in ["train", "val", "test"]
    ]

    report = {
        "splits": reports,
        "emission_feature_names": EMISSION_FEATURE_NAMES,
        "transition_feature_names": TRANSITION_FEATURE_NAMES,
        "transition_mask_mode": args.transition_mask_mode,
        "speed_margin_m": args.speed_margin_m,
    }

    with args.report.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    print("[OK] Built training tensors")
    for row in reports:
        print(f"[OK] {row['split']}: {row['trajectories']} trajectories, {row['points']} points -> {row['output']}")
    print(f"[OK] transition_mask_mode={args.transition_mask_mode}")


if __name__ == "__main__":
    main()
