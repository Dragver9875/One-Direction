from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2 * math.pi) - math.pi)


def load_transition_set(path: Path) -> set[tuple[int, int]]:
    df = pd.read_parquet(path)
    return set(zip(df["prev_edge_idx"].astype(int), df["curr_edge_idx"].astype(int)))


def build_one_trajectory(
    tid: int,
    candidates: pd.DataFrame,
    edge_df: pd.DataFrame,
    transition_set: set[tuple[int, int]],
    max_candidates: int,
) -> dict:
    g = candidates[candidates["trajectory_id"] == tid].sort_values(["t", "candidate_rank"]).copy()
    timesteps = sorted(g["t"].unique().tolist())
    t_to_rows = {t: g[g["t"] == t].sort_values("candidate_rank") for t in timesteps}
    T, K = len(timesteps), max_candidates

    candidate_edge_idx = torch.full((T, K), -1, dtype=torch.long)
    candidate_mask = torch.zeros((T, K), dtype=torch.bool)
    emission_features = torch.zeros((T, K, 10), dtype=torch.float32)
    candidate_proj_xy = torch.full((T, K, 2), float("nan"), dtype=torch.float32)
    gt_candidate_pos = torch.full((T,), -1, dtype=torch.long)
    gt_edge_idx = torch.full((T,), -1, dtype=torch.long)
    gt_proj_xy = torch.full((T, 2), float("nan"), dtype=torch.float32)
    timestamps: list[str] = []
    xy = torch.zeros((T, 2), dtype=torch.float32)
    yaw = torch.zeros((T,), dtype=torch.float32)
    speed = torch.zeros((T,), dtype=torch.float32)

    for ti, t in enumerate(timesteps):
        rows = t_to_rows[t].head(K).reset_index(drop=True)
        first = rows.iloc[0]
        timestamps.append(str(first["timestamp"]))
        xy[ti] = torch.tensor([float(first["x"]), float(first["y"])])
        yaw[ti] = float(first["yaw"])
        speed[ti] = float(first["speed_mps"])
        gt_edge = int(first.get("gt_edge_idx", -1))
        gt_edge_idx[ti] = gt_edge
        gt_proj_xy[ti] = torch.tensor([float(first.get("gt_proj_x", np.nan)), float(first.get("gt_proj_y", np.nan))])

        for ci, row in rows.iterrows():
            candidate_edge_idx[ti, ci] = int(row["edge_idx"])
            candidate_mask[ti, ci] = True
            candidate_proj_xy[ti, ci] = torch.tensor([float(row["proj_x"]), float(row["proj_y"])])
            if int(row.get("is_gt", 0)) == 1:
                gt_candidate_pos[ti] = ci
            emission_features[ti, ci] = torch.tensor(
                [
                    float(row["distance_m"]) / 50.0,
                    float(row["yaw_diff_rad"]) / math.pi,
                    float(row["offset_ratio"]),
                    float(row["speed_mps"]) / 30.0,
                    math.sin(float(row["yaw"])),
                    math.cos(float(row["yaw"])),
                    float(row["sin_segment_bearing"]),
                    float(row["cos_segment_bearing"]),
                    float(row["candidate_rank"]) / max(K - 1, 1),
                    float(row.get("speed_consistency", 0.0)),
                ],
                dtype=torch.float32,
            )

    transition_features = torch.zeros((max(T - 1, 0), K, K, 9), dtype=torch.float32)
    transition_mask = torch.zeros((max(T - 1, 0), K, K), dtype=torch.bool)

    edge_lookup = edge_df.set_index("edge_idx")
    for ti in range(1, T):
        gps_dist = float(torch.linalg.vector_norm(xy[ti] - xy[ti - 1]).item())
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
                turn = abs(angle_diff(float(curr_row["bearing_rad"]), float(prev_row["bearing_rad"])))
                if same:
                    route_dist = gps_dist
                elif connected:
                    route_dist = min(float(curr_row["length_m"]), max(gps_dist, 1.0))
                else:
                    route_dist = float(abs(float(curr_row["length_m"]) - float(prev_row["length_m"])))
                absdiff = abs(route_dist - gps_dist)
                maxspeed = curr_row.get("maxspeed", np.nan)
                maxspeed_mps = 50.0 / 3.6 if pd.isna(maxspeed) else float(maxspeed) / 3.6
                speed_consistency = abs(float(speed[ti]) - maxspeed_mps) / max(maxspeed_mps, 1.0)
                transition_features[ti - 1, pi, ci] = torch.tensor(
                    [
                        gps_dist / 100.0,
                        route_dist / 100.0,
                        absdiff / 100.0,
                        turn / math.pi,
                        yaw_change / math.pi,
                        speed_consistency,
                        float(connected),
                        float(legal),
                        float(same),
                    ],
                    dtype=torch.float32,
                )
                transition_mask[ti - 1, pi, ci] = True

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
    }


def build_split(split: str, candidate_dir: Path, edge_df: pd.DataFrame, transition_set: set[tuple[int, int]], max_candidates: int, output_dir: Path) -> dict:
    path = candidate_dir / f"candidates_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = pd.read_parquet(path)
    datasets = []
    for tid in sorted(candidates["trajectory_id"].unique().tolist()):
        datasets.append(build_one_trajectory(int(tid), candidates, edge_df, transition_set, max_candidates))
    out_path = output_dir / f"{split}_dataset.pt"
    torch.save(datasets, out_path)
    labelled = sum(int((d["gt_candidate_pos"] >= 0).sum().item()) for d in datasets)
    total = sum(int(d["gt_candidate_pos"].numel()) for d in datasets)
    return {"split": split, "trajectories": len(datasets), "points": total, "labelled_points": labelled, "output": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--transition-table", type=Path, default=Path("data/processed/line_graph/transition_table.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/tensors"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/tensor_report.json"))
    parser.add_argument("--max-candidates", type=int, default=10)
    args = parser.parse_args()

    ensure_dir(args.output)
    ensure_dir(args.report.parent)
    edge_df = pd.read_parquet(args.edges).reset_index(drop=True)
    edge_df["edge_idx"] = np.arange(len(edge_df), dtype=np.int64)
    transition_set = load_transition_set(args.transition_table)

    reports = [build_split(split, args.candidates, edge_df, transition_set, args.max_candidates, args.output) for split in ["train", "val", "test"]]
    report = {"splits": reports, "max_candidates": args.max_candidates}
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("[OK] Built training tensors")
    for r in reports:
        print(f"[OK] {r['split']}: {r['trajectories']} trajectories, {r['points']} points -> {r['output']}")


if __name__ == "__main__":
    main()
