from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=Path("outputs/matches/gnn_hmm_matches.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/gnn_hmm_metrics.json"))
    parser.add_argument("--error-cases", type=Path, default=Path("outputs/metrics/error_cases.csv"))
    parser.add_argument("--projection-threshold-m", type=float, default=10.0)
    parser.add_argument("--trajectory-success-acc", type=float, default=0.90)
    args = parser.parse_args()

    if not args.pred.exists():
        raise FileNotFoundError(args.pred)
    ensure_dir(args.output.parent)
    ensure_dir(args.error_cases.parent)

    df = pd.read_parquet(args.pred)
    labelled = df[df["gt_edge_idx"] >= 0].copy()
    if len(labelled) == 0:
        raise RuntimeError("No labelled points found in prediction file.")

    labelled["edge_correct"] = labelled["pred_edge_idx"].astype(int) == labelled["gt_edge_idx"].astype(int)
    labelled["projection_error_m"] = np.sqrt((labelled["pred_proj_x"] - labelled["gt_proj_x"]) ** 2 + (labelled["pred_proj_y"] - labelled["gt_proj_y"]) ** 2)
    labelled["projection_success"] = labelled["projection_error_m"] <= args.projection_threshold_m

    traj_metrics = []
    path_edit_distances = []
    for tid, g in labelled.groupby("trajectory_id"):
        g = g.sort_values("t")
        pred_seq = g["pred_edge_idx"].astype(int).tolist()
        gt_seq = g["gt_edge_idx"].astype(int).tolist()
        edit = levenshtein(pred_seq, gt_seq)
        path_edit_distances.append(edit)
        acc = float(g["edge_correct"].mean())
        traj_metrics.append(
            {
                "trajectory_id": int(tid),
                "points": int(len(g)),
                "edge_accuracy": acc,
                "mean_projection_error_m": float(g["projection_error_m"].mean()),
                "path_edit_distance": int(edit),
                "success": bool(acc >= args.trajectory_success_acc),
            }
        )

    traj_df = pd.DataFrame(traj_metrics)
    error_cases = labelled[~labelled["edge_correct"]].sort_values(["trajectory_id", "t"])
    error_cases.to_csv(args.error_cases, index=False)

    metrics = {
        "num_points": int(len(df)),
        "num_labelled_points": int(len(labelled)),
        "num_trajectories": int(labelled["trajectory_id"].nunique()),
        "point_edge_accuracy": float(labelled["edge_correct"].mean()),
        "mean_projection_error_m": float(labelled["projection_error_m"].mean()),
        "median_projection_error_m": float(labelled["projection_error_m"].median()),
        "p90_projection_error_m": float(labelled["projection_error_m"].quantile(0.90)),
        "projection_success_rate": float(labelled["projection_success"].mean()),
        "mean_confidence": float(labelled["confidence"].mean()),
        "path_edit_distance_mean": float(np.mean(path_edit_distances)),
        "path_edit_distance_median": float(np.median(path_edit_distances)),
        "trajectory_success_rate": float(traj_df["success"].mean()),
        "num_error_points": int((~labelled["edge_correct"]).sum()),
        "error_cases_csv": str(args.error_cases),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    traj_df.to_csv(args.output.with_name("trajectory_metrics.csv"), index=False)

    print("[OK] Evaluation complete")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
