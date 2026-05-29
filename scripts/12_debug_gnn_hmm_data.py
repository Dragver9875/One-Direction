from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def bool_rate(x) -> float:
    if len(x) == 0:
        return float("nan")
    return float(np.mean(np.asarray(x, dtype=bool)))


def tensor_summary(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    points = sum(int(item["candidate_edge_idx"].shape[0]) for item in data)
    labelled = sum(int((item["gt_candidate_pos"] >= 0).sum().item()) for item in data)
    candidate_mask_density = np.mean([float(item["candidate_mask"].float().mean().item()) for item in data])
    transition_mask_values = [
        float(item["transition_mask"].float().mean().item())
        for item in data
        if "transition_mask" in item and item["transition_mask"].numel() > 0
    ]

    gt_candidate_valid = []
    gt_transition_valid = []
    for item in data:
        labels = item["gt_candidate_pos"]
        mask = item["candidate_mask"]
        for t, pos in enumerate(labels.tolist()):
            if pos >= 0:
                gt_candidate_valid.append(bool(mask[t, pos].item()))
        if item.get("transition_mask") is not None and item["transition_mask"].numel() > 0:
            tm = item["transition_mask"]
            for t in range(len(labels) - 1):
                a = int(labels[t].item())
                b = int(labels[t + 1].item())
                if a >= 0 and b >= 0:
                    gt_transition_valid.append(bool(tm[t, a, b].item()))

    return {
        "path": str(path),
        "trajectories": len(data),
        "points": points,
        "labelled_points": labelled,
        "candidate_mask_density": float(candidate_mask_density),
        "transition_mask_density": float(np.mean(transition_mask_values)) if transition_mask_values else float("nan"),
        "gt_candidate_position_valid_rate": bool_rate(gt_candidate_valid),
        "gt_transition_mask_valid_rate": bool_rate(gt_transition_valid),
        "emission_feature_dim": int(data[0]["emission_features"].shape[-1]),
        "transition_feature_dim": int(data[0]["transition_features"].shape[-1]),
        "emission_feature_names": data[0].get("emission_feature_names", []),
        "transition_feature_names": data[0].get("transition_feature_names", []),
    }


def edge_summary(path: Path) -> dict:
    df = pd.read_parquet(path)
    out = {
        "edge_count": int(len(df)),
        "edge_id_unique": bool(df["edge_id"].is_unique) if "edge_id" in df.columns else None,
        "edge_idx_unique": bool(df["edge_idx"].is_unique) if "edge_idx" in df.columns else None,
    }
    if "direction" in df.columns:
        out["direction_counts"] = df["direction"].value_counts(dropna=False).to_dict()
    if "oneway" in df.columns:
        out["oneway_counts"] = df["oneway"].value_counts(dropna=False).to_dict()
    if "road_class" in df.columns:
        out["road_class_counts"] = df["road_class"].value_counts(dropna=False).head(30).to_dict()
    return out


def candidate_recall_by_trajectory(candidate_dir: Path) -> dict:
    frames = []
    for split in ["train", "val", "test"]:
        path = candidate_dir / f"candidates_{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["split"] = split
            frames.append(df)
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    point = df.groupby(["split", "trajectory_id", "t"])["is_gt"].max().reset_index()
    traj = point.groupby(["split", "trajectory_id"])["is_gt"].mean().reset_index(name="recall")
    worst = traj.sort_values("recall").head(20)

    return {
        "global_point_recall": float(point["is_gt"].mean()),
        "trajectory_recall_mean": float(traj["recall"].mean()),
        "trajectory_recall_min": float(traj["recall"].min()),
        "worst_trajectories": worst.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/processed/tensors/train_dataset.pt"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/tensors/val_dataset.pt"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/tensors/test_dataset.pt"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/gnn_hmm_data_debug_report.json"))
    args = parser.parse_args()

    report = {
        "tensors": {},
        "edges": edge_summary(args.edges) if args.edges.exists() else {},
        "candidate_recall": candidate_recall_by_trajectory(args.candidates),
    }

    for split, path in [("train", args.train), ("val", args.val), ("test", args.test)]:
        if path.exists():
            report["tensors"][split] = tensor_summary(path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)

    print("[OK] Wrote debug report:", args.output)
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
