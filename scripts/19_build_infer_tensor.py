#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_tensor_builder(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    spec = importlib.util.spec_from_file_location("one_direction_tensor_builder", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an inference tensor dataset from candidates_all.parquet.")
    parser.add_argument("--builder-script", type=Path, default=Path("scripts/06_build_training_tensors.py"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates/candidates_all.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--transition-table", type=Path, default=Path("data/processed/line_graph/transition_table.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/tensors/infer_dataset.pt"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/infer_tensor_report.json"))
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--transition-mask-mode", choices=["all", "legal", "speed"], default="all")
    parser.add_argument("--speed-margin-m", type=float, default=30.0)
    parser.add_argument("--trajectory-id", default=None, help="Optional trajectory_id filter.")
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    ensure_dir(args.report.parent)

    builder = load_tensor_builder(args.builder_script)

    candidates = pd.read_parquet(args.candidates)
    if args.trajectory_id is not None:
        candidates = candidates[candidates["trajectory_id"].astype(str) == str(args.trajectory_id)].copy()

    if candidates.empty:
        raise RuntimeError(f"No candidates found in {args.candidates} after filtering.")

    edge_df = pd.read_parquet(args.edges).reset_index(drop=True)
    edge_df["edge_idx"] = np.arange(len(edge_df), dtype=np.int64)
    transition_set = builder.load_transition_set(args.transition_table)

    datasets = []
    for tid in sorted(candidates["trajectory_id"].unique().tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        datasets.append(
            builder.build_one_trajectory(
                tid=int(tid),
                candidates=candidates,
                edge_df=edge_df,
                transition_set=transition_set,
                max_candidates=args.max_candidates,
                transition_mask_mode=args.transition_mask_mode,
                speed_margin_m=args.speed_margin_m,
            )
        )

    torch.save(datasets, args.output)

    total_points = sum(int(d["gt_candidate_pos"].numel()) for d in datasets)
    labelled = sum(int((d["gt_candidate_pos"] >= 0).sum().item()) for d in datasets)
    report = {
        "output": str(args.output),
        "candidate_source": str(args.candidates),
        "num_trajectories": len(datasets),
        "num_points": total_points,
        "num_labelled_points": labelled,
        "trajectory_ids": [int(d["trajectory_id"]) for d in datasets],
        "max_candidates": args.max_candidates,
        "transition_mask_mode": args.transition_mask_mode,
        "speed_margin_m": args.speed_margin_m,
    }

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    print(f"[OK] Built inference dataset: {args.output}")
    print(f"[OK] trajectories={len(datasets)} points={total_points} labelled={labelled}")
    print(f"[OK] report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
