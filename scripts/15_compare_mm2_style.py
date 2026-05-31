#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_WORKFLOWS = {
    "hmm": {
        "matches": "HMM/outputs/matches/hmm_matches_test.parquet",
        "output": "HMM/outputs/metrics/hmm_mm2_style_metrics_test.json",
        "trajectory_output": "HMM/outputs/metrics/hmm_mm2_style_trajectory_test.csv",
    },
    "gnn_hmm": {
        "matches": "outputs/matches/gnn_hmm_matches.parquet",
        "output": "outputs/metrics/gnn_hmm_mm2_style_metrics_test.json",
        "trajectory_output": "outputs/metrics/gnn_hmm_mm2_style_trajectory_test.csv",
    },
    "dsac": {
        "matches": "D-SAC/outputs/matches/dsac_asym_matches.parquet",
        "output": "D-SAC/outputs/metrics/dsac_mm2_style_metrics_test.json",
        "trajectory_output": "D-SAC/outputs/metrics/dsac_mm2_style_trajectory_test.csv",
    },
    "ppo": {
        "matches": "RL/outputs/matches/ppo_asym_matches.parquet",
        "output": "RL/outputs/metrics/ppo_mm2_style_metrics_test.json",
        "trajectory_output": "RL/outputs/metrics/ppo_mm2_style_trajectory_test.csv",
    },
}


def fmt_threshold(value: str) -> str:
    x = float(value)
    return str(int(x)) if x.is_integer() else str(x).replace(".", "p")


def criterion_value(metrics: dict, criterion: str, field: str = "weighted_mean_correct_fraction"):
    try:
        return metrics["criteria"][criterion][field]
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and compare MM2-style metrics across One-Direction workflows.")
    p.add_argument("--workflows", nargs="+", default=["hmm", "gnn_hmm", "dsac"], choices=list(DEFAULT_WORKFLOWS))
    p.add_argument("--thresholds-m", nargs="+", default=["2", "5", "10"])
    p.add_argument("--require-gt-candidate", action="store_true", default=True)
    p.add_argument("--include-gt-missing", action="store_true")
    p.add_argument("--summary-output", type=Path, default=Path("outputs/metrics/mm2_style_comparison.csv"))
    p.add_argument("--json-output", type=Path, default=Path("outputs/metrics/mm2_style_comparison.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    evaluator = repo / "scripts" / "14_evaluate_mm2_style.py"

    if not evaluator.exists():
        raise FileNotFoundError(evaluator)

    rows = []
    all_metrics = {}

    for workflow in args.workflows:
        spec = DEFAULT_WORKFLOWS[workflow]
        matches = repo / spec["matches"]

        if not matches.exists():
            print(f"[SKIP] {workflow}: missing {matches}", flush=True)
            continue

        output = repo / spec["output"]
        traj_output = repo / spec["trajectory_output"]

        cmd = [
            sys.executable, str(evaluator),
            "--workflow", workflow,
            "--matches", str(matches),
            "--output", str(output),
            "--trajectory-output", str(traj_output),
            "--thresholds-m", *args.thresholds_m,
        ]

        if args.include_gt_missing:
            cmd.append("--include-gt-missing")
        else:
            cmd.append("--require-gt-candidate")

        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=repo)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        metrics = json.loads(output.read_text(encoding="utf-8"))
        all_metrics[workflow] = metrics

        row = {
            "workflow": workflow,
            "num_points_evaluated": metrics.get("num_points_evaluated"),
            "num_trajectories": metrics.get("num_trajectories"),
            "edge_wmcf": criterion_value(metrics, "edge_correct"),
            "edge_mcf": criterion_value(metrics, "edge_correct", "mean_correct_fraction"),
            "edge_traj_success": criterion_value(metrics, "edge_correct", "trajectory_success_rate"),
        }

        for threshold in args.thresholds_m:
            tok = fmt_threshold(threshold)
            row[f"geometry_{tok}m_wmcf"] = criterion_value(metrics, f"mm2_geometry_correct_{tok}m")
            row[f"combined_{tok}m_wmcf"] = criterion_value(metrics, f"mm2_combined_correct_{tok}m")

        projection = metrics.get("projection_error", {})
        row["mean_projection_error_m"] = projection.get("mean_m")
        row["p90_projection_error_m"] = projection.get("p90_m")
        rows.append(row)

    summary = pd.DataFrame(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    args.json_output.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8", newline="\n")

    print(f"[OK] Wrote comparison CSV: {args.summary_output}", flush=True)
    print(f"[OK] Wrote comparison JSON: {args.json_output}", flush=True)
    if len(summary):
        print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
