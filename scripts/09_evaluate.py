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


def load_edge_lookup(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    edge_df = pd.read_parquet(path).copy()
    if "edge_idx" not in edge_df.columns:
        edge_df["edge_idx"] = np.arange(len(edge_df), dtype=np.int64)
    return edge_df


def add_edge_taxonomy(df: pd.DataFrame, edge_df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    out["same_osm_way"] = False
    out["same_road_class"] = False
    out["reverse_pair"] = False
    out["same_undirected_uv"] = False

    if edge_df is None:
        return out

    keep = ["edge_idx", "edge_id", "osm_way_id", "road_class", "u", "v", "direction", "is_reverse"]
    keep = [c for c in keep if c in edge_df.columns]
    edges = edge_df[keep].copy()

    pred = edges.add_prefix("pred_")
    gt = edges.add_prefix("gt_")

    out = out.merge(pred, left_on="pred_edge_idx", right_on="pred_edge_idx", how="left")
    out = out.merge(gt, left_on="gt_edge_idx", right_on="gt_edge_idx", how="left")

    if {"pred_osm_way_id", "gt_osm_way_id"}.issubset(out.columns):
        out["same_osm_way"] = out["pred_osm_way_id"].astype(str) == out["gt_osm_way_id"].astype(str)

    if {"pred_road_class", "gt_road_class"}.issubset(out.columns):
        out["same_road_class"] = out["pred_road_class"].astype(str) == out["gt_road_class"].astype(str)

    if {"pred_u", "pred_v", "gt_u", "gt_v"}.issubset(out.columns):
        pred_uv = out["pred_u"].astype(str) + ":" + out["pred_v"].astype(str)
        gt_uv = out["gt_u"].astype(str) + ":" + out["gt_v"].astype(str)
        pred_rev = out["pred_v"].astype(str) + ":" + out["pred_u"].astype(str)
        out["same_undirected_uv"] = (pred_uv == gt_uv) | (pred_rev == gt_uv)
        out["reverse_pair"] = pred_rev == gt_uv

    return out


def add_transition_diagnostics(df: pd.DataFrame, transition_table: Path | None) -> pd.DataFrame:
    out = df.sort_values(["trajectory_id", "t"]).copy()
    out["pred_prev_edge_idx"] = out.groupby("trajectory_id")["pred_edge_idx"].shift(1)
    out["gt_prev_edge_idx"] = out.groupby("trajectory_id")["gt_edge_idx"].shift(1)

    out["pred_transition_same"] = out["pred_prev_edge_idx"] == out["pred_edge_idx"]
    out["gt_transition_same"] = out["gt_prev_edge_idx"] == out["gt_edge_idx"]
    out["pred_transition_legal"] = True
    out["gt_transition_legal"] = True

    if transition_table is None or not transition_table.exists():
        return out

    transitions = pd.read_parquet(transition_table)
    legal_pairs = set(zip(transitions["prev_edge_idx"].astype(int), transitions["curr_edge_idx"].astype(int)))

    pred_legal = []
    gt_legal = []
    for _, row in out.iterrows():
        if pd.isna(row["pred_prev_edge_idx"]):
            pred_legal.append(True)
            gt_legal.append(True)
            continue

        pp = int(row["pred_prev_edge_idx"])
        pc = int(row["pred_edge_idx"])
        gp = int(row["gt_prev_edge_idx"])
        gc = int(row["gt_edge_idx"])

        pred_legal.append(pp == pc or (pp, pc) in legal_pairs)
        gt_legal.append(gp == gc or (gp, gc) in legal_pairs)

    out["pred_transition_legal"] = pred_legal
    out["gt_transition_legal"] = gt_legal
    return out


def summarize_bool(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.astype(bool).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=Path("outputs/matches/gnn_hmm_matches.parquet"))
    parser.add_argument("--edges", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--transition-table", type=Path, default=Path("data/processed/line_graph/transition_table.parquet"))
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

    edge_df = load_edge_lookup(args.edges)
    labelled = add_edge_taxonomy(labelled, edge_df)
    labelled = add_transition_diagnostics(labelled, args.transition_table)

    labelled["edge_correct"] = labelled["pred_edge_idx"].astype(int) == labelled["gt_edge_idx"].astype(int)
    labelled["projection_error_m"] = np.sqrt(
        (labelled["pred_proj_x"] - labelled["gt_proj_x"]) ** 2
        + (labelled["pred_proj_y"] - labelled["gt_proj_y"]) ** 2
    )
    labelled["projection_success"] = labelled["projection_error_m"] <= args.projection_threshold_m
    labelled["within_2m"] = labelled["projection_error_m"] <= 2.0
    labelled["within_5m"] = labelled["projection_error_m"] <= 5.0
    labelled["within_10m"] = labelled["projection_error_m"] <= 10.0
    labelled["near_but_wrong_edge"] = (~labelled["edge_correct"]) & labelled["within_5m"]
    labelled["same_way_but_wrong_edge"] = (~labelled["edge_correct"]) & labelled["same_osm_way"].astype(bool)
    labelled["reverse_edge_error"] = (~labelled["edge_correct"]) & labelled["reverse_pair"].astype(bool)
    labelled["severe_error"] = (~labelled["edge_correct"]) & (~labelled["within_10m"])

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
                "same_way_rate": summarize_bool(g["same_osm_way"]),
                "within_5m_rate": summarize_bool(g["within_5m"]),
                "pred_transition_legal_rate": summarize_bool(g[g["t"] > 0]["pred_transition_legal"]),
                "mean_projection_error_m": float(g["projection_error_m"].mean()),
                "path_edit_distance": int(edit),
                "success": bool(acc >= args.trajectory_success_acc),
            }
        )

    traj_df = pd.DataFrame(traj_metrics)

    error_cases = labelled[~labelled["edge_correct"]].sort_values(["trajectory_id", "t"])
    error_cases.to_csv(args.error_cases, index=False)
    traj_df.to_csv(args.output.with_name("trajectory_metrics.csv"), index=False)

    errors = labelled[~labelled["edge_correct"]]

    metrics = {
        "num_points": int(len(df)),
        "num_labelled_points": int(len(labelled)),
        "num_trajectories": int(labelled["trajectory_id"].nunique()),
        "point_edge_accuracy": float(labelled["edge_correct"].mean()),
        "same_osm_way_accuracy": summarize_bool(labelled["same_osm_way"]),
        "same_road_class_accuracy": summarize_bool(labelled["same_road_class"]),
        "same_undirected_uv_accuracy": summarize_bool(labelled["same_undirected_uv"]),
        "mean_projection_error_m": float(labelled["projection_error_m"].mean()),
        "median_projection_error_m": float(labelled["projection_error_m"].median()),
        "p90_projection_error_m": float(labelled["projection_error_m"].quantile(0.90)),
        "within_2m_rate": summarize_bool(labelled["within_2m"]),
        "within_5m_rate": summarize_bool(labelled["within_5m"]),
        "within_10m_rate": summarize_bool(labelled["within_10m"]),
        "projection_success_rate": float(labelled["projection_success"].mean()),
        "mean_confidence": float(labelled["confidence"].mean()) if "confidence" in labelled.columns else float("nan"),
        "path_edit_distance_mean": float(np.mean(path_edit_distances)),
        "path_edit_distance_median": float(np.median(path_edit_distances)),
        "trajectory_success_rate": float(traj_df["success"].mean()),
        "pred_transition_legal_rate": summarize_bool(labelled[labelled["t"] > 0]["pred_transition_legal"]),
        "gt_transition_legal_rate": summarize_bool(labelled[labelled["t"] > 0]["gt_transition_legal"]),
        "num_error_points": int((~labelled["edge_correct"]).sum()),
        "error_near_but_wrong_edge_rate": summarize_bool(errors["near_but_wrong_edge"]) if len(errors) else 0.0,
        "error_same_way_rate": summarize_bool(errors["same_way_but_wrong_edge"]) if len(errors) else 0.0,
        "error_reverse_edge_rate": summarize_bool(errors["reverse_edge_error"]) if len(errors) else 0.0,
        "error_severe_rate": summarize_bool(errors["severe_error"]) if len(errors) else 0.0,
        "error_cases_csv": str(args.error_cases),
    }

    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] Evaluation complete")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
