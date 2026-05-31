#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_threshold(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def load_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported match file type: {path.suffix}")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "trajectory_id": ["trajectory_id", "traj_id", "track_id", "id"],
        "t": ["t", "step", "point_idx", "idx", "time_idx"],
        "pred_edge_idx": ["pred_edge_idx", "pred_edge_id", "matched_edge_idx", "matched_edge_id"],
        "gt_edge_idx": ["gt_edge_idx", "gt_edge_id", "true_edge_idx", "true_edge_id"],
        "pred_candidate_pos": ["pred_candidate_pos", "pred_action", "action", "pred_candidate"],
        "gt_candidate_pos": ["gt_candidate_pos", "gt_action", "target", "gt_candidate"],
        "pred_proj_x": ["pred_proj_x", "matched_x", "proj_x", "x_pred"],
        "pred_proj_y": ["pred_proj_y", "matched_y", "proj_y", "y_pred"],
        "gt_proj_x": ["gt_proj_x", "true_proj_x", "gt_x", "x_gt"],
        "gt_proj_y": ["gt_proj_y", "true_proj_y", "gt_y", "y_gt"],
        "confidence": ["confidence", "prob", "posterior"],
    }
    for canonical, names in aliases.items():
        if canonical in out.columns:
            continue
        for name in names:
            if name in out.columns:
                out[canonical] = out[name]
                break
    if "trajectory_id" not in out.columns:
        raise ValueError("Missing trajectory_id column.")
    if "t" not in out.columns:
        out["t"] = out.groupby("trajectory_id").cumcount()
    return out


def add_correctness_columns(df: pd.DataFrame, thresholds_m: list[float], require_gt_candidate: bool) -> pd.DataFrame:
    out = df.copy()
    if "gt_edge_idx" in out.columns:
        out = out[out["gt_edge_idx"].notna()].copy()
        out = out[out["gt_edge_idx"].astype(int) >= 0].copy()
    if require_gt_candidate and "gt_candidate_pos" in out.columns:
        out = out[out["gt_candidate_pos"].notna()].copy()
        out = out[out["gt_candidate_pos"].astype(int) >= 0].copy()
    if "pred_edge_idx" in out.columns and "gt_edge_idx" in out.columns:
        out["edge_correct"] = out["pred_edge_idx"].astype(int) == out["gt_edge_idx"].astype(int)
    elif "pred_candidate_pos" in out.columns and "gt_candidate_pos" in out.columns:
        out["edge_correct"] = out["pred_candidate_pos"].astype(int) == out["gt_candidate_pos"].astype(int)
    else:
        out["edge_correct"] = False

    has_proj = {"pred_proj_x", "pred_proj_y", "gt_proj_x", "gt_proj_y"}.issubset(out.columns)
    if has_proj:
        out["projection_error_m"] = np.sqrt(
            (out["pred_proj_x"].astype(float) - out["gt_proj_x"].astype(float)) ** 2
            + (out["pred_proj_y"].astype(float) - out["gt_proj_y"].astype(float)) ** 2
        )
    elif "projection_error_m" not in out.columns:
        out["projection_error_m"] = np.nan

    for threshold in thresholds_m:
        tok = fmt_threshold(threshold)
        out[f"mm2_geometry_correct_{tok}m"] = out["projection_error_m"] <= threshold
        out[f"mm2_combined_correct_{tok}m"] = out["edge_correct"] | (out["projection_error_m"] <= threshold)
    return out


def add_route_length_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mm2_point_weight_m"] = 1.0
    out["mm2_segment_to_next_m"] = np.nan
    if not {"gt_proj_x", "gt_proj_y"}.issubset(out.columns):
        return out

    weights = pd.Series(index=out.index, dtype=float)
    seg_to_next = pd.Series(index=out.index, dtype=float)

    for _, group in out.sort_values(["trajectory_id", "t"]).groupby("trajectory_id", sort=False):
        idx = group.index.to_list()
        coords = group[["gt_proj_x", "gt_proj_y"]].astype(float).to_numpy()
        if len(group) == 1:
            weights.loc[idx] = 1.0
            seg_to_next.loc[idx] = np.nan
            continue

        seg = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        w = np.zeros(len(group), dtype=float)
        w[0] = 0.5 * seg[0]
        w[-1] = 0.5 * seg[-1]
        if len(group) > 2:
            w[1:-1] = 0.5 * seg[:-1] + 0.5 * seg[1:]

        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(group), dtype=float)

        weights.loc[idx] = w
        seg_full = np.full(len(group), np.nan, dtype=float)
        seg_full[:-1] = seg
        seg_to_next.loc[idx] = seg_full

    out["mm2_point_weight_m"] = weights.fillna(1.0).clip(lower=0.0)
    out["mm2_segment_to_next_m"] = seg_to_next
    if float(out["mm2_point_weight_m"].sum()) <= 0:
        out["mm2_point_weight_m"] = 1.0
    return out


def summarize_criterion(df: pd.DataFrame, criterion: str, trajectory_success_threshold: float) -> dict[str, Any]:
    if criterion not in df.columns:
        return {"criterion": criterion, "available": False}

    labelled = df[df[criterion].notna()].copy()
    if len(labelled) == 0:
        return {"criterion": criterion, "available": False}

    labelled[criterion] = labelled[criterion].astype(bool)
    weights = labelled["mm2_point_weight_m"].astype(float).clip(lower=0.0)
    weighted_den = float(weights.sum())
    weighted_num = float((weights * labelled[criterion].astype(float)).sum())

    traj_rows = []
    for tid, group in labelled.groupby("trajectory_id"):
        gw = group["mm2_point_weight_m"].astype(float).clip(lower=0.0)
        den = float(gw.sum())
        if den <= 0:
            gw = pd.Series(np.ones(len(group)), index=group.index)
            den = float(len(group))
        weighted_fraction = float((gw * group[criterion].astype(float)).sum() / den)
        correct_fraction = float(group[criterion].astype(float).mean())
        traj_rows.append({
            "trajectory_id": tid,
            "num_points": int(len(group)),
            "route_length_weight_m": den,
            "correct_fraction": correct_fraction,
            "weighted_correct_fraction": weighted_fraction,
            "success": bool(weighted_fraction >= trajectory_success_threshold),
        })
    traj = pd.DataFrame(traj_rows)

    return {
        "criterion": criterion,
        "available": True,
        "num_points": int(len(labelled)),
        "num_trajectories": int(labelled["trajectory_id"].nunique()),
        "mean_correct_fraction": float(traj["correct_fraction"].mean()),
        "weighted_mean_correct_fraction": float(weighted_num / weighted_den) if weighted_den > 0 else float(labelled[criterion].mean()),
        "mean_trajectory_weighted_correct_fraction": float(traj["weighted_correct_fraction"].mean()),
        "trajectory_success_rate": float(traj["success"].mean()),
        "total_route_weight_m": weighted_den,
        "correct_route_weight_m": weighted_num,
    }


def make_trajectory_table(df: pd.DataFrame, criteria: list[str], trajectory_success_threshold: float) -> pd.DataFrame:
    rows = []
    for tid, group in df.sort_values(["trajectory_id", "t"]).groupby("trajectory_id", sort=False):
        row: dict[str, Any] = {
            "trajectory_id": tid,
            "num_points": int(len(group)),
            "route_length_weight_m": float(group["mm2_point_weight_m"].sum()),
        }
        for criterion in criteria:
            if criterion not in group.columns:
                continue
            gw = group["mm2_point_weight_m"].astype(float).clip(lower=0.0)
            den = float(gw.sum())
            if den <= 0:
                gw = pd.Series(np.ones(len(group)), index=group.index)
                den = float(len(group))
            correct = group[criterion].astype(float)
            row[f"{criterion}_fraction"] = float(correct.mean())
            row[f"{criterion}_weighted_fraction"] = float((gw * correct).sum() / den)
            row[f"{criterion}_success"] = bool(row[f"{criterion}_weighted_fraction"] >= trajectory_success_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_mm2_style(matches: pd.DataFrame, workflow: str, thresholds_m: list[float], require_gt_candidate: bool, trajectory_success_threshold: float):
    df = canonicalize_columns(matches)
    df = add_correctness_columns(df, thresholds_m, require_gt_candidate)
    df = add_route_length_weights(df)

    criteria = ["edge_correct"]
    for threshold in thresholds_m:
        tok = fmt_threshold(threshold)
        criteria += [f"mm2_geometry_correct_{tok}m", f"mm2_combined_correct_{tok}m"]

    summaries = {criterion: summarize_criterion(df, criterion, trajectory_success_threshold) for criterion in criteria}
    traj = make_trajectory_table(df, criteria, trajectory_success_threshold)

    metrics = {
        "workflow": workflow,
        "metric_family": "map_matching_2_style",
        "note": "Map Matching 2-style mean correct fraction and weighted mean correct fraction from available matched point outputs. Geometry variants are route-length-weighted approximations, not exact MM2 route-overlap.",
        "num_points_loaded": int(len(matches)),
        "num_points_evaluated": int(len(df)),
        "num_trajectories": int(df["trajectory_id"].nunique()) if len(df) else 0,
        "require_gt_candidate": bool(require_gt_candidate),
        "trajectory_success_threshold": float(trajectory_success_threshold),
        "thresholds_m": thresholds_m,
        "has_projection_columns": {"pred_proj_x", "pred_proj_y", "gt_proj_x", "gt_proj_y"}.issubset(df.columns),
        "criteria": summaries,
    }

    if "projection_error_m" in df.columns:
        err = df["projection_error_m"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(err):
            metrics["projection_error"] = {
                "mean_m": float(err.mean()),
                "median_m": float(err.median()),
                "p90_m": float(err.quantile(0.90)),
                "p95_m": float(err.quantile(0.95)),
            }

    if "confidence" in df.columns:
        conf = df["confidence"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(conf):
            metrics["confidence"] = {"mean": float(conf.mean()), "median": float(conf.median())}

    return metrics, traj, df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map Matching 2-style evaluator for One-Direction outputs.")
    p.add_argument("--workflow", required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory-output", type=Path, default=None)
    p.add_argument("--annotated-output", type=Path, default=None)
    p.add_argument("--thresholds-m", nargs="+", default=["2", "5", "10"])
    p.add_argument("--require-gt-candidate", action="store_true")
    p.add_argument("--include-gt-missing", action="store_true")
    p.add_argument("--trajectory-success-threshold", type=float, default=0.90)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    thresholds_m = sorted({float(x) for x in args.thresholds_m})
    require_gt_candidate = bool(args.require_gt_candidate and not args.include_gt_missing)

    matches = load_matches(args.matches)
    metrics, traj, annotated = evaluate_mm2_style(matches, args.workflow, thresholds_m, require_gt_candidate, args.trajectory_success_threshold)

    ensure_dir(args.output.parent)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")

    if args.trajectory_output is not None:
        ensure_dir(args.trajectory_output.parent)
        traj.to_csv(args.trajectory_output, index=False)

    if args.annotated_output is not None:
        ensure_dir(args.annotated_output.parent)
        annotated.to_parquet(args.annotated_output, index=False)

    print(f"[OK] Wrote MM2-style metrics: {args.output}", flush=True)
    print(f"workflow: {args.workflow}", flush=True)
    print(f"edge_correct weighted_mean_correct_fraction: {metrics['criteria']['edge_correct'].get('weighted_mean_correct_fraction')}", flush=True)
    for threshold in thresholds_m:
        key = f"mm2_geometry_correct_{fmt_threshold(threshold)}m"
        print(f"{key} weighted_mean_correct_fraction: {metrics['criteria'][key].get('weighted_mean_correct_fraction')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
