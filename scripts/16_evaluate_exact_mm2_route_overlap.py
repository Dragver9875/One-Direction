#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union


DEFAULT_WORKFLOWS = {
    "hmm": {
        "matches": "HMM/outputs/matches/hmm_matches_test.parquet",
        "metrics": "HMM/outputs/metrics/hmm_exact_mm2_route_overlap_test.json",
        "trajectory": "HMM/outputs/metrics/hmm_exact_mm2_route_overlap_trajectory_test.csv",
    },
    "gnn_hmm": {
        "matches": "outputs/matches/gnn_hmm_matches.parquet",
        "metrics": "outputs/metrics/gnn_hmm_exact_mm2_route_overlap_test.json",
        "trajectory": "outputs/metrics/gnn_hmm_exact_mm2_route_overlap_trajectory_test.csv",
    },
    "dsac": {
        "matches": "D-SAC/outputs/matches/dsac_asym_matches.parquet",
        "metrics": "D-SAC/outputs/metrics/dsac_exact_mm2_route_overlap_test.json",
        "trajectory": "D-SAC/outputs/metrics/dsac_exact_mm2_route_overlap_trajectory_test.csv",
    },
    "ppo": {
        "matches": "RL/outputs/matches/ppo_asym_matches.parquet",
        "metrics": "RL/outputs/metrics/ppo_exact_mm2_route_overlap_test.json",
        "trajectory": "RL/outputs/metrics/ppo_exact_mm2_route_overlap_trajectory_test.csv",
    },
}


EDGE_ID_ALIASES = ["edge_idx", "edge_id", "id", "fid", "index"]
GEOMETRY_ALIASES = ["geometry_wkt", "geometry", "geom_wkt", "wkt"]
LENGTH_ALIASES = ["length_m", "length", "edge_length_m", "geometry_length_m"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_tol(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")

    raise ValueError(f"Unsupported file type: {path}")


def first_present(columns: list[str], aliases: list[str]) -> str | None:
    for name in aliases:
        if name in columns:
            return name
    return None


def parse_wkt_geometry(value: Any):
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        geom = wkt.loads(text)
    except Exception:
        return None

    if geom.is_empty or geom.length <= 0:
        return None

    if geom.geom_type not in {"LineString", "MultiLineString"}:
        return None

    return geom


def load_edge_geometries(edge_table_path: Path | None) -> dict[int, Any]:
    if edge_table_path is None:
        return {}

    edges = load_table(edge_table_path)
    edge_col = first_present(list(edges.columns), EDGE_ID_ALIASES)
    geom_col = first_present(list(edges.columns), GEOMETRY_ALIASES)

    if edge_col is None:
        raise ValueError(
            f"Could not find edge ID column in {edge_table_path}. "
            f"Tried: {EDGE_ID_ALIASES}"
        )

    if geom_col is None:
        raise ValueError(
            f"Could not find geometry WKT column in {edge_table_path}. "
            f"Tried: {GEOMETRY_ALIASES}"
        )

    edge_geom: dict[int, Any] = {}

    for _, row in edges.iterrows():
        edge_id = int(row[edge_col])
        geom = parse_wkt_geometry(row[geom_col])
        if geom is not None:
            edge_geom[edge_id] = geom

    if not edge_geom:
        raise RuntimeError(f"No usable edge geometries loaded from {edge_table_path}")

    return edge_geom


def canonicalize_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    aliases = {
        "trajectory_id": ["trajectory_id", "traj_id", "track_id", "id"],
        "t": ["t", "step", "point_idx", "idx", "time_idx"],

        "pred_edge_idx": ["pred_edge_idx", "pred_edge_id", "matched_edge_idx", "matched_edge_id"],
        "gt_edge_idx": ["gt_edge_idx", "gt_edge_id", "true_edge_idx", "true_edge_id"],

        "gt_candidate_pos": ["gt_candidate_pos", "gt_action", "target", "gt_candidate"],

        "pred_route_wkt": ["pred_route_wkt", "pred_geometry_wkt", "matched_route_wkt"],
        "gt_route_wkt": ["gt_route_wkt", "gt_geometry_wkt", "true_route_wkt"],

        "pred_proj_x": ["pred_proj_x", "matched_x", "proj_x", "x_pred", "pred_x"],
        "pred_proj_y": ["pred_proj_y", "matched_y", "proj_y", "y_pred", "pred_y"],
        "gt_proj_x": ["gt_proj_x", "true_proj_x", "gt_x", "x_gt"],
        "gt_proj_y": ["gt_proj_y", "true_proj_y", "gt_y", "y_gt"],
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


def filter_evaluable(df: pd.DataFrame, require_gt_candidate: bool) -> pd.DataFrame:
    out = df.copy()

    if "gt_edge_idx" in out.columns:
        out = out[out["gt_edge_idx"].notna()].copy()
        out = out[out["gt_edge_idx"].astype(int) >= 0].copy()

    if require_gt_candidate and "gt_candidate_pos" in out.columns:
        out = out[out["gt_candidate_pos"].notna()].copy()
        out = out[out["gt_candidate_pos"].astype(int) >= 0].copy()

    return out


def unique_consecutive(values: list[int]) -> list[int]:
    out = []

    for value in values:
        if value < 0:
            continue
        if not out or out[-1] != value:
            out.append(value)

    return out


def route_from_edge_sequence(edge_ids: list[int], edge_geom: dict[int, Any]):
    geoms = []

    for edge_id in unique_consecutive(edge_ids):
        geom = edge_geom.get(int(edge_id))
        if geom is not None and not geom.is_empty and geom.length > 0:
            geoms.append(geom)

    if not geoms:
        return None

    merged = unary_union(geoms)

    try:
        merged = linemerge(merged)
    except Exception:
        pass

    if merged.is_empty or merged.length <= 0:
        return None

    if merged.geom_type not in {"LineString", "MultiLineString"}:
        return None

    return merged


def route_from_wkt_series(values: pd.Series):
    geoms = []

    for value in values.dropna().astype(str).unique():
        geom = parse_wkt_geometry(value)
        if geom is not None:
            geoms.append(geom)

    if not geoms:
        return None

    merged = unary_union(geoms)

    try:
        merged = linemerge(merged)
    except Exception:
        pass

    if merged.is_empty or merged.length <= 0:
        return None

    if merged.geom_type not in {"LineString", "MultiLineString"}:
        return None

    return merged


def clean_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []

    for x, y in points:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        p = (float(x), float(y))
        if not out or out[-1] != p:
            out.append(p)

    return out


def route_from_points(group: pd.DataFrame, x_col: str, y_col: str):
    points = clean_points(list(zip(group[x_col].astype(float), group[y_col].astype(float))))

    if len(points) < 2:
        return None

    line = LineString(points)

    if line.is_empty or line.length <= 0:
        return None

    return line


def build_route_geometry(
    group: pd.DataFrame,
    kind: str,
    edge_geom: dict[int, Any],
    allow_point_fallback: bool,
):
    if kind == "pred":
        wkt_col = "pred_route_wkt"
        edge_col = "pred_edge_idx"
        x_col = "pred_proj_x"
        y_col = "pred_proj_y"
    elif kind == "gt":
        wkt_col = "gt_route_wkt"
        edge_col = "gt_edge_idx"
        x_col = "gt_proj_x"
        y_col = "gt_proj_y"
    else:
        raise ValueError(kind)

    if wkt_col in group.columns:
        route = route_from_wkt_series(group[wkt_col])
        if route is not None:
            return route, "wkt"

    if edge_geom and edge_col in group.columns:
        edge_ids = group[edge_col].dropna().astype(int).tolist()
        route = route_from_edge_sequence(edge_ids, edge_geom)
        if route is not None:
            return route, "edge_geometry"

    if allow_point_fallback and {x_col, y_col}.issubset(group.columns):
        route = route_from_points(group, x_col, y_col)
        if route is not None:
            return route, "projected_points"

    return None, "missing"


def geom_length(geom) -> float:
    if geom is None or geom.is_empty:
        return 0.0

    value = float(geom.length)

    if not np.isfinite(value):
        return 0.0

    return max(value, 0.0)


def overlap_length(line, other, tolerance_m: float) -> float:
    if line is None or other is None:
        return 0.0

    if line.is_empty or other.is_empty:
        return 0.0

    if tolerance_m <= 0:
        inter = line.intersection(other)
    else:
        inter = line.intersection(other.buffer(tolerance_m))

    return min(geom_length(inter), geom_length(line))


def exact_overlap_metrics(pred, gt, tolerance_m: float) -> dict[str, float]:
    pred_len = geom_length(pred)
    gt_len = geom_length(gt)

    pred_correct = overlap_length(pred, gt, tolerance_m) if pred_len > 0 and gt_len > 0 else 0.0
    gt_correct = overlap_length(gt, pred, tolerance_m) if pred_len > 0 and gt_len > 0 else 0.0

    pred_extra = max(pred_len - pred_correct, 0.0)
    gt_missed = max(gt_len - gt_correct, 0.0)

    precision = pred_correct / pred_len if pred_len > 0 else 0.0
    recall = gt_correct / gt_len if gt_len > 0 else 0.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)

    denom_union = pred_len + gt_len - 0.5 * (pred_correct + gt_correct)
    iou_like = 0.5 * (pred_correct + gt_correct) / denom_union if denom_union > 0 else 0.0

    return {
        "pred_len_m": float(pred_len),
        "gt_len_m": float(gt_len),
        "pred_correct_len_m": float(pred_correct),
        "gt_correct_len_m": float(gt_correct),
        "pred_extra_len_m": float(pred_extra),
        "gt_missed_len_m": float(gt_missed),

        "precision_overlap": float(precision),
        "correct_fraction": float(recall),
        "recall_overlap": float(recall),
        "symmetric_overlap_f1": float(f1),
        "iou_like_overlap": float(iou_like),

        "extra_fraction": float(pred_extra / pred_len) if pred_len > 0 else 1.0,
        "missed_fraction": float(gt_missed / gt_len) if gt_len > 0 else 1.0,
    }


def evaluate_workflow(
    matches_path: Path,
    workflow: str,
    edge_table_path: Path | None,
    tolerance_m: float,
    require_gt_candidate: bool,
    allow_point_fallback: bool,
    trajectory_success_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    matches = canonicalize_matches(load_table(matches_path))
    matches = filter_evaluable(matches, require_gt_candidate=require_gt_candidate)
    edge_geom = load_edge_geometries(edge_table_path)

    rows = []

    for tid, group in matches.sort_values(["trajectory_id", "t"]).groupby("trajectory_id", sort=False):
        pred_route, pred_source = build_route_geometry(group, "pred", edge_geom, allow_point_fallback)
        gt_route, gt_source = build_route_geometry(group, "gt", edge_geom, allow_point_fallback)

        has_geometry = pred_route is not None and gt_route is not None

        if has_geometry:
            m = exact_overlap_metrics(pred_route, gt_route, tolerance_m)
        else:
            m = {
                "pred_len_m": 0.0,
                "gt_len_m": 0.0,
                "pred_correct_len_m": 0.0,
                "gt_correct_len_m": 0.0,
                "pred_extra_len_m": 0.0,
                "gt_missed_len_m": 0.0,
                "precision_overlap": 0.0,
                "correct_fraction": 0.0,
                "recall_overlap": 0.0,
                "symmetric_overlap_f1": 0.0,
                "iou_like_overlap": 0.0,
                "extra_fraction": 1.0,
                "missed_fraction": 1.0,
            }

        rows.append(
            {
                "workflow": workflow,
                "trajectory_id": tid,
                "num_points": int(len(group)),
                "has_geometry": bool(has_geometry),
                "pred_geometry_source": pred_source,
                "gt_geometry_source": gt_source,
                "success": bool(m["correct_fraction"] >= trajectory_success_threshold),
                **m,
            }
        )

    traj = pd.DataFrame(rows)

    valid = traj[traj["has_geometry"]].copy()

    if len(valid) == 0:
        metrics = {
            "workflow": workflow,
            "metric_family": "exact_mm2_route_geometry_overlap",
            "available": False,
            "reason": "No route geometry could be reconstructed.",
            "num_input_rows": int(len(matches)),
            "num_trajectories": int(matches["trajectory_id"].nunique()) if len(matches) else 0,
        }
        return metrics, traj

    pred_len_total = float(valid["pred_len_m"].sum())
    gt_len_total = float(valid["gt_len_m"].sum())
    pred_correct_total = float(valid["pred_correct_len_m"].sum())
    gt_correct_total = float(valid["gt_correct_len_m"].sum())
    pred_extra_total = float(valid["pred_extra_len_m"].sum())
    gt_missed_total = float(valid["gt_missed_len_m"].sum())

    weighted_precision = pred_correct_total / pred_len_total if pred_len_total > 0 else 0.0
    weighted_correct_fraction = gt_correct_total / gt_len_total if gt_len_total > 0 else 0.0
    weighted_f1 = (
        2.0 * weighted_precision * weighted_correct_fraction
        / max(weighted_precision + weighted_correct_fraction, 1.0e-12)
    )

    denom_union = pred_len_total + gt_len_total - 0.5 * (pred_correct_total + gt_correct_total)
    weighted_iou_like = (
        0.5 * (pred_correct_total + gt_correct_total) / denom_union
        if denom_union > 0
        else 0.0
    )

    metrics = {
        "workflow": workflow,
        "metric_family": "exact_mm2_route_geometry_overlap",
        "available": True,
        "note": (
            "This computes exact route-geometry overlap for the geometries available to the project. "
            "It uses pred_route_wkt/gt_route_wkt when present, otherwise edge geometries from --edge-table, "
            "otherwise projected-point polylines if --allow-point-fallback is set. "
            "weighted_mean_correct_fraction is GT route length recovered by the predicted route, which is the closest "
            "project-level analogue to Map Matching 2 weighted mean correct fraction."
        ),
        "matches_path": str(matches_path),
        "edge_table_path": str(edge_table_path) if edge_table_path else None,
        "tolerance_m": float(tolerance_m),
        "require_gt_candidate": bool(require_gt_candidate),
        "allow_point_fallback": bool(allow_point_fallback),
        "trajectory_success_threshold": float(trajectory_success_threshold),

        "num_input_rows": int(len(matches)),
        "num_trajectories": int(matches["trajectory_id"].nunique()) if len(matches) else 0,
        "num_valid_trajectories": int(len(valid)),

        "mean_correct_fraction": float(valid["correct_fraction"].mean()),
        "weighted_mean_correct_fraction": float(weighted_correct_fraction),
        "mean_precision_overlap": float(valid["precision_overlap"].mean()),
        "weighted_precision_overlap": float(weighted_precision),
        "mean_symmetric_overlap_f1": float(valid["symmetric_overlap_f1"].mean()),
        "weighted_symmetric_overlap_f1": float(weighted_f1),
        "mean_iou_like_overlap": float(valid["iou_like_overlap"].mean()),
        "weighted_iou_like_overlap": float(weighted_iou_like),

        "mean_extra_fraction": float(valid["extra_fraction"].mean()),
        "weighted_extra_fraction": float(pred_extra_total / pred_len_total) if pred_len_total > 0 else 1.0,
        "mean_missed_fraction": float(valid["missed_fraction"].mean()),
        "weighted_missed_fraction": float(gt_missed_total / gt_len_total) if gt_len_total > 0 else 1.0,

        "trajectory_success_rate": float(valid["success"].mean()),

        "total_pred_len_m": pred_len_total,
        "total_gt_len_m": gt_len_total,
        "correct_pred_len_m": pred_correct_total,
        "correct_gt_len_m": gt_correct_total,
        "extra_pred_len_m": pred_extra_total,
        "missed_gt_len_m": gt_missed_total,

        "geometry_sources": {
            "pred": valid["pred_geometry_source"].value_counts().to_dict(),
            "gt": valid["gt_geometry_source"].value_counts().to_dict(),
        },
    }

    return metrics, traj


def write_outputs(metrics: dict[str, Any], traj: pd.DataFrame, metrics_path: Path, trajectory_path: Path) -> None:
    ensure_dir(metrics_path.parent)
    ensure_dir(trajectory_path.parent)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8", newline="\n")
    traj.to_csv(trajectory_path, index=False)


def compare_summary(all_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for workflow, metrics in all_metrics.items():
        rows.append(
            {
                "workflow": workflow,
                "available": metrics.get("available"),
                "num_valid_trajectories": metrics.get("num_valid_trajectories"),
                "weighted_mean_correct_fraction": metrics.get("weighted_mean_correct_fraction"),
                "mean_correct_fraction": metrics.get("mean_correct_fraction"),
                "weighted_precision_overlap": metrics.get("weighted_precision_overlap"),
                "weighted_symmetric_overlap_f1": metrics.get("weighted_symmetric_overlap_f1"),
                "weighted_iou_like_overlap": metrics.get("weighted_iou_like_overlap"),
                "weighted_extra_fraction": metrics.get("weighted_extra_fraction"),
                "weighted_missed_fraction": metrics.get("weighted_missed_fraction"),
                "trajectory_success_rate": metrics.get("trajectory_success_rate"),
                "total_gt_len_m": metrics.get("total_gt_len_m"),
                "correct_gt_len_m": metrics.get("correct_gt_len_m"),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact route-geometry overlap metric comparison for One-Direction.")

    parser.add_argument(
        "mode",
        choices=["one", "compare"],
        help="Evaluate one match file, or compare default HMM/GNN-HMM/D-SAC outputs.",
    )

    parser.add_argument("--workflow", default=None)
    parser.add_argument("--matches", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trajectory-output", type=Path, default=None)

    parser.add_argument("--workflows", nargs="+", default=["hmm", "gnn_hmm", "dsac"], choices=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--edge-table", type=Path, default=None)
    parser.add_argument("--tolerance-m", type=float, default=0.0)
    parser.add_argument("--require-gt-candidate", action="store_true", default=True)
    parser.add_argument("--include-gt-missing", action="store_true")
    parser.add_argument("--allow-point-fallback", action="store_true")

    parser.add_argument("--trajectory-success-threshold", type=float, default=0.90)
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/metrics/exact_mm2_route_overlap_comparison.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("outputs/metrics/exact_mm2_route_overlap_comparison.json"))

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    require_gt_candidate = bool(args.require_gt_candidate and not args.include_gt_missing)

    if args.mode == "one":
        if args.workflow is None or args.matches is None:
            raise ValueError("mode=one requires --workflow and --matches.")

        output = args.output or Path(f"outputs/metrics/{args.workflow}_exact_mm2_route_overlap.json")
        trajectory_output = args.trajectory_output or Path(f"outputs/metrics/{args.workflow}_exact_mm2_route_overlap_trajectory.csv")

        metrics, traj = evaluate_workflow(
            matches_path=args.matches,
            workflow=args.workflow,
            edge_table_path=args.edge_table,
            tolerance_m=args.tolerance_m,
            require_gt_candidate=require_gt_candidate,
            allow_point_fallback=args.allow_point_fallback,
            trajectory_success_threshold=args.trajectory_success_threshold,
        )

        write_outputs(metrics, traj, output, trajectory_output)

        print(f"[OK] Wrote exact route-overlap metrics: {output}", flush=True)
        print(f"weighted_mean_correct_fraction: {metrics.get('weighted_mean_correct_fraction')}", flush=True)

        return 0

    all_metrics = {}

    for workflow in args.workflows:
        spec = DEFAULT_WORKFLOWS[workflow]
        matches_path = Path(spec["matches"])

        if not matches_path.exists():
            print(f"[SKIP] {workflow}: missing {matches_path}", flush=True)
            continue

        metrics, traj = evaluate_workflow(
            matches_path=matches_path,
            workflow=workflow,
            edge_table_path=args.edge_table,
            tolerance_m=args.tolerance_m,
            require_gt_candidate=require_gt_candidate,
            allow_point_fallback=args.allow_point_fallback,
            trajectory_success_threshold=args.trajectory_success_threshold,
        )

        write_outputs(metrics, traj, Path(spec["metrics"]), Path(spec["trajectory"]))
        all_metrics[workflow] = metrics

        print(
            f"[OK] {workflow}: weighted_mean_correct_fraction="
            f"{metrics.get('weighted_mean_correct_fraction')}",
            flush=True,
        )

    summary = compare_summary(all_metrics)

    ensure_dir(args.summary_output.parent)
    ensure_dir(args.json_output.parent)

    summary.to_csv(args.summary_output, index=False)
    args.json_output.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8", newline="\n")

    print(f"[OK] Wrote comparison CSV: {args.summary_output}", flush=True)
    print(f"[OK] Wrote comparison JSON: {args.json_output}", flush=True)

    if len(summary):
        print(summary.to_string(index=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
