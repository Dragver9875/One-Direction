from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .projection_metrics import compute_projection_errors, projection_error_summary, projection_success_rate
from .sequence_metrics import per_trajectory_sequence_metrics, trajectory_success_rate


@dataclass(frozen=True)
class EvaluationConfig:
    trajectory_col: str = "trajectory_id"
    timestep_col: str = "t"
    pred_edge_col: str = "pred_edge_id"
    gt_edge_col: str = "gt_edge_id"
    pred_x_col: str = "pred_proj_x"
    pred_y_col: str = "pred_proj_y"
    gt_x_col: str = "gt_proj_x"
    gt_y_col: str = "gt_proj_y"
    confidence_col: str = "confidence"
    projection_success_threshold_m: float = 10.0
    trajectory_success_edge_accuracy: float = 0.9
    low_confidence_threshold: float = 0.5


def merge_predictions_with_gt(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> pd.DataFrame:
    merge_cols = [cfg.trajectory_col]
    if cfg.timestep_col in pred.columns and cfg.timestep_col in gt.columns:
        merge_cols.append(cfg.timestep_col)
    elif "timestamp" in pred.columns and "timestamp" in gt.columns:
        merge_cols.append("timestamp")
    else:
        raise ValueError("Cannot merge predictions and GT. Need trajectory_id+t or trajectory_id+timestamp.")

    return pred.merge(gt, on=merge_cols, how="inner", suffixes=("", "_gt"))


def point_edge_accuracy(
    matches: pd.DataFrame,
    pred_col: str = "pred_edge_id",
    gt_col: str = "gt_edge_id",
) -> float:
    if pred_col not in matches.columns or gt_col not in matches.columns:
        return float("nan")

    if len(matches) == 0:
        return 0.0

    return float((matches[pred_col].astype(str) == matches[gt_col].astype(str)).mean())


def confidence_summary(
    matches: pd.DataFrame,
    confidence_col: str = "confidence",
    low_threshold: float = 0.5,
) -> dict[str, float]:
    if confidence_col not in matches.columns:
        return {}

    conf = pd.to_numeric(matches[confidence_col], errors="coerce").dropna()

    if conf.empty:
        return {
            "confidence_mean": float("nan"),
            "confidence_median": float("nan"),
            "low_confidence_rate": float("nan"),
        }

    return {
        "confidence_mean": float(conf.mean()),
        "confidence_median": float(conf.median()),
        "confidence_p10": float(conf.quantile(0.10)),
        "confidence_p90": float(conf.quantile(0.90)),
        "low_confidence_rate": float((conf < low_threshold).mean()),
    }


def evaluate_predictions(
    pred: pd.DataFrame,
    gt: pd.DataFrame | None = None,
    cfg: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    if gt is not None:
        matches = merge_predictions_with_gt(pred, gt, cfg)
    else:
        matches = pred.copy()

    metrics: dict[str, Any] = {
        "config": asdict(cfg),
        "num_predictions": int(len(pred)),
        "num_evaluated_points": int(len(matches)),
        "num_trajectories": int(matches[cfg.trajectory_col].nunique()) if cfg.trajectory_col in matches.columns else 0,
    }

    if cfg.pred_edge_col in matches.columns and cfg.gt_edge_col in matches.columns:
        metrics["point_edge_accuracy"] = point_edge_accuracy(
            matches,
            pred_col=cfg.pred_edge_col,
            gt_col=cfg.gt_edge_col,
        )

        traj_metrics = per_trajectory_sequence_metrics(
            matches,
            trajectory_col=cfg.trajectory_col,
            pred_col=cfg.pred_edge_col,
            gt_col=cfg.gt_edge_col,
            compress=True,
        )

        if not traj_metrics.empty:
            metrics["path_edit_distance_mean"] = float(traj_metrics["path_edit_distance"].mean())
            metrics["path_edit_distance_median"] = float(traj_metrics["path_edit_distance"].median())
            metrics["normalized_path_edit_distance_mean"] = float(traj_metrics["normalized_path_edit_distance"].mean())
            metrics["trajectory_edge_accuracy_mean"] = float(traj_metrics["edge_accuracy"].mean())
            metrics["trajectory_success_rate"] = trajectory_success_rate(
                traj_metrics,
                accuracy_col="edge_accuracy",
                threshold=cfg.trajectory_success_edge_accuracy,
            )

    projection_cols = {cfg.pred_x_col, cfg.pred_y_col, cfg.gt_x_col, cfg.gt_y_col}
    if projection_cols.issubset(matches.columns):
        matches_with_errors = compute_projection_errors(
            matches,
            pred_x_col=cfg.pred_x_col,
            pred_y_col=cfg.pred_y_col,
            gt_x_col=cfg.gt_x_col,
            gt_y_col=cfg.gt_y_col,
        )
        metrics.update(projection_error_summary(matches_with_errors["projection_error_m"]))
        metrics["projection_success_rate"] = projection_success_rate(
            matches_with_errors["projection_error_m"],
            threshold_m=cfg.projection_success_threshold_m,
        )

    metrics.update(
        confidence_summary(
            matches,
            confidence_col=cfg.confidence_col,
            low_threshold=cfg.low_confidence_threshold,
        )
    )

    return metrics


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=convert)
