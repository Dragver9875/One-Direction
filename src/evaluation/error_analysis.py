from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .projection_metrics import compute_projection_errors


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    trajectory_col: str = "trajectory_id"
    timestep_col: str = "t"
    pred_edge_col: str = "pred_edge_id"
    gt_edge_col: str = "gt_edge_id"
    pred_x_col: str = "pred_proj_x"
    pred_y_col: str = "pred_proj_y"
    gt_x_col: str = "gt_proj_x"
    gt_y_col: str = "gt_proj_y"
    confidence_col: str = "confidence"
    projection_error_threshold_m: float = 15.0
    low_confidence_threshold: float = 0.5


def build_error_cases(
    matches: pd.DataFrame,
    cfg: ErrorAnalysisConfig = ErrorAnalysisConfig(),
) -> pd.DataFrame:
    df = matches.copy()

    if {cfg.pred_x_col, cfg.pred_y_col, cfg.gt_x_col, cfg.gt_y_col}.issubset(df.columns):
        df = compute_projection_errors(
            df,
            pred_x_col=cfg.pred_x_col,
            pred_y_col=cfg.pred_y_col,
            gt_x_col=cfg.gt_x_col,
            gt_y_col=cfg.gt_y_col,
        )
    else:
        df["projection_error_m"] = np.nan

    if cfg.pred_edge_col in df.columns and cfg.gt_edge_col in df.columns:
        df["edge_mismatch"] = df[cfg.pred_edge_col].astype(str) != df[cfg.gt_edge_col].astype(str)
    else:
        df["edge_mismatch"] = False

    if cfg.confidence_col in df.columns:
        df["low_confidence"] = pd.to_numeric(df[cfg.confidence_col], errors="coerce") < cfg.low_confidence_threshold
    else:
        df["low_confidence"] = False

    df["large_projection_error"] = df["projection_error_m"] > cfg.projection_error_threshold_m

    df["is_error_case"] = (
        df["edge_mismatch"].fillna(False)
        | df["large_projection_error"].fillna(False)
        | df["low_confidence"].fillna(False)
    )

    return df[df["is_error_case"]].copy().reset_index(drop=True)


def summarize_error_cases(
    error_cases: pd.DataFrame,
    cfg: ErrorAnalysisConfig = ErrorAnalysisConfig(),
) -> dict[str, float | int]:
    if error_cases.empty:
        return {
            "num_error_cases": 0,
            "num_error_trajectories": 0,
            "edge_mismatch_count": 0,
            "large_projection_error_count": 0,
            "low_confidence_count": 0,
        }

    return {
        "num_error_cases": int(len(error_cases)),
        "num_error_trajectories": int(error_cases[cfg.trajectory_col].nunique()) if cfg.trajectory_col in error_cases.columns else 0,
        "edge_mismatch_count": int(error_cases.get("edge_mismatch", pd.Series(dtype=bool)).sum()),
        "large_projection_error_count": int(error_cases.get("large_projection_error", pd.Series(dtype=bool)).sum()),
        "low_confidence_count": int(error_cases.get("low_confidence", pd.Series(dtype=bool)).sum()),
        "projection_error_mean_m": float(pd.to_numeric(error_cases.get("projection_error_m", pd.Series(dtype=float)), errors="coerce").mean()),
        "projection_error_max_m": float(pd.to_numeric(error_cases.get("projection_error_m", pd.Series(dtype=float)), errors="coerce").max()),
    }


def top_error_trajectories(
    error_cases: pd.DataFrame,
    trajectory_col: str = "trajectory_id",
    top_n: int = 20,
) -> pd.DataFrame:
    if error_cases.empty or trajectory_col not in error_cases.columns:
        return pd.DataFrame(columns=[trajectory_col, "error_count"])

    return (
        error_cases.groupby(trajectory_col)
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
