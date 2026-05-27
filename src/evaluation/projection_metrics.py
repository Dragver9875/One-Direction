from __future__ import annotations

import numpy as np
import pandas as pd


def compute_projection_errors(
    df: pd.DataFrame,
    pred_x_col: str = "pred_proj_x",
    pred_y_col: str = "pred_proj_y",
    gt_x_col: str = "gt_proj_x",
    gt_y_col: str = "gt_proj_y",
    output_col: str = "projection_error_m",
) -> pd.DataFrame:
    required = {pred_x_col, pred_y_col, gt_x_col, gt_y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing projection columns: {sorted(missing)}")

    out = df.copy()
    dx = out[pred_x_col].astype(float) - out[gt_x_col].astype(float)
    dy = out[pred_y_col].astype(float) - out[gt_y_col].astype(float)
    out[output_col] = np.sqrt(dx * dx + dy * dy)
    return out


def projection_error_summary(
    errors: pd.Series | np.ndarray,
    prefix: str = "projection_error",
) -> dict[str, float]:
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            f"{prefix}_mean_m": float("nan"),
            f"{prefix}_median_m": float("nan"),
            f"{prefix}_p90_m": float("nan"),
            f"{prefix}_p95_m": float("nan"),
            f"{prefix}_p99_m": float("nan"),
            f"{prefix}_max_m": float("nan"),
        }

    return {
        f"{prefix}_mean_m": float(np.mean(arr)),
        f"{prefix}_median_m": float(np.median(arr)),
        f"{prefix}_p90_m": float(np.percentile(arr, 90)),
        f"{prefix}_p95_m": float(np.percentile(arr, 95)),
        f"{prefix}_p99_m": float(np.percentile(arr, 99)),
        f"{prefix}_max_m": float(np.max(arr)),
    }


def projection_success_rate(
    errors: pd.Series | np.ndarray,
    threshold_m: float = 10.0,
) -> float:
    arr = np.asarray(errors, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return 0.0

    return float(np.mean(arr <= threshold_m))


def grouped_projection_summary(
    df: pd.DataFrame,
    group_col: str = "trajectory_id",
    error_col: str = "projection_error_m",
) -> pd.DataFrame:
    if group_col not in df.columns or error_col not in df.columns:
        raise ValueError(f"Expected columns {group_col!r} and {error_col!r}.")

    grouped = df.groupby(group_col)[error_col]
    return grouped.agg(
        projection_error_mean_m="mean",
        projection_error_median_m="median",
        projection_error_max_m="max",
        point_count="size",
    ).reset_index()
