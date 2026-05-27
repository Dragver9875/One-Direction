from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def edit_distance(a: Sequence, b: Sequence) -> int:
    n = len(a)
    m = len(b)

    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return int(prev[m])


def compress_consecutive(sequence: Sequence) -> list:
    out = []
    previous = object()

    for item in sequence:
        if item != previous:
            out.append(item)
        previous = item

    return out


def path_edit_distance(
    pred_edges: Sequence,
    gt_edges: Sequence,
    compress: bool = True,
) -> int:
    pred_seq = compress_consecutive(pred_edges) if compress else list(pred_edges)
    gt_seq = compress_consecutive(gt_edges) if compress else list(gt_edges)
    return edit_distance(pred_seq, gt_seq)


def normalized_path_edit_distance(
    pred_edges: Sequence,
    gt_edges: Sequence,
    compress: bool = True,
) -> float:
    pred_seq = compress_consecutive(pred_edges) if compress else list(pred_edges)
    gt_seq = compress_consecutive(gt_edges) if compress else list(gt_edges)
    denom = max(len(pred_seq), len(gt_seq), 1)
    return float(edit_distance(pred_seq, gt_seq) / denom)


def route_continuity_breaks(
    edge_sequence: Sequence,
    transition_lookup: set[tuple] | None = None,
) -> int:
    if len(edge_sequence) < 2:
        return 0

    if transition_lookup is None:
        return 0

    breaks = 0
    for prev_edge, curr_edge in zip(edge_sequence[:-1], edge_sequence[1:]):
        if prev_edge == curr_edge:
            continue
        if (prev_edge, curr_edge) not in transition_lookup:
            breaks += 1

    return int(breaks)


def trajectory_success_rate(
    trajectory_metrics: pd.DataFrame,
    accuracy_col: str = "edge_accuracy",
    threshold: float = 0.9,
) -> float:
    if trajectory_metrics.empty or accuracy_col not in trajectory_metrics.columns:
        return 0.0

    return float((trajectory_metrics[accuracy_col].astype(float) >= threshold).mean())


def per_trajectory_sequence_metrics(
    matches: pd.DataFrame,
    trajectory_col: str = "trajectory_id",
    pred_col: str = "pred_edge_id",
    gt_col: str = "gt_edge_id",
    compress: bool = True,
) -> pd.DataFrame:
    required = {trajectory_col, pred_col, gt_col}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Missing sequence metric columns: {sorted(missing)}")

    records = []

    for trajectory_id, group in matches.groupby(trajectory_col, sort=True):
        group = group.sort_values("t") if "t" in group.columns else group

        pred = group[pred_col].astype(str).tolist()
        gt = group[gt_col].astype(str).tolist()

        dist = path_edit_distance(pred, gt, compress=compress)
        norm = normalized_path_edit_distance(pred, gt, compress=compress)
        edge_accuracy = float(np.mean(np.asarray(pred, dtype=object) == np.asarray(gt, dtype=object)))

        records.append(
            {
                trajectory_col: trajectory_id,
                "path_edit_distance": int(dist),
                "normalized_path_edit_distance": float(norm),
                "edge_accuracy": edge_accuracy,
                "point_count": int(len(group)),
                "pred_unique_edges": int(len(set(pred))),
                "gt_unique_edges": int(len(set(gt))),
            }
        )

    return pd.DataFrame.from_records(records)
