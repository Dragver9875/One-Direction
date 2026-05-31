from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def evaluate_matches(
    matches: pd.DataFrame,
    projection_threshold_m: float = 10.0,
    trajectory_success_accuracy: float = 0.90,
    require_gt_candidate: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = matches.copy()

    labelled = df[df['gt_edge_idx'] >= 0].copy()
    if require_gt_candidate and 'gt_candidate_pos' in labelled.columns:
        labelled = labelled[labelled['gt_candidate_pos'] >= 0].copy()

    if len(labelled) == 0:
        raise RuntimeError('No labelled points found.')

    labelled['edge_correct'] = labelled['pred_edge_idx'].astype(int) == labelled['gt_edge_idx'].astype(int)

    if 'gt_candidate_pos' in labelled.columns:
        labelled['action_correct'] = labelled['pred_candidate_pos'].astype(int) == labelled['gt_candidate_pos'].astype(int)
    else:
        labelled['action_correct'] = labelled['edge_correct']

    if {'pred_proj_x', 'pred_proj_y', 'gt_proj_x', 'gt_proj_y'}.issubset(labelled.columns):
        labelled['projection_error_m'] = np.sqrt(
            (labelled['pred_proj_x'] - labelled['gt_proj_x']) ** 2
            + (labelled['pred_proj_y'] - labelled['gt_proj_y']) ** 2
        )
    else:
        labelled['projection_error_m'] = np.nan

    labelled['projection_success'] = labelled['projection_error_m'] <= projection_threshold_m
    labelled['within_2m'] = labelled['projection_error_m'] <= 2.0
    labelled['within_5m'] = labelled['projection_error_m'] <= 5.0
    labelled['within_10m'] = labelled['projection_error_m'] <= 10.0
    labelled['near_but_wrong_edge'] = (~labelled['edge_correct']) & labelled['within_5m']

    traj_rows = []
    path_edits = []

    for tid, g in labelled.groupby('trajectory_id'):
        g = g.sort_values('t')
        pred_seq = g['pred_edge_idx'].astype(int).tolist()
        gt_seq = g['gt_edge_idx'].astype(int).tolist()
        edit = levenshtein(pred_seq, gt_seq)
        path_edits.append(edit)
        edge_acc = float(g['edge_correct'].mean())

        traj_rows.append(
            {
                'trajectory_id': int(tid),
                'points': int(len(g)),
                'edge_accuracy': edge_acc,
                'action_accuracy': float(g['action_correct'].mean()),
                'mean_projection_error_m': float(g['projection_error_m'].mean()),
                'within_5m_rate': float(g['within_5m'].mean()),
                'path_edit_distance': int(edit),
                'success': bool(edge_acc >= trajectory_success_accuracy),
            }
        )

    traj_df = pd.DataFrame(traj_rows)
    error_cases = labelled[~labelled['edge_correct']].copy()

    metrics = {
        'num_points': int(len(df)),
        'num_labelled_points': int(len(labelled)),
        'num_unlabelled_or_gt_missing_points': int(len(df) - len(labelled)),
        'num_trajectories': int(labelled['trajectory_id'].nunique()),
        'point_action_accuracy': float(labelled['action_correct'].mean()),
        'point_edge_accuracy': float(labelled['edge_correct'].mean()),
        'mean_projection_error_m': float(labelled['projection_error_m'].mean()),
        'median_projection_error_m': float(labelled['projection_error_m'].median()),
        'p90_projection_error_m': float(labelled['projection_error_m'].quantile(0.90)),
        'within_2m_rate': float(labelled['within_2m'].mean()),
        'within_5m_rate': float(labelled['within_5m'].mean()),
        'within_10m_rate': float(labelled['within_10m'].mean()),
        'projection_success_rate': float(labelled['projection_success'].mean()),
        'mean_confidence': float(labelled['confidence'].mean()) if 'confidence' in labelled.columns else float('nan'),
        'path_edit_distance_mean': float(np.mean(path_edits)),
        'path_edit_distance_median': float(np.median(path_edits)),
        'trajectory_success_rate': float(traj_df['success'].mean()),
        'num_error_points': int((~labelled['edge_correct']).sum()),
        'error_near_but_wrong_edge_rate': float(error_cases['near_but_wrong_edge'].mean()) if len(error_cases) else 0.0,
    }

    return metrics, traj_df, error_cases


def save_metrics(
    metrics: dict,
    trajectory_metrics: pd.DataFrame,
    error_cases: pd.DataFrame,
    metric_path: str | Path,
    trajectory_path: str | Path,
    error_path: str | Path,
) -> None:
    metric_path = Path(metric_path)
    trajectory_path = Path(trajectory_path)
    error_path = Path(error_path)

    metric_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    with metric_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(metrics, f, indent=2)

    trajectory_metrics.to_csv(trajectory_path, index=False)
    error_cases.to_csv(error_path, index=False)
