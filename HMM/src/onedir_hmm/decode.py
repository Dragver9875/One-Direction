from __future__ import annotations
from pathlib import Path
import pandas as pd
from .data import HMMDataset
from .scoring import HMMParams, compute_emission_scores, compute_transition_scores
from .viterbi import confidence_from_scores, viterbi_decode

def decode_dataset(dataset: HMMDataset, params: HMMParams, transition_mode: str = "soft", confidence_temperature: float = 1.0) -> pd.DataFrame:
    rows = []
    for sample in dataset.episodes:
        emissions = compute_emission_scores(sample, params)
        transitions = compute_transition_scores(sample, params, mode=transition_mode)
        path, score_history = viterbi_decode(emissions, transitions)
        conf = confidence_from_scores(score_history, path, confidence_temperature)

        for t, action in enumerate(path):
            gt_action = int(sample.gt_candidate_pos[t].item())
            pred_edge_idx = int(sample.candidate_edge_idx[t, action].item()) if action < sample.num_candidates else -1
            if sample.gt_edge_idx is not None:
                gt_edge_idx = int(sample.gt_edge_idx[t].item())
            elif 0 <= gt_action < sample.num_candidates:
                gt_edge_idx = int(sample.candidate_edge_idx[t, gt_action].item())
            else:
                gt_edge_idx = -1
            row = {
                "trajectory_id": int(sample.trajectory_id),
                "t": int(t),
                "pred_candidate_pos": int(action),
                "gt_candidate_pos": int(gt_action),
                "pred_edge_idx": pred_edge_idx,
                "gt_edge_idx": gt_edge_idx,
                "confidence": float(conf[t]) if t < len(conf) else 0.0,
                "emission_score": float(emissions[t, action].item()),
                "path_score": float(score_history[t, action].item()),
            }
            if sample.candidate_proj_xy is not None and action < sample.num_candidates:
                row["pred_proj_x"] = float(sample.candidate_proj_xy[t, action, 0].item())
                row["pred_proj_y"] = float(sample.candidate_proj_xy[t, action, 1].item())
            if sample.gt_proj_xy is not None:
                row["gt_proj_x"] = float(sample.gt_proj_xy[t, 0].item())
                row["gt_proj_y"] = float(sample.gt_proj_xy[t, 1].item())
            rows.append(row)
    return pd.DataFrame(rows)

def save_matches(matches: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(path, index=False)
