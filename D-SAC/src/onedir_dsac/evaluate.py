from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import torch
from .data import RLDataset
from .features import action_mask, actor_observation
from .models import DiscreteSACModel
from .reward import legal_transition, projection_error_m


def build_model_from_checkpoint(path: str | Path, device: str = "cpu") -> DiscreteSACModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = DiscreteSACModel(int(ckpt["actor_obs_dim"]), int(ckpt["critic_obs_dim"]), int(ckpt["action_dim"]), int(cfg.get("hidden_dim", 256)), int(cfg.get("num_layers", 2)), float(cfg.get("dropout", 0.1)), bool(cfg.get("use_privileged_critic", True)))
    model.load_state_dict(ckpt["model_state_dict"])
    return model


@torch.no_grad()
def evaluate_policy(model: DiscreteSACModel, dataset: RLDataset, device: str = "cpu", greedy: bool = True, k_max: int | None = None) -> tuple[pd.DataFrame, dict]:
    model.to(device)
    model.eval()
    rows = []
    for sample in dataset.episodes:
        previous_action = None
        for t in range(sample.length):
            obs = actor_observation(sample, t, previous_action, k_max).to(device)
            mask = action_mask(sample, t, k_max).to(device)
            action, confidence = model.act(obs, mask, greedy=greedy)
            gt_action = int(sample.gt_candidate_pos[t].item())
            pred_edge_idx = int(sample.candidate_edge_idx[t, action].item()) if action < sample.num_candidates else -1
            gt_edge_idx = int(sample.candidate_edge_idx[t, gt_action].item()) if 0 <= gt_action < sample.num_candidates else -1
            row = {"trajectory_id": int(sample.trajectory_id), "t": int(t), "pred_candidate_pos": int(action), "gt_candidate_pos": int(gt_action), "pred_edge_idx": pred_edge_idx, "gt_edge_idx": gt_edge_idx, "correct": int(action == gt_action), "legal_transition": int(legal_transition(sample, t, previous_action, action)), "confidence": float(confidence), "projection_error_m": float(projection_error_m(sample, t, action))}
            if sample.candidate_proj_xy is not None and action < sample.num_candidates:
                row["pred_proj_x"] = float(sample.candidate_proj_xy[t, action, 0].item())
                row["pred_proj_y"] = float(sample.candidate_proj_xy[t, action, 1].item())
            if sample.gt_proj_xy is not None:
                row["gt_proj_x"] = float(sample.gt_proj_xy[t, 0].item())
                row["gt_proj_y"] = float(sample.gt_proj_xy[t, 1].item())
            rows.append(row)
            previous_action = action
    df = pd.DataFrame(rows)
    transitions = df[df["t"] > 0]
    metrics = {"num_points": int(len(df)), "num_trajectories": int(df["trajectory_id"].nunique()) if len(df) else 0, "point_action_accuracy": float(df["correct"].mean()) if len(df) else 0.0, "point_edge_accuracy": float((df["pred_edge_idx"] == df["gt_edge_idx"]).mean()) if len(df) else 0.0, "legal_transition_rate": float(transitions["legal_transition"].mean()) if len(transitions) else 1.0, "mean_projection_error_m": float(df["projection_error_m"].mean()) if len(df) else 0.0, "median_projection_error_m": float(df["projection_error_m"].median()) if len(df) else 0.0, "p90_projection_error_m": float(df["projection_error_m"].quantile(0.90)) if len(df) else 0.0, "within_5m_rate": float((df["projection_error_m"] <= 5.0).mean()) if len(df) else 0.0, "mean_confidence": float(df["confidence"].mean()) if len(df) else 0.0}
    return df, metrics


def save_eval_outputs(matches: pd.DataFrame, metrics: dict, match_path: str | Path, metric_path: str | Path) -> None:
    match_path, metric_path = Path(match_path), Path(metric_path)
    match_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(match_path, index=False)
    with metric_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)
