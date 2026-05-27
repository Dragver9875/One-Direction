from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .losses import GNNHMMLossConfig, compute_total_loss


@dataclass(frozen=True)
class ValidationConfig:
    max_batches: int | None = None
    decode_during_validation: bool = False


def compute_emission_accuracy(
    emission_scores: Tensor,
    gt_candidate_pos: Tensor,
    candidate_mask: Tensor | None = None,
    topk: tuple[int, ...] = (1, 3),
    ignore_index: int = -1,
) -> dict[str, float]:
    if candidate_mask is not None:
        emission_scores = emission_scores.masked_fill(~candidate_mask.bool(), -1.0e9)

    bsz, t_count, k_count = emission_scores.shape
    target = gt_candidate_pos.long()
    valid = (target != ignore_index) & (target >= 0) & (target < k_count)

    if not torch.any(valid):
        return {f"emission_top{k}_accuracy": 0.0 for k in topk}

    metrics: dict[str, float] = {}
    max_k = min(max(topk), k_count)
    pred_topk = torch.topk(emission_scores, k=max_k, dim=-1).indices

    for k in topk:
        k_eff = min(k, k_count)
        hit = pred_topk[..., :k_eff].eq(target.unsqueeze(-1)).any(dim=-1)
        metrics[f"emission_top{k}_accuracy"] = float(hit[valid].float().mean().item())

    pred_top1 = pred_topk[..., 0]
    metrics["point_edge_accuracy"] = float(pred_top1[valid].eq(target[valid]).float().mean().item())

    return metrics


def compute_transition_accuracy(
    transition_scores: Tensor,
    gt_candidate_pos: Tensor,
    transition_mask: Tensor | None = None,
    ignore_index: int = -1,
) -> dict[str, float]:
    bsz, trans_count, k_prev, k_curr = transition_scores.shape

    prev_pos = gt_candidate_pos[:, :-1].long()
    curr_pos = gt_candidate_pos[:, 1:].long()

    valid = (prev_pos != ignore_index) & (curr_pos != ignore_index)
    valid = valid & (prev_pos >= 0) & (prev_pos < k_prev)
    valid = valid & (curr_pos >= 0) & (curr_pos < k_curr)

    if not torch.any(valid):
        return {"transition_top1_accuracy": 0.0}

    batch_idx = torch.arange(bsz, device=transition_scores.device).view(-1, 1).expand(bsz, trans_count)
    time_idx = torch.arange(trans_count, device=transition_scores.device).view(1, -1).expand(bsz, trans_count)
    safe_prev = prev_pos.clamp(0, k_prev - 1)

    conditioned = transition_scores[batch_idx, time_idx, safe_prev, :]

    if transition_mask is not None:
        conditioned_mask = transition_mask[batch_idx, time_idx, safe_prev, :]
        conditioned = conditioned.masked_fill(~conditioned_mask.bool(), -1.0e9)

    pred = torch.argmax(conditioned, dim=-1)
    acc = pred[valid].eq(curr_pos[valid]).float().mean().item()

    return {"transition_top1_accuracy": float(acc)}


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    road_data,
    device: torch.device,
    cfg: ValidationConfig = ValidationConfig(),
    loss_cfg: GNNHMMLossConfig = GNNHMMLossConfig(),
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_emission_loss = 0.0
    total_transition_loss = 0.0
    total_batches = 0

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    if hasattr(road_data, "to"):
        road_data = road_data.to(device)
    elif isinstance(road_data, Tensor):
        road_data = road_data.to(device)

    for batch_idx, batch in enumerate(dataloader):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break

        batch = _move_batch_to_device(batch, device)

        outputs = model(
            road_x_or_data=road_data,
            candidate_edge_idx=batch["candidate_edge_idx"],
            emission_features=batch["emission_features"],
            prev_edge_idx=batch["prev_edge_idx"],
            curr_edge_idx=batch["curr_edge_idx"],
            transition_features=batch["transition_features"],
            candidate_mask=batch.get("candidate_mask"),
            transition_mask=batch.get("transition_mask"),
        )

        losses = compute_total_loss(outputs, batch, model=None, cfg=loss_cfg)

        total_loss += float(losses["loss"].item())
        total_emission_loss += float(losses["emission_loss"].item())
        total_transition_loss += float(losses["transition_loss"].item())
        total_batches += 1

        batch_metrics = {}
        batch_metrics.update(
            compute_emission_accuracy(
                outputs["emission_scores"],
                batch["gt_candidate_pos"],
                batch.get("candidate_mask"),
                topk=(1, 3),
                ignore_index=loss_cfg.ignore_index,
            )
        )
        batch_metrics.update(
            compute_transition_accuracy(
                outputs["transition_scores"],
                batch["gt_candidate_pos"],
                batch.get("transition_mask"),
                ignore_index=loss_cfg.ignore_index,
            )
        )

        for key, value in batch_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            metric_counts[key] = metric_counts.get(key, 0) + 1

    denom = max(total_batches, 1)
    metrics = {
        "val_loss": total_loss / denom,
        "val_emission_loss": total_emission_loss / denom,
        "val_transition_loss": total_transition_loss / denom,
    }

    for key, value in metric_sums.items():
        metrics[f"val_{key}"] = value / max(metric_counts.get(key, 1), 1)

    return metrics
