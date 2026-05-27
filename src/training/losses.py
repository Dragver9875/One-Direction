from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GNNHMMLossConfig:
    emission_weight: float = 1.0
    transition_weight: float = 1.0
    l2_weight: float = 0.0
    ignore_index: int = -1
    invalid_score: float = -1.0e9
    label_smoothing: float = 0.0


def _masked_scores(scores: Tensor, mask: Tensor | None, invalid_score: float) -> Tensor:
    if mask is None:
        return scores
    return scores.masked_fill(~mask.bool(), invalid_score)


def compute_emission_loss(
    emission_scores: Tensor,
    gt_candidate_pos: Tensor,
    candidate_mask: Tensor | None = None,
    cfg: GNNHMMLossConfig = GNNHMMLossConfig(),
) -> Tensor:
    if emission_scores.ndim != 3:
        raise ValueError("emission_scores must have shape [B, T, K].")
    if gt_candidate_pos.shape != emission_scores.shape[:2]:
        raise ValueError("gt_candidate_pos must have shape [B, T].")

    scores = _masked_scores(emission_scores, candidate_mask, cfg.invalid_score)
    bsz, t_count, k_count = scores.shape

    scores_2d = scores.reshape(bsz * t_count, k_count)
    target = gt_candidate_pos.long().reshape(bsz * t_count)

    valid = target != cfg.ignore_index
    valid = valid & (target >= 0) & (target < k_count)

    if candidate_mask is not None:
        flat_mask = candidate_mask.reshape(bsz * t_count, k_count).bool()
        target_safe = target.clamp(0, k_count - 1)
        valid = valid & flat_mask.gather(1, target_safe.view(-1, 1)).squeeze(1)

    if not torch.any(valid):
        return scores.sum() * 0.0

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    return loss_fn(scores_2d[valid], target[valid])


def compute_transition_loss(
    transition_scores: Tensor,
    gt_candidate_pos: Tensor,
    transition_mask: Tensor | None = None,
    cfg: GNNHMMLossConfig = GNNHMMLossConfig(),
) -> Tensor:
    if transition_scores.ndim != 4:
        raise ValueError("transition_scores must have shape [B, T - 1, K, K].")
    if gt_candidate_pos.ndim != 2:
        raise ValueError("gt_candidate_pos must have shape [B, T].")

    bsz, trans_count, k_prev, k_curr = transition_scores.shape
    if gt_candidate_pos.shape[0] != bsz or gt_candidate_pos.shape[1] != trans_count + 1:
        raise ValueError("gt_candidate_pos shape must be [B, T] where T = transition_scores.shape[1] + 1.")

    prev_pos = gt_candidate_pos[:, :-1].long()
    curr_pos = gt_candidate_pos[:, 1:].long()

    valid = prev_pos != cfg.ignore_index
    valid = valid & (curr_pos != cfg.ignore_index)
    valid = valid & (prev_pos >= 0) & (prev_pos < k_prev)
    valid = valid & (curr_pos >= 0) & (curr_pos < k_curr)

    if transition_mask is not None:
        if transition_mask.shape != transition_scores.shape:
            raise ValueError("transition_mask must have shape [B, T - 1, K, K].")
        safe_prev = prev_pos.clamp(0, k_prev - 1)
        safe_curr = curr_pos.clamp(0, k_curr - 1)
        batch_idx = torch.arange(bsz, device=transition_scores.device).view(-1, 1).expand(bsz, trans_count)
        time_idx = torch.arange(trans_count, device=transition_scores.device).view(1, -1).expand(bsz, trans_count)
        valid = valid & transition_mask[batch_idx, time_idx, safe_prev, safe_curr].bool()

    if not torch.any(valid):
        return transition_scores.sum() * 0.0

    safe_prev = prev_pos.clamp(0, k_prev - 1)
    batch_idx = torch.arange(bsz, device=transition_scores.device).view(-1, 1).expand(bsz, trans_count)
    time_idx = torch.arange(trans_count, device=transition_scores.device).view(1, -1).expand(bsz, trans_count)

    conditioned_scores = transition_scores[batch_idx, time_idx, safe_prev, :]
    if transition_mask is not None:
        conditioned_mask = transition_mask[batch_idx, time_idx, safe_prev, :]
        conditioned_scores = conditioned_scores.masked_fill(~conditioned_mask.bool(), cfg.invalid_score)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scores_2d = conditioned_scores.reshape(bsz * trans_count, k_curr)
    target = curr_pos.reshape(bsz * trans_count)
    valid_flat = valid.reshape(bsz * trans_count)

    return loss_fn(scores_2d[valid_flat], target[valid_flat])


def compute_l2_regularization(model: torch.nn.Module) -> Tensor:
    params = [p.pow(2).sum() for p in model.parameters() if p.requires_grad]
    if not params:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(params).sum()


def compute_total_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    model: torch.nn.Module | None = None,
    cfg: GNNHMMLossConfig = GNNHMMLossConfig(),
) -> dict[str, Tensor]:
    emission_loss = compute_emission_loss(
        emission_scores=outputs["emission_scores"],
        gt_candidate_pos=batch["gt_candidate_pos"],
        candidate_mask=batch.get("candidate_mask"),
        cfg=cfg,
    )

    transition_loss = compute_transition_loss(
        transition_scores=outputs["transition_scores"],
        gt_candidate_pos=batch["gt_candidate_pos"],
        transition_mask=batch.get("transition_mask"),
        cfg=cfg,
    )

    total = cfg.emission_weight * emission_loss + cfg.transition_weight * transition_loss

    l2_loss = torch.zeros_like(total)
    if model is not None and cfg.l2_weight > 0:
        l2_loss = compute_l2_regularization(model)
        total = total + cfg.l2_weight * l2_loss

    return {
        "loss": total,
        "emission_loss": emission_loss.detach(),
        "transition_loss": transition_loss.detach(),
        "l2_loss": l2_loss.detach(),
    }
