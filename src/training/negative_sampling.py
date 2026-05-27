from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NegativeSamplingConfig:
    num_negatives: int = 5
    replacement: bool = False
    ignore_index: int = -1
    seed: int | None = None


def _make_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def sample_candidate_negatives(
    candidate_mask: Tensor,
    gt_candidate_pos: Tensor,
    cfg: NegativeSamplingConfig = NegativeSamplingConfig(),
) -> Tensor:
    if candidate_mask.ndim != 3:
        raise ValueError("candidate_mask must have shape [B, T, K].")
    if gt_candidate_pos.shape != candidate_mask.shape[:2]:
        raise ValueError("gt_candidate_pos must have shape [B, T].")

    device = candidate_mask.device
    bsz, t_count, k_count = candidate_mask.shape
    negatives = torch.full(
        (bsz, t_count, cfg.num_negatives),
        cfg.ignore_index,
        dtype=torch.long,
        device=device,
    )

    gen = _make_generator(device, cfg.seed)

    for b in range(bsz):
        for t in range(t_count):
            valid_idx = torch.where(candidate_mask[b, t].bool())[0]
            gt = int(gt_candidate_pos[b, t].item())
            valid_idx = valid_idx[valid_idx != gt]

            if valid_idx.numel() == 0:
                continue

            if cfg.replacement:
                choice = valid_idx[torch.randint(0, valid_idx.numel(), (cfg.num_negatives,), device=device, generator=gen)]
            else:
                count = min(cfg.num_negatives, valid_idx.numel())
                perm = torch.randperm(valid_idx.numel(), device=device, generator=gen)[:count]
                choice = valid_idx[perm]

            negatives[b, t, : choice.numel()] = choice

    return negatives


def sample_transition_negatives(
    transition_mask: Tensor,
    gt_candidate_pos: Tensor,
    cfg: NegativeSamplingConfig = NegativeSamplingConfig(),
) -> Tensor:
    if transition_mask.ndim != 4:
        raise ValueError("transition_mask must have shape [B, T - 1, K, K].")
    if gt_candidate_pos.ndim != 2:
        raise ValueError("gt_candidate_pos must have shape [B, T].")

    device = transition_mask.device
    bsz, trans_count, k_prev, k_curr = transition_mask.shape

    negatives = torch.full(
        (bsz, trans_count, cfg.num_negatives),
        cfg.ignore_index,
        dtype=torch.long,
        device=device,
    )

    gen = _make_generator(device, cfg.seed)

    for b in range(bsz):
        for t in range(trans_count):
            prev_gt = int(gt_candidate_pos[b, t].item())
            curr_gt = int(gt_candidate_pos[b, t + 1].item())

            if prev_gt < 0 or prev_gt >= k_prev:
                continue

            valid_next = torch.where(transition_mask[b, t, prev_gt].bool())[0]
            valid_next = valid_next[valid_next != curr_gt]

            if valid_next.numel() == 0:
                continue

            if cfg.replacement:
                choice = valid_next[torch.randint(0, valid_next.numel(), (cfg.num_negatives,), device=device, generator=gen)]
            else:
                count = min(cfg.num_negatives, valid_next.numel())
                perm = torch.randperm(valid_next.numel(), device=device, generator=gen)[:count]
                choice = valid_next[perm]

            negatives[b, t, : choice.numel()] = choice

    return negatives


def build_all_negative_mask(
    candidate_mask: Tensor,
    gt_candidate_pos: Tensor,
) -> Tensor:
    if candidate_mask.ndim != 3:
        raise ValueError("candidate_mask must have shape [B, T, K].")

    mask = candidate_mask.bool().clone()
    bsz, t_count, k_count = mask.shape

    safe_target = gt_candidate_pos.long().clamp(0, k_count - 1)
    valid_target = (gt_candidate_pos >= 0) & (gt_candidate_pos < k_count)

    b_idx = torch.arange(bsz, device=mask.device).view(-1, 1).expand(bsz, t_count)
    t_idx = torch.arange(t_count, device=mask.device).view(1, -1).expand(bsz, t_count)

    mask[b_idx[valid_target], t_idx[valid_target], safe_target[valid_target]] = False
    return mask
