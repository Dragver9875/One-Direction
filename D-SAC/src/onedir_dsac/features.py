from __future__ import annotations

import torch
from torch import Tensor
from .data import EpisodeSample


def max_candidates(sample: EpisodeSample, configured: int | None = None) -> int:
    return int(configured or sample.num_candidates)


def pad_2d(x: Tensor, rows: int, cols: int) -> Tensor:
    out = torch.zeros((rows, cols), dtype=x.dtype)
    out[: min(rows, x.shape[0]), : min(cols, x.shape[1])] = x[: min(rows, x.shape[0]), : min(cols, x.shape[1])]
    return out


def pad_1d(x: Tensor, length: int) -> Tensor:
    out = torch.zeros((length,), dtype=x.dtype)
    out[: min(length, x.shape[0])] = x[: min(length, x.shape[0])]
    return out


def actor_observation(sample: EpisodeSample, t: int, previous_action: int | None, k_max: int | None = None) -> Tensor:
    k = max_candidates(sample, k_max)
    feats = sample.emission_features[t].float()
    feat_pad = pad_2d(feats, k, feats.shape[-1]).reshape(-1)
    mask_pad = pad_1d(sample.candidate_mask[t].float(), k)

    prev = torch.zeros((k,), dtype=torch.float32)
    if previous_action is not None and 0 <= previous_action < k:
        prev[previous_action] = 1.0

    t_frac = torch.tensor([t / max(sample.length - 1, 1)], dtype=torch.float32)
    return torch.cat([feat_pad, mask_pad, prev, t_frac], dim=0)


def action_mask(sample: EpisodeSample, t: int, k_max: int | None = None) -> Tensor:
    return pad_1d(sample.candidate_mask[t].bool(), max_candidates(sample, k_max)).bool()


def projection_distance_feature(sample: EpisodeSample, t: int, k: int, cap_m: float = 50.0) -> float:
    if sample.candidate_proj_xy is None or sample.gt_proj_xy is None or k < 0 or k >= sample.num_candidates:
        return 0.0
    gt = sample.gt_proj_xy[t]
    cand = sample.candidate_proj_xy[t, k]
    if torch.isnan(gt).any() or torch.isnan(cand).any():
        return 0.0
    return min(float(torch.linalg.vector_norm(cand - gt).item()), cap_m) / cap_m


def privileged_per_candidate(sample: EpisodeSample, t: int, k_max: int | None = None) -> Tensor:
    k = max_candidates(sample, k_max)
    out = torch.zeros((k, 7), dtype=torch.float32)
    gt_pos = int(sample.gt_candidate_pos[t].item())

    for a in range(k):
        valid = bool(a < sample.num_candidates and sample.candidate_mask[t, a].item())
        if not valid:
            continue

        same_edge = 0.0
        if 0 <= gt_pos < sample.num_candidates:
            same_edge = float(sample.candidate_edge_idx[t, a].item() == sample.candidate_edge_idx[t, gt_pos].item())

        prev_legal = 1.0
        if t > 0 and sample.transition_mask is not None:
            prev_gt = int(sample.gt_candidate_pos[t - 1].item())
            if 0 <= prev_gt < sample.num_candidates:
                prev_legal = float(sample.transition_mask[t - 1, prev_gt, a].item())

        next_legal = 1.0
        if t < sample.length - 1 and sample.transition_mask is not None:
            next_gt = int(sample.gt_candidate_pos[t + 1].item())
            if 0 <= next_gt < sample.num_candidates:
                next_legal = float(sample.transition_mask[t, a, next_gt].item())

        rank_gap = abs(a - gt_pos) / max(k - 1, 1) if gt_pos >= 0 else 1.0
        out[a] = torch.tensor([float(a == gt_pos), rank_gap, same_edge, prev_legal, next_legal, projection_distance_feature(sample, t, a), float(valid)])

    return out


def privileged_observation(sample: EpisodeSample, t: int, previous_action: int | None, k_max: int | None = None) -> Tensor:
    return torch.cat([actor_observation(sample, t, previous_action, k_max), privileged_per_candidate(sample, t, k_max).reshape(-1)], dim=0)


def observation_dims(sample: EpisodeSample, k_max: int | None = None) -> tuple[int, int, int]:
    k = max_candidates(sample, k_max)
    return int(actor_observation(sample, 0, None, k).numel()), int(privileged_observation(sample, 0, None, k).numel()), k
