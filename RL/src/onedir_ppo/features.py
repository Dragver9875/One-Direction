from __future__ import annotations
import torch
from torch import Tensor
from .data import EpisodeSample


def _k(sample: EpisodeSample, configured: int | None = None) -> int:
    return int(configured or sample.num_candidates)


def _pad1(x: Tensor, n: int) -> Tensor:
    out = torch.zeros((n,), dtype=x.dtype)
    out[: min(n, x.shape[0])] = x[: min(n, x.shape[0])]
    return out


def _pad2(x: Tensor, rows: int, cols: int) -> Tensor:
    out = torch.zeros((rows, cols), dtype=x.dtype)
    r, c = min(rows, x.shape[0]), min(cols, x.shape[1])
    out[:r, :c] = x[:r, :c]
    return out


def actor_observation(sample: EpisodeSample, t: int, previous_action: int | None, k_max: int | None = None) -> Tensor:
    k = _k(sample, k_max)
    feats = sample.emission_features[t].float()
    mask = sample.candidate_mask[t].float()
    feat_pad = _pad2(feats, k, feats.shape[-1]).reshape(-1)
    mask_pad = _pad1(mask, k)
    prev = torch.zeros((k,), dtype=torch.float32)
    if previous_action is not None and 0 <= previous_action < k:
        prev[previous_action] = 1.0
    t_frac = torch.tensor([t / max(sample.length - 1, 1)], dtype=torch.float32)
    return torch.cat([feat_pad, mask_pad, prev, t_frac], dim=0)


def action_mask(sample: EpisodeSample, t: int, k_max: int | None = None) -> Tensor:
    return _pad1(sample.candidate_mask[t].bool(), _k(sample, k_max)).bool()


def _projection_feature(sample: EpisodeSample, t: int, a: int, cap_m: float = 50.0) -> float:
    if sample.candidate_proj_xy is None or sample.gt_proj_xy is None or a >= sample.num_candidates:
        return 0.0
    pred, gt = sample.candidate_proj_xy[t, a], sample.gt_proj_xy[t]
    if torch.isnan(pred).any() or torch.isnan(gt).any():
        return 0.0
    return min(float(torch.linalg.vector_norm(pred - gt).item()), cap_m) / cap_m


def privileged_per_candidate(sample: EpisodeSample, t: int, k_max: int | None = None) -> Tensor:
    k = _k(sample, k_max)
    out = torch.zeros((k, 7), dtype=torch.float32)
    gt = int(sample.gt_candidate_pos[t].item())
    for a in range(k):
        valid = bool(a < sample.num_candidates and sample.candidate_mask[t, a].item())
        if not valid:
            continue
        same_edge_as_gt = 0.0
        if 0 <= gt < sample.num_candidates:
            same_edge_as_gt = float(sample.candidate_edge_idx[t, a].item() == sample.candidate_edge_idx[t, gt].item())
        from_prev_gt = 1.0
        if t > 0 and sample.transition_mask is not None:
            prev_gt = int(sample.gt_candidate_pos[t - 1].item())
            if 0 <= prev_gt < sample.num_candidates:
                from_prev_gt = float(sample.transition_mask[t - 1, prev_gt, a].item())
        to_next_gt = 1.0
        if t < sample.length - 1 and sample.transition_mask is not None:
            next_gt = int(sample.gt_candidate_pos[t + 1].item())
            if 0 <= next_gt < sample.num_candidates:
                to_next_gt = float(sample.transition_mask[t, a, next_gt].item())
        out[a] = torch.tensor([
            float(a == gt),
            abs(a - gt) / max(k - 1, 1) if gt >= 0 else 1.0,
            same_edge_as_gt,
            from_prev_gt,
            to_next_gt,
            _projection_feature(sample, t, a),
            float(valid),
        ])
    return out


def privileged_observation(sample: EpisodeSample, t: int, previous_action: int | None, k_max: int | None = None) -> Tensor:
    return torch.cat([actor_observation(sample, t, previous_action, k_max), privileged_per_candidate(sample, t, k_max).reshape(-1)])


def observation_dims(sample: EpisodeSample, k_max: int | None = None) -> tuple[int, int, int]:
    return actor_observation(sample, 0, None, k_max).numel(), privileged_observation(sample, 0, None, k_max).numel(), _k(sample, k_max)
