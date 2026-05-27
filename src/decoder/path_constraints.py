from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConstraintConfig:
    illegal_transition_score: float = -1.0e9
    disconnected_transition_penalty: float = -1000.0
    max_route_distance_m: float = 300.0
    max_speed_mps: float = 70.0
    allow_candidate_jumps: bool = True
    require_legal_transition: bool = True


def build_transition_mask(
    transition_features: Tensor,
    is_connected_index: int | None = None,
    is_legal_index: int | None = None,
    route_distance_index: int | None = None,
    gps_dt_s: Tensor | None = None,
    cfg: ConstraintConfig = ConstraintConfig(),
) -> Tensor:
    if transition_features.ndim < 3:
        raise ValueError("transition_features must have at least shape [T - 1, K, K, F].")

    mask_shape = transition_features.shape[:-1]
    mask = torch.ones(mask_shape, dtype=torch.bool, device=transition_features.device)

    if is_connected_index is not None and not cfg.allow_candidate_jumps:
        is_connected = transition_features[..., is_connected_index] > 0.5
        mask = mask & is_connected

    if is_legal_index is not None and cfg.require_legal_transition:
        is_legal = transition_features[..., is_legal_index] > 0.5
        mask = mask & is_legal

    if route_distance_index is not None:
        route_distance = transition_features[..., route_distance_index]
        mask = mask & (route_distance <= cfg.max_route_distance_m)

        if gps_dt_s is not None:
            if gps_dt_s.ndim == 1:
                dt = gps_dt_s.view(-1, 1, 1)
            elif gps_dt_s.ndim == 2:
                dt = gps_dt_s.view(gps_dt_s.shape[0], gps_dt_s.shape[1], 1, 1)
            else:
                raise ValueError("gps_dt_s must have shape [T - 1] or [B, T - 1].")

            max_distance = torch.clamp(dt, min=0.0) * cfg.max_speed_mps
            mask = mask & (route_distance <= max_distance)

    return mask


def apply_transition_constraints(
    transition_scores: Tensor,
    transition_mask: Tensor,
    cfg: ConstraintConfig = ConstraintConfig(),
) -> Tensor:
    if transition_scores.shape != transition_mask.shape:
        raise ValueError("transition_scores and transition_mask must have identical shape.")

    return transition_scores.masked_fill(~transition_mask.bool(), cfg.illegal_transition_score)


def build_candidate_mask(candidate_edge_idx: Tensor) -> Tensor:
    return candidate_edge_idx >= 0


def combine_transition_scores(
    learned_transition_scores: Tensor,
    classical_penalty: Tensor | None = None,
    transition_mask: Tensor | None = None,
    cfg: ConstraintConfig = ConstraintConfig(),
) -> Tensor:
    scores = learned_transition_scores

    if classical_penalty is not None:
        if classical_penalty.shape != scores.shape:
            raise ValueError("classical_penalty must have the same shape as learned_transition_scores.")
        scores = scores + classical_penalty

    if transition_mask is not None:
        scores = apply_transition_constraints(scores, transition_mask, cfg)

    return scores
