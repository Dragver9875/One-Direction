from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .data import EpisodeSample


MASK_VALUE = -1.0e9


@dataclass
class HMMParams:
    emission_scale: float = 1.0
    distance_weight: float = 3.0
    log_distance_weight: float = 0.8
    yaw_weight: float = 1.2
    rank_weight: float = 1.2
    speed_consistency_weight: float = 0.15
    oneway_weight: float = 0.05
    yaw_reliability_weight: float = 0.35
    bias: float = 0.0

    transition_scale: float = 0.35
    legal_bonus: float = 0.35
    illegal_penalty: float = 2.0
    same_edge_bonus: float = 0.35
    same_osm_way_bonus: float = 0.15
    same_road_class_bonus: float = 0.05
    route_distance_weight: float = 0.25
    route_gps_ratio_weight: float = 0.10
    route_minus_gps_weight: float = 0.15
    turn_weight: float = 0.10
    yaw_change_weight: float = 0.05
    time_feasible_bonus: float = 0.10
    time_infeasible_penalty: float = 0.50
    rank_delta_weight: float = 0.05
    distance_delta_weight: float = 0.05


def feature_index(names: list[str], name: str, fallback: int | None = None) -> int | None:
    if name in names:
        return names.index(name)
    return fallback


def get_feature(features: Tensor, names: list[str], name: str, fallback: int | None = None) -> Tensor:
    idx = feature_index(names, name, fallback)
    if idx is None or idx >= features.shape[-1]:
        return torch.zeros(features.shape[:-1], dtype=features.dtype)
    return features[..., idx]


def compute_emission_scores(sample: EpisodeSample, params: HMMParams) -> Tensor:
    f = sample.emission_features.float()
    names = sample.emission_feature_names

    distance = get_feature(f, names, 'distance_norm', 0).clamp_min(0.0)
    log_distance = get_feature(f, names, 'log_distance_norm', 1).clamp_min(0.0)
    abs_yaw = get_feature(f, names, 'abs_yaw_diff_norm', 3).clamp_min(0.0)
    rank = get_feature(f, names, 'candidate_rank_norm', 10).clamp_min(0.0)
    speed_consistency = get_feature(f, names, 'speed_consistency', 11).clamp_min(0.0)
    oneway = get_feature(f, names, 'oneway', 12).clamp_min(0.0)
    yaw_reliability = get_feature(f, names, 'yaw_reliability', 15).clamp(0.0, 1.0)

    yaw_penalty = abs_yaw * (params.yaw_reliability_weight + (1.0 - params.yaw_reliability_weight) * yaw_reliability)

    score = (
        params.bias
        - params.distance_weight * distance
        - params.log_distance_weight * log_distance
        - params.yaw_weight * yaw_penalty
        - params.rank_weight * rank
        - params.speed_consistency_weight * speed_consistency
        + params.oneway_weight * oneway
    )

    score = params.emission_scale * score
    return score.masked_fill(~sample.candidate_mask.bool(), MASK_VALUE)


def compute_transition_scores(sample: EpisodeSample, params: HMMParams, mode: str = 'soft') -> Tensor:
    tfeat = sample.transition_features.float()
    if tfeat.numel() == 0:
        return torch.empty(0)

    names = sample.transition_feature_names

    route_dist = get_feature(tfeat, names, 'route_dist_norm', 3).clamp_min(0.0)
    route_minus_gps = get_feature(tfeat, names, 'route_minus_gps_norm', 4).clamp_min(0.0)
    route_ratio = get_feature(tfeat, names, 'route_gps_ratio_norm', 5).clamp_min(0.0)
    turn = get_feature(tfeat, names, 'turn_norm', 6).clamp_min(0.0)
    yaw_change = get_feature(tfeat, names, 'yaw_change_norm', 7).clamp_min(0.0)

    connected = get_feature(tfeat, names, 'connected', 9)
    legal = get_feature(tfeat, names, 'legal', 10)
    same_edge = get_feature(tfeat, names, 'same_edge', 11)
    same_osm_way = get_feature(tfeat, names, 'same_osm_way', 12)
    same_road_class = get_feature(tfeat, names, 'same_road_class', 13)
    rank_delta = get_feature(tfeat, names, 'candidate_rank_delta_norm', 15).clamp_min(0.0)
    distance_delta = get_feature(tfeat, names, 'distance_delta_norm', 18).clamp_min(0.0)
    time_feasible = get_feature(tfeat, names, 'time_feasible', 19).clamp(0.0, 1.0)

    legal_like = ((legal > 0.5) | (same_edge > 0.5) | (connected > 0.5)).float()
    infeasible = 1.0 - time_feasible

    score = (
        params.legal_bonus * legal_like
        + params.same_edge_bonus * same_edge
        + params.same_osm_way_bonus * same_osm_way
        + params.same_road_class_bonus * same_road_class
        + params.time_feasible_bonus * time_feasible
        - params.illegal_penalty * (1.0 - legal_like)
        - params.time_infeasible_penalty * infeasible
        - params.route_distance_weight * route_dist
        - params.route_minus_gps_weight * route_minus_gps
        - params.route_gps_ratio_weight * route_ratio
        - params.turn_weight * turn
        - params.yaw_change_weight * yaw_change
        - params.rank_delta_weight * rank_delta
        - params.distance_delta_weight * distance_delta
    )

    score = params.transition_scale * score
    mask = sample.transition_mask.bool()

    if mode == 'hard':
        score = score.masked_fill(~mask, MASK_VALUE)
    elif mode == 'soft':
        score = score - (~mask).float() * (params.transition_scale * params.illegal_penalty)
    elif mode == 'none':
        pass
    else:
        raise ValueError(f'Unsupported transition mode: {mode}')

    return score


def params_from_config(config: dict) -> HMMParams:
    emission = config.get('emission', {})
    transition = config.get('transition', {})
    return HMMParams(
        emission_scale=float(emission.get('emission_scale', 1.0)),
        distance_weight=float(emission.get('distance_weight', 3.0)),
        log_distance_weight=float(emission.get('log_distance_weight', 0.8)),
        yaw_weight=float(emission.get('yaw_weight', 1.2)),
        rank_weight=float(emission.get('rank_weight', 1.2)),
        speed_consistency_weight=float(emission.get('speed_consistency_weight', 0.15)),
        oneway_weight=float(emission.get('oneway_weight', 0.05)),
        yaw_reliability_weight=float(emission.get('yaw_reliability_weight', 0.35)),
        bias=float(emission.get('bias', 0.0)),
        transition_scale=float(transition.get('transition_scale', 0.35)),
        legal_bonus=float(transition.get('legal_bonus', 0.35)),
        illegal_penalty=float(transition.get('illegal_penalty', 2.0)),
        same_edge_bonus=float(transition.get('same_edge_bonus', 0.35)),
        same_osm_way_bonus=float(transition.get('same_osm_way_bonus', 0.15)),
        same_road_class_bonus=float(transition.get('same_road_class_bonus', 0.05)),
        route_distance_weight=float(transition.get('route_distance_weight', 0.25)),
        route_gps_ratio_weight=float(transition.get('route_gps_ratio_weight', 0.10)),
        route_minus_gps_weight=float(transition.get('route_minus_gps_weight', 0.15)),
        turn_weight=float(transition.get('turn_weight', 0.10)),
        yaw_change_weight=float(transition.get('yaw_change_weight', 0.05)),
        time_feasible_bonus=float(transition.get('time_feasible_bonus', 0.10)),
        time_infeasible_penalty=float(transition.get('time_infeasible_penalty', 0.50)),
        rank_delta_weight=float(transition.get('rank_delta_weight', 0.05)),
        distance_delta_weight=float(transition.get('distance_delta_weight', 0.05)),
    )
