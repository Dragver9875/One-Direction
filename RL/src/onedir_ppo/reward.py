from __future__ import annotations
from dataclasses import dataclass
import torch
from .data import EpisodeSample


@dataclass(frozen=True)
class RewardConfig:
    correct_action_reward: float = 1.0
    wrong_action_penalty: float = -0.2
    invalid_action_penalty: float = -2.0
    legal_transition_bonus: float = 0.2
    illegal_transition_penalty: float = -1.0
    projection_error_weight: float = 0.02
    projection_error_cap_m: float = 25.0
    terminal_success_accuracy: float = 0.90
    terminal_success_bonus: float = 2.0
    terminal_accuracy_weight: float = 1.0
    terminal_illegal_transition_weight: float = 0.2


def projection_error_m(sample: EpisodeSample, t: int, action: int) -> float:
    if sample.candidate_proj_xy is None or sample.gt_proj_xy is None or action >= sample.num_candidates:
        return 0.0
    pred, gt = sample.candidate_proj_xy[t, action], sample.gt_proj_xy[t]
    if torch.isnan(pred).any() or torch.isnan(gt).any():
        return 0.0
    return float(torch.linalg.vector_norm(pred - gt).item())


def legal_transition(sample: EpisodeSample, t: int, previous_action: int | None, action: int) -> bool:
    if t <= 0 or previous_action is None or action == previous_action:
        return True
    if sample.transition_mask is None:
        return True
    if previous_action >= sample.transition_mask.shape[1] or action >= sample.transition_mask.shape[2]:
        return False
    return bool(sample.transition_mask[t - 1, previous_action, action].item())


def step_reward(sample: EpisodeSample, t: int, action: int, previous_action: int | None, cfg: RewardConfig) -> tuple[float, dict]:
    valid = 0 <= action < sample.num_candidates and bool(sample.candidate_mask[t, action].item())
    gt = int(sample.gt_candidate_pos[t].item())
    if not valid:
        return cfg.invalid_action_penalty, {'correct': False, 'valid_action': False, 'legal_transition': False, 'projection_error_m': 0.0}
    correct = action == gt
    legal = legal_transition(sample, t, previous_action, action)
    err = projection_error_m(sample, t, action)
    reward = cfg.correct_action_reward if correct else cfg.wrong_action_penalty
    reward += cfg.legal_transition_bonus if legal else cfg.illegal_transition_penalty
    reward -= cfg.projection_error_weight * min(err, cfg.projection_error_cap_m)
    return float(reward), {'correct': correct, 'valid_action': True, 'legal_transition': legal, 'projection_error_m': err}


def terminal_reward(correct_flags: list[bool], legal_flags: list[bool], cfg: RewardConfig) -> float:
    if not correct_flags:
        return 0.0
    acc = sum(correct_flags) / len(correct_flags)
    illegal = sum(1 for x in legal_flags if not x)
    reward = cfg.terminal_accuracy_weight * acc
    if acc >= cfg.terminal_success_accuracy:
        reward += cfg.terminal_success_bonus
    reward -= cfg.terminal_illegal_transition_weight * illegal
    return float(reward)
