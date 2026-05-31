#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor


MASK_VALUE = -1.0e9


@dataclass
class EpisodeSample:
    trajectory_id: int
    candidate_edge_idx: Tensor
    candidate_mask: Tensor
    candidate_proj_xy: Tensor | None
    emission_features: Tensor
    transition_features: Tensor
    transition_mask: Tensor
    gt_candidate_pos: Tensor
    gt_edge_idx: Tensor | None
    gt_proj_xy: Tensor | None
    emission_feature_names: list[str]
    transition_feature_names: list[str]
    timestamps: list[Any] | None = None

    @property
    def length(self) -> int:
        return int(self.candidate_edge_idx.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_edge_idx.shape[1])


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_value(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def deep_set(config: dict[str, Any], key: str, value: Any) -> None:
    current = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, value = item.split("=", 1)
        deep_set(config, key, parse_value(value))
    return config


def as_tensor(value: Any, dtype: torch.dtype | None = None) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        out = value
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def extract_raw(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    if isinstance(payload, dict):
        for key in ["episodes", "trajectories", "samples", "items", "data"]:
            if key in payload and isinstance(payload[key], (list, tuple)):
                return list(payload[key])
        if "candidate_edge_idx" in payload:
            return [payload]
    raise ValueError(f"Unsupported tensor dataset payload type: {type(payload)}")


def make_episode(raw: Any, fallback_id: int) -> EpisodeSample:
    if not isinstance(raw, dict):
        if hasattr(raw, "__dict__"):
            raw = vars(raw)
        else:
            raise ValueError(f"Unsupported episode type: {type(raw)}")

    candidate_edge_idx = as_tensor(raw["candidate_edge_idx"], torch.long)
    candidate_mask = as_tensor(raw.get("candidate_mask", candidate_edge_idx >= 0), torch.bool)
    emission_features = as_tensor(raw["emission_features"], torch.float32)
    transition_features = as_tensor(raw.get("transition_features"), torch.float32)
    transition_mask = as_tensor(raw.get("transition_mask"), torch.bool)

    if transition_features is None:
        t_len, k = candidate_edge_idx.shape
        transition_features = torch.zeros(max(t_len - 1, 0), k, k, 0, dtype=torch.float32)

    if transition_mask is None:
        t_len, k = candidate_edge_idx.shape
        transition_mask = torch.ones(max(t_len - 1, 0), k, k, dtype=torch.bool)

    gt_key = "gt_candidate_pos" if "gt_candidate_pos" in raw else "gt_pos"

    return EpisodeSample(
        trajectory_id=int(raw.get("trajectory_id", raw.get("id", fallback_id))),
        candidate_edge_idx=candidate_edge_idx,
        candidate_mask=candidate_mask,
        candidate_proj_xy=as_tensor(raw.get("candidate_proj_xy"), torch.float32),
        emission_features=emission_features,
        transition_features=transition_features,
        transition_mask=transition_mask,
        gt_candidate_pos=as_tensor(raw[gt_key], torch.long).reshape(-1),
        gt_edge_idx=as_tensor(raw.get("gt_edge_idx"), torch.long),
        gt_proj_xy=as_tensor(raw.get("gt_proj_xy"), torch.float32),
        emission_feature_names=list(raw.get("emission_feature_names", [])),
        transition_feature_names=list(raw.get("transition_feature_names", [])),
        timestamps=raw.get("timestamps", None),
    )


def load_dataset(path: Path) -> list[EpisodeSample]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [make_episode(item, idx) for idx, item in enumerate(extract_raw(payload))]


def dataset_summary(dataset: list[EpisodeSample]) -> dict[str, Any]:
    points = sum(ep.length for ep in dataset)
    labelled = sum(int((ep.gt_candidate_pos >= 0).sum().item()) for ep in dataset)
    return {
        "episodes": len(dataset),
        "points": points,
        "labelled_points": labelled,
        "min_length": min(ep.length for ep in dataset),
        "max_length": max(ep.length for ep in dataset),
        "max_candidates": max(ep.num_candidates for ep in dataset),
        "emission_feature_dim": int(dataset[0].emission_features.shape[-1]),
        "transition_feature_dim": int(dataset[0].transition_features.shape[-1]),
    }


def feature_index(names: list[str], name: str, fallback: int | None = None) -> int | None:
    if name in names:
        return names.index(name)
    return fallback


def get_feature(features: Tensor, names: list[str], name: str, fallback: int | None = None) -> Tensor:
    idx = feature_index(names, name, fallback)
    if idx is None or idx >= features.shape[-1]:
        return torch.zeros(features.shape[:-1], dtype=features.dtype)
    return features[..., idx]


def params_from_config(config: dict[str, Any]) -> HMMParams:
    emission = config.get("emission", {})
    transition = config.get("transition", {})
    return HMMParams(
        emission_scale=float(emission.get("emission_scale", 1.0)),
        distance_weight=float(emission.get("distance_weight", 3.0)),
        log_distance_weight=float(emission.get("log_distance_weight", 0.8)),
        yaw_weight=float(emission.get("yaw_weight", 1.2)),
        rank_weight=float(emission.get("rank_weight", 1.2)),
        speed_consistency_weight=float(emission.get("speed_consistency_weight", 0.15)),
        oneway_weight=float(emission.get("oneway_weight", 0.05)),
        yaw_reliability_weight=float(emission.get("yaw_reliability_weight", 0.35)),
        bias=float(emission.get("bias", 0.0)),
        transition_scale=float(transition.get("transition_scale", 0.35)),
        legal_bonus=float(transition.get("legal_bonus", 0.35)),
        illegal_penalty=float(transition.get("illegal_penalty", 2.0)),
        same_edge_bonus=float(transition.get("same_edge_bonus", 0.35)),
        same_osm_way_bonus=float(transition.get("same_osm_way_bonus", 0.15)),
        same_road_class_bonus=float(transition.get("same_road_class_bonus", 0.05)),
        route_distance_weight=float(transition.get("route_distance_weight", 0.25)),
        route_gps_ratio_weight=float(transition.get("route_gps_ratio_weight", 0.10)),
        route_minus_gps_weight=float(transition.get("route_minus_gps_weight", 0.15)),
        turn_weight=float(transition.get("turn_weight", 0.10)),
        yaw_change_weight=float(transition.get("yaw_change_weight", 0.05)),
        time_feasible_bonus=float(transition.get("time_feasible_bonus", 0.10)),
        time_infeasible_penalty=float(transition.get("time_infeasible_penalty", 0.50)),
        rank_delta_weight=float(transition.get("rank_delta_weight", 0.05)),
        distance_delta_weight=float(transition.get("distance_delta_weight", 0.05)),
    )


def compute_emission_scores(sample: EpisodeSample, params: HMMParams) -> Tensor:
    f = sample.emission_features.float()
    names = sample.emission_feature_names

    distance = get_feature(f, names, "distance_norm", 0).clamp_min(0.0)
    log_distance = get_feature(f, names, "log_distance_norm", 1).clamp_min(0.0)
    abs_yaw = get_feature(f, names, "abs_yaw_diff_norm", 3).clamp_min(0.0)
    rank = get_feature(f, names, "candidate_rank_norm", 10).clamp_min(0.0)
    speed_consistency = get_feature(f, names, "speed_consistency", 11).clamp_min(0.0)
    oneway = get_feature(f, names, "oneway", 12).clamp_min(0.0)
    yaw_reliability = get_feature(f, names, "yaw_reliability", 15).clamp(0.0, 1.0)

    yaw_factor = params.yaw_reliability_weight + (1.0 - params.yaw_reliability_weight) * yaw_reliability
    yaw_penalty = abs_yaw * yaw_factor

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


def compute_transition_scores(sample: EpisodeSample, params: HMMParams, mode: str) -> Tensor:
    tfeat = sample.transition_features.float()
    if tfeat.numel() == 0:
        return torch.empty(0)

    names = sample.transition_feature_names

    route_dist = get_feature(tfeat, names, "route_dist_norm", 3).clamp_min(0.0)
    route_minus_gps = get_feature(tfeat, names, "route_minus_gps_norm", 4).clamp_min(0.0)
    route_ratio = get_feature(tfeat, names, "route_gps_ratio_norm", 5).clamp_min(0.0)
    turn = get_feature(tfeat, names, "turn_norm", 6).clamp_min(0.0)
    yaw_change = get_feature(tfeat, names, "yaw_change_norm", 7).clamp_min(0.0)

    connected = get_feature(tfeat, names, "connected", 9)
    legal = get_feature(tfeat, names, "legal", 10)
    same_edge = get_feature(tfeat, names, "same_edge", 11)
    same_osm_way = get_feature(tfeat, names, "same_osm_way", 12)
    same_road_class = get_feature(tfeat, names, "same_road_class", 13)
    rank_delta = get_feature(tfeat, names, "candidate_rank_delta_norm", 15).clamp_min(0.0)
    distance_delta = get_feature(tfeat, names, "distance_delta_norm", 18).clamp_min(0.0)
    time_feasible = get_feature(tfeat, names, "time_feasible", 19).clamp(0.0, 1.0)

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

    if mode == "hard":
        score = score.masked_fill(~mask, MASK_VALUE)
    elif mode == "soft":
        score = score - (~mask).float() * (params.transition_scale * params.illegal_penalty)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported transition mode: {mode}")

    return score


def viterbi_decode(emissions: Tensor, transitions: Tensor) -> tuple[list[int], Tensor]:
    if emissions.ndim != 2:
        raise ValueError("emissions must have shape [T, K].")

    t_len, _ = emissions.shape
    if t_len == 0:
        return [], torch.empty(0)

    dp = emissions[0].clone()
    history = [dp.clone()]
    backpointers: list[Tensor] = []

    for t in range(1, t_len):
        scores = dp[:, None] + transitions[t - 1] if transitions.numel() else dp[:, None]
        best_scores, best_prev = scores.max(dim=0)
        dp = emissions[t] + best_scores
        backpointers.append(best_prev)
        history.append(dp.clone())

    last = int(dp.argmax().item())
    path = [last]

    for bp in reversed(backpointers):
        last = int(bp[last].item())
        path.append(last)

    path.reverse()
    return path, torch.stack(history)


def confidence_from_scores(scores: Tensor, path: list[int], temperature: float) -> list[float]:
    if not path:
        return []
    temp = max(float(temperature), 1.0e-6)
    probs = torch.softmax(scores / temp, dim=-1)
    return [float(probs[t, a].item()) for t, a in enumerate(path)]


def decode_dataset(
    dataset: list[EpisodeSample],
    params: HMMParams,
    transition_mode: str,
    confidence_temperature: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for sample in dataset:
        emissions = compute_emission_scores(sample, params)
        transitions = compute_transition_scores(sample, params, transition_mode)
        path, score_history = viterbi_decode(emissions, transitions)
        confidences = confidence_from_scores(score_history, path, confidence_temperature)

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
                "confidence": float(confidences[t]) if t < len(confidences) else 0.0,
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


def levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = curr
    return prev[-1]


def evaluate_matches(
    matches: pd.DataFrame,
    projection_threshold_m: float,
    trajectory_success_accuracy: float,
    require_gt_candidate: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    df = matches.copy()

    labelled = df[df["gt_edge_idx"] >= 0].copy()
    if require_gt_candidate:
        labelled = labelled[labelled["gt_candidate_pos"] >= 0].copy()

    if len(labelled) == 0:
        raise RuntimeError("No labelled points found.")

    labelled["edge_correct"] = labelled["pred_edge_idx"].astype(int) == labelled["gt_edge_idx"].astype(int)
    labelled["action_correct"] = labelled["pred_candidate_pos"].astype(int) == labelled["gt_candidate_pos"].astype(int)

    if {"pred_proj_x", "pred_proj_y", "gt_proj_x", "gt_proj_y"}.issubset(labelled.columns):
        labelled["projection_error_m"] = np.sqrt(
            (labelled["pred_proj_x"] - labelled["gt_proj_x"]) ** 2
            + (labelled["pred_proj_y"] - labelled["gt_proj_y"]) ** 2
        )
    else:
        labelled["projection_error_m"] = np.nan

    labelled["projection_success"] = labelled["projection_error_m"] <= projection_threshold_m
    labelled["within_2m"] = labelled["projection_error_m"] <= 2.0
    labelled["within_5m"] = labelled["projection_error_m"] <= 5.0
    labelled["within_10m"] = labelled["projection_error_m"] <= 10.0
    labelled["near_but_wrong_edge"] = (~labelled["edge_correct"]) & labelled["within_5m"]

    trajectory_rows: list[dict[str, Any]] = []
    path_edits = []

    for tid, g in labelled.groupby("trajectory_id"):
        g = g.sort_values("t")
        pred_seq = g["pred_edge_idx"].astype(int).tolist()
        gt_seq = g["gt_edge_idx"].astype(int).tolist()
        edit = levenshtein(pred_seq, gt_seq)
        edge_acc = float(g["edge_correct"].mean())
        path_edits.append(edit)

        trajectory_rows.append(
            {
                "trajectory_id": int(tid),
                "points": int(len(g)),
                "edge_accuracy": edge_acc,
                "action_accuracy": float(g["action_correct"].mean()),
                "mean_projection_error_m": float(g["projection_error_m"].mean()),
                "within_5m_rate": float(g["within_5m"].mean()),
                "path_edit_distance": int(edit),
                "success": bool(edge_acc >= trajectory_success_accuracy),
            }
        )

    trajectory_df = pd.DataFrame(trajectory_rows)
    error_cases = labelled[~labelled["edge_correct"]].copy()

    metrics = {
        "num_points": int(len(df)),
        "num_labelled_points": int(len(labelled)),
        "num_unlabelled_or_gt_missing_points": int(len(df) - len(labelled)),
        "num_trajectories": int(labelled["trajectory_id"].nunique()),
        "point_action_accuracy": float(labelled["action_correct"].mean()),
        "point_edge_accuracy": float(labelled["edge_correct"].mean()),
        "mean_projection_error_m": float(labelled["projection_error_m"].mean()),
        "median_projection_error_m": float(labelled["projection_error_m"].median()),
        "p90_projection_error_m": float(labelled["projection_error_m"].quantile(0.90)),
        "within_2m_rate": float(labelled["within_2m"].mean()),
        "within_5m_rate": float(labelled["within_5m"].mean()),
        "within_10m_rate": float(labelled["within_10m"].mean()),
        "projection_success_rate": float(labelled["projection_success"].mean()),
        "mean_confidence": float(labelled["confidence"].mean()) if "confidence" in labelled.columns else float("nan"),
        "path_edit_distance_mean": float(np.mean(path_edits)),
        "path_edit_distance_median": float(np.median(path_edits)),
        "trajectory_success_rate": float(trajectory_df["success"].mean()),
        "num_error_points": int((~labelled["edge_correct"]).sum()),
        "error_near_but_wrong_edge_rate": float(error_cases["near_but_wrong_edge"].mean()) if len(error_cases) else 0.0,
    }

    return metrics, trajectory_df, error_cases


def product_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def params_with_overrides(base: HMMParams, overrides: dict[str, Any]) -> HMMParams:
    values = asdict(base)
    values.update(overrides)
    return HMMParams(**values)


def run_check(config: dict[str, Any]) -> None:
    for split in ["train", "val", "test"]:
        dataset = load_dataset(Path(deep_get(config, f"paths.{split}_dataset")))
        print(f"[OK] {split}: {dataset_summary(dataset)}", flush=True)


def run_decode(config: dict[str, Any], split: str, params: HMMParams | None = None) -> Path:
    dataset = load_dataset(Path(deep_get(config, f"paths.{split}_dataset")))
    params = params or params_from_config(config)

    matches = decode_dataset(
        dataset=dataset,
        params=params,
        transition_mode=str(deep_get(config, "decode.transition_mode", "soft")),
        confidence_temperature=float(deep_get(config, "decode.confidence_temperature", 1.0)),
    )

    match_dir = Path(deep_get(config, "paths.match_dir", "HMM/outputs/matches"))
    ensure_dir(match_dir)
    out = match_dir / f"hmm_matches_{split}.parquet"
    matches.to_parquet(out, index=False)
    print(f"[OK] Decoded {split}: {out}", flush=True)
    return out


def run_evaluate(config: dict[str, Any], split: str) -> dict[str, Any]:
    match_path = Path(deep_get(config, "paths.match_dir", "HMM/outputs/matches")) / f"hmm_matches_{split}.parquet"
    if not match_path.exists():
        raise FileNotFoundError(f"Missing matches file. Run decode first: {match_path}")

    matches = pd.read_parquet(match_path)

    metrics, trajectories, errors = evaluate_matches(
        matches,
        projection_threshold_m=float(deep_get(config, "evaluation.projection_threshold_m", 10.0)),
        trajectory_success_accuracy=float(deep_get(config, "evaluation.trajectory_success_accuracy", 0.90)),
        require_gt_candidate=bool(deep_get(config, "evaluation.require_gt_candidate", True)),
    )

    metric_dir = Path(deep_get(config, "paths.metric_dir", "HMM/outputs/metrics"))
    ensure_dir(metric_dir)

    metric_path = metric_dir / f"hmm_metrics_{split}.json"
    traj_path = metric_dir / f"hmm_trajectory_metrics_{split}.csv"
    error_path = metric_dir / f"hmm_error_cases_{split}.csv"

    with metric_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)

    trajectories.to_csv(traj_path, index=False)
    errors.to_csv(error_path, index=False)

    print("[OK] HMM evaluation complete", flush=True)
    for key, value in metrics.items():
        print(f"{key}: {value}", flush=True)

    return metrics


def run_tune(config: dict[str, Any]) -> None:
    split = str(deep_get(config, "grid_search.split", "val"))
    dataset = load_dataset(Path(deep_get(config, f"paths.{split}_dataset")))

    base_params = params_from_config(config)

    grid_config = deep_get(config, "grid_search", {})
    ignored = {"split", "max_trials"}
    grid = {k: v for k, v in grid_config.items() if k not in ignored and isinstance(v, list)}

    trials = product_grid(grid)
    random.seed(int(deep_get(config, "project.seed", 42)))
    random.shuffle(trials)

    max_trials = deep_get(config, "grid_search.max_trials", None)
    if max_trials is not None:
        trials = trials[: int(max_trials)]

    metric_dir = Path(deep_get(config, "paths.metric_dir", "HMM/outputs/metrics"))
    ensure_dir(metric_dir)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1.0

    for idx, overrides in enumerate(trials, start=1):
        params = params_with_overrides(base_params, overrides)
        matches = decode_dataset(
            dataset=dataset,
            params=params,
            transition_mode=str(deep_get(config, "decode.transition_mode", "soft")),
            confidence_temperature=float(deep_get(config, "decode.confidence_temperature", 1.0)),
        )
        metrics, _, _ = evaluate_matches(
            matches,
            projection_threshold_m=float(deep_get(config, "evaluation.projection_threshold_m", 10.0)),
            trajectory_success_accuracy=float(deep_get(config, "evaluation.trajectory_success_accuracy", 0.90)),
            require_gt_candidate=bool(deep_get(config, "evaluation.require_gt_candidate", True)),
        )

        score = float(metrics["point_edge_accuracy"])
        rows.append({"trial": idx, **overrides, **metrics})

        print(f"[grid] {idx}/{len(trials)} point_edge_accuracy={score:.6f}", flush=True)

        if score > best_score:
            best_score = score
            best = {
                "params": asdict(params),
                "overrides": overrides,
                "metrics": metrics,
            }

    pd.DataFrame(rows).to_csv(metric_dir / "hmm_grid_search.csv", index=False)

    with (metric_dir / "hmm_best_params.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(best or {}, f, indent=2)

    print(f"[OK] HMM grid tuning complete. Best point_edge_accuracy={best_score:.6f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file classical HMM/Viterbi workflow for One-Direction.")
    parser.add_argument("stage", nargs="?", default="all", choices=["check", "decode", "evaluate", "all", "tune"])
    parser.add_argument("--config", type=Path, default=Path("HMM/configs/hmm_default.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.override)
    split = args.split or str(deep_get(config, "decode.split", "test"))

    if args.stage == "check":
        run_check(config)
    elif args.stage == "decode":
        run_decode(config, split)
    elif args.stage == "evaluate":
        run_evaluate(config, split)
    elif args.stage == "all":
        run_check(config)
        run_decode(config, split)
        run_evaluate(config, split)
    elif args.stage == "tune":
        run_tune(config)
    else:
        raise ValueError(args.stage)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
