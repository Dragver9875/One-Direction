from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass
class EpisodeSample:
    trajectory_id: int
    candidate_edge_idx: Tensor
    candidate_mask: Tensor
    emission_features: Tensor
    gt_candidate_pos: Tensor
    transition_mask: Tensor | None
    candidate_proj_xy: Tensor | None
    gt_proj_xy: Tensor | None
    gt_edge_idx: Tensor | None
    timestamps: list[Any] | None

    @property
    def length(self) -> int:
        return int(self.candidate_edge_idx.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_edge_idx.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.emission_features.shape[-1])


class RLDataset:
    def __init__(self, episodes: list[EpisodeSample]) -> None:
        if not episodes:
            raise ValueError("RLDataset received no episodes.")
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> EpisodeSample:
        return self.episodes[index]

    def summary(self) -> dict:
        points = sum(ep.length for ep in self.episodes)
        labelled = sum(int((ep.gt_candidate_pos >= 0).sum().item()) for ep in self.episodes)
        return {
            "episodes": len(self.episodes),
            "points": points,
            "labelled_points": labelled,
            "min_length": min(ep.length for ep in self.episodes),
            "max_length": max(ep.length for ep in self.episodes),
            "max_candidates": max(ep.num_candidates for ep in self.episodes),
            "feature_dim": self.episodes[0].feature_dim,
        }


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> Tensor | None:
    if value is None:
        return None
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _extract_raw_episodes(payload: Any) -> list[Any]:
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


def _make_episode(raw: Any, fallback_id: int) -> EpisodeSample:
    if not isinstance(raw, dict):
        if hasattr(raw, "__dict__"):
            raw = vars(raw)
        else:
            raise ValueError(f"Unsupported episode type: {type(raw)}")

    candidate_edge_idx = _as_tensor(raw["candidate_edge_idx"], torch.long)
    candidate_mask = _as_tensor(raw.get("candidate_mask", candidate_edge_idx >= 0), torch.bool)
    emission_features = _as_tensor(raw["emission_features"], torch.float32)

    gt_key = "gt_candidate_pos" if "gt_candidate_pos" in raw else "gt_pos"
    gt_candidate_pos = _as_tensor(raw[gt_key], torch.long).reshape(-1)

    timestamps = raw.get("timestamps", raw.get("timestamp", None))
    if isinstance(timestamps, Tensor):
        timestamps = timestamps.detach().cpu().tolist()

    return EpisodeSample(
        trajectory_id=int(raw.get("trajectory_id", raw.get("id", fallback_id))),
        candidate_edge_idx=candidate_edge_idx,
        candidate_mask=candidate_mask,
        emission_features=emission_features,
        gt_candidate_pos=gt_candidate_pos,
        transition_mask=_as_tensor(raw.get("transition_mask"), torch.bool),
        candidate_proj_xy=_as_tensor(raw.get("candidate_proj_xy"), torch.float32),
        gt_proj_xy=_as_tensor(raw.get("gt_proj_xy"), torch.float32),
        gt_edge_idx=_as_tensor(raw.get("gt_edge_idx"), torch.long),
        timestamps=timestamps,
    )


def load_rl_dataset(path: str | Path, map_location: str = "cpu") -> RLDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return RLDataset([_make_episode(item, idx) for idx, item in enumerate(_extract_raw_episodes(payload))])
