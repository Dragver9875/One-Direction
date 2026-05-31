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

class HMMDataset:
    def __init__(self, episodes: list[EpisodeSample]) -> None:
        if not episodes:
            raise ValueError("HMMDataset received no episodes.")
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> EpisodeSample:
        return self.episodes[index]

    def summary(self) -> dict:
        return {
            "episodes": len(self.episodes),
            "points": sum(ep.length for ep in self.episodes),
            "labelled_points": sum(int((ep.gt_candidate_pos >= 0).sum().item()) for ep in self.episodes),
            "min_length": min(ep.length for ep in self.episodes),
            "max_length": max(ep.length for ep in self.episodes),
            "max_candidates": max(ep.num_candidates for ep in self.episodes),
            "emission_feature_dim": int(self.episodes[0].emission_features.shape[-1]),
            "transition_feature_dim": int(self.episodes[0].transition_features.shape[-1]),
        }

def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> Tensor | None:
    if value is None:
        return None
    x = value if isinstance(value, Tensor) else torch.as_tensor(value)
    return x.to(dtype=dtype) if dtype is not None else x

def _extract(payload: Any) -> list[Any]:
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

def _episode(raw: Any, fallback_id: int) -> EpisodeSample:
    if not isinstance(raw, dict):
        if hasattr(raw, "__dict__"):
            raw = vars(raw)
        else:
            raise ValueError(f"Unsupported episode type: {type(raw)}")

    edge_idx = _as_tensor(raw["candidate_edge_idx"], torch.long)
    mask = _as_tensor(raw.get("candidate_mask", edge_idx >= 0), torch.bool)
    emissions = _as_tensor(raw["emission_features"], torch.float32)
    trans = _as_tensor(raw.get("transition_features"), torch.float32)
    trans_mask = _as_tensor(raw.get("transition_mask"), torch.bool)

    if trans is None:
        t, k = edge_idx.shape
        trans = torch.zeros(max(t - 1, 0), k, k, 0, dtype=torch.float32)
    if trans_mask is None:
        t, k = edge_idx.shape
        trans_mask = torch.ones(max(t - 1, 0), k, k, dtype=torch.bool)

    gt_key = "gt_candidate_pos" if "gt_candidate_pos" in raw else "gt_pos"

    return EpisodeSample(
        trajectory_id=int(raw.get("trajectory_id", raw.get("id", fallback_id))),
        candidate_edge_idx=edge_idx,
        candidate_mask=mask,
        candidate_proj_xy=_as_tensor(raw.get("candidate_proj_xy"), torch.float32),
        emission_features=emissions,
        transition_features=trans,
        transition_mask=trans_mask,
        gt_candidate_pos=_as_tensor(raw[gt_key], torch.long).reshape(-1),
        gt_edge_idx=_as_tensor(raw.get("gt_edge_idx"), torch.long),
        gt_proj_xy=_as_tensor(raw.get("gt_proj_xy"), torch.float32),
        emission_feature_names=list(raw.get("emission_feature_names", [])),
        transition_feature_names=list(raw.get("transition_feature_names", [])),
        timestamps=raw.get("timestamps", None),
    )

def load_hmm_dataset(path: str | Path) -> HMMDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return HMMDataset([_episode(item, i) for i, item in enumerate(_extract(payload))])
