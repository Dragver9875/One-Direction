from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import Dataset


class OneDirectionDataset(Dataset):

    def __init__(self, path_or_object: str | Path | dict[str, Any]):
        if isinstance(path_or_object, (str, Path)):
            path = Path(path_or_object)
            if not path.exists():
                raise FileNotFoundError(f"Dataset tensor file not found: {path}")
            obj = torch.load(path, map_location="cpu")
            self.path = path
        elif isinstance(path_or_object, dict):
            obj = path_or_object
            self.path = None
        else:
            raise TypeError(
                "path_or_object must be a path to a torch object or a dataset dict."
            )

        if "samples" not in obj:
            raise ValueError("Dataset object must contain a 'samples' field.")

        self.obj = obj
        self.samples: list[dict[str, Any]] = list(obj["samples"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    @property
    def num_trajectories(self) -> int:
        return len(self.samples)

    @property
    def max_candidates(self) -> int | None:
        value = self.obj.get("max_candidates")
        return int(value) if value is not None else None

    def trajectory_ids(self) -> list[int]:
        return [int(sample["trajectory_id"]) for sample in self.samples]


def one_direction_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def move_sample_to_device(sample: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def iter_samples_on_device(
    batch: Iterable[dict[str, Any]],
    device: torch.device | str,
) -> Iterator[dict[str, Any]]:
    for sample in batch:
        yield move_sample_to_device(sample, device)
