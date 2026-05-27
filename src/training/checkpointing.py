from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: str | Path = "outputs/checkpoints"
    best_name: str = "gnn_hmm_best.pt"
    last_name: str = "gnn_hmm_last.pt"
    monitor: str = "val_point_edge_accuracy"
    mode: str = "max"
    save_best: bool = True
    save_last: bool = True


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)

    model.load_state_dict(payload["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig = CheckpointConfig()) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_value: float | None = None

        if cfg.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'.")

    @property
    def best_path(self) -> Path:
        return self.output_dir / self.cfg.best_name

    @property
    def last_path(self) -> Path:
        return self.output_dir / self.cfg.last_name

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.cfg.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        saved = {"best": False, "last": False}

        if self.cfg.save_last:
            save_checkpoint(
                path=self.last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                extra=extra,
            )
            saved["last"] = True

        monitor_value = metrics.get(self.cfg.monitor)

        if self.cfg.save_best and monitor_value is not None and self.is_better(float(monitor_value)):
            self.best_value = float(monitor_value)
            save_checkpoint(
                path=self.best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                extra=extra,
            )
            saved["best"] = True

        return saved
