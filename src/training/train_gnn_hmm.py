from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpointing import CheckpointConfig, CheckpointManager
from .losses import GNNHMMLossConfig, compute_total_loss
from .scheduler import OptimizerConfig, SchedulerConfig, build_optimizer, build_scheduler, current_lr, scheduler_step
from .validation import ValidationConfig, validate_epoch


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 50
    device: str = "auto"
    amp: bool = False
    grad_clip_norm: float | None = 5.0
    log_every_steps: int = 20
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    loss: GNNHMMLossConfig = GNNHMMLossConfig()
    checkpointing: CheckpointConfig = CheckpointConfig()
    validation: ValidationConfig = ValidationConfig()


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


class GNNHMMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        road_data,
        cfg: TrainerConfig = TrainerConfig(),
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.road_data = road_data
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

        self.model.to(self.device)

        if hasattr(self.road_data, "to"):
            self.road_data = self.road_data.to(self.device)
        elif isinstance(self.road_data, Tensor):
            self.road_data = self.road_data.to(self.device)

        self.optimizer = build_optimizer(self.model, cfg.optimizer)
        self.scheduler = build_scheduler(self.optimizer, cfg.scheduler)
        self.checkpoints = CheckpointManager(cfg.checkpointing)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and self.device.type == "cuda")

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        running_loss = 0.0
        running_emission_loss = 0.0
        running_transition_loss = 0.0
        num_batches = 0

        progress = tqdm(
            self.train_loader,
            desc=f"train epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.amp and self.device.type == "cuda"):
                outputs = self.model(
                    road_x_or_data=self.road_data,
                    candidate_edge_idx=batch["candidate_edge_idx"],
                    emission_features=batch["emission_features"],
                    prev_edge_idx=batch["prev_edge_idx"],
                    curr_edge_idx=batch["curr_edge_idx"],
                    transition_features=batch["transition_features"],
                    candidate_mask=batch.get("candidate_mask"),
                    transition_mask=batch.get("transition_mask"),
                )

                losses = compute_total_loss(
                    outputs=outputs,
                    batch=batch,
                    model=self.model,
                    cfg=self.cfg.loss,
                )
                loss = losses["loss"]

            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.cfg.grad_clip_norm),
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.detach().item())
            running_emission_loss += float(losses["emission_loss"].item())
            running_transition_loss += float(losses["transition_loss"].item())
            num_batches += 1

            if self.cfg.log_every_steps > 0 and step % self.cfg.log_every_steps == 0:
                progress.set_postfix(
                    loss=running_loss / max(num_batches, 1),
                    lr=current_lr(self.optimizer),
                )

        denom = max(num_batches, 1)
        return {
            "train_loss": running_loss / denom,
            "train_emission_loss": running_emission_loss / denom,
            "train_transition_loss": running_transition_loss / denom,
            "lr": current_lr(self.optimizer),
        }

    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        return validate_epoch(
            model=self.model,
            dataloader=self.val_loader,
            road_data=self.road_data,
            device=self.device,
            cfg=self.cfg.validation,
            loss_cfg=self.cfg.loss,
        )

    def fit(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            metrics = {**train_metrics, **val_metrics}

            if self.scheduler is not None:
                monitor_value = metrics.get(self.cfg.checkpointing.monitor)
                scheduler_step(self.scheduler, monitor_value)

            saved = self.checkpoints.step(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
                extra={"saved_by": "GNNHMMTrainer"},
            )

            metrics["epoch"] = float(epoch)
            metrics["saved_best"] = float(saved["best"])
            metrics["saved_last"] = float(saved["last"])
            history.append(metrics)

            message = " ".join(
                f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in metrics.items()
                if key not in {"saved_best", "saved_last"}
            )
            print(f"[epoch {epoch}] {message}")

        return history
