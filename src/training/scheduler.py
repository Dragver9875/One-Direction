from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch.optim import Optimizer


OptimizerName = Literal["adam", "adamw", "sgd"]
SchedulerName = Literal["none", "cosine", "step", "plateau"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: OptimizerName = "adamw"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    momentum: float = 0.9


@dataclass(frozen=True)
class SchedulerConfig:
    name: SchedulerName = "cosine"
    epochs: int = 50
    warmup_epochs: int = 3
    min_lr: float = 1.0e-5
    step_size: int = 15
    gamma: float = 0.5
    monitor_mode: str = "max"


def build_optimizer(
    model: torch.nn.Module,
    cfg: OptimizerConfig = OptimizerConfig(),
) -> Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]

    if cfg.name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )

    if cfg.name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )

    if cfg.name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {cfg.name}")


def build_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig = SchedulerConfig(),
):
    if cfg.name == "none":
        return None

    if cfg.name == "cosine":
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        min_lr_ratio = min(cfg.min_lr / max(base_lrs), 1.0) if base_lrs else 0.0

        def lr_lambda(epoch: int) -> float:
            if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
                return float(epoch + 1) / float(cfg.warmup_epochs)
            denom = max(cfg.epochs - cfg.warmup_epochs, 1)
            progress = float(epoch - cfg.warmup_epochs) / float(denom)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if cfg.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )

    if cfg.name == "plateau":
        mode = "max" if cfg.monitor_mode == "max" else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=cfg.gamma,
            patience=5,
        )

    raise ValueError(f"Unsupported scheduler: {cfg.name}")


def scheduler_step(scheduler, metric: float | None = None) -> None:
    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if metric is None:
            raise ValueError("Plateau scheduler requires a metric.")
        scheduler.step(metric)
    else:
        scheduler.step()


def current_lr(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])
