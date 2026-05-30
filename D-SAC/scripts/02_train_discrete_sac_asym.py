from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import add_dsac_src_to_path

add_dsac_src_to_path()

from onedir_dsac.config import apply_overrides, deep_get, load_config, resolve_device
from onedir_dsac.data import load_rl_dataset
from onedir_dsac.dsac import train_discrete_sac_asym
from onedir_dsac.reward import RewardConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("D-SAC/configs/dsac_default.yaml"))
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(deep_get(cfg, "project.device", "auto"))
    train_dataset = load_rl_dataset(deep_get(cfg, "paths.train_dataset"))
    val_dataset = load_rl_dataset(deep_get(cfg, "paths.val_dataset"))
    reward_cfg = RewardConfig(**{k: float(deep_get(cfg, f"reward.{k}", getattr(RewardConfig(), k))) for k in RewardConfig.__dataclass_fields__})
    train_discrete_sac_asym(train_dataset, val_dataset, output_dir=deep_get(cfg, "paths.output_dir", "D-SAC/outputs"), device=device, seed=int(deep_get(cfg, "project.seed", 42)), hidden_dim=int(deep_get(cfg, "model.hidden_dim", 256)), num_layers=int(deep_get(cfg, "model.num_layers", 2)), dropout=float(deep_get(cfg, "model.dropout", 0.1)), use_privileged_critic=bool(deep_get(cfg, "features.use_privileged_critic", True)), epochs=int(deep_get(cfg, "dsac.epochs", 60)), rollouts_per_epoch=int(deep_get(cfg, "dsac.rollouts_per_epoch", 200)), updates_per_epoch=int(deep_get(cfg, "dsac.updates_per_epoch", 800)), batch_size=int(deep_get(cfg, "dsac.batch_size", 512)), replay_size=int(deep_get(cfg, "dsac.replay_size", 200000)), warmup_rollouts=int(deep_get(cfg, "dsac.warmup_rollouts", 200)), gamma=float(deep_get(cfg, "dsac.gamma", 0.98)), tau=float(deep_get(cfg, "dsac.tau", 0.005)), actor_lr=float(deep_get(cfg, "dsac.actor_lr", 3e-4)), critic_lr=float(deep_get(cfg, "dsac.critic_lr", 3e-4)), alpha_lr=float(deep_get(cfg, "dsac.alpha_lr", 3e-4)), init_alpha=float(deep_get(cfg, "dsac.init_alpha", 0.2)), auto_alpha=bool(deep_get(cfg, "dsac.auto_alpha", True)), target_entropy_scale=float(deep_get(cfg, "dsac.target_entropy_scale", 0.7)), grad_clip_norm=float(deep_get(cfg, "dsac.grad_clip_norm", 5.0)), reward_cfg=reward_cfg, load_bc_checkpoint=bool(deep_get(cfg, "dsac.load_bc_checkpoint", True)), bc_checkpoint=deep_get(cfg, "bc.checkpoint", "D-SAC/outputs/checkpoints/bc_actor.pt"), k_max=deep_get(cfg, "features.max_candidates", None))


if __name__ == "__main__":
    main()
