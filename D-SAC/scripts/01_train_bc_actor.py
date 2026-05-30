from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import add_dsac_src_to_path

add_dsac_src_to_path()

from onedir_dsac.bc import train_behavior_cloning
from onedir_dsac.config import apply_overrides, deep_get, load_config, resolve_device
from onedir_dsac.data import load_rl_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("D-SAC/configs/dsac_default.yaml"))
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    device = resolve_device(deep_get(cfg, "project.device", "auto"))
    train_dataset = load_rl_dataset(deep_get(cfg, "paths.train_dataset"))
    val_dataset = load_rl_dataset(deep_get(cfg, "paths.val_dataset"))
    train_behavior_cloning(train_dataset, val_dataset, deep_get(cfg, "bc.checkpoint", "D-SAC/outputs/checkpoints/bc_actor.pt"), device=device, epochs=int(deep_get(cfg, "bc.epochs", 10)), batch_size=int(deep_get(cfg, "bc.batch_size", 512)), lr=float(deep_get(cfg, "bc.lr", 1e-3)), hidden_dim=int(deep_get(cfg, "model.hidden_dim", 256)), num_layers=int(deep_get(cfg, "model.num_layers", 2)), dropout=float(deep_get(cfg, "model.dropout", 0.1)), use_privileged_critic=bool(deep_get(cfg, "features.use_privileged_critic", True)), seed=int(deep_get(cfg, "project.seed", 42)), k_max=deep_get(cfg, "features.max_candidates", None))


if __name__ == "__main__":
    main()
