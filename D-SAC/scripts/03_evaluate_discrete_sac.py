from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import add_dsac_src_to_path

add_dsac_src_to_path()

from onedir_dsac.config import deep_get, load_config, resolve_device
from onedir_dsac.data import load_rl_dataset
from onedir_dsac.evaluate import build_model_from_checkpoint, evaluate_policy, save_eval_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("D-SAC/configs/dsac_default.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("D-SAC/outputs/checkpoints/dsac_asym_best.pt"))
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(deep_get(cfg, "project.device", "auto"))
    split = args.split or deep_get(cfg, "evaluation.split", "test")
    dataset = load_rl_dataset(deep_get(cfg, f"paths.{split}_dataset"))
    model = build_model_from_checkpoint(args.checkpoint, device=device)
    matches, metrics = evaluate_policy(model, dataset, device=device, greedy=bool(deep_get(cfg, "evaluation.greedy", True)), k_max=deep_get(cfg, "features.max_candidates", None))
    match_path = Path(deep_get(cfg, "paths.match_dir", "D-SAC/outputs/matches")) / "dsac_asym_matches.parquet"
    metric_path = Path(deep_get(cfg, "paths.metric_dir", "D-SAC/outputs/metrics")) / "dsac_asym_metrics.json"
    save_eval_outputs(matches, metrics, match_path, metric_path)
    print("[OK] D-SAC evaluation complete", flush=True)
    for key, value in metrics.items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
