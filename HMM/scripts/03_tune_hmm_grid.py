from __future__ import annotations
import argparse
from pathlib import Path
from _bootstrap import add_hmm_src_to_path
add_hmm_src_to_path()
from onedir_hmm.config import apply_overrides, deep_get, load_config
from onedir_hmm.data import load_hmm_dataset
from onedir_hmm.scoring import params_from_config
from onedir_hmm.tune import tune_grid

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("HMM/configs/hmm_default.yaml"))
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    split = deep_get(cfg, "grid_search.split", "val")
    ds = load_hmm_dataset(deep_get(cfg, f"paths.{split}_dataset"))
    grid_cfg = deep_get(cfg, "grid_search", {})
    grid = {k: v for k, v in grid_cfg.items() if k not in {"split", "max_trials"} and isinstance(v, list)}
    metric_dir = Path(deep_get(cfg, "paths.metric_dir", "HMM/outputs/metrics"))
    tune_grid(ds, params_from_config(cfg), grid, deep_get(cfg, "decode.transition_mode", "soft"), deep_get(cfg, "grid_search.max_trials", None), metric_dir / "hmm_grid_search.csv", metric_dir / "hmm_best_params.json")
    print("[OK] HMM grid tuning complete", flush=True)

if __name__ == "__main__":
    main()
