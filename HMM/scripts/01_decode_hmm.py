from __future__ import annotations
import argparse
from pathlib import Path
from _bootstrap import add_hmm_src_to_path
add_hmm_src_to_path()
from onedir_hmm.config import apply_overrides, deep_get, load_config
from onedir_hmm.data import load_hmm_dataset
from onedir_hmm.decode import decode_dataset, save_matches
from onedir_hmm.scoring import params_from_config

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("HMM/configs/hmm_default.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    split = args.split or deep_get(cfg, "decode.split", "test")
    ds = load_hmm_dataset(deep_get(cfg, f"paths.{split}_dataset"))
    matches = decode_dataset(ds, params_from_config(cfg), deep_get(cfg, "decode.transition_mode", "soft"), float(deep_get(cfg, "decode.confidence_temperature", 1.0)))
    out = Path(deep_get(cfg, "paths.match_dir", "HMM/outputs/matches")) / f"hmm_matches_{split}.parquet"
    save_matches(matches, out)
    print(f"[OK] Decoded {split}: {out}", flush=True)

if __name__ == "__main__":
    main()
