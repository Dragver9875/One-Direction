from __future__ import annotations
import argparse
from pathlib import Path
from _bootstrap import add_hmm_src_to_path
add_hmm_src_to_path()
from onedir_hmm.config import deep_get, load_config
from onedir_hmm.data import load_hmm_dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("HMM/configs/hmm_default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    for split in ["train", "val", "test"]:
        ds = load_hmm_dataset(deep_get(cfg, f"paths.{split}_dataset"))
        print(f"[OK] {split}: {ds.summary()}", flush=True)

if __name__ == "__main__":
    main()
