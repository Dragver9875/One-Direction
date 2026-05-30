from __future__ import annotations

import argparse
from pathlib import Path
from _bootstrap import add_dsac_src_to_path

add_dsac_src_to_path()

from onedir_dsac.config import deep_get, load_config
from onedir_dsac.data import load_rl_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("D-SAC/configs/dsac_default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    for split in ["train", "val", "test"]:
        dataset = load_rl_dataset(deep_get(cfg, f"paths.{split}_dataset"))
        print(f"[OK] {split}: {dataset.summary()}", flush=True)


if __name__ == "__main__":
    main()
