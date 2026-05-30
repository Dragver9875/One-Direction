from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = {
    "check": ["D-SAC/scripts/00_check_inputs.py"],
    "bc": ["D-SAC/scripts/01_train_bc_actor.py"],
    "train": ["D-SAC/scripts/02_train_discrete_sac_asym.py"],
    "eval": ["D-SAC/scripts/03_evaluate_discrete_sac.py"],
}
ALIASES = {"all": ["check", "bc", "train", "eval"], "smoke": ["check", "bc", "train", "eval"]}


def resolve(items):
    out = []
    for item in items:
        if item in ALIASES:
            out.extend(ALIASES[item])
        elif item in STAGES:
            out.append(item)
        else:
            raise ValueError(f"Unknown D-SAC stage: {item}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="*", default=["all"])
    parser.add_argument("--config", default="D-SAC/configs/dsac_default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("Stages:")
        for key in STAGES:
            print(f"  {key}")
        print("Aliases:")
        for key, value in ALIASES.items():
            print(f"  {key}: {', '.join(value)}")
        return 0
    repo_root = Path(__file__).resolve().parents[2]
    for stage in resolve(args.stages):
        cmd = [sys.executable, "-u", *STAGES[stage], "--config", args.config]
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=repo_root)
            if result.returncode != 0:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
