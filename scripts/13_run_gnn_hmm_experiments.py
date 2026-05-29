from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--weights", nargs="+", type=float, default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary", type=Path, default=Path("outputs/metrics/gnn_hmm_experiment_summary.csv"))
    args = parser.parse_args()

    rows = []
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    for w in args.weights:
        tag = f"tw{str(w).replace('.', 'p')}"
        ckpt_dir = Path("outputs/checkpoints") / tag
        metrics_path = Path("outputs/metrics") / f"gnn_hmm_metrics_{tag}.json"
        matches_path = Path("outputs/matches") / f"gnn_hmm_matches_{tag}.parquet"

        run([
            sys.executable,
            "scripts/07_train_gnn_hmm.py",
            "--output", str(ckpt_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--transition-weight", str(w),
            "--device", args.device,
        ])

        run([
            sys.executable,
            "scripts/08_decode_gnn_hmm.py",
            "--checkpoint", str(ckpt_dir / "gnn_hmm_best.pt"),
            "--output", str(matches_path),
            "--device", args.device,
        ])

        run([
            sys.executable,
            "scripts/09_evaluate.py",
            "--pred", str(matches_path),
            "--output", str(metrics_path),
            "--error-cases", str(Path("outputs/metrics") / f"error_cases_{tag}.csv"),
        ])

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        row = {"transition_weight": w, **metrics}
        rows.append(row)

        with args.summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*[r.keys() for r in rows])))
            writer.writeheader()
            writer.writerows(rows)

    print("[OK] Wrote experiment summary:", args.summary)


if __name__ == "__main__":
    main()
