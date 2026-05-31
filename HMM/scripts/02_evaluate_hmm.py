from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_hmm_src_to_path

add_hmm_src_to_path()

from onedir_hmm.config import apply_overrides, deep_get, load_config
from onedir_hmm.evaluate import evaluate_matches, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('HMM/configs/hmm_default.yaml'))
    parser.add_argument('--split', choices=['train', 'val', 'test'], default=None)
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    split = args.split or deep_get(cfg, 'decode.split', 'test')

    match_path = Path(deep_get(cfg, 'paths.match_dir', 'HMM/outputs/matches')) / f'hmm_matches_{split}.parquet'
    if not match_path.exists():
        raise FileNotFoundError(f'Run 01_decode_hmm.py first. Missing: {match_path}')

    matches = pd.read_parquet(match_path)

    metrics, traj, errors = evaluate_matches(
        matches,
        projection_threshold_m=float(deep_get(cfg, 'evaluation.projection_threshold_m', 10.0)),
        trajectory_success_accuracy=float(deep_get(cfg, 'evaluation.trajectory_success_accuracy', 0.90)),
        require_gt_candidate=bool(deep_get(cfg, 'evaluation.require_gt_candidate', True)),
    )

    metric_dir = Path(deep_get(cfg, 'paths.metric_dir', 'HMM/outputs/metrics'))
    save_metrics(
        metrics=metrics,
        trajectory_metrics=traj,
        error_cases=errors,
        metric_path=metric_dir / f'hmm_metrics_{split}.json',
        trajectory_path=metric_dir / f'hmm_trajectory_metrics_{split}.csv',
        error_path=metric_dir / f'hmm_error_cases_{split}.csv',
    )

    print('[OK] HMM evaluation complete', flush=True)
    for key, value in metrics.items():
        print(f'{key}: {value}', flush=True)


if __name__ == '__main__':
    main()
