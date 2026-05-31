from __future__ import annotations
import itertools, json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from .data import HMMDataset
from .decode import decode_dataset
from .evaluate import evaluate_matches
from .scoring import HMMParams

def product_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]

def params_with_overrides(base: HMMParams, overrides: dict) -> HMMParams:
    d = asdict(base)
    d.update(overrides)
    return HMMParams(**d)

def tune_grid(dataset: HMMDataset, base_params: HMMParams, grid: dict, transition_mode: str, max_trials: int | None, output_csv: str | Path, output_best_json: str | Path):
    trials = product_grid(grid)
    if max_trials is not None:
        trials = trials[: int(max_trials)]
    rows, best, best_score = [], None, -1.0
    for idx, overrides in enumerate(trials, start=1):
        params = params_with_overrides(base_params, overrides)
        matches = decode_dataset(dataset, params, transition_mode=transition_mode)
        metrics, _, _ = evaluate_matches(matches)
        score = float(metrics["point_edge_accuracy"])
        row = {"trial": idx, **overrides, **metrics}
        rows.append(row)
        print(f"[grid] {idx}/{len(trials)} point_edge_accuracy={score:.6f}", flush=True)
        if score > best_score:
            best_score = score
            best = {"params": asdict(params), "metrics": metrics, "overrides": overrides}
    df = pd.DataFrame(rows)
    output_csv, output_best_json = Path(output_csv), Path(output_best_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    with output_best_json.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(best, f, indent=2)
    return df, best or {}
