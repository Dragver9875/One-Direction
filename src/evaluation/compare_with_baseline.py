from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .metrics import EvaluationConfig, evaluate_predictions


@dataclass(frozen=True)
class BaselineComparisonConfig:
    baseline_dir: str | Path = "model_outputs"
    output_csv: str | Path = "outputs/metrics/comparison_report.csv"
    output_json: str | Path = "outputs/metrics/comparison_report.json"
    file_glob: str = "*_matches.parquet"


def load_prediction_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported prediction file: {path}")


def model_name_from_path(path: str | Path) -> str:
    path = Path(path)
    name = path.stem
    suffix = "_matches"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def compare_against_baselines(
    one_direction_pred: pd.DataFrame,
    gt: pd.DataFrame,
    baseline_files: Iterable[str | Path],
    cfg: EvaluationConfig = EvaluationConfig(),
) -> pd.DataFrame:
    records = []

    main_metrics = evaluate_predictions(one_direction_pred, gt, cfg)
    records.append(
        {
            "model": "one_direction",
            **{k: v for k, v in main_metrics.items() if isinstance(v, (int, float, str, bool))},
        }
    )

    for path in baseline_files:
        pred = load_prediction_file(path)
        metrics = evaluate_predictions(pred, gt, cfg)
        records.append(
            {
                "model": model_name_from_path(path),
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
            }
        )

    return pd.DataFrame.from_records(records)


def find_baseline_files(cfg: BaselineComparisonConfig = BaselineComparisonConfig()) -> list[Path]:
    baseline_dir = Path(cfg.baseline_dir)
    if not baseline_dir.exists():
        return []
    return sorted(baseline_dir.glob(cfg.file_glob))


def save_comparison_report(report: pd.DataFrame, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_csv, index=False)
