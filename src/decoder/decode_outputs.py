from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True)
class DecodeOutputConfig:
    output_parquet: str | Path = "outputs/matches/gnn_hmm_matches.parquet"
    output_geojson: str | Path = "outputs/matches/gnn_hmm_matches.geojson"
    include_geojson: bool = True
    crs: str = "EPSG:32632"


def _to_numpy_1d(value: Tensor | Sequence | np.ndarray) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def build_match_dataframe(
    trajectory_id: int,
    timestamps: Sequence | np.ndarray,
    candidate_edge_idx: Tensor,
    candidate_edge_id: Sequence[Sequence[str]] | None,
    candidate_proj_x: Tensor,
    candidate_proj_y: Tensor,
    emission_scores: Tensor,
    transition_scores: Tensor,
    path_positions: Tensor,
    path_edge_idx: Tensor,
    confidence: Tensor | None = None,
    path_score: Tensor | float | None = None,
) -> pd.DataFrame:
    positions = _to_numpy_1d(path_positions).astype(int)
    edge_idx = _to_numpy_1d(path_edge_idx).astype(int)

    timestamps_arr = np.asarray(list(timestamps))
    if len(timestamps_arr) != len(positions):
        raise ValueError("timestamps length must match decoded path length.")

    cand_idx_np = candidate_edge_idx.detach().cpu().numpy()
    proj_x_np = candidate_proj_x.detach().cpu().numpy()
    proj_y_np = candidate_proj_y.detach().cpu().numpy()
    emission_np = emission_scores.detach().cpu().numpy()

    transition_np = transition_scores.detach().cpu().numpy()
    conf_np = None if confidence is None else _to_numpy_1d(confidence)

    records = []
    total_path_score = float(path_score.detach().cpu().item()) if isinstance(path_score, Tensor) else path_score

    for t, pos in enumerate(positions):
        pred_edge_idx = int(edge_idx[t])

        if candidate_edge_id is not None and pos >= 0:
            pred_edge_id = str(candidate_edge_id[t][pos])
        else:
            pred_edge_id = str(pred_edge_idx)

        transition_score = np.nan
        if t > 0 and transition_np.size > 0:
            prev_pos = positions[t - 1]
            curr_pos = positions[t]
            transition_score = float(transition_np[t - 1, prev_pos, curr_pos])

        records.append(
            {
                "trajectory_id": int(trajectory_id),
                "t": int(t),
                "timestamp": timestamps_arr[t],
                "pred_edge_id": pred_edge_id,
                "pred_edge_idx": pred_edge_idx,
                "pred_candidate_position": int(pos),
                "pred_proj_x": float(proj_x_np[t, pos]),
                "pred_proj_y": float(proj_y_np[t, pos]),
                "confidence": float(conf_np[t]) if conf_np is not None else np.nan,
                "emission_score": float(emission_np[t, pos]),
                "transition_score": transition_score,
                "total_path_score": float(total_path_score) if total_path_score is not None else np.nan,
            }
        )

    return pd.DataFrame.from_records(records)


def save_decode_outputs(
    matches: pd.DataFrame,
    cfg: DecodeOutputConfig = DecodeOutputConfig(),
) -> None:
    output_parquet = Path(cfg.output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(output_parquet, index=False)

    if cfg.include_geojson:
        save_matches_geojson(matches, cfg.output_geojson, cfg.crs)


def save_matches_geojson(
    matches: pd.DataFrame,
    output_geojson: str | Path,
    crs: str = "EPSG:32632",
) -> None:
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError("geopandas and shapely are required for GeoJSON output.") from exc

    required = {"pred_proj_x", "pred_proj_y"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing required columns for GeoJSON: {sorted(missing)}")

    gdf = gpd.GeoDataFrame(
        matches.copy(),
        geometry=[
            Point(float(x), float(y))
            for x, y in zip(matches["pred_proj_x"], matches["pred_proj_y"])
        ],
        crs=crs,
    )

    output_geojson = Path(output_geojson)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_geojson, driver="GeoJSON")


def concatenate_match_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
