from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

DEFAULT_EMISSION_FEATURES = [
    "distance_m",
    "yaw_diff_rad",
    "offset_m",
    "offset_ratio",
    "candidate_rank",
    "speed_mps",
    "segment_bearing_rad",
    "sin_segment_bearing",
    "cos_segment_bearing",
    "sin_yaw",
    "cos_yaw",
    "oneway_compatible",
]

DEFAULT_TRANSITION_FEATURES = [
    "same_edge",
    "is_connected",
    "is_legal",
    "prev_candidate_rank",
    "curr_candidate_rank",
    "rank_abs_diff",
    "prev_distance_m",
    "curr_distance_m",
    "distance_delta_m",
    "prev_yaw_diff_rad",
    "curr_yaw_diff_rad",
    "yaw_diff_delta_rad",
    "projection_distance_m",
]

EDGE_IDX_COLUMNS = ["edge_idx", "segment_idx", "road_segment_idx"]
EDGE_ID_COLUMNS = ["edge_id", "osm_edge_id", "segment_id"]


@dataclass(frozen=True)
class TensorBuildConfig:

    max_candidates: int = 10
    pad_edge_idx: int = -1
    pad_label: int = -1
    emission_feature_candidates: tuple[str, ...] = tuple(DEFAULT_EMISSION_FEATURES)
    include_transition_features: bool = True
    transition_feature_names: tuple[str, ...] = tuple(DEFAULT_TRANSITION_FEATURES)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def write_torch_object(obj: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    return path


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> str:
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"None of these required columns were found: {list(candidates)}")


def _available_columns(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def _normalize_candidate_table(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"trajectory_id", "t"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing required columns: {sorted(missing)}")

    df = candidates.copy()
    df["trajectory_id"] = df["trajectory_id"].astype(int)
    df["t"] = df["t"].astype(int)

    edge_idx_col = _first_existing_column(df.columns, EDGE_IDX_COLUMNS)
    if edge_idx_col != "edge_idx":
        df["edge_idx"] = df[edge_idx_col]
    df["edge_idx"] = df["edge_idx"].astype(int)

    edge_id_cols_present = [col for col in EDGE_ID_COLUMNS if col in df.columns]
    if edge_id_cols_present and edge_id_cols_present[0] != "edge_id":
        df["edge_id"] = df[edge_id_cols_present[0]]
    elif not edge_id_cols_present:
        df["edge_id"] = df["edge_idx"].astype(str)

    if "candidate_rank" not in df.columns:
        sort_cols = ["trajectory_id", "t"]
        if "distance_m" in df.columns:
            sort_cols.append("distance_m")
        df = df.sort_values(sort_cols, kind="mergesort")
        df["candidate_rank"] = df.groupby(["trajectory_id", "t"]).cumcount()

    if "is_gt" not in df.columns:
        df["is_gt"] = False

    if "yaw" in df.columns and "sin_yaw" not in df.columns:
        df["sin_yaw"] = np.sin(df["yaw"].astype(float))
        df["cos_yaw"] = np.cos(df["yaw"].astype(float))

    if "segment_bearing_rad" in df.columns:
        if "sin_segment_bearing" not in df.columns:
            df["sin_segment_bearing"] = np.sin(df["segment_bearing_rad"].astype(float))
        if "cos_segment_bearing" not in df.columns:
            df["cos_segment_bearing"] = np.cos(df["segment_bearing_rad"].astype(float))

    return df.sort_values(["trajectory_id", "t", "candidate_rank"], kind="mergesort")


def load_transition_edges(path: str | Path | None) -> set[tuple[int, int]]:

    if path is None:
        return set()

    path = Path(path)
    if not path.exists():
        return set()

    df = read_table(path)
    pairs = [
        ("prev_edge_idx", "curr_edge_idx"),
        ("src_edge_idx", "dst_edge_idx"),
        ("source", "target"),
        ("src", "dst"),
    ]
    for src_col, dst_col in pairs:
        if src_col in df.columns and dst_col in df.columns:
            return set(zip(df[src_col].astype(int), df[dst_col].astype(int)))

    return set()


def _candidate_label_index(step_df: pd.DataFrame, pad_label: int = -1) -> int:
    gt = step_df[step_df["is_gt"].astype(bool)]
    if gt.empty:
        return pad_label
    return int(gt.iloc[0]["candidate_rank"])


def _pad_step_candidates(
    step_df: pd.DataFrame,
    max_candidates: int,
    emission_feature_names: Sequence[str],
    pad_edge_idx: int,
) -> dict[str, object]:
    step_df = step_df.sort_values("candidate_rank", kind="mergesort").head(max_candidates)
    k = len(step_df)

    edge_idx = np.full(max_candidates, pad_edge_idx, dtype=np.int64)
    edge_ids: list[str | None] = [None] * max_candidates
    mask = np.zeros(max_candidates, dtype=bool)
    emission_features = np.zeros((max_candidates, len(emission_feature_names)), dtype=np.float32)

    if k:
        edge_idx[:k] = step_df["edge_idx"].to_numpy(dtype=np.int64)
        edge_ids[:k] = [str(x) for x in step_df["edge_id"].tolist()]
        mask[:k] = True
        if emission_feature_names:
            values = step_df[list(emission_feature_names)].fillna(0.0).to_numpy(dtype=np.float32)
            emission_features[:k, :] = values

    label = _candidate_label_index(step_df, pad_label=-1)
    if label >= max_candidates:
        label = -1

    timestamp = None
    if "timestamp" in step_df.columns and k:
        timestamp = str(step_df.iloc[0]["timestamp"])

    return {
        "edge_idx": edge_idx,
        "edge_ids": edge_ids,
        "candidate_mask": mask,
        "emission_features": emission_features,
        "emission_label": label,
        "timestamp": timestamp,
    }


def _build_transition_tensor(
    prev_step: pd.DataFrame,
    curr_step: pd.DataFrame,
    max_candidates: int,
    legal_edges: set[tuple[int, int]],
    feature_names: Sequence[str] = DEFAULT_TRANSITION_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    prev_step = prev_step.sort_values("candidate_rank", kind="mergesort").head(max_candidates)
    curr_step = curr_step.sort_values("candidate_rank", kind="mergesort").head(max_candidates)

    features = np.zeros((max_candidates, max_candidates, len(feature_names)), dtype=np.float32)
    mask = np.zeros((max_candidates, max_candidates), dtype=bool)

    for i, (_, prev_row) in enumerate(prev_step.iterrows()):
        prev_edge = int(prev_row["edge_idx"])
        for j, (_, curr_row) in enumerate(curr_step.iterrows()):
            curr_edge = int(curr_row["edge_idx"])

            same_edge = float(prev_edge == curr_edge)
            if legal_edges:
                connected = float((prev_edge, curr_edge) in legal_edges)
            else:
                connected = 1.0
            legal = connected

            prev_dist = float(prev_row.get("distance_m", 0.0) or 0.0)
            curr_dist = float(curr_row.get("distance_m", 0.0) or 0.0)
            prev_yaw = float(prev_row.get("yaw_diff_rad", 0.0) or 0.0)
            curr_yaw = float(curr_row.get("yaw_diff_rad", 0.0) or 0.0)

            proj_dist = 0.0
            if all(col in prev_row.index for col in ["proj_x", "proj_y"]) and all(
                col in curr_row.index for col in ["proj_x", "proj_y"]
            ):
                dx = float(curr_row["proj_x"] or 0.0) - float(prev_row["proj_x"] or 0.0)
                dy = float(curr_row["proj_y"] or 0.0) - float(prev_row["proj_y"] or 0.0)
                proj_dist = float(np.hypot(dx, dy))

            values = {
                "same_edge": same_edge,
                "is_connected": connected,
                "is_legal": legal,
                "prev_candidate_rank": float(prev_row.get("candidate_rank", i)),
                "curr_candidate_rank": float(curr_row.get("candidate_rank", j)),
                "rank_abs_diff": abs(
                    float(prev_row.get("candidate_rank", i))
                    - float(curr_row.get("candidate_rank", j))
                ),
                "prev_distance_m": prev_dist,
                "curr_distance_m": curr_dist,
                "distance_delta_m": curr_dist - prev_dist,
                "prev_yaw_diff_rad": prev_yaw,
                "curr_yaw_diff_rad": curr_yaw,
                "yaw_diff_delta_rad": curr_yaw - prev_yaw,
                "projection_distance_m": proj_dist,
            }

            features[i, j, :] = np.array([values[name] for name in feature_names], dtype=np.float32)
            mask[i, j] = bool(legal)

    return features, mask


def build_trajectory_tensor(
    trajectory_candidates: pd.DataFrame,
    config: TensorBuildConfig,
    legal_edges: set[tuple[int, int]] | None = None,
) -> dict[str, object]:
    if legal_edges is None:
        legal_edges = set()

    trajectory_id = int(trajectory_candidates["trajectory_id"].iloc[0])
    trajectory_candidates = trajectory_candidates.sort_values(["t", "candidate_rank"], kind="mergesort")

    emission_feature_names = _available_columns(
        trajectory_candidates,
        list(config.emission_feature_candidates),
    )

    timesteps = sorted(int(t) for t in trajectory_candidates["t"].unique())
    T = len(timesteps)
    K = config.max_candidates
    F = len(emission_feature_names)

    candidate_edge_idx = np.full((T, K), config.pad_edge_idx, dtype=np.int64)
    candidate_mask = np.zeros((T, K), dtype=bool)
    emission_features = np.zeros((T, K, F), dtype=np.float32)
    emission_labels = np.full(T, config.pad_label, dtype=np.int64)
    timestamps: list[str | None] = [None] * T
    candidate_edge_ids: list[list[str | None]] = []

    grouped_by_t = {
        int(t): g.copy()
        for t, g in trajectory_candidates.groupby("t", sort=True)
    }

    for local_t, actual_t in enumerate(timesteps):
        step = grouped_by_t[actual_t]
        padded = _pad_step_candidates(
            step,
            max_candidates=K,
            emission_feature_names=emission_feature_names,
            pad_edge_idx=config.pad_edge_idx,
        )
        candidate_edge_idx[local_t] = padded["edge_idx"]  
        candidate_mask[local_t] = padded["candidate_mask"]  
        emission_features[local_t] = padded["emission_features"]
        emission_labels[local_t] = int(padded["emission_label"])
        timestamps[local_t] = padded["timestamp"]
        candidate_edge_ids.append(padded["edge_ids"])

    sample: dict[str, object] = {
        "trajectory_id": trajectory_id,
        "t": torch.tensor(timesteps, dtype=torch.long),
        "timestamps": timestamps,
        "candidate_edge_idx": torch.tensor(candidate_edge_idx, dtype=torch.long),
        "candidate_edge_ids": candidate_edge_ids,
        "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
        "emission_features": torch.tensor(emission_features, dtype=torch.float32),
        "emission_feature_names": emission_feature_names,
        "emission_labels": torch.tensor(emission_labels, dtype=torch.long),
    }

    if config.include_transition_features and T >= 2:
        Ft = len(config.transition_feature_names)
        transition_features = np.zeros((T - 1, K, K, Ft), dtype=np.float32)
        transition_mask = np.zeros((T - 1, K, K), dtype=bool)
        transition_labels = np.full((T - 1, 2), config.pad_label, dtype=np.int64)

        for local_t in range(1, T):
            prev_step = grouped_by_t[timesteps[local_t - 1]]
            curr_step = grouped_by_t[timesteps[local_t]]
            tf, tm = _build_transition_tensor(
                prev_step,
                curr_step,
                max_candidates=K,
                legal_edges=legal_edges,
                feature_names=config.transition_feature_names,
            )
            transition_features[local_t - 1] = tf
            transition_mask[local_t - 1] = tm

            prev_label = _candidate_label_index(prev_step, pad_label=config.pad_label)
            curr_label = _candidate_label_index(curr_step, pad_label=config.pad_label)
            if prev_label >= K:
                prev_label = config.pad_label
            if curr_label >= K:
                curr_label = config.pad_label
            transition_labels[local_t - 1] = [prev_label, curr_label]

        sample.update(
            {
                "transition_features": torch.tensor(transition_features, dtype=torch.float32),
                "transition_feature_names": list(config.transition_feature_names),
                "transition_mask": torch.tensor(transition_mask, dtype=torch.bool),
                "transition_labels": torch.tensor(transition_labels, dtype=torch.long),
            }
        )
    else:
        sample.update(
            {
                "transition_features": torch.empty((0, K, K, 0), dtype=torch.float32),
                "transition_feature_names": list(config.transition_feature_names),
                "transition_mask": torch.empty((0, K, K), dtype=torch.bool),
                "transition_labels": torch.empty((0, 2), dtype=torch.long),
            }
        )

    return sample


def build_dataset_tensors(
    candidates: pd.DataFrame,
    trajectory_ids: Sequence[int] | None = None,
    max_candidates: int = 10,
    transition_table_path: str | Path | None = None,
) -> dict[str, object]:
    df = _normalize_candidate_table(candidates)

    if trajectory_ids is not None:
        allowed = set(int(x) for x in trajectory_ids)
        df = df[df["trajectory_id"].isin(allowed)].copy()

    config = TensorBuildConfig(max_candidates=max_candidates)
    legal_edges = load_transition_edges(transition_table_path)

    samples: list[dict[str, object]] = []
    for _, group in df.groupby("trajectory_id", sort=True):
        samples.append(build_trajectory_tensor(group, config=config, legal_edges=legal_edges))

    return {
        "samples": samples,
        "num_trajectories": len(samples),
        "max_candidates": max_candidates,
        "transition_table_path": str(transition_table_path) if transition_table_path else None,
    }


def save_dataset_tensors(
    candidates_path: str | Path,
    output_path: str | Path,
    trajectory_ids: Sequence[int] | None = None,
    max_candidates: int = 10,
    transition_table_path: str | Path | None = None,
) -> Path:
    candidates = read_table(candidates_path)
    dataset = build_dataset_tensors(
        candidates,
        trajectory_ids=trajectory_ids,
        max_candidates=max_candidates,
        transition_table_path=transition_table_path,
    )
    return write_torch_object(dataset, output_path)
