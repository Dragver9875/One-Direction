from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .graph_features import RoadFeatureConfig, build_segment_feature_table
from .transition_builder import TransitionBuildConfig, build_transition_table


@dataclass(frozen=True)
class LineGraphBuildConfig:
    edge_table_path: str | Path
    output_dir: str | Path = "data/processed/line_graph"

    allow_u_turns: bool = False
    include_self_transition: bool = True
    max_turn_angle_deg_for_non_uturn: float = 160.0
    connect_only_legal_transitions: bool = True

    normalize_numeric_features: bool = True

    output_graph_name: str = "line_graph.pt"
    output_edge_index_name: str = "line_edge_index.pt"
    output_segment_features_name: str = "segment_features.pt"
    output_feature_table_name: str = "segment_feature_table.parquet"
    output_transition_table_name: str = "transition_table.parquet"
    output_edge_id_to_idx_name: str = "edge_id_to_idx.json"
    output_idx_to_edge_id_name: str = "idx_to_edge_id.json"
    output_report_name: str = "line_graph_report.json"


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _read_edge_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"edge_table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported edge_table format: {path}")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()

    return str(value)


def _save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def build_line_graph(cfg: LineGraphBuildConfig) -> Dict[str, Path]:
    edge_table_path = _to_path(cfg.edge_table_path)
    output_dir = _to_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_table = _read_edge_table(edge_table_path)

    if "edge_idx" not in edge_table.columns:
        edge_table = edge_table.copy()
        edge_table["edge_idx"] = np.arange(len(edge_table), dtype=np.int64)

    edge_table = edge_table.sort_values("edge_idx").reset_index(drop=True)

    feature_cfg = RoadFeatureConfig(normalize_numeric=cfg.normalize_numeric_features)
    feature_table, feature_columns = build_segment_feature_table(edge_table, feature_cfg)

    x = torch.tensor(
        feature_table[feature_columns].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    trans_cfg = TransitionBuildConfig(
        allow_u_turns=cfg.allow_u_turns,
        include_self_transition=cfg.include_self_transition,
        max_turn_angle_deg_for_non_uturn=cfg.max_turn_angle_deg_for_non_uturn,
        connect_only_legal_transitions=cfg.connect_only_legal_transitions,
    )
    transition_table = build_transition_table(edge_table, trans_cfg)

    edge_index = torch.tensor(
        transition_table[["prev_edge_idx", "curr_edge_idx"]].to_numpy(dtype=np.int64).T,
        dtype=torch.long,
    )

    edge_id_to_idx = {
        str(row["edge_id"]): int(row["edge_idx"])
        for _, row in edge_table.iterrows()
    }
    idx_to_edge_id = {
        str(int(row["edge_idx"])): str(row["edge_id"])
        for _, row in edge_table.iterrows()
    }
    try:
        from torch_geometric.data import Data  # type: ignore
        line_graph = Data(x=x, edge_index=edge_index)
        line_graph.num_nodes = int(len(feature_table))
    except Exception:
        line_graph = {
            "x": x,
            "edge_index": edge_index,
            "num_nodes": int(len(feature_table)),
        }

    graph_path = output_dir / cfg.output_graph_name
    edge_index_path = output_dir / cfg.output_edge_index_name
    feature_tensor_path = output_dir / cfg.output_segment_features_name
    feature_table_path = output_dir / cfg.output_feature_table_name
    transition_table_path = output_dir / cfg.output_transition_table_name
    edge_id_to_idx_path = output_dir / cfg.output_edge_id_to_idx_name
    idx_to_edge_id_path = output_dir / cfg.output_idx_to_edge_id_name
    report_path = output_dir / cfg.output_report_name

    torch.save(line_graph, graph_path)
    torch.save(edge_index, edge_index_path)
    torch.save(x, feature_tensor_path)
    feature_table.to_parquet(feature_table_path, index=False)
    transition_table.to_parquet(transition_table_path, index=False)
    _save_json(edge_id_to_idx, edge_id_to_idx_path)
    _save_json(idx_to_edge_id, idx_to_edge_id_path)

    report = {
        "config": asdict(cfg),
        "num_segment_nodes": int(x.shape[0]),
        "num_transition_edges": int(edge_index.shape[1]),
        "feature_dim": int(x.shape[1]),
        "feature_columns": feature_columns,
        "outputs": {
            "line_graph": str(graph_path),
            "edge_index": str(edge_index_path),
            "segment_features": str(feature_tensor_path),
            "feature_table": str(feature_table_path),
            "transition_table": str(transition_table_path),
            "edge_id_to_idx": str(edge_id_to_idx_path),
            "idx_to_edge_id": str(idx_to_edge_id_path),
        },
    }
    _save_json(report, report_path)

    return {
        "line_graph": graph_path,
        "edge_index": edge_index_path,
        "segment_features": feature_tensor_path,
        "feature_table": feature_table_path,
        "transition_table": transition_table_path,
        "edge_id_to_idx": edge_id_to_idx_path,
        "idx_to_edge_id": idx_to_edge_id_path,
        "report": report_path,
    }


__all__ = ["LineGraphBuildConfig", "build_line_graph"]
