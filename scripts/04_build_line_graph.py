from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2 * math.pi) - math.pi)


def build_features(edge_df: pd.DataFrame) -> tuple[torch.Tensor, dict[str, int]]:
    road_classes = sorted(edge_df["road_class"].fillna("unknown").unique())
    class_map = {name: i for i, name in enumerate(road_classes)}
    max_len = max(float(edge_df["length_m"].quantile(0.95)), 1.0)
    max_speed = edge_df["maxspeed"].replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(1, 160)
    lanes = edge_df["lanes"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(1, 10)
    feats = pd.DataFrame(
        {
            "length_norm": edge_df["length_m"].clip(0, max_len) / max_len,
            "sin_bearing": edge_df["sin_bearing"],
            "cos_bearing": edge_df["cos_bearing"],
            "road_class_id_norm": edge_df["road_class"].fillna("unknown").map(class_map) / max(len(class_map) - 1, 1),
            "oneway": edge_df["oneway"].fillna(0).astype(float),
            "maxspeed_norm": max_speed / 160.0,
            "lanes_norm": lanes / 10.0,
            "curvature": edge_df["curvature"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(1.0, 5.0) / 5.0,
            "in_degree_norm": edge_df["in_degree_v"].fillna(0).clip(0, 10) / 10.0,
            "out_degree_norm": edge_df["out_degree_u"].fillna(0).clip(0, 10) / 10.0,
            "bridge": edge_df["bridge"].fillna(0).astype(float),
            "tunnel": edge_df["tunnel"].fillna(0).astype(float),
        }
    )
    return torch.tensor(feats.to_numpy(np.float32)), class_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-table", type=Path, default=Path("data/processed/road_graph/edge_table.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/line_graph"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/line_graph_report.json"))
    parser.add_argument("--allow-uturns", action="store_true")
    parser.add_argument("--include-self-transition", action="store_true", default=True)
    args = parser.parse_args()

    if not args.edge_table.exists():
        raise FileNotFoundError(args.edge_table)
    ensure_dir(args.output)
    ensure_dir(args.report.parent)

    edge_df = pd.read_parquet(args.edge_table).reset_index(drop=True)
    edge_df["edge_idx"] = np.arange(len(edge_df), dtype=np.int64)
    edge_id_to_idx = {str(eid): int(idx) for eid, idx in zip(edge_df["edge_id"], edge_df["edge_idx"])}
    idx_to_edge_id = {str(idx): str(eid) for eid, idx in zip(edge_df["edge_id"], edge_df["edge_idx"])}

    by_u: dict[int, list[int]] = {}
    for row in edge_df.itertuples(index=False):
        by_u.setdefault(int(row.u), []).append(int(row.edge_idx))

    src, dst = [], []
    transition_records = []
    for row in edge_df.itertuples(index=False):
        i = int(row.edge_idx)
        if args.include_self_transition:
            src.append(i)
            dst.append(i)
            transition_records.append(
                {
                    "prev_edge_idx": i,
                    "curr_edge_idx": i,
                    "prev_edge_id": row.edge_id,
                    "curr_edge_id": row.edge_id,
                    "is_self": 1,
                    "is_uturn": 0,
                    "turn_angle_rad": 0.0,
                }
            )
        for j in by_u.get(int(row.v), []):
            curr = edge_df.iloc[j]
            is_uturn = int(int(curr["v"]) == int(row.u))
            if is_uturn and not args.allow_uturns:
                continue
            turn = angle_diff(float(curr["bearing_rad"]), float(row.bearing_rad))
            src.append(i)
            dst.append(j)
            transition_records.append(
                {
                    "prev_edge_idx": i,
                    "curr_edge_idx": j,
                    "prev_edge_id": row.edge_id,
                    "curr_edge_id": curr["edge_id"],
                    "is_self": int(i == j),
                    "is_uturn": is_uturn,
                    "turn_angle_rad": turn,
                }
            )

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x, class_map = build_features(edge_df)
    transition_df = pd.DataFrame(transition_records)

    torch.save({"x": x, "edge_index": edge_index, "num_nodes": int(len(edge_df))}, args.output / "line_graph.pt")
    torch.save(edge_index, args.output / "line_edge_index.pt")
    torch.save(x, args.output / "segment_features.pt")
    transition_df.to_parquet(args.output / "transition_table.parquet", index=False)
    with (args.output / "edge_id_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(edge_id_to_idx, f, indent=2)
    with (args.output / "idx_to_edge_id.json").open("w", encoding="utf-8") as f:
        json.dump(idx_to_edge_id, f, indent=2)
    with (args.output / "road_class_map.json").open("w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    report = {
        "num_segment_nodes": int(len(edge_df)),
        "num_transition_edges": int(edge_index.shape[1]),
        "feature_dim": int(x.shape[1]),
        "allow_uturns": bool(args.allow_uturns),
        "include_self_transition": bool(args.include_self_transition),
        "outputs": {
            "line_graph": str(args.output / "line_graph.pt"),
            "segment_features": str(args.output / "segment_features.pt"),
            "transition_table": str(args.output / "transition_table.parquet"),
        },
    }
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Built line graph: {len(edge_df)} nodes, {edge_index.shape[1]} transitions")
    print(f"[OK] Output: {args.output}")


if __name__ == "__main__":
    main()
