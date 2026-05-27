from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransitionBuildConfig:
    allow_u_turns: bool = False
    include_self_transition: bool = True
    max_turn_angle_deg_for_non_uturn: float = 160.0
    connect_only_legal_transitions: bool = True


def normalize_angle_rad(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def angular_difference_rad(a: float, b: float) -> float:
    return float(abs(normalize_angle_rad(a - b)))


def signed_turn_angle_rad(prev_bearing: float, curr_bearing: float) -> float:
    return float(normalize_angle_rad(curr_bearing - prev_bearing))


def _build_incoming_by_node(edge_table: pd.DataFrame) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for _, row in edge_table.iterrows():
        node = str(row["v"])
        out.setdefault(node, []).append(int(row["edge_idx"]))
    return out


def _build_outgoing_by_node(edge_table: pd.DataFrame) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for _, row in edge_table.iterrows():
        node = str(row["u"])
        out.setdefault(node, []).append(int(row["edge_idx"]))
    return out


def build_transition_table(
    edge_table: pd.DataFrame,
    cfg: TransitionBuildConfig = TransitionBuildConfig(),
) -> pd.DataFrame:
    required = {"edge_idx", "edge_id", "u", "v", "bearing_rad"}
    missing = required - set(edge_table.columns)
    if missing:
        raise ValueError(f"edge_table missing required columns: {sorted(missing)}")

    df = edge_table.copy().sort_values("edge_idx").reset_index(drop=True)

    row_by_idx = {int(row["edge_idx"]): row for _, row in df.iterrows()}
    outgoing_by_node = _build_outgoing_by_node(df)

    records: List[dict] = []

    for prev_idx, prev_row in row_by_idx.items():
        prev_v = str(prev_row["v"])
        possible_next = outgoing_by_node.get(prev_v, [])

        if cfg.include_self_transition:
            possible_next = list(set(possible_next + [prev_idx]))

        for curr_idx in possible_next:
            curr_row = row_by_idx[curr_idx]

            same_edge = int(prev_idx == curr_idx)
            reverse_of_prev = int(
                str(prev_row["u"]) == str(curr_row["v"])
                and str(prev_row["v"]) == str(curr_row["u"])
            )

            turn_angle = signed_turn_angle_rad(
                float(prev_row.get("bearing_rad", 0.0)),
                float(curr_row.get("bearing_rad", 0.0)),
            )
            turn_abs = abs(turn_angle)
            turn_abs_deg = math.degrees(turn_abs)

            is_uturn_like = reverse_of_prev or (
                turn_abs_deg >= cfg.max_turn_angle_deg_for_non_uturn
            )

            if is_uturn_like and not cfg.allow_u_turns and not same_edge:
                continue

            is_connected = int(str(prev_row["v"]) == str(curr_row["u"]) or same_edge)
            is_legal = int(is_connected and (cfg.allow_u_turns or not is_uturn_like or same_edge))

            if cfg.connect_only_legal_transitions and not is_legal:
                continue

            records.append(
                {
                    "prev_edge_idx": int(prev_idx),
                    "curr_edge_idx": int(curr_idx),
                    "prev_edge_id": str(prev_row["edge_id"]),
                    "curr_edge_id": str(curr_row["edge_id"]),
                    "via_node": str(prev_row["v"]),
                    "is_connected": is_connected,
                    "is_legal": is_legal,
                    "same_edge": same_edge,
                    "reverse_of_prev": reverse_of_prev,
                    "turn_angle_rad": float(turn_angle),
                    "turn_angle_abs_rad": float(turn_abs),
                    "turn_angle_abs_deg": float(turn_abs_deg),
                    "prev_road_class": str(prev_row.get("road_class", "unknown")),
                    "curr_road_class": str(curr_row.get("road_class", "unknown")),
                    "prev_length_m": float(prev_row.get("length_m", 0.0)),
                    "curr_length_m": float(curr_row.get("length_m", 0.0)),
                }
            )

    transition_table = pd.DataFrame.from_records(records)

    if transition_table.empty:
        raise ValueError("No legal transitions were produced. Check edge endpoints/config.")

    return transition_table.sort_values(
        ["prev_edge_idx", "curr_edge_idx"]
    ).reset_index(drop=True)


def transition_lookup_dict(
    transition_table: pd.DataFrame,
) -> Dict[Tuple[int, int], dict]:
    """Create a dictionary keyed by (prev_edge_idx, curr_edge_idx)."""
    return {
        (int(row["prev_edge_idx"]), int(row["curr_edge_idx"])): row.to_dict()
        for _, row in transition_table.iterrows()
    }


__all__ = [
    "TransitionBuildConfig",
    "normalize_angle_rad",
    "angular_difference_rad",
    "signed_turn_angle_rad",
    "build_transition_table",
    "transition_lookup_dict",
]
