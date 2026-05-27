from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


@dataclass(frozen=True)
class CandidateGenerationConfig:
    radius_m: float = 50.0
    max_candidates: int = 10
    min_candidates: int = 1
    include_geometry_wkt: bool = False
    require_gt_in_candidates: bool = False
    candidate_recall_topk: Tuple[int, ...] = (1, 3, 5, 10)


def normalize_angle_rad(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_difference_rad(yaw: float, bearing: float) -> float:
    return float(abs(normalize_angle_rad(yaw - bearing)))


def _geometry_from_value(value: object) -> Optional[LineString]:
    if isinstance(value, LineString):
        return value
    if isinstance(value, str):
        geom = wkt.loads(value)
        if isinstance(geom, LineString):
            return geom
    return None


def _point_projection_features(point: Point, line: LineString) -> Dict[str, float]:
    offset_m = float(line.project(point))
    proj = line.interpolate(offset_m)
    length_m = float(line.length) if line.length > 0 else 1.0
    offset_ratio = float(offset_m / length_m)
    return {
        "distance_m": float(point.distance(line)),
        "proj_x": float(proj.x),
        "proj_y": float(proj.y),
        "offset_m": offset_m,
        "offset_ratio": offset_ratio,
    }


class CandidateGenerator:

    def __init__(
        self,
        edge_table: pd.DataFrame,
        cfg: CandidateGenerationConfig = CandidateGenerationConfig(),
    ) -> None:
        self.cfg = cfg
        self.edge_table = self._prepare_edge_table(edge_table)

        self._geometries: List[LineString] = list(self.edge_table["geometry"])
        self._tree = STRtree(self._geometries)

        self._geom_id_to_idx = {id(g): i for i, g in enumerate(self._geometries)}

    @staticmethod
    def _prepare_edge_table(edge_table: pd.DataFrame) -> pd.DataFrame:
        df = edge_table.copy()

        if "geometry" not in df.columns:
            if "geometry_wkt" not in df.columns:
                raise ValueError("edge_table must contain geometry or geometry_wkt.")
            df["geometry"] = df["geometry_wkt"].apply(_geometry_from_value)
        else:
            df["geometry"] = df["geometry"].apply(_geometry_from_value)

        df = df[df["geometry"].notna()].copy()

        if "edge_idx" not in df.columns:
            df["edge_idx"] = np.arange(len(df), dtype=np.int64)
        if "edge_id" not in df.columns:
            df["edge_id"] = df["edge_idx"].map(lambda i: f"e_{int(i):09d}")
        if "bearing_rad" not in df.columns:
            df["bearing_rad"] = df["geometry"].apply(_bearing_rad)

        return df.reset_index(drop=True)

    def _query_indices(self, point: Point, radius_m: float) -> List[int]:
        query_geom = point.buffer(radius_m)
        results = self._tree.query(query_geom)

        indices: List[int] = []
        for item in results:
            if isinstance(item, (int, np.integer)):
                indices.append(int(item))
            else:
                indices.append(self._geom_id_to_idx[id(item)])

        return indices

    def candidates_for_point(
        self,
        trajectory_id: int,
        t: int,
        timestamp: object,
        x: float,
        y: float,
        yaw: float,
        speed_mps: float | None = None,
        gt_edge_id: str | None = None,
    ) -> pd.DataFrame:
        point = Point(float(x), float(y))
        indices = self._query_indices(point, self.cfg.radius_m)

        records: List[Dict[str, object]] = []
        for local_idx in indices:
            edge = self.edge_table.iloc[local_idx]
            line = edge["geometry"]
            proj_features = _point_projection_features(point, line)
            bearing = float(edge.get("bearing_rad", 0.0))
            yaw_diff = yaw_difference_rad(float(yaw), bearing)

            road_class = str(edge.get("road_class", "unknown"))

            records.append(
                {
                    "trajectory_id": int(trajectory_id),
                    "t": int(t),
                    "timestamp": timestamp,
                    "edge_idx": int(edge["edge_idx"]),
                    "edge_id": str(edge["edge_id"]),
                    "distance_m": proj_features["distance_m"],
                    "yaw_diff_rad": yaw_diff,
                    "proj_x": proj_features["proj_x"],
                    "proj_y": proj_features["proj_y"],
                    "offset_m": proj_features["offset_m"],
                    "offset_ratio": proj_features["offset_ratio"],
                    "segment_bearing_rad": bearing,
                    "speed_mps": float(speed_mps) if speed_mps is not None else np.nan,
                    "road_class": road_class,
                    "is_gt": int(gt_edge_id is not None and str(edge["edge_id"]) == str(gt_edge_id)),
                }
            )

        if not records:
            indices = self._query_indices(point, self.cfg.radius_m * 2.0)
            for local_idx in indices:
                edge = self.edge_table.iloc[local_idx]
                line = edge["geometry"]
                proj_features = _point_projection_features(point, line)
                bearing = float(edge.get("bearing_rad", 0.0))
                yaw_diff = yaw_difference_rad(float(yaw), bearing)

                records.append(
                    {
                        "trajectory_id": int(trajectory_id),
                        "t": int(t),
                        "timestamp": timestamp,
                        "edge_idx": int(edge["edge_idx"]),
                        "edge_id": str(edge["edge_id"]),
                        "distance_m": proj_features["distance_m"],
                        "yaw_diff_rad": yaw_diff,
                        "proj_x": proj_features["proj_x"],
                        "proj_y": proj_features["proj_y"],
                        "offset_m": proj_features["offset_m"],
                        "offset_ratio": proj_features["offset_ratio"],
                        "segment_bearing_rad": bearing,
                        "speed_mps": float(speed_mps) if speed_mps is not None else np.nan,
                        "road_class": str(edge.get("road_class", "unknown")),
                        "is_gt": int(gt_edge_id is not None and str(edge["edge_id"]) == str(gt_edge_id)),
                    }
                )

        out = pd.DataFrame.from_records(records)

        if out.empty:
            raise ValueError(
                f"No road candidates found for trajectory={trajectory_id}, t={t}. "
                "Increase radius_m or inspect projected CRS alignment."
            )

        out = out.sort_values(
            ["distance_m", "yaw_diff_rad", "edge_idx"],
            ascending=[True, True, True],
        ).head(self.cfg.max_candidates).reset_index(drop=True)

        out["candidate_rank"] = np.arange(len(out), dtype=np.int64)

        if self.cfg.include_geometry_wkt:
            geom_lookup = self.edge_table.set_index("edge_idx")["geometry"].to_dict()
            out["edge_geometry_wkt"] = out["edge_idx"].map(lambda idx: geom_lookup[int(idx)].wkt)

        return out

    def generate(
        self,
        trajectory_df: pd.DataFrame,
        gt_point_labels: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        required = {"trajectory_id", "t", "timestamp", "x", "y", "yaw"}
        missing = required - set(trajectory_df.columns)
        if missing:
            raise ValueError(f"trajectory_df missing required columns: {sorted(missing)}")

        gt_lookup: Dict[Tuple[int, int], str] = {}
        if gt_point_labels is not None and not gt_point_labels.empty:
            for _, row in gt_point_labels.iterrows():
                gt_lookup[(int(row["trajectory_id"]), int(row["t"]))] = str(row["gt_edge_id"])

        chunks: List[pd.DataFrame] = []
        for _, row in trajectory_df.sort_values(["trajectory_id", "t"]).iterrows():
            tid = int(row["trajectory_id"])
            t = int(row["t"])
            gt_edge_id = gt_lookup.get((tid, t))
            speed = row["speed_mps"] if "speed_mps" in row else None

            chunks.append(
                self.candidates_for_point(
                    trajectory_id=tid,
                    t=t,
                    timestamp=row["timestamp"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw=float(row["yaw"]),
                    speed_mps=None if pd.isna(speed) else float(speed),
                    gt_edge_id=gt_edge_id,
                )
            )

        return pd.concat(chunks, ignore_index=True)


def _bearing_rad(line: LineString) -> float:
    coords = list(line.coords)
    if len(coords) < 2:
        return 0.0
    x0, y0 = coords[0][:2]
    x1, y1 = coords[-1][:2]
    return math.atan2(y1 - y0, x1 - x0)


def candidate_recall_report(
    candidates: pd.DataFrame,
    topk: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, object]:
    if "is_gt" not in candidates.columns:
        return {"available": False, "reason": "is_gt column not found"}

    records = []
    grouped = candidates.groupby(["trajectory_id", "t"], sort=False)
    total_points = grouped.ngroups

    report: Dict[str, object] = {
        "available": True,
        "total_points": int(total_points),
    }

    for k in topk:
        hit = 0
        for _, group in grouped:
            top = group.sort_values("candidate_rank").head(int(k))
            if int(top["is_gt"].max()) == 1:
                hit += 1
        report[f"top_{int(k)}_recall"] = float(hit / max(total_points, 1))
        report[f"top_{int(k)}_hits"] = int(hit)

    missing = 0
    for _, group in grouped:
        if int(group["is_gt"].max()) != 1:
            missing += 1
    report["missing_gt_candidate_count"] = int(missing)

    return report


def save_candidate_outputs(
    candidates: pd.DataFrame,
    output_path: str | Path,
    report_path: str | Path | None = None,
    topk: Iterable[int] = (1, 3, 5, 10),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        candidates.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        candidates.to_csv(output_path, index=False)
    else:
        raise ValueError("candidate output must be .parquet or .csv")

    if report_path is not None:
        report = candidate_recall_report(candidates, topk)
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


__all__ = [
    "CandidateGenerationConfig",
    "CandidateGenerator",
    "normalize_angle_rad",
    "yaw_difference_rad",
    "candidate_recall_report",
    "save_candidate_outputs",
]
