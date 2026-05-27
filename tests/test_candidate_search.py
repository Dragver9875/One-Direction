import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.graph.candidate_search import (
    CandidateGenerationConfig,
    CandidateGenerator,
    candidate_recall_report,
    yaw_difference_rad,
)


def make_edge_table():
    return pd.DataFrame(
        {
            "edge_idx": [0, 1],
            "edge_id": ["e0", "e1"],
            "geometry_wkt": [
                LineString([(0.0, 0.0), (10.0, 0.0)]).wkt,
                LineString([(0.0, 10.0), (10.0, 10.0)]).wkt,
            ],
            "bearing_rad": [0.0, 0.0],
            "road_class": ["residential", "residential"],
        }
    )


def test_yaw_difference_rad_wraps():
    diff = yaw_difference_rad(np.deg2rad(359.0), np.deg2rad(1.0))
    assert diff == pytest.approx(np.deg2rad(2.0))


def test_candidate_generator_single_point():
    edge_table = make_edge_table()
    generator = CandidateGenerator(
        edge_table,
        CandidateGenerationConfig(radius_m=5.0, max_candidates=2),
    )

    out = generator.candidates_for_point(
        trajectory_id=0,
        t=0,
        timestamp="2024-01-01",
        x=5.0,
        y=1.0,
        yaw=0.0,
        gt_edge_id="e0",
    )

    assert len(out) >= 1
    assert out.iloc[0]["edge_id"] == "e0"
    assert out.iloc[0]["distance_m"] == pytest.approx(1.0)
    assert out.iloc[0]["is_gt"] == 1


def test_candidate_generator_dataframe():
    edge_table = make_edge_table()
    generator = CandidateGenerator(
        edge_table,
        CandidateGenerationConfig(radius_m=5.0, max_candidates=2),
    )

    trajectory = pd.DataFrame(
        {
            "trajectory_id": [0, 0],
            "t": [0, 1],
            "timestamp": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [1.0, 1.0],
            "yaw": [0.0, 0.0],
            "speed_mps": [1.0, 1.0],
        }
    )

    gt = pd.DataFrame(
        {
            "trajectory_id": [0, 0],
            "t": [0, 1],
            "gt_edge_id": ["e0", "e0"],
        }
    )

    out = generator.generate(trajectory, gt)
    assert set(out["trajectory_id"]) == {0}
    assert out["is_gt"].sum() == 2


def test_candidate_recall_report():
    candidates = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 1, 1],
            "t": [0, 0, 0, 0],
            "candidate_rank": [0, 1, 0, 1],
            "is_gt": [0, 1, 1, 0],
        }
    )
    report = candidate_recall_report(candidates, topk=(1, 2))
    assert report["top_1_recall"] == pytest.approx(0.5)
    assert report["top_2_recall"] == pytest.approx(1.0)
