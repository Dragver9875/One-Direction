import pandas as pd
import pytest
from shapely.geometry import LineString

from src.graph.graph_features import build_segment_feature_table
from src.graph.transition_builder import build_transition_table
from src.graph.line_graph_builder import LineGraphBuildConfig, build_line_graph


def make_edge_table():
    return pd.DataFrame(
        {
            "edge_idx": [0, 1, 2],
            "edge_id": ["e0", "e1", "e2"],
            "u": ["a", "b", "c"],
            "v": ["b", "c", "d"],
            "geometry_wkt": [
                LineString([(0.0, 0.0), (1.0, 0.0)]).wkt,
                LineString([(1.0, 0.0), (2.0, 0.0)]).wkt,
                LineString([(2.0, 0.0), (3.0, 0.0)]).wkt,
            ],
            "length_m": [1.0, 1.0, 1.0],
            "bearing_rad": [0.0, 0.0, 0.0],
            "road_class": ["residential", "residential", "residential"],
            "oneway": [False, False, False],
            "maxspeed": [50.0, 50.0, 50.0],
            "lanes": [1.0, 1.0, 1.0],
            "bridge": [0, 0, 0],
            "tunnel": [0, 0, 0],
        }
    )


def test_build_segment_feature_table():
    table, columns = build_segment_feature_table(make_edge_table())
    assert len(table) == 3
    assert "road_class_id" in columns
    assert "sin_bearing" in columns


def test_build_transition_table():
    transitions = build_transition_table(make_edge_table())
    pairs = set(zip(transitions["prev_edge_idx"], transitions["curr_edge_idx"]))
    assert (0, 1) in pairs
    assert (1, 2) in pairs


def test_build_line_graph_outputs(tmp_path):
    edge_path = tmp_path / "edge_table.parquet"
    make_edge_table().to_parquet(edge_path, index=False)

    outputs = build_line_graph(
        LineGraphBuildConfig(
            edge_table_path=edge_path,
            output_dir=tmp_path / "line_graph",
        )
    )

    assert outputs["line_graph"].exists()
    assert outputs["edge_index"].exists()
    assert outputs["segment_features"].exists()
    assert outputs["transition_table"].exists()
