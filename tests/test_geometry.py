from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.geometry.bearing import (
    angular_difference_rad,
    bearing_from_xy,
    linestring_bearing_rad,
    normalize_angle_rad,
)
from src.geometry.distances import (
    euclidean_distance,
    path_length_xy,
    project_point_to_linestring,
)
from src.geometry.polyline_ops import (
    ensure_linestring,
    flatten_multilinestring,
    sample_linestring_by_distance,
)
from src.geometry.projection import detect_utm_epsg, project_dataframe_points
from src.geometry.yaw import YawDerivationConfig, derive_yaw_and_speed


def test_normalize_angle_rad_range():
    values = np.array([-4 * np.pi, -np.pi, 0.0, np.pi, 4 * np.pi])
    out = normalize_angle_rad(values)
    assert np.all(out >= -np.pi)
    assert np.all(out < np.pi)


def test_angular_difference_wraparound():
    a = np.deg2rad(359.0)
    b = np.deg2rad(1.0)
    diff = angular_difference_rad(a, b)
    assert diff == pytest.approx(np.deg2rad(2.0))


def test_bearing_from_xy_east_and_north():
    assert bearing_from_xy(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)
    assert bearing_from_xy(0.0, 0.0, 0.0, 1.0) == pytest.approx(np.pi / 2)


def test_linestring_bearing_rad():
    line = LineString([(0.0, 0.0), (10.0, 0.0)])
    assert linestring_bearing_rad(line) == pytest.approx(0.0)


def test_project_point_to_linestring():
    line = LineString([(0.0, 0.0), (10.0, 0.0)])
    point = Point(5.0, 3.0)
    result = project_point_to_linestring(point, line)
    assert result.distance_m == pytest.approx(3.0)
    assert result.proj_x == pytest.approx(5.0)
    assert result.proj_y == pytest.approx(0.0)
    assert result.offset_ratio == pytest.approx(0.5)


def test_euclidean_and_path_length():
    assert euclidean_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)
    assert path_length_xy([0.0, 3.0, 6.0], [0.0, 4.0, 8.0]) == pytest.approx(10.0)


def test_projection_detect_utm():
    assert detect_utm_epsg(11.9, 50.3) == "EPSG:32632"


def test_project_dataframe_points_adds_xy():
    df = pd.DataFrame({"lon": [11.925816], "lat": [50.301888]})
    out = project_dataframe_points(df, target_crs="EPSG:32632")
    assert "x" in out.columns
    assert "y" in out.columns
    assert np.isfinite(out.loc[0, "x"])
    assert np.isfinite(out.loc[0, "y"])


def test_polyline_ops():
    line = ensure_linestring("LINESTRING (0 0, 10 0)")
    samples = sample_linestring_by_distance(line, spacing_m=5.0)
    assert len(samples) == 3
    assert samples[-1].x == pytest.approx(10.0)


def test_derive_yaw_and_speed():
    df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:01Z",
                    "2024-01-01T00:00:02Z",
                ],
                utc=True,
            ),
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
        }
    )
    out = derive_yaw_and_speed(df, cfg=YawDerivationConfig(min_step_m=0.01))
    assert np.allclose(out["yaw"].to_numpy(), 0.0)
    assert np.allclose(out["speed_mps"].to_numpy(), 1.0)
