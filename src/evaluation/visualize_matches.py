from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class VisualizationConfig:
    output_dir: str | Path = "outputs/figures"
    crs: str = "EPSG:32632"
    max_trajectories: int = 20
    figsize: tuple[int, int] = (10, 10)
    dpi: int = 150


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for visualization.") from exc
    return plt


def _read_edges(edges_path: str | Path | None):
    if edges_path is None:
        return None

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to read edge GeoJSON.") from exc

    edges_path = Path(edges_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    return gpd.read_file(edges_path)


def plot_matched_trajectories(
    matches: pd.DataFrame,
    edges_path: str | Path | None = None,
    cfg: VisualizationConfig = VisualizationConfig(),
    output_name: str = "matched_paths.png",
) -> Path:
    plt = _require_matplotlib()
    edges = _read_edges(edges_path)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    fig, ax = plt.subplots(figsize=cfg.figsize)

    if edges is not None and not edges.empty:
        edges.plot(ax=ax, linewidth=0.5, alpha=0.4)

    required = {"trajectory_id", "pred_proj_x", "pred_proj_y"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing required columns: {sorted(missing)}")

    trajectory_ids = list(matches["trajectory_id"].drop_duplicates())[: cfg.max_trajectories]

    for trajectory_id in trajectory_ids:
        group = matches[matches["trajectory_id"] == trajectory_id]
        if "t" in group.columns:
            group = group.sort_values("t")
        ax.plot(group["pred_proj_x"], group["pred_proj_y"], linewidth=1.2, marker="o", markersize=2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("One-Direction matched trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.dpi)
    plt.close(fig)

    return output_path


def plot_error_cases(
    error_cases: pd.DataFrame,
    edges_path: str | Path | None = None,
    cfg: VisualizationConfig = VisualizationConfig(),
    output_name: str = "error_cases.png",
) -> Path:
    plt = _require_matplotlib()
    edges = _read_edges(edges_path)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    fig, ax = plt.subplots(figsize=cfg.figsize)

    if edges is not None and not edges.empty:
        edges.plot(ax=ax, linewidth=0.5, alpha=0.4)

    required = {"pred_proj_x", "pred_proj_y"}
    missing = required - set(error_cases.columns)
    if missing:
        raise ValueError(f"error_cases missing required columns: {sorted(missing)}")

    ax.scatter(error_cases["pred_proj_x"], error_cases["pred_proj_y"], s=12)

    if {"gt_proj_x", "gt_proj_y"}.issubset(error_cases.columns):
        ax.scatter(error_cases["gt_proj_x"], error_cases["gt_proj_y"], s=8, marker="x")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("One-Direction error cases")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.dpi)
    plt.close(fig)

    return output_path


def save_matches_geojson(
    matches: pd.DataFrame,
    output_path: str | Path,
    x_col: str = "pred_proj_x",
    y_col: str = "pred_proj_y",
    crs: str = "EPSG:32632",
) -> Path:
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError("geopandas and shapely are required for GeoJSON export.") from exc

    required = {x_col, y_col}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing GeoJSON coordinate columns: {sorted(missing)}")

    gdf = gpd.GeoDataFrame(
        matches.copy(),
        geometry=[Point(float(x), float(y)) for x, y in zip(matches[x_col], matches[y_col])],
        crs=crs,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")

    return output_path
