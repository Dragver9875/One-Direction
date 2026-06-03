#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import yaml
from pyproj import Transformer

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as RosPath
from visualization_msgs.msg import Marker, MarkerArray


def load_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported match file: {path}")


def load_origin(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def projected_origin(origin: dict) -> tuple[float, float]:
    target_crs = origin.get("target_crs", "EPSG:32643")
    lon = float(origin["longitude"])
    lat = float(origin["latitude"])

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x0, y0 = transformer.transform(lon, lat)

    return float(x0), float(y0)


def subtract_origin(df: pd.DataFrame, x0: float, y0: float) -> pd.DataFrame:
    df = df.copy()

    pairs = [
        ("pred_proj_x", "pred_proj_y"),
        ("gt_proj_x", "gt_proj_y"),
        ("matched_x", "matched_y"),
        ("proj_x", "proj_y"),
        ("x_pred", "y_pred"),
        ("gt_x", "gt_y"),
        ("x_gt", "y_gt"),
    ]

    for x_col, y_col in pairs:
        if x_col in df.columns and y_col in df.columns:
            df[x_col] = df[x_col].astype(float) - x0
            df[y_col] = df[y_col].astype(float) - y0

    return df


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def build_path(df: pd.DataFrame, x_col: str, y_col: str, frame_id: str) -> RosPath:
    msg = RosPath()
    msg.header.frame_id = frame_id

    for _, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])

        if not math.isfinite(x) or not math.isfinite(y):
            continue

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)

    return msg


def build_points_marker(df: pd.DataFrame, x_col: str, y_col: str, frame_id: str) -> MarkerArray:
    arr = MarkerArray()

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = "one_direction_pred_points"
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = 1.25
    marker.scale.y = 1.25
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    for _, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])

        if not math.isfinite(x) or not math.isfinite(y):
            continue

        p = Point()
        p.x = x
        p.y = y
        p.z = 0.0
        marker.points.append(p)

    arr.markers.append(marker)
    return arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--origin-yaml", type=Path, required=True)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--trajectory-id", default=None)
    parser.add_argument("--no-localize", action="store_true")
    parser.add_argument("--pred-path-topic", default="/one_direction/pred_path")
    parser.add_argument("--gt-path-topic", default="/one_direction/gt_path")
    parser.add_argument("--pred-marker-topic", default="/one_direction/pred_points")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = load_matches(args.matches)

    if args.trajectory_id is not None and "trajectory_id" in df.columns:
        df = df[df["trajectory_id"].astype(str) == str(args.trajectory_id)].copy()

    if "t" in df.columns:
        sort_cols = ["trajectory_id", "t"] if "trajectory_id" in df.columns else ["t"]
        df = df.sort_values(sort_cols)

    if not args.no_localize:
        origin = load_origin(args.origin_yaml)
        x0, y0 = projected_origin(origin)
        df = subtract_origin(df, x0, y0)
        print(f"[OK] projected origin subtracted: x0={x0:.3f}, y0={y0:.3f}")
    else:
        print("[WARN] --no-localize used; publishing raw coordinates")

    pred_x = first_existing(df, ["pred_proj_x", "matched_x", "proj_x", "x_pred"])
    pred_y = first_existing(df, ["pred_proj_y", "matched_y", "proj_y", "y_pred"])
    gt_x = first_existing(df, ["gt_proj_x", "true_proj_x", "gt_x", "x_gt"])
    gt_y = first_existing(df, ["gt_proj_y", "true_proj_y", "gt_y", "y_gt"])

    if pred_x is None or pred_y is None:
        raise ValueError("Could not find predicted coordinate columns.")

    print("[OK] local pred x min/max:", float(df[pred_x].min()), float(df[pred_x].max()))
    print("[OK] local pred y min/max:", float(df[pred_y].min()), float(df[pred_y].max()))
    print("[OK] first local pred:", float(df[pred_x].iloc[0]), float(df[pred_y].iloc[0]))

    rclpy.init()
    node = rclpy.create_node("one_direction_mapviz_publisher")

    pred_path_pub = node.create_publisher(RosPath, args.pred_path_topic, 10)
    gt_path_pub = node.create_publisher(RosPath, args.gt_path_topic, 10)
    pred_marker_pub = node.create_publisher(MarkerArray, args.pred_marker_topic, 10)

    pred_path = build_path(df, pred_x, pred_y, args.frame_id)
    pred_markers = build_points_marker(df, pred_x, pred_y, args.frame_id)

    gt_path = None
    if gt_x is not None and gt_y is not None:
        valid_gt = df[[gt_x, gt_y]].replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(valid_gt) > 0:
            gt_path = build_path(df, gt_x, gt_y, args.frame_id)

    print(f"[OK] publishing predicted path poses: {len(pred_path.poses)} on {args.pred_path_topic}")
    print(f"[OK] publishing predicted points on {args.pred_marker_topic}")

    if gt_path is not None:
        print(f"[OK] publishing GT path poses: {len(gt_path.poses)} on {args.gt_path_topic}")
    else:
        print("[OK] no valid GT path detected; publishing prediction only")

    period = 1.0 / max(args.rate, 0.1)

    try:
        while rclpy.ok():
            now = node.get_clock().now().to_msg()

            pred_path.header.stamp = now
            for pose in pred_path.poses:
                pose.header.stamp = now
            pred_path_pub.publish(pred_path)

            for marker in pred_markers.markers:
                marker.header.stamp = now
            pred_marker_pub.publish(pred_markers)

            if gt_path is not None:
                gt_path.header.stamp = now
                for pose in gt_path.poses:
                    pose.header.stamp = now
                gt_path_pub.publish(gt_path)

            rclpy.spin_once(node, timeout_sec=period)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
