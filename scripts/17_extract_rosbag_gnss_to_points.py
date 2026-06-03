#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import yaml


def import_rosbag_tools():
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except Exception as exc:
        raise RuntimeError(
            "Run inside a sourced ROS 2 environment:\n"
            "  source /opt/ros/$ROS_DISTRO/setup.bash"
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def stamp_to_float(msg: Any, fallback_ns: int) -> float:
    try:
        sec = int(msg.header.stamp.sec)
        nsec = int(msg.header.stamp.nanosec)
        if sec != 0 or nsec != 0:
            return sec + nsec * 1.0e-9
    except Exception:
        pass
    return float(fallback_ns) * 1.0e-9


def yaw_from_quaternion(q: Any) -> float:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_from_delta(dx: float, dy: float) -> float:
    if abs(dx) < 1.0e-9 and abs(dy) < 1.0e-9:
        return float("nan")
    return math.atan2(dy, dx)


def nearest_yaw(timestamp: float, odom_times: list[float], odom_yaws: list[float], max_dt: float) -> float:
    if not odom_times:
        return float("nan")
    best_i = min(range(len(odom_times)), key=lambda i: abs(odom_times[i] - timestamp))
    if abs(odom_times[best_i] - timestamp) > max_dt:
        return float("nan")
    return odom_yaws[best_i]


def make_transformer(target_crs: str):
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError("Install pyproj: python -m pip install pyproj") from exc
    return Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)


def read_bag(bag_dir: Path, gnss_topic: str, odom_topic: str | None):
    rosbag2_py, deserialize_message, get_message = import_rosbag_tools()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if gnss_topic not in topic_types:
        available = "\n".join(f"  {name}: {typ}" for name, typ in sorted(topic_types.items()))
        raise ValueError(f"GNSS topic {gnss_topic!r} not found. Available topics:\n{available}")

    gnss_msg_type = get_message(topic_types[gnss_topic])
    odom_msg_type = get_message(topic_types[odom_topic]) if odom_topic and odom_topic in topic_types else None

    gnss_rows, odom_times, odom_yaws = [], [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == gnss_topic:
            msg = deserialize_message(data, gnss_msg_type)
            ts = stamp_to_float(msg, t_ns)
            gnss_rows.append(
                {
                    "timestamp": ts,
                    "latitude": float(msg.latitude),
                    "longitude": float(msg.longitude),
                    "altitude": float(msg.altitude),
                    "fix_status": int(msg.status.status),
                    "position_covariance_0": float(msg.position_covariance[0]) if len(msg.position_covariance) else float("nan"),
                    "position_covariance_4": float(msg.position_covariance[4]) if len(msg.position_covariance) > 4 else float("nan"),
                    "position_covariance_8": float(msg.position_covariance[8]) if len(msg.position_covariance) > 8 else float("nan"),
                }
            )
        elif odom_msg_type is not None and topic == odom_topic:
            msg = deserialize_message(data, odom_msg_type)
            odom_times.append(stamp_to_float(msg, t_ns))
            odom_yaws.append(yaw_from_quaternion(msg.pose.pose.orientation))

    gnss_rows.sort(key=lambda r: r["timestamp"])
    return gnss_rows, odom_times, odom_yaws


def fill_xy_and_yaw(rows, target_crs, yaw_source, odom_times, odom_yaws, max_odom_dt):
    transformer = make_transformer(target_crs)

    for row in rows:
        x, y = transformer.transform(row["longitude"], row["latitude"])
        row["x"], row["y"] = float(x), float(y)

    for i, row in enumerate(rows):
        yaw = float("nan")
        if yaw_source in {"odom", "auto"}:
            yaw = nearest_yaw(row["timestamp"], odom_times, odom_yaws, max_odom_dt)
        if yaw_source in {"gnss_course", "auto"} and not math.isfinite(yaw):
            if i == 0 and len(rows) > 1:
                dx, dy = rows[i + 1]["x"] - row["x"], rows[i + 1]["y"] - row["y"]
            elif i > 0:
                dx, dy = row["x"] - rows[i - 1]["x"], row["y"] - rows[i - 1]["y"]
            else:
                dx, dy = 0.0, 0.0
            yaw = angle_from_delta(dx, dy)
        row["yaw"] = float(yaw if math.isfinite(yaw) else 0.0)

    return rows


def write_points(rows, out_csv: Path, trajectory_id: str):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory_id", "timestamp", "latitude", "longitude", "altitude",
        "x", "y", "yaw", "fix_status",
        "position_covariance_0", "position_covariance_4", "position_covariance_8",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            out["trajectory_id"] = trajectory_id
            writer.writerow(out)


def write_origin(rows, out_yaml: Path, target_crs: str, gnss_topic: str, odom_topic: str | None):
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    origin = {
        "latitude": float(first["latitude"]),
        "longitude": float(first["longitude"]),
        "altitude": float(first.get("altitude", 0.0)),
        "target_crs": target_crs,
        "gnss_topic": gnss_topic,
        "odom_topic": odom_topic,
    }
    out_yaml.write_text(yaml.safe_dump(origin, sort_keys=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract map_data6 /gnss bag data into One-Direction points.csv.")
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--gnss-topic", default="/gnss")
    parser.add_argument("--odom-topic", default="/copernicus_base_controller/odom")
    parser.add_argument("--no-odom", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("data/raw/trajectories/points.csv"))
    parser.add_argument("--origin-out", type=Path, default=Path("outputs/mapviz/map_data6_local_origin.yaml"))
    parser.add_argument("--trajectory-id", default="map_data6")
    parser.add_argument("--target-crs", default="EPSG:32643")
    parser.add_argument("--yaw-source", choices=["auto", "odom", "gnss_course"], default="auto")
    parser.add_argument("--max-odom-dt", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    odom_topic = None if args.no_odom else args.odom_topic
    gnss_rows, odom_times, odom_yaws = read_bag(args.bag, args.gnss_topic, odom_topic)
    if not gnss_rows:
        raise RuntimeError(f"No NavSatFix messages found on {args.gnss_topic}")
    rows = fill_xy_and_yaw(gnss_rows, args.target_crs, args.yaw_source, odom_times, odom_yaws, args.max_odom_dt)
    write_points(rows, args.out, args.trajectory_id)
    write_origin(rows, args.origin_out, args.target_crs, args.gnss_topic, odom_topic)
    print(f"[OK] wrote points: {args.out}")
    print(f"[OK] wrote origin: {args.origin_out}")
    print(f"[OK] rows: {len(rows)}")
    print(f"[OK] first lat/lon: {rows[0]['latitude']}, {rows[0]['longitude']}")
    print(f"[OK] target_crs: {args.target_crs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
