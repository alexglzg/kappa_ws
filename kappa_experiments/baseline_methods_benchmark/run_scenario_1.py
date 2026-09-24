#!/usr/bin/env python3
"""Call Nav2 SmacPlannerLattice repeatedly for Scenario 1."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node


ROOT = Path(__file__).resolve().parent


def yaw_to_quaternion(yaw: float):
    from geometry_msgs.msg import Quaternion
    return Quaternion(z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw))


class BenchmarkClient(Node):
    def __init__(self):
        super().__init__("scenario_1_lattice_benchmark")
        self.client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")

    def run_once(self, start: list[float], goal: list[float]) -> dict:
        request = ComputePathToPose.Goal()
        request.planner_id = "SmacLattice"
        request.use_start = True
        now = self.get_clock().now().to_msg()
        for message, pose in ((request.start, start), (request.goal, goal)):
            message.header.frame_id = "map"
            message.header.stamp = now
            message.pose.position.x = float(pose[0])
            message.pose.position.y = float(pose[1])
            message.pose.orientation = yaw_to_quaternion(float(pose[2]))

        before = time.perf_counter_ns()
        goal_future = self.client.send_goal_async(request)
        rclpy.spin_until_future_complete(self, goal_future)
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            return {"success": False, "error": "goal_rejected"}
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        after = time.perf_counter_ns()
        wrapped = result_future.result()
        if wrapped is None:
            return {"success": False, "error": "missing_action_result"}
        result = wrapped.result
        poses = []
        for stamped in result.path.poses:
            q = stamped.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            poses.append([stamped.pose.position.x, stamped.pose.position.y, yaw])
        planning_time = (result.planning_time.sec +
                         result.planning_time.nanosec * 1e-9)
        error_code = int(getattr(result, "error_code", 0))
        return {
            "success": bool(poses) and error_code == 0,
            "error_code": error_code,
            "error_msg": str(getattr(result, "error_msg", "")),
            "planning_time_s": planning_time,
            "round_trip_time_s": (after - before) * 1e-9,
            "poses": poses,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "scenario_1_runs.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = ROOT / "generated" / "scenario_1.json"
    if not metadata_path.exists():
        raise SystemExit(f"Generate the map first: missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rclpy.init()
    node = BenchmarkClient()
    try:
        if not node.client.wait_for_server(timeout_sec=args.server_timeout):
            raise RuntimeError("compute_path_to_pose action server is unavailable")
        warmups = [node.run_once(metadata["start_pose"], metadata["goal_pose"])
                   for _ in range(args.warmups)]
        runs = []
        for index in range(args.runs):
            result = node.run_once(metadata["start_pose"], metadata["goal_pose"])
            result["run_index"] = index + 1
            runs.append(result)
            print(f"run {index + 1:03d}: success={result['success']} "
                  f"planner={result.get('planning_time_s', math.nan):.6f} s "
                  f"round_trip={result.get('round_trip_time_s', math.nan):.6f} s")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    payload = {
        "planner": "nav2_smac_planner::SmacPlannerLattice",
        "planner_id": "SmacLattice",
        "scenario_metadata": str(metadata_path.relative_to(ROOT)),
        "warmup_count": args.warmups,
        "warmup_success": [item.get("success", False) for item in warmups],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

