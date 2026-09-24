#!/usr/bin/env python3
"""Validate Scenario-1 Smac lattice paths and create CSV/figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent


def wrap_to_pi(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def main() -> int:
    metadata = json.loads((ROOT / "generated" / "scenario_1.json").read_text())
    results_path = ROOT / "results" / "scenario_1_runs.json"
    if not results_path.exists():
        raise SystemExit(f"Run the benchmark first: missing {results_path}")
    results = json.loads(results_path.read_text())
    free_space = unary_union([Polygon(corridor) for corridor in metadata["corridors"]])
    radius = float(metadata["robot"]["radius"])
    goal = np.asarray(metadata["goal_pose"], dtype=float)
    rows = []
    valid_paths = []
    for run in results["runs"]:
        poses = np.asarray(run.get("poses", []), dtype=float)
        row = {
            "run_index": run.get("run_index"),
            "planner_success": bool(run.get("success")),
            "planning_time_s": run.get("planning_time_s"),
            "round_trip_time_s": run.get("round_trip_time_s"),
            "pose_count": len(poses),
        }
        if len(poses):
            line = LineString(poses[:, :2])
            swept = line.buffer(radius, cap_style=1, join_style=1, resolution=16)
            footprint_valid = free_space.buffer(1e-9).covers(swept)
            clearance = min(Point(x, y).distance(free_space.boundary)
                            for x, y in poses[:, :2]) - radius
            row.update({
                "path_length_m": float(np.sum(np.hypot(
                    np.diff(poses[:, 0]), np.diff(poses[:, 1])))),
                "footprint_valid": footprint_valid,
                "minimum_footprint_clearance_m": float(clearance),
                "final_position_error_m": float(np.linalg.norm(poses[-1, :2] - goal[:2])),
                "final_heading_error_rad": abs(wrap_to_pi(float(poses[-1, 2] - goal[2]))),
            })
            if footprint_valid:
                valid_paths.append((run, poses))
        rows.append(row)

    output_dir = ROOT / "results"
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "scenario_1_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row.get("planner_success")]
    print(f"attempted={len(rows)} planner_success={len(successful)} "
          f"footprint_valid={sum(bool(row.get('footprint_valid')) for row in rows)}")
    for key in ("planning_time_s", "round_trip_time_s", "path_length_m",
                "minimum_footprint_clearance_m", "final_position_error_m",
                "final_heading_error_rad"):
        values = np.asarray([float(row[key]) for row in successful if row.get(key) is not None])
        if len(values):
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            print(f"{key}: {median:.6f} [{q1:.6f}, {q3:.6f}], max {np.max(values):.6f}")

    if valid_paths:
        selected_run, selected = min(
            valid_paths, key=lambda item: float(item[0]["planning_time_s"])
        )
        fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
        for index, corridor in enumerate(metadata["corridors"]):
            polygon = np.asarray(corridor)
            closed = np.vstack((polygon, polygon[0]))
            ax.plot(closed[:, 0], closed[:, 1], "--", color="0.45", lw=1.0,
                    label="Corridor boundaries" if index == 0 else None)
        ax.plot(selected[:, 0], selected[:, 1], color="#0072B2", lw=1.8,
                label="Smac state-lattice path")
        start = metadata["start_pose"]
        ax.scatter(start[0], start[1], color="#009E73", s=35, label="Start", zorder=3)
        ax.scatter(goal[0], goal[1], color="#D55E00", marker="*", s=80,
                   label="Goal", zorder=3)
        ax.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]",
               title=f"Scenario 1 — Smac lattice run {selected_run['run_index']}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(output_dir / "scenario_1_lattice_path.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "scenario_1_lattice_path.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
