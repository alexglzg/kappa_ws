#!/usr/bin/env python3
"""Launch Scenario 1 with Nav2 SmacPlannerLattice."""

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "generated" / "scenario_1.yaml"
PARAMS = ROOT / "config" / "smac_lattice.yaml"
LATTICE = ROOT / "config" / "diff_drive_lattice.json"


def generate_launch_description():
    if not MAP.exists():
        raise RuntimeError(f"Generate the benchmark map first: missing {MAP}")
    return LaunchDescription([
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[str(PARAMS), {"yaml_filename": str(MAP)}],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[str(PARAMS), {
                "SmacLattice.lattice_filepath": str(LATTICE),
            }],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="scenario_1_start_tf",
            arguments=[
                "--x", "-0.15", "--y", "-0.35", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "-2.356194490192345",
                "--frame-id", "map", "--child-frame-id", "base_link",
            ],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager",
            output="screen",
            parameters=[str(PARAMS)],
        ),
    ])

