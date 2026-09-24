#!/usr/bin/env python3
"""Generate the Scenario-1 corridor-union occupancy map and metadata."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from kappa_experiments.scenarios import build_scenario, build_unicycle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "generated"
RESOLUTION = 0.01
ROBOT_RADIUS = 0.17
MAP_PADDING = 0.30
FOOTPRINT_VERTICES = 16


def footprint(radius: float, vertices: int) -> list[list[float]]:
    return [
        [radius * math.cos(2.0 * math.pi * index / vertices),
         radius * math.sin(2.0 * math.pi * index / vertices)]
        for index in range(vertices)
    ]


def main() -> None:
    scenario = build_scenario(1, build_unicycle())
    polygons = [Polygon(np.asarray(corridor.corners)[:, :2])
                for corridor in scenario.corridor_list]
    free_space = unary_union(polygons)
    min_x, min_y, max_x, max_y = free_space.bounds
    origin_x = math.floor((min_x - MAP_PADDING) / RESOLUTION) * RESOLUTION
    origin_y = math.floor((min_y - MAP_PADDING) / RESOLUTION) * RESOLUTION
    upper_x = math.ceil((max_x + MAP_PADDING) / RESOLUTION) * RESOLUTION
    upper_y = math.ceil((max_y + MAP_PADDING) / RESOLUTION) * RESOLUTION
    width = int(round((upper_x - origin_x) / RESOLUTION))
    height = int(round((upper_y - origin_y) / RESOLUTION))

    # PGM row zero represents the top of the image. The map YAML origin is the
    # lower-left world coordinate; map_server handles the image/world flip.
    image = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        y = upper_y - (row + 0.5) * RESOLUTION
        for column in range(width):
            x = origin_x + (column + 0.5) * RESOLUTION
            if free_space.covers(Point(x, y)):
                image[row, column] = 254

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pgm_path = OUTPUT / "scenario_1.pgm"
    with pgm_path.open("wb") as stream:
        stream.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        stream.write(image.tobytes())

    yaml_data = {
        "image": pgm_path.name,
        "mode": "trinary",
        "resolution": RESOLUTION,
        "origin": [origin_x, origin_y, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    with (OUTPUT / "scenario_1.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(yaml_data, stream, sort_keys=False)

    metadata = {
        "scenario": scenario.number,
        "title": scenario.title,
        "start_pose": [float(value) for value in scenario.start_pose],
        "goal_pose": [float(value) for value in scenario.goal_pose],
        "corridors": [np.asarray(corridor.corners)[:, :2].tolist()
                      for corridor in scenario.corridor_list],
        "map": {
            "resolution": RESOLUTION,
            "origin": [origin_x, origin_y, 0.0],
            "width_cells": width,
            "height_cells": height,
            "bounds": [origin_x, origin_y, upper_x, upper_y],
            "free_cells": int(np.count_nonzero(image == 254)),
            "occupied_cells": int(np.count_nonzero(image == 0)),
        },
        "robot": {
            "model": "circular",
            "radius": ROBOT_RADIUS,
            "footprint": footprint(ROBOT_RADIUS, FOOTPRINT_VERTICES),
            "footprint_padding": 0.0,
        },
    }
    with (OUTPUT / "scenario_1.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    ax.imshow(image, cmap="gray", origin="upper",
              extent=[origin_x, upper_x, origin_y, upper_y], interpolation="nearest")
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, "--", color="#D55E00", lw=1.2)
    start = metadata["start_pose"]
    goal = metadata["goal_pose"]
    ax.scatter(start[0], start[1], color="#009E73", s=35, label="Start", zorder=3)
    ax.scatter(goal[0], goal[1], color="#D55E00", marker="*", s=80,
               label="Goal", zorder=3)
    ax.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]",
           title="Scenario 1 lattice benchmark map")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "scenario_1.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "scenario_1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {width} x {height} map at {RESOLUTION:.3f} m/cell")
    print(f"Free cells: {metadata['map']['free_cells']}; output: {OUTPUT}")


if __name__ == "__main__":
    main()

