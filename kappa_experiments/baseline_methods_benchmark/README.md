# Baseline methods benchmark

This directory contains an isolated first benchmark of Kappa's predefined
Scenario 1 against Nav2's `SmacPlannerLattice`. The baseline uses the
differential-drive lattice distributed with
[`alexglzg/corridor_navigation`](https://github.com/alexglzg/corridor_navigation):

- 0.01 m grid resolution;
- 16 heading bins;
- 192 motion primitives;
- 0.25 m turning radius;
- 32 exact zero-translation rotation primitives.

## Fair-space convention

The occupancy map is the union of the original Scenario-1 corridor polygons.
Cells inside the union are free and cells outside it are occupied. Corridors
are **not eroded**. Nav2 instead receives a 16-sided approximation of Kappa's
0.17 m circular robot footprint, with zero footprint padding and no inflation
layer. This avoids applying the footprint twice.

## Generate and inspect the map

From the workspace root, after building/sourcing the workspace:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/kappa_experiments/baseline_methods_benchmark/generate_scenario_1.py
```

This writes `generated/scenario_1.{pgm,yaml,json,pdf,png}`. The JSON file is
the authoritative benchmark metadata and records the exact polygons, poses,
map bounds, grid resolution, and footprint.

## Run SmacPlannerLattice

The ROS execution requires the Jazzy Nav2 packages, notably
`nav2_smac_planner`, `nav2_planner`, `nav2_map_server`, `nav2_lifecycle_manager`,
and `nav2_msgs`.

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch \
  src/kappa_experiments/baseline_methods_benchmark/launch/scenario_1_lattice.launch.py
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/kappa_experiments/baseline_methods_benchmark/run_scenario_1.py \
  --warmups 1 --runs 20
```

The runner saves every returned pose path, Nav2's reported planning duration,
and the external action round-trip duration to `results/scenario_1_runs.json`.

## Validate and plot

```bash
MPLCONFIGDIR=/tmp python3 \
  src/kappa_experiments/baseline_methods_benchmark/analyze_scenario_1.py
```

The validator checks the complete 0.17 m buffered path against the corridor
union, rather than checking only path vertices. It writes a CSV summary and a
PDF/PNG plot under `results/`.

## Timing interpretation

The first action call is a warm-up and is not reported. `planning_time_s` is
the duration returned by Nav2. `round_trip_time_s` additionally contains ROS
action communication and serialization overhead. Map loading, lifecycle
startup, lattice loading, plotting, and file writing are outside both values.

