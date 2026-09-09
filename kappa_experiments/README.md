# kappa_experiments

Standalone real-robot tests of the kappa analytical planner (unicycle, circular
footprint) on predefined corridor scenarios projected on the lab floor.

## Workspace layout

```
ws/src/
  kappa-motion-planner/      # local library (kappa_planner importable from src/)
  kappa_experiments/         # this package
  demo_mpc/                  # rosbot_interface (pose -> state) + mpc_test_node
  box_sim/                   # simulator: publishes /robot_pose, takes /cmd_vel (Twist)
                             # and /rosbot3/cmd_vel (TwistStamped), like the robot
```

`kappa_experiments/__init__.py` adds `ws/src/kappa-motion-planner/src` to
`sys.path` at import time (same trick as `corridor_planner`).

```
cd ws && colcon build --symlink-install --packages-select kappa_experiments box_sim
source install/setup.bash
```

## Launch

Lab:  `ros2 launch kappa_experiments experiment.launch.py scenario:=1 session_name:=sept02`
Sim:  `ros2 launch kappa_experiments sim.launch.py scenario:=3`

Both bring up the whole stack: `rosbot_interface` + `mpc_test_node` +
`experiment_node` + `metrics_node` + RViz (the sim launch adds box_sim).
Switches: `mpc:=false` (drops interface and MPC — planning and drawing only),
`metrics:=false`, `rviz:=false`.

**Lab and sim are wired identically — the sim launch overrides no topic.**
The measured pose arrives on `/robot_pose` (PoseStamped, `map` frame) from the
lab publisher or from box_sim, and the single `pose_topic` argument (default
`/robot_pose`) is given to `experiment_node` and `metrics_node` *and* remapped
onto `rosbot_interface`'s input, so one argument moves all three. Commands are
the same on both sides too: `mpc_test_node` publishes TwistStamped on
`/rosbot3/cmd_vel` and box_sim subscribes to exactly that (as well as to its
original `Twist` on `/cmd_vel`), so `cmd_topic` / `cmd_type` never change.

> `/vive/pose` is the **raw tracker output in the tilted `vive_world` frame**
> and must never be used by anything here. Everything in this workspace works
> in `map`, and `/robot_pose` is the pose already expressed in it.
Remaining arguments: `goal_radius` (0.05 = the MPC's `tolerance_radius`),
`output_dir` (`~/kappa_experiment_logs`), plus the three below.

```
rosbot_interface   /robot_pose  -> /rosbot2pro/state
mpc_test_node      /rosbot2pro/state + /rosbot2pro/planned_path
                                -> /rosbot3/cmd_vel (TwistStamped), /finished_tracking
experiment_node    scenarios    -> /rosbot2pro/planned_path, plan_info, markers
metrics_node       /robot_pose + /rosbot3/cmd_vel -> run summaries
```

`map_frame` (default `map`) is the frame of everything RViz shows — markers,
planned path and robot ring all come from `experiment_node` — so that one
argument moves the whole scene if the lab ever renames the frame.

`control_rate` (default `10.0` Hz) is the single knob for the control
frequency. The launch derives the MPC's timer period and horizon
discretisation (`control_period = 1/rate`, `n_horizon = horizon_time * rate`,
`horizon_time` default 1.0 s) and `experiment_node`'s `sampling_dt = 1/rate`
from it, so the reference spacing and the control period cannot drift apart.
`sampling_dt` in `experiment_params.yaml` is overridden by this. Going to
20 Hz is `control_rate:=20.0` and nothing else.

### RViz configs

`rviz_config` selects the view:

- **`config/projector_lab.rviz`** (lab default) — the *calibrated* projector
  config: TopDownOrtho angle −1.5708, scale 557.14, view offset, 3840×2123
  window at X=1920. **Do not regenerate or reformat this file** (do not save
  over it from RViz); edit only display entries if a topic is ever added.
- **`config/floor_projection.rviz`** (sim default) — the laptop-screen view,
  no projector geometry.

Both use fixed frame `map`, matching the `map_frame` default.

## Run protocol

| step | command |
|---|---|
| select scenario (redraws the floor, wipes previous markers) | `ros2 param set /experiment_node scenario 2` |
| place the robot on the green START footprint; the white ring follows the Vive pose | — |
| start recording | `ros2 bag record /robot_pose /rosbot3/cmd_vel /rosbot2pro/planned_path /rosbot2pro/state /finished_tracking /experiment_node/plan_info /experiment_node/planned_controls /experiment_node/experiment_markers /metrics_node/run_summary /metrics_node/live -o <bag>` (same list in lab and sim) |
| plan (from the measured pose by default) | `ros2 service call /experiment_node/plan std_srvs/srv/Trigger` |
| (sim only) teleport the box robot to the nominal start | `ros2 service call /experiment_node/teleport_to_start std_srvs/srv/Trigger` |
| remove the trajectory drawing, keep the corridors | `ros2 service call /experiment_node/clear_plan std_srvs/srv/Trigger` |

The `/plan` response and the log line contain: run index, theoretical
duration, number of samples, planner computation time, per-piece durations,
and the placement error of the robot w.r.t. the predefined start pose.
`~/plan_info` (std_msgs/String, JSON, latched) carries the same plus the full
reference `(t, x, y, theta, v, omega)`, the arc circles and the corridor
polygons (`corridors`, `shrunken_corridors`), so postprocessing needs nothing
but the bag.

## Parameters worth knowing

- `start_pose_source`: `measured` (default) plans from the Vive pose at the
  time of the call; `predefined` always plans the nominal trajectory.
- `placement_tol_xy` / `placement_tol_yaw` / `enforce_placement_tolerance`:
  the placement error is always logged; with `enforce_...: true` the call is
  refused when the robot is outside tolerance.
- `assumptions`: passed to `MotionPlanner` (`standing`, as in the experiment script).
- `sampling_dt`: reference sample time; must equal the MPC period.
- `plot_*`, `line_width`, `text_height`, `marker_z`: floor drawing options,
  changeable at runtime with `ros2 param set` (markers are redrawn).

## Topics

| topic | type | note |
|---|---|---|
| `/robot_pose` (param `pose_topic`) | PoseStamped, `map` frame | measured pose in; same topic in lab and sim |
| `/rosbot3/cmd_vel` | TwistStamped | MPC output; box_sim subscribes to it too |
| `/rosbot2pro/planned_path` (param `path_topic`) | nav_msgs/Path, latched | pose `k` is at time `k*sampling_dt`; stamps = plan time + `t_k` |
| `/experiment_node/planned_controls` | Float64MultiArray, latched | rows `[t, v, omega]` |
| `/experiment_node/plan_info` | String (JSON), latched | full run summary |
| `/experiment_node/experiment_markers` | MarkerArray, latched | DELETEALL + corridors, shrunk corridors, start/goal, trajectory, circles |
| `/experiment_node/robot_marker` | Marker | ring around the measured pose |
| `/metrics_node/run_summary` | String (JSON), latched | one message per finished run |
| `/metrics_node/live` | String (JSON) | live cross-track / distance-to-goal while a run is active |

## Metrics

`metrics_node` (online) and `postprocess` (rosbag) share
`metrics.RunRecorder.summarize`, so their numbers agree by construction.

Run lifecycle: a `plan_info` message **arms** a run; it **starts** at
`start_event` (`first_nonzero_cmd` by default; `first_cmd` and `first_motion`
are computed and reported too); it **ends** on the node's own `end_criterion`
check, on `~/abort_run`, or on timeout (`3 x theoretical + 10 s`).
`/finished_tracking` is recorded but does not end the run, keeping the metric
independent of the MPC's internal state.

Per run: a log line, `~/run_summary` (String JSON, latched — record it),
one row in `<output_dir>/<session>.csv`, and
`<output_dir>/<session>_runNNN.json` with the full error series and the raw
measured poses and commands.

Offline, on a recorded bag:

```
ros2 run kappa_experiments postprocess <bag_dir>
```
(`--pose-topic` defaults to `/robot_pose` and `--cmd-topic` to
`/rosbot3/cmd_vel`, which are right for both lab and sim bags)

writes `summary.csv`, per-run map / error / command figures and per-run JSON,
and prints per-scenario mean±std. The map figures are drawn from `plan_info`
alone (it carries `corridors`, `shrunken_corridors` and `circles`), so
postprocessing needs no kappa install.

### End criterion

`end_criterion` decides when a run is over, and must match what the MPC stops
on — `mpc_test_node` finishes only once the reference is exhausted and then
requires **both** `tolerance_radius` (0.05 m) and `heading_tolerance`
(0.10 rad), giving up after `settle_timeout` (3 s) with whatever error
remains. The defaults here are the same two numbers:

- `pose` (default): within `goal_radius` **and** `heading_tolerance` of the
  goal pose — a trajectory ending in a turn on the spot is measured including
  that turn.
- `position`: the goal circle only, ignoring the heading.

The timing baseline applies the *same* criterion to the plan, so it stays fair
in either mode: `theoretical_time_to_radius` is when the reference itself
first satisfies it (interpolated between samples — in the distance, in the
heading error, or at the later of the two), and `time_ratio_to_radius` /
`time_overhead_to_radius` are measured against that. The ratio therefore
compares only the portion of the trajectory both the plan and the tracker
cover; nothing is subtracted by hand. Full-plan `theoretical_time`,
`time_ratio` and `time_overhead` are kept alongside, and a reference that
never satisfies the criterion falls back to the full `theoretical_time`.

`final_heading_error` is measured against the goal heading;
`final_heading_error_vs_entry` against `reference_heading_at_entry`, the
heading the tracker was actually asked to reach at the moment the criterion
was met.

A perfect tracker, both modes, `goal_radius` 0.05 m / `heading_tolerance`
0.10 rad:

| scenario | plan | mode | baseline | executed | `time_ratio_to_radius` | final heading err |
|---|---|---|---|---|---|---|
| 1 (0.25 s trailing turn) | 11.73 s | position | 11.38 s | 11.40 s | 1.00 | 37.6° |
| | | **pose** | 11.68 s | 11.68 s | 1.00 | 5.5° |
| 3 (0.40 s trailing turn) | 13.56 s | position | 13.06 s | 13.08 s | 1.00 | 55.0° |
| | | **pose** | 13.51 s | 13.52 s | 1.00 | 4.6° |

In `position` mode the trailing rotation is in neither number (hence the large
heading error, and a full-plan `time_ratio` of 0.97 that looks faster than the
plan); in `pose` mode it is in both. `postprocess` takes `--end-criterion` /
`--heading-tolerance` and defaults to `pose` as well — pass `--end-criterion
position` when re-analysing a bag recorded in position mode.

### Where each recording requirement is saved

| requirement | where |
|---|---|
| 1. timestamps per measurement | rosbag (all topics) + `raw.measured.t` / `raw.commands.t` per run JSON |
| 2. x_ref, y_ref, theta_ref | `plan_info.reference` + `/rosbot2pro/planned_path` |
| 3. x_meas, y_meas, theta_meas | `/robot_pose` in the bag + `raw.measured` |
| 4. v_ref, omega_ref | `plan_info.reference.v/omega` + `/experiment_node/planned_controls` |
| 5. v_cmd, omega_cmd | `/rosbot3/cmd_vel` in the bag + `raw.commands` |
| 6. run ID, script start/end poses | `plan_info`: session, scenario, run_index, start_pose_predefined, start_pose_used, goal_pose |
| 7. analytical traversal time | `plan_info.theoretical_time` (+ `piece_times`) |
| 8. actual start/end time | `summary.t_start` / `t_end` / `execution_time`, plus `first_cmd_time`, `first_nonzero_cmd_time`, `first_motion_time` |

The Vive gives poses only, so measured v/omega are finite differences of the
poses (`metrics.derive_velocities`, smoothed); they are stored in the
postprocess JSON as `raw.derived_velocities`, not treated as measurements.
If requirement 7 means the reference must start exactly at the script poses,
run with `start_pose_source:=predefined`.

## Checked against kappa

The three open points have been resolved against `kappa-motion-planner`:

- **Arc circles.** `plot_helpers.plot_analytical_trajectory(plot_circles=True)`
  draws a circle only for `CurvilinearArcUnicycle` and `BackwardArc`, through
  `plot_circle`, which uses `xc`, `yc`, `radius`.
  `trajectory_utils.piece_circles` now reads exactly those, selecting the two
  types by their `label` (`arc` / `backward arc`). Matching on `radius` alone
  would have been wrong: `LinearSegmentUnicycle` also carries one (`-100000`,
  "for consistency").
- **Shrunken corridors.** `MotionPlanner` shrinks with
  `shrink_corridor_list(corridor_list, 0.5 * vehicle.width)` and keeps the
  result in `shrunken_corridor_list`; `CorridorWorld.shrink(margin)` takes
  `2 * margin` off width *and* height, centre and tilt unchanged. The node now
  calls that same helper with `0.5 * unicycle.width` instead of using the
  robot radius `r`, and warns when a corridor is too narrow to leave free
  space. Numerically identical while the footprint is square
  (`0.34 x 0.34` -> `r = margin = 0.17`), but no longer a coincidence.
- **Uniform sampling.** Every piece rebuilds `time_grid` as
  `linspace(t0, t0 + maneuver_time, n)` in both `__init__` and `resample()`,
  so `resample(new_samples_number=n)` is uniform in time. `piece_to_arrays`
  no longer assumes it: it takes the piece's own `time_grid`, shifted to
  start at 0, and falls back to a uniform axis if the grid does not match the
  samples (checked over scenarios 1-4: 32/32 pieces use the grid, deviation
  from uniform 1.5e-15 s).
