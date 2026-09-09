# Metrics layer — integration notes (for Claude Code)

> **Status: integrated (2026-09-02).** The three files were moved out of the
> nested `kappa_experiments/kappa_experiments/kappa_experiments/` directory
> into the package module; §1 setup.py, §2 plan_info, §3 wiring (launch files +
> a `metrics_node` block in `config/experiment_params.yaml`), §4 record list
> and §5/§7 documentation are done — see the *Metrics* section of the package
> README. Of §6, only the offline checks could be run here (no ROS session):
> the ROS-level sanity runs are still open, and are listed at the end of this
> file.

New files (drop into `kappa_experiments/kappa_experiments/`):
- `metrics.py` — pure metric functions, no ROS imports. Unit-tested outside ROS.
- `metrics_node.py` — online node.
- `postprocess.py` — rosbag postprocessor (rosbag2_py).

## 1. setup.py

Add to `console_scripts`:
```python
'metrics_node = kappa_experiments.metrics_node:main',
'postprocess = kappa_experiments.postprocess:main',
```

## 2. experiment_node.py — extend plan_info (small edit)

Postprocessing draws the corridor map from `plan_info` alone (no kappa needed
offline). In `plan()`, where the `info` dict is built, add:

```python
info['corridors'] = [np.asarray(c.corners)[:, :2].tolist() for c in s.corridor_list]
info['shrunken_corridors'] = [np.asarray(c.corners)[:, :2].tolist()
                              for c in <the planner's shrunken_corridor_list>]
```

Use whatever variable now holds `planner.shrunken_corridor_list` after the
shrink fix. Keys are optional — `postprocess` skips them if absent — but the
map figures are much more useful with them.

## 3. Wiring / defaults

metrics_node defaults match the robot workspace:
`pose_topic:=/robot_pose` (the same in lab and sim),
`cmd_topic:=/rosbot3/cmd_vel` with `cmd_type:=TwistStamped`
(`/cmd_vel` + `Twist` when driving box_sim), `goal_radius:=0.05`
(= MPC `tolerance_radius` — keep them equal), `plan_info_topic` and
`finished_topic` as published today.

Run lifecycle: a `plan_info` message arms a run; the run starts at
`start_event` (`first_nonzero_cmd` default; `first_cmd` / `first_motion`
also computed and reported); it ends on the node's own goal-radius check,
`~/abort_run`, or timeout (`3 x theoretical + 10 s`). `/finished_tracking`
is recorded but does not end the run, so the metric stays independent of the
MPC's internal state.

Outputs per run: log line + `~/run_summary` (String JSON, latched → include it
in the bag), `<output_dir>/<session>.csv` (one row per run) and
`<session>_runNNN.json` (full error series). Default
`output_dir: ~/kappa_experiment_logs`.

Add to the launch file (both lab and sim), e.g.:
```python
Node(package='kappa_experiments', executable='metrics_node', name='metrics_node',
     output='screen', parameters=[{'pose_topic': ..., 'cmd_topic': ..., 'cmd_type': ...}]),
```

## 4. rosbag record list

```
/robot_pose
/rosbot3/cmd_vel  (or /cmd_vel in sim)
/rosbot2pro/planned_path
/rosbot2pro/state
/finished_tracking
/experiment_node/plan_info
/experiment_node/planned_controls
/experiment_node/experiment_markers
/metrics_node/run_summary
/metrics_node/live
```

## 5. Postprocess

```
ros2 run kappa_experiments postprocess <bag_dir> \
    --pose-topic /robot_pose --cmd-topic /rosbot3/cmd_vel
```
Produces `summary.csv`, per-run map / error / command figures, per-run JSON,
and prints per-scenario mean±std (execution time, time ratio, cross-track rms,
final error). Online and offline numbers come from the same
`metrics.RunRecorder.summarize`, so they agree by construction.

## 6. Sanity checks worth running in the ws  — STILL OPEN (need a ROS session)

- One sim run end-to-end: launch sim + experiment_node + metrics_node + the
  MPC (with `cmd` remapped/relayed to plain `Twist` for box_sim), call
  `/experiment_node/plan`, verify a CSV row appears and
  `postprocess` on the recorded bag reproduces the online summary.
- With the robot idle after `/plan`, verify the run stays `armed` (no start)
  and the timeout closes it with `end_reason: timeout`.
- `postprocess` on a bag with two runs in it (two `/plan` calls) splits them
  correctly.


## 7. Mapping to the recording requirements

| requirement | where it is saved |
|---|---|
| 1. timestamps per measurement | rosbag (all topics) + `raw.measured.t` / `raw.commands.t` in each run JSON |
| 2. x_ref, y_ref, theta_ref | `plan_info.reference` (per run) + `/rosbot2pro/planned_path` in the bag |
| 3. x_meas, y_meas, theta_meas | `/robot_pose` in the bag + `raw.measured` in each run JSON |
| 4. v_ref, omega_ref | `plan_info.reference.v/omega` + `/experiment_node/planned_controls` |
| 5. v_cmd, omega_cmd | `/rosbot3/cmd_vel` in the bag + `raw.commands` in each run JSON |
| 6. experiment/run ID, script start/end poses | `plan_info`: session, scenario, run_index, start_pose_predefined, start_pose_used, goal_pose |
| 7. total traversal time of the analytical solution | `plan_info.theoretical_time` (+ `piece_times`) |
| 8. actual start/end time | `summary.t_start` / `summary.t_end` / `execution_time`; `first_cmd_time`, `first_nonzero_cmd_time`, `first_motion_time` all stored |

If the requirement in 7 means the reference must start exactly at the script
poses, run with `start_pose_source:=predefined`.


## 8. What was verified offline (no ROS session available)

`test/test_metrics.py` (11 tests, `pytest-3 kappa_experiments/test/test_metrics.py`,
no ROS and no kappa needed — the reference is built analytically):

- `point_to_polyline` matches a brute-force 2001-point-per-segment search on
  200 random query points (< 1e-6 m).
- A real `plan_info` (scenario 1 planned with kappa) survives the JSON
  round-trip and rebuilds into a `Reference`.
- On a synthetic run (2 s idle, then reference tracking with a 0.35 s lag and
  a 1 cm wobble), the online path (`metrics_node`) and the offline path
  (`postprocess`) produce identical summaries to 1e-12, and the 2 s idle
  window is correctly excluded by `first_nonzero_cmd`.
- `csv_row` fills exactly `CSV_FIELDS`, and an invalid summary leaves the
  metric cells empty instead of raising.
- The goal check is position-only, so a trajectory ending in a turn on the spot
  is cut before the final rotation; the timing baseline is therefore
  `theoretical_time_to_radius` (when the reference itself enters the goal
  circle), with the full-plan fields kept alongside (see the README note).
- An all-idle run produces no start event, no goal event, and still
  summarizes (the `timeout` path).
- `derive_velocities` recovers the 0.5 m/s reference speed from finite
  differences of a wobbling pose signal, and survives a 1-sample input.

The three checks of §6 (sim run end-to-end, idle timeout on the real
lifecycle, two `/plan` calls in one bag) need a running ROS 2 + MPC.
