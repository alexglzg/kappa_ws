"""Offline postprocessing of experiment rosbags.

Usage (inside the sourced workspace):
    ros2 run kappa_experiments postprocess /path/to/bag_dir [-o out_dir] \
        [--pose-topic /robot_pose] [--cmd-topic /rosbot3/cmd_vel]

Reads pose, command, plan_info and finished_tracking messages from a rosbag2
(sqlite3 or mcap), splits the bag into runs at each plan_info message, and
recomputes exactly the metrics of metrics_node (same code path,
kappa_experiments.metrics). Outputs:
  out_dir/summary.csv                one row per run
  out_dir/runNNN_map.png             corridors + reference + executed path
  out_dir/runNNN_errors.png          cross-track / tracking / heading vs time
  out_dir/runNNN_commands.png        commanded vs reference v, omega
  out_dir/runNNN.json                full summary incl. error series
Aggregate mean/std per scenario are printed at the end.
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from .metrics import (CSV_FIELDS, END_CRITERIA, Reference, RunRecorder, csv_row,
                      derive_velocities, summary_line)
from .trajectory_utils import yaw_from_quaternion


# ---------------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------------
def read_bag(bag_path, topics):
    """Yield (topic, t_seconds, message) for the requested topics."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [t for t in topics if t not in type_map]
    if missing:
        print(f'warning: topics not in bag: {missing}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in type_map]))
    classes = {t: get_message(type_map[t]) for t in topics if t in type_map}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, t_ns * 1e-9, deserialize_message(data, classes[topic])


def load_runs(bag_path, pose_topic, cmd_topic, plan_info_topic, finished_topic):
    """Split the bag into runs: one per plan_info message."""
    runs = []
    current = None
    for topic, t, msg in read_bag(bag_path, [pose_topic, cmd_topic,
                                             plan_info_topic, finished_topic]):
        if topic == plan_info_topic:
            info = json.loads(msg.data)
            current = {'info': info, 'recorder': RunRecorder(Reference.from_plan_info(info)),
                       'arm_time': t, 'finished_time': None}
            runs.append(current)
        elif current is None:
            continue
        elif topic == pose_topic:
            current['recorder'].add_pose(t, msg.pose.position.x, msg.pose.position.y,
                                         yaw_from_quaternion(msg.pose.orientation))
        elif topic == cmd_topic:
            twist = getattr(msg, 'twist', msg)          # TwistStamped or Twist
            stamp = getattr(msg, 'header', None)
            if stamp is not None and (stamp.stamp.sec or stamp.stamp.nanosec):
                t_cmd = stamp.stamp.sec + stamp.stamp.nanosec * 1e-9
            else:
                t_cmd = t
            current['recorder'].add_cmd(t_cmd, twist.linear.x, twist.angular.z)
        elif topic == finished_topic and getattr(msg, 'data', False):
            if current['finished_time'] is None:
                current['finished_time'] = t
    return runs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_run(run, summary, out_prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rec: RunRecorder = run['recorder']
    ref: Reference = rec.reference
    info = run['info']

    # ---- map ----
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.set_aspect('equal', adjustable='box')
    for key, style in (('corridors', dict(color='0.3', lw=1.5)),
                       ('shrunken_corridors', dict(color='0.6', lw=0.8, ls='--'))):
        for corners in info.get(key, []):
            c = np.asarray(corners)
            ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), **style)
    for circ in info.get('circles', []):
        ang = np.linspace(0, 2 * np.pi, 60)
        ax.plot(circ['center'][0] + circ['radius'] * np.cos(ang),
                circ['center'][1] + circ['radius'] * np.sin(ang), color='tab:cyan', lw=0.8)
    ax.plot(ref.x, ref.y, color='tab:orange', lw=2, label='reference')
    ax.plot(rec.x, rec.y, color='tab:blue', lw=1.2, label='executed')
    sp = info.get('start_pose_used', [ref.x[0], ref.y[0]])
    ax.plot(sp[0], sp[1], 'go', label='start')
    ax.plot(ref.goal_pose[0], ref.goal_pose[1], 'r*', ms=12, label='goal')
    ax.legend(loc='best', fontsize=8)
    ax.set_title(f"run {info.get('run_index')} scenario {info.get('scenario')} | "
                 + summary_line(summary).split('|')[0])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    fig.savefig(out_prefix + '_map.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    if not summary.get('valid'):
        return

    # ---- errors ----
    s = summary['series']
    t0 = summary['t_start']
    tt = np.asarray(s['t']) - t0
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(tt, np.asarray(s['cross_track']) * 100)
    axes[0].set_ylabel('cross-track [cm]')
    axes[1].plot(tt, np.asarray(s['tracking']) * 100)
    axes[1].set_ylabel('tracking [cm]')
    axes[2].plot(tt, np.rad2deg(s['heading_error']))
    axes[2].set_ylabel('heading err [deg]')
    axes[2].set_xlabel('t - t_start [s]')
    for tb in run['info'].get('piece_boundaries', []):
        for ax in axes:
            ax.axvline(tb, color='0.8', lw=0.6, zorder=0)
    axes[0].set_title(f"run {run['info'].get('run_index')} errors")
    fig.savefig(out_prefix + '_errors.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- commands ----
    if rec.cmd_t:
        ct = np.asarray(rec.cmd_t) - t0
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        axes[0].step(ct, rec.cmd_v, where='post', label='cmd')
        axes[0].step(ref.t, ref.v, where='post', label='reference', alpha=0.7)
        axes[1].step(ct, rec.cmd_w, where='post', label='cmd')
        axes[1].step(ref.t, ref.omega, where='post', label='reference', alpha=0.7)
        if rec.t:
            td, vd, wd = derive_velocities(rec.t, rec.x, rec.y, rec.theta)
            axes[0].plot(td - t0, vd, lw=0.7, alpha=0.7, label='measured (derived)')
            axes[1].plot(td - t0, wd, lw=0.7, alpha=0.7, label='measured (derived)')
        axes[0].set_ylabel('v [m/s]')
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel('omega [rad/s]')
        axes[1].set_xlabel('t - t_start [s]')
        fig.savefig(out_prefix + '_commands.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='rosbag2 directory')
    ap.add_argument('-o', '--out', default=None, help='output directory (default: <bag>/postprocess)')
    ap.add_argument('--pose-topic', default='/robot_pose')
    ap.add_argument('--cmd-topic', default='/rosbot3/cmd_vel')
    ap.add_argument('--plan-info-topic', default='/experiment_node/plan_info')
    ap.add_argument('--finished-topic', default='/finished_tracking')
    ap.add_argument('--goal-radius', type=float, default=0.05)
    ap.add_argument('--end-criterion', default='pose', choices=list(END_CRITERIA),
                    help='pose also requires the goal heading (metrics_node default)')
    ap.add_argument('--heading-tolerance', type=float, default=0.10,
                    help='rad, only used by --end-criterion pose')
    ap.add_argument('--start-event', default='first_nonzero_cmd',
                    choices=['first_cmd', 'first_nonzero_cmd', 'first_motion'])
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(args.bag, 'postprocess')
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.bag, args.pose_topic, args.cmd_topic,
                     args.plan_info_topic, args.finished_topic)
    if not runs:
        print('No plan_info messages found in the bag; nothing to do.')
        return 1

    rows = []
    by_scenario = defaultdict(list)
    for run in runs:
        rec: RunRecorder = run['recorder']
        info = run['info']
        if args.start_event == 'first_cmd':
            t_start = rec.first_cmd_time()
        elif args.start_event == 'first_motion':
            t_start = rec.first_motion_time()
        else:
            t_start = rec.first_nonzero_cmd_time()
        t_end = rec.goal_reached_time(args.goal_radius, args.end_criterion,
                                      args.heading_tolerance)
        reason = 'goal_reached' if t_end is not None else 'not_reached'
        summary = rec.summarize(t_start, t_end, args.goal_radius, args.end_criterion,
                                args.heading_tolerance)
        summary['end_reason'] = reason
        summary['finished_tracking_time'] = run['finished_time']

        run_idx = info.get('run_index', len(rows) + 1)
        prefix = os.path.join(out_dir, f'run{run_idx:03d}')
        print(f"run {run_idx} scenario {info.get('scenario')} [{reason}]: {summary_line(summary)}")
        raw = rec.raw_dict()
        if rec.t:
            td, vd, wd = derive_velocities(rec.t, rec.x, rec.y, rec.theta)
            raw['derived_velocities'] = {'t': td.tolist(), 'v': vd.tolist(),
                                         'omega': wd.tolist(),
                                         'note': 'finite differences of Vive poses, smoothed'}
        with open(prefix + '.json', 'w') as f:
            json.dump({'info': info, 'summary': summary, 'raw': raw}, f)
        rows.append(csv_row(info, summary, reason))
        if summary.get('valid'):
            by_scenario[info.get('scenario')].append(summary)
        if not args.no_plots:
            try:
                plot_run(run, summary, prefix)
            except Exception as exc:
                print(f'  (plotting failed: {exc})')

    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nPer-scenario aggregates (mean +/- std over valid runs, '
          f'end criterion {args.end_criterion}):')
    for scen in sorted(by_scenario):
        ss = by_scenario[scen]
        def agg(getter):
            vals = [getter(s) for s in ss if getter(s) is not None]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (math.nan, math.nan)
        et = agg(lambda s: s.get('execution_time'))
        # against the reference's own time to the goal circle, the part of the
        # plan the tracker actually executes (metrics.summarize)
        tr = agg(lambda s: s.get('time_ratio_to_radius'))
        cx = agg(lambda s: s['cross_track']['rms'])
        fe = agg(lambda s: s.get('final_position_error'))
        print(f'  scenario {scen} ({len(ss)} runs): exec {et[0]:.2f}+/-{et[1]:.2f} s | '
              f'time ratio (to radius) {tr[0]:.2f}+/-{tr[1]:.2f} | cross-track rms '
              f'{cx[0] * 100:.1f}+/-{cx[1] * 100:.1f} cm | final err '
              f'{fe[0] * 100:.1f}+/-{fe[1] * 100:.1f} cm')
    print(f'\nOutputs in {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
