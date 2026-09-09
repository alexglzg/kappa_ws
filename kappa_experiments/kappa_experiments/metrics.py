"""Run metrics for the analytical-planner experiments. No ROS imports here:
the same functions are used by metrics_node (online) and postprocess (rosbag),
and they are unit-testable without a ROS environment.

Definitions
-----------
cross-track error   distance from the measured position to the *geometric*
                    reference path (point-to-polyline). Insensitive to timing;
                    this is the "deviation due to non-smoothness" metric.
tracking error      distance to the *time-indexed* reference pose at
                    tau = t - t_start. Contains lag as well as geometry.
end criterion       'position': the robot is within goal_radius of the goal
                    position; 'pose': within goal_radius *and* within
                    heading_tolerance of the goal heading. Keep it equal to
                    what the MPC uses (mpc_test_node: 0.05 m / 0.10 rad, both
                    checked, and only once the reference is exhausted).
execution time      t_end - t_start, with t_start given by the chosen start
                    event (first command / first non-zero command / first
                    actual motion) and t_end when the robot first satisfies
                    the end criterion.
time to radius      the same criterion applied to the *reference*: the time at
                    which the analytical trajectory itself first satisfies it.
                    This is the fair baseline for the executed time, since the
                    tracker stops on the same condition and never executes what
                    the plan does afterwards. Under 'position' that drops a
                    trailing turn on the spot and the last goal_radius metres
                    of a straight approach; under 'pose' the rotation is part
                    of both the plan and the run again.
                    ``theoretical_time`` (the full plan) is kept alongside it.
"""
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


DEFAULT_GOAL_RADIUS = 0.05          # m, = mpc_test_node.tolerance_radius
DEFAULT_HEADING_TOLERANCE = 0.10    # rad, = mpc_test_node.heading_tolerance
END_CRITERIA = ('position', 'pose')


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _crossing_fraction(before: float, after: float, threshold: float) -> float:
    """Where a quantity crosses ``threshold`` between two samples, in [0, 1]."""
    span = before - after
    if abs(span) < 1e-12:
        return 1.0
    return min(max((before - threshold) / span, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def point_to_polyline(px: float, py: float, xs: np.ndarray, ys: np.ndarray):
    """Min distance from (px, py) to the polyline (xs, ys).

    Returns (distance, arc_length_at_closest_point, segment_index).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 1:
        return float(np.hypot(px - xs[0], py - ys[0])), 0.0, 0
    ax, ay = xs[:-1], ys[:-1]
    bx, by = xs[1:], ys[1:]
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    seg_len2 = np.where(seg_len2 < 1e-18, 1e-18, seg_len2)
    u = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    u = np.clip(u, 0.0, 1.0)
    cx, cy = ax + u * dx, ay + u * dy
    d = np.hypot(px - cx, py - cy)
    k = int(np.argmin(d))
    seg_len = np.sqrt(seg_len2)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return float(d[k]), float(cum[k] + u[k] * seg_len[k]), k


# ---------------------------------------------------------------------------
# Reference (from plan_info JSON)
# ---------------------------------------------------------------------------
@dataclass
class Reference:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    theoretical_time: float
    goal_pose: List[float]
    plan_info: dict = field(default_factory=dict)

    @classmethod
    def from_plan_info(cls, info) -> 'Reference':
        if isinstance(info, str):
            info = json.loads(info)
        ref = info['reference']
        return cls(
            t=np.asarray(ref['t'], dtype=float),
            x=np.asarray(ref['x'], dtype=float),
            y=np.asarray(ref['y'], dtype=float),
            theta=np.asarray(ref['theta'], dtype=float),
            v=np.asarray(ref['v'], dtype=float),
            omega=np.asarray(ref['omega'], dtype=float),
            theoretical_time=float(info['theoretical_time']),
            goal_pose=[float(v) for v in info['goal_pose']],
            plan_info=info,
        )

    def pose_at(self, tau: float):
        tau = min(max(tau, self.t[0]), self.t[-1])
        return (float(np.interp(tau, self.t, self.x)),
                float(np.interp(tau, self.t, self.y)),
                float(np.interp(tau, self.t, self.theta)))

    def path_length(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    def entry_into_goal_radius(self, goal_radius: float,
                               end_criterion: str = 'position',
                               heading_tolerance: float = DEFAULT_HEADING_TOLERANCE):
        """When the reference first satisfies the end criterion.

        'position': within ``goal_radius`` of the goal position. 'pose': that
        and within ``heading_tolerance`` of the goal heading, so that the
        baseline covers the same part of the plan the tracker executes in
        either mode.

        Returns ``(t, theta)`` at that crossing, linearly interpolated between
        the two bracketing samples (in the distance, in the heading error, or
        at the later of the two when both are still violated), or ``None`` if
        the reference never satisfies it.
        """
        gx, gy, gth = self.goal_pose
        d = np.hypot(self.x - gx, self.y - gy)
        h = np.abs(wrap_angle(self.theta - gth))
        satisfied = d <= goal_radius
        if end_criterion == 'pose':
            satisfied = satisfied & (h <= heading_tolerance)
        inside = np.flatnonzero(satisfied)
        if inside.size == 0:
            return None
        i = int(inside[0])
        if i == 0:
            return float(self.t[0]), float(self.theta[0])
        u = 0.0
        if d[i - 1] > goal_radius:
            u = max(u, _crossing_fraction(d[i - 1], d[i], goal_radius))
        if end_criterion == 'pose' and h[i - 1] > heading_tolerance:
            u = max(u, _crossing_fraction(h[i - 1], h[i], heading_tolerance))
        t = float(self.t[i - 1] + u * (self.t[i] - self.t[i - 1]))
        theta = float(self.theta[i - 1] + u * (self.theta[i] - self.theta[i - 1]))
        return t, theta


# ---------------------------------------------------------------------------
# Run accumulation
# ---------------------------------------------------------------------------
@dataclass
class RunRecorder:
    """Collects raw samples during a run; all statistics are computed at the
    end in ``summarize`` so that online and offline results are identical."""
    reference: Reference
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    cmd_t: List[float] = field(default_factory=list)
    cmd_v: List[float] = field(default_factory=list)
    cmd_w: List[float] = field(default_factory=list)

    def add_pose(self, t, x, y, theta):
        self.t.append(float(t))
        self.x.append(float(x))
        self.y.append(float(y))
        self.theta.append(float(theta))

    def add_cmd(self, t, v, w):
        self.cmd_t.append(float(t))
        self.cmd_v.append(float(v))
        self.cmd_w.append(float(w))

    # -- events ------------------------------------------------------------
    def first_cmd_time(self) -> Optional[float]:
        return self.cmd_t[0] if self.cmd_t else None

    def first_nonzero_cmd_time(self, v_eps=0.01, w_eps=0.05) -> Optional[float]:
        for t, v, w in zip(self.cmd_t, self.cmd_v, self.cmd_w):
            if abs(v) > v_eps or abs(w) > w_eps:
                return t
        return None

    def first_motion_time(self, eps_xy=0.01, eps_yaw=0.035) -> Optional[float]:
        if not self.t:
            return None
        x0, y0, th0 = self.x[0], self.y[0], self.theta[0]
        for t, x, y, th in zip(self.t, self.x, self.y, self.theta):
            if (math.hypot(x - x0, y - y0) > eps_xy or
                    abs(float(wrap_angle(th - th0))) > eps_yaw):
                return t
        return None

    def goal_reached_time(self, goal_radius=DEFAULT_GOAL_RADIUS,
                          end_criterion: str = 'position',
                          heading_tolerance=DEFAULT_HEADING_TOLERANCE) -> Optional[float]:
        """First measured sample satisfying the end criterion (see the module
        docstring): the goal circle alone, or the circle and the heading."""
        gx, gy, gth = self.reference.goal_pose
        check_heading = end_criterion == 'pose'
        for t, x, y, th in zip(self.t, self.x, self.y, self.theta):
            if math.hypot(x - gx, y - gy) > goal_radius:
                continue
            if check_heading and abs(float(wrap_angle(th - gth))) > heading_tolerance:
                continue
            return t
        return None

    def distance_to_goal(self) -> Optional[float]:
        if not self.t:
            return None
        gx, gy = self.reference.goal_pose[0], self.reference.goal_pose[1]
        return math.hypot(self.x[-1] - gx, self.y[-1] - gy)

    def cross_track_now(self) -> Optional[float]:
        if not self.t:
            return None
        d, _, _ = point_to_polyline(self.x[-1], self.y[-1],
                                    self.reference.x, self.reference.y)
        return d

    # -- summary -----------------------------------------------------------
    def summarize(self, t_start: Optional[float], t_end: Optional[float],
                  goal_radius=DEFAULT_GOAL_RADIUS, end_criterion: str = 'position',
                  heading_tolerance=DEFAULT_HEADING_TOLERANCE) -> dict:
        ref = self.reference
        t = np.asarray(self.t)
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        th = np.asarray(self.theta)

        # Baseline the tracker can actually be compared against: the plan up to
        # the point where it first enters the goal circle (see module docstring).
        entry = ref.entry_into_goal_radius(goal_radius, end_criterion, heading_tolerance)
        if entry is None:                      # never gets that close: full plan
            t_entry, theta_entry = ref.theoretical_time, float(ref.theta[-1])
        else:
            t_entry, theta_entry = entry

        out = {
            'n_pose_samples': int(t.size),
            't_start': t_start,
            't_end': t_end,
            'goal_radius': goal_radius,
            'end_criterion': end_criterion,
            'heading_tolerance': heading_tolerance,
            'theoretical_time': ref.theoretical_time,
            'theoretical_time_to_radius': t_entry,
            'reference_heading_at_entry': theta_entry,
            'reference_path_length': ref.path_length(),
        }
        if t.size == 0:
            out['valid'] = False
            return out

        if t_start is None:
            t_start = t[0]
        if t_end is None:
            t_end = t[-1]
        mask = (t >= t_start) & (t <= t_end)
        if not np.any(mask):
            out['valid'] = False
            return out
        tm, xm, ym, thm = t[mask], x[mask], y[mask], th[mask]

        cross = np.empty(tm.size)
        arc = np.empty(tm.size)
        for i in range(tm.size):
            cross[i], arc[i], _ = point_to_polyline(xm[i], ym[i], ref.x, ref.y)

        rx = np.interp(tm - t_start, ref.t, ref.x)
        ry = np.interp(tm - t_start, ref.t, ref.y)
        rth = np.interp(tm - t_start, ref.t, ref.theta)
        track = np.hypot(xm - rx, ym - ry)
        heading_err = np.abs(wrap_angle(thm - rth))

        executed_length = float(np.sum(np.hypot(np.diff(xm), np.diff(ym))))
        execution_time = float(t_end - t_start)
        gx, gy, gth = ref.goal_pose

        def stats(a):
            return {'mean': float(np.mean(a)), 'rms': float(np.sqrt(np.mean(a ** 2))),
                    'max': float(np.max(a))}

        out.update({
            'valid': True,
            'execution_time': execution_time,
            'time_overhead': execution_time - ref.theoretical_time,
            'time_ratio': (execution_time / ref.theoretical_time
                           if ref.theoretical_time > 0 else None),
            'time_overhead_to_radius': execution_time - t_entry,
            'time_ratio_to_radius': (execution_time / t_entry if t_entry > 0 else None),
            'cross_track': stats(cross),
            'tracking': stats(track),
            'heading_error': stats(heading_err),
            'executed_path_length': executed_length,
            'path_length_ratio': (executed_length / out['reference_path_length']
                                  if out['reference_path_length'] > 0 else None),
            'final_position_error': float(math.hypot(xm[-1] - gx, ym[-1] - gy)),
            # vs the goal heading (includes a trailing rotation the tracker
            # never executes) and vs the reference heading at circle entry
            # (what the tracker was actually asked to reach).
            'final_heading_error': float(abs(wrap_angle(thm[-1] - gth))),
            'final_heading_error_vs_entry': float(abs(wrap_angle(thm[-1] - theta_entry))),
            'series': {'t': tm.tolist(), 'cross_track': cross.tolist(),
                       'tracking': track.tolist(), 'heading_error': heading_err.tolist(),
                       'arc_position': arc.tolist()},
        })
        return out


    def raw_dict(self) -> dict:
        """Raw recorded data for the per-run file: measured poses and the
        commands sent to the robot, with their timestamps."""
        return {
            'measured': {'t': list(self.t), 'x': list(self.x), 'y': list(self.y),
                         'theta': list(self.theta)},
            'commands': {'t': list(self.cmd_t), 'v': list(self.cmd_v),
                         'omega': list(self.cmd_w)},
        }


def derive_velocities(t, x, y, theta, smooth_window: int = 5):
    """v_meas / omega_meas DERIVED from measured poses by central differences
    (the Vive gives poses only). ``smooth_window`` is a moving-average length
    in samples applied to the derivatives; use an odd number.

    v is signed by projection of the displacement on the heading, so reversing
    shows up as negative v.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    th = np.unwrap(np.asarray(theta, dtype=float))
    if t.size < 3:
        z = np.zeros_like(t)
        return t, z, z
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-6, 1e-6, dt)
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    v = vx * np.cos(th) + vy * np.sin(th)
    omega = np.gradient(th) / dt
    if smooth_window and smooth_window > 1:
        k = int(smooth_window) | 1
        kern = np.ones(k) / k
        v = np.convolve(v, kern, mode='same')
        omega = np.convolve(omega, kern, mode='same')
    return t, v, omega


def summary_line(s: dict) -> str:
    if not s.get('valid'):
        return 'run invalid (no samples in [t_start, t_end])'
    t_ref = s.get('theoretical_time_to_radius', s['theoretical_time'])
    ratio = s.get('time_ratio_to_radius', s['time_ratio'])
    return (f"exec {s['execution_time']:.2f} s vs {t_ref:.2f} s to radius "
            f"(x{ratio:.2f}) | cross-track mean {s['cross_track']['mean'] * 100:.1f} / "
            f"rms {s['cross_track']['rms'] * 100:.1f} / max {s['cross_track']['max'] * 100:.1f} cm | "
            f"tracking rms {s['tracking']['rms'] * 100:.1f} cm | "
            f"final err {s['final_position_error'] * 100:.1f} cm, "
            f"{s['final_heading_error'] * 180 / math.pi:.1f} deg")


CSV_FIELDS = [
    'session', 'run_index', 'scenario', 'start_pose_source', 'stamp', 'end_criterion',
    'placement_error_xy', 'placement_error_yaw',
    'theoretical_time', 'theoretical_time_to_radius', 'execution_time',
    'time_overhead', 'time_ratio', 'time_overhead_to_radius', 'time_ratio_to_radius',
    'cross_track_mean', 'cross_track_rms', 'cross_track_max',
    'tracking_mean', 'tracking_rms', 'tracking_max',
    'heading_error_rms', 'final_position_error', 'final_heading_error',
    'final_heading_error_vs_entry',
    'reference_path_length', 'executed_path_length', 'path_length_ratio',
    'end_reason',
]


def csv_row(info: dict, summary: dict, end_reason: str) -> dict:
    def g(d, *keys):
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if not isinstance(d, dict) or d else ''
    row = {
        'session': info.get('session', ''),
        'run_index': info.get('run_index', ''),
        'scenario': info.get('scenario', ''),
        'start_pose_source': info.get('start_pose_source', ''),
        'stamp': info.get('stamp', ''),
        'end_criterion': summary.get('end_criterion', ''),
        'placement_error_xy': info.get('placement_error_xy', ''),
        'placement_error_yaw': info.get('placement_error_yaw', ''),
        'theoretical_time': summary.get('theoretical_time', ''),
        'theoretical_time_to_radius': summary.get('theoretical_time_to_radius', ''),
        'execution_time': summary.get('execution_time', ''),
        'time_overhead': summary.get('time_overhead', ''),
        'time_ratio': summary.get('time_ratio', ''),
        'time_overhead_to_radius': summary.get('time_overhead_to_radius', ''),
        'time_ratio_to_radius': summary.get('time_ratio_to_radius', ''),
        'cross_track_mean': g(summary, 'cross_track', 'mean'),
        'cross_track_rms': g(summary, 'cross_track', 'rms'),
        'cross_track_max': g(summary, 'cross_track', 'max'),
        'tracking_mean': g(summary, 'tracking', 'mean'),
        'tracking_rms': g(summary, 'tracking', 'rms'),
        'tracking_max': g(summary, 'tracking', 'max'),
        'heading_error_rms': g(summary, 'heading_error', 'rms'),
        'final_position_error': summary.get('final_position_error', ''),
        'final_heading_error': summary.get('final_heading_error', ''),
        'final_heading_error_vs_entry': summary.get('final_heading_error_vs_entry', ''),
        'reference_path_length': summary.get('reference_path_length', ''),
        'executed_path_length': summary.get('executed_path_length', ''),
        'path_length_ratio': summary.get('path_length_ratio', ''),
        'end_reason': end_reason,
    }
    return row


def append_csv_row(path: str, row: dict, fieldnames=CSV_FIELDS) -> Optional[str]:
    """Append one run to ``<session>.csv``, writing the header on creation.

    A file whose header does not match ``fieldnames`` was written by an older
    schema: appending to it would silently shift every value one column. Such a
    file is rotated aside and a fresh one started instead. Returns the path the
    old file was moved to, or None.
    """
    rotated = None
    if os.path.exists(path):
        with open(path, newline='') as f:
            header = next(csv.reader(f), None)
        if header is not None and header != list(fieldnames):
            rotated = f'{path}.{time.strftime("%Y%m%d_%H%M%S")}.old'
            os.rename(path, rotated)
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return rotated
