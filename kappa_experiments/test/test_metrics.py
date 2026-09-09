"""Offline tests of the metrics layer (no ROS, no kappa needed).

They cover what §6 of INTEGRATION_METRICS.md can be checked without a running
ROS 2 session: the geometry, the run lifecycle events, and above all that the
online path (metrics_node) and the offline path (postprocess) produce the same
summary, since both call metrics.RunRecorder.summarize.

    pytest-3 kappa_experiments/test/test_metrics.py
"""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kappa_experiments.metrics import (CSV_FIELDS, END_CRITERIA, Reference,  # noqa: E402
                                       RunRecorder, append_csv_row, csv_row,
                                       derive_velocities, point_to_polyline,
                                       summary_line)

DT = 0.1
GOAL_RADIUS = 0.05
HEADING_TOL = 0.10   # rad, = mpc_test_node.heading_tolerance
T0 = 1000.0          # wall clock at which the plan is published
LAG = 0.35           # tracking lag of the synthetic follower [s]


def make_plan_info():
    """A reference an experiment could produce: straight, quarter turn,
    straight, then a small turn on the spot at the goal."""
    v, omega, R = 0.5, 2.0, 0.25
    t, x, y, th = [0.0], [0.0], [0.0], [0.0]

    def push(dt, dx, dy, dth):
        t.append(t[-1] + dt)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        th.append(th[-1] + dth)

    for _ in range(20):                       # straight, 1 m
        push(DT, v * DT, 0.0, 0.0)
    for _ in range(int(round((math.pi / 2) / omega / DT))):   # left quarter turn
        push(DT, v * DT * math.cos(th[-1]), v * DT * math.sin(th[-1]), omega * DT)
    for _ in range(20):                       # straight again
        push(DT, v * DT * math.cos(th[-1]), v * DT * math.sin(th[-1]), 0.0)
    n_turn = 3
    for _ in range(n_turn):                   # turn on the spot at the goal
        push(DT, 0.0, 0.0, omega * DT)

    t = np.asarray(t)
    piece_times = [2.0, (math.pi / 2) / omega, 2.0, n_turn * DT]
    return {
        'session': 'test', 'run_index': 7, 'scenario': 1, 'start_pose_source': 'predefined',
        'stamp': T0, 'placement_error_xy': 0.0, 'placement_error_yaw': 0.0,
        'goal_pose': [x[-1], y[-1], th[-1]],
        'theoretical_time': float(t[-1]),
        'piece_times': piece_times,
        'piece_types': ['LinearSegmentUnicycle', 'CurvilinearArcUnicycle',
                        'LinearSegmentUnicycle', 'TurnOnTheSpot'],
        'piece_boundaries': np.cumsum([0.0] + piece_times).tolist(),
        'reference': {'t': t.tolist(), 'x': x, 'y': y, 'theta': th,
                      'v': [v] * len(t), 'omega': [0.0] * len(t)},
        'circles': [{'center': [1.0, R], 'radius': R, 'piece': 'arc'}],
        'corridors': [], 'shrunken_corridors': [],
    }


def simulate(rec: RunRecorder, ref: Reference, idle=2.0, dt=0.02):
    """2 s armed and idle, then follow the reference with a lag and a wobble."""
    t = 0.0
    while t < idle:
        rec.add_pose(T0 + t, ref.x[0], ref.y[0], ref.theta[0])
        rec.add_cmd(T0 + t, 0.0, 0.0)
        t += dt
    t_go = t
    while t - t_go <= ref.t[-1] + 1.0:
        tau = max(0.0, (t - t_go) - LAG)
        x, y, th = ref.pose_at(tau)
        wobble = 0.01 * math.sin(6.0 * tau)
        rec.add_pose(T0 + t, x + wobble, y - wobble, th + 0.02 * math.sin(3.0 * tau))
        k = min(int(np.searchsorted(ref.t, tau)), len(ref.v) - 1)
        rec.add_cmd(T0 + t, float(ref.v[k]), float(ref.omega[k]))
        t += dt
    return T0 + t_go


@pytest.fixture
def info():
    # round-trip, exactly as the JSON travels over ~/plan_info
    return json.loads(json.dumps(make_plan_info()))


@pytest.fixture
def ref(info):
    return Reference.from_plan_info(info)


def test_point_to_polyline_matches_brute_force():
    rng = np.random.default_rng(0)
    xs = np.cumsum(rng.normal(0, 1, 12))
    ys = np.cumsum(rng.normal(0, 1, 12))
    u = np.linspace(0, 1, 2001)
    for _ in range(100):
        px, py = rng.normal(0, 3, 2)
        d, _, _ = point_to_polyline(px, py, xs, ys)
        brute = min(np.min(np.hypot(px - (xs[i] + u * (xs[i + 1] - xs[i])),
                                    py - (ys[i] + u * (ys[i + 1] - ys[i]))))
                    for i in range(xs.size - 1))
        assert abs(d - brute) < 1e-6


def test_point_to_polyline_single_point():
    d, arc, k = point_to_polyline(1.0, 1.0, np.array([0.0]), np.array([0.0]))
    assert (round(d, 9), arc, k) == (round(math.sqrt(2), 9), 0.0, 0)


def test_reference_round_trip(info, ref):
    assert ref.t.size == len(info['reference']['t'])
    assert ref.theoretical_time == pytest.approx(info['theoretical_time'])
    assert ref.path_length() > 2.0
    x, y, th = ref.pose_at(-5.0)              # clamped to the ends
    assert (x, y) == (ref.x[0], ref.y[0])
    x, y, th = ref.pose_at(1e6)
    assert (x, y) == (ref.x[-1], ref.y[-1])


def test_online_and_offline_summaries_are_identical(info, ref):
    online = RunRecorder(ref)
    simulate(online, ref)
    t_start = online.first_nonzero_cmd_time(0.01, 0.05)
    t_end = online.goal_reached_time(GOAL_RADIUS)
    s_online = online.summarize(t_start, t_end, GOAL_RADIUS)

    # postprocess rebuilds the same recorder from the bag
    offline = RunRecorder(Reference.from_plan_info(json.loads(json.dumps(info))))
    for row in zip(online.t, online.x, online.y, online.theta):
        offline.add_pose(*row)
    for row in zip(online.cmd_t, online.cmd_v, online.cmd_w):
        offline.add_cmd(*row)
    assert offline.first_nonzero_cmd_time() == t_start
    assert offline.goal_reached_time(GOAL_RADIUS) == t_end
    s_offline = offline.summarize(t_start, t_end, GOAL_RADIUS)

    assert s_online['valid'] and s_offline['valid']
    for key in ('execution_time', 'time_ratio', 'final_position_error',
                'final_heading_error', 'executed_path_length', 'reference_path_length'):
        assert s_online[key] == pytest.approx(s_offline[key], abs=1e-12)
    for key in ('cross_track', 'tracking', 'heading_error'):
        for stat in ('mean', 'rms', 'max'):
            assert s_online[key][stat] == pytest.approx(s_offline[key][stat], abs=1e-12)
    assert summary_line(s_online) == summary_line(s_offline)


def test_start_event_ignores_the_idle_window(info, ref):
    rec = RunRecorder(ref)
    t_go = simulate(rec, ref, idle=2.0)
    assert rec.first_cmd_time() == pytest.approx(T0)             # first zero command
    assert rec.first_nonzero_cmd_time(0.01, 0.05) == pytest.approx(t_go, abs=0.03)
    # first_motion is necessarily the latest of the three: the robot has to
    # actually move motion_eps_xy after the command, here through a 0.35 s lag.
    assert rec.first_motion_time() == pytest.approx(t_go + LAG, abs=0.1)
    assert (rec.first_cmd_time() <= rec.first_nonzero_cmd_time(0.01, 0.05)
            <= rec.first_motion_time())


def test_lag_shows_in_tracking_not_in_cross_track(info, ref):
    rec = RunRecorder(ref)
    simulate(rec, ref)
    s = rec.summarize(rec.first_nonzero_cmd_time(), rec.goal_reached_time(GOAL_RADIUS),
                      GOAL_RADIUS)
    assert s['cross_track']['max'] < 0.02          # only the wobble
    assert s['tracking']['rms'] > s['cross_track']['rms']


def test_goal_check_is_position_only(info, ref):
    """The run closes on entering the goal circle, so a trajectory ending in a
    turn on the spot is cut before its final rotation. Documented behaviour:
    it matches the MPC tolerance_radius, which is why the timing baseline is
    theoretical_time_to_radius rather than the full theoretical_time."""
    rec = RunRecorder(ref)
    simulate(rec, ref)
    t_end = rec.goal_reached_time(GOAL_RADIUS)
    assert t_end is not None
    s = rec.summarize(rec.first_nonzero_cmd_time(), t_end, GOAL_RADIUS)
    d_ref = np.hypot(ref.x - ref.goal_pose[0], ref.y - ref.goal_pose[1])
    t_ref_in_circle = float(ref.t[np.argmax(d_ref <= GOAL_RADIUS)])
    assert t_ref_in_circle < ref.theoretical_time - info['piece_times'][-1] + 1e-9
    assert s['execution_time'] < ref.theoretical_time + LAG + 0.05
    assert s['final_heading_error'] > math.radians(5)   # last turn not executed yet


def test_time_to_radius_drops_the_trailing_rotation(info, ref):
    """A reference that ends in a turn on the spot enters the goal circle
    before the plan ends: the rotation costs distance-free time."""
    t_entry, theta_entry = ref.entry_into_goal_radius(GOAL_RADIUS)
    turn = info['piece_times'][-1]
    assert t_entry < ref.theoretical_time
    # the whole rotation is after the entry, and nothing much before it
    assert ref.theoretical_time - t_entry == pytest.approx(turn, abs=2 * DT)
    assert theta_entry == pytest.approx(ref.goal_pose[2] - turn * 2.0, abs=0.2)

    rec = RunRecorder(ref)
    simulate(rec, ref)
    s = rec.summarize(rec.first_nonzero_cmd_time(), rec.goal_reached_time(GOAL_RADIUS),
                      GOAL_RADIUS)
    assert s['theoretical_time_to_radius'] == pytest.approx(t_entry)
    assert s['reference_heading_at_entry'] == pytest.approx(theta_entry)
    # the to-radius baseline is the shorter one, so the ratio is the larger
    assert s['time_ratio_to_radius'] > s['time_ratio']
    assert s['time_overhead_to_radius'] > s['time_overhead']
    assert s['time_ratio_to_radius'] == pytest.approx(
        s['execution_time'] / s['theoretical_time_to_radius'])
    # the unexecuted rotation shows up against the goal heading, not against
    # the heading the tracker was actually asked to reach
    assert s['final_heading_error_vs_entry'] < s['final_heading_error']
    assert s['final_heading_error_vs_entry'] < math.radians(5)


def test_time_to_radius_on_a_straight_ending_reference():
    """With no trailing rotation the plan only loses the last goal_radius
    metres of the approach, i.e. goal_radius / v seconds."""
    v = 0.5
    t = np.arange(0, 4.0 + 1e-9, DT)
    x = v * t
    info = {
        'session': 'straight', 'run_index': 1, 'scenario': 9,
        'goal_pose': [float(x[-1]), 0.0, 0.0],
        'theoretical_time': float(t[-1]),
        'reference': {'t': t.tolist(), 'x': x.tolist(), 'y': [0.0] * t.size,
                      'theta': [0.0] * t.size, 'v': [v] * t.size,
                      'omega': [0.0] * t.size},
    }
    ref = Reference.from_plan_info(json.loads(json.dumps(info)))
    t_entry, theta_entry = ref.entry_into_goal_radius(GOAL_RADIUS)
    # entry is interpolated between samples, so it is exact, not sample-snapped
    assert ref.theoretical_time - t_entry == pytest.approx(GOAL_RADIUS / v, abs=1e-9)
    assert theta_entry == pytest.approx(0.0)

    rec = RunRecorder(ref)
    for i, ti in enumerate(np.arange(0.0, t[-1] + 1.0, 0.02)):
        xr, yr, th = ref.pose_at(float(ti))
        rec.add_pose(T0 + ti, xr, yr, th)
        rec.add_cmd(T0 + ti, v, 0.0)
    s = rec.summarize(rec.first_nonzero_cmd_time(), rec.goal_reached_time(GOAL_RADIUS),
                      GOAL_RADIUS)
    # a perfect tracker needs exactly the reference's time to the circle
    assert s['execution_time'] == pytest.approx(s['theoretical_time_to_radius'], abs=0.03)
    assert s['time_ratio_to_radius'] == pytest.approx(1.0, abs=0.02)
    assert s['time_ratio'] < s['time_ratio_to_radius']
    assert s['final_heading_error_vs_entry'] == pytest.approx(s['final_heading_error'], abs=1e-9)


def test_reference_that_never_enters_the_circle_falls_back(info):
    """Goal far off the path: the baseline degrades to the full plan time."""
    info = json.loads(json.dumps(info))
    info['goal_pose'] = [99.0, 99.0, 0.0]
    ref = Reference.from_plan_info(info)
    assert ref.entry_into_goal_radius(GOAL_RADIUS) is None
    rec = RunRecorder(ref)
    simulate(rec, ref)
    s = rec.summarize(rec.first_nonzero_cmd_time(), None, GOAL_RADIUS)
    assert s['theoretical_time_to_radius'] == pytest.approx(ref.theoretical_time)
    assert s['reference_heading_at_entry'] == pytest.approx(ref.theta[-1])
    assert s['time_ratio_to_radius'] == pytest.approx(s['time_ratio'])


def test_idle_run_never_starts_and_still_summarizes(ref):
    rec = RunRecorder(ref)
    for i in range(300):
        rec.add_pose(T0 + 0.02 * i, ref.x[0], ref.y[0], ref.theta[0])
        rec.add_cmd(T0 + 0.02 * i, 0.0, 0.0)
    assert rec.first_nonzero_cmd_time(0.01, 0.05) is None
    assert rec.first_motion_time() is None
    assert rec.goal_reached_time(GOAL_RADIUS) is None
    s = rec.summarize(None, None, GOAL_RADIUS)     # the timeout path
    assert s['valid'] and s['final_position_error'] > GOAL_RADIUS


def test_empty_run_is_invalid(ref):
    s = RunRecorder(ref).summarize(None, None, GOAL_RADIUS)
    assert s['valid'] is False
    assert 'run invalid' in summary_line(s)


def test_csv_row_fields(info, ref):
    rec = RunRecorder(ref)
    simulate(rec, ref)
    s = rec.summarize(rec.first_nonzero_cmd_time(), rec.goal_reached_time(GOAL_RADIUS),
                      GOAL_RADIUS)
    row = csv_row(info, s, 'goal_reached')
    assert set(row) == set(CSV_FIELDS)
    assert row['run_index'] == 7 and row['end_reason'] == 'goal_reached'
    # an invalid summary must not raise, just leave the metric cells empty
    empty = csv_row(info, RunRecorder(ref).summarize(None, None, GOAL_RADIUS), 'timeout')
    assert empty['cross_track_rms'] == '' and empty['end_reason'] == 'timeout'


def test_derived_velocities(ref):
    rec = RunRecorder(ref)
    t_go = simulate(rec, ref)
    t, v, w = derive_velocities(rec.t, rec.x, rec.y, rec.theta)
    mid = (t > t_go + 1.0) & (t < t_go + ref.t[-1] - 1.0)
    assert 0.3 < float(np.median(v[mid])) < 0.7        # reference runs at 0.5 m/s
    assert derive_velocities([0.0], [0.0], [0.0], [0.0])[1].size == 1   # too few samples


# ---------------------------------------------------------------------------
# end_criterion: 'pose' (goal circle + goal heading, what the MPC stops on)
# ---------------------------------------------------------------------------
def test_pose_mode_baseline_keeps_the_trailing_turn(info, ref):
    """In pose mode the reference only satisfies the criterion at the very end
    of its final rotation, so the baseline is the full plan minus the sliver
    where it is already inside the circle *and* already aligned."""
    t_pos, _ = ref.entry_into_goal_radius(GOAL_RADIUS)
    t_pose, theta_pose = ref.entry_into_goal_radius(GOAL_RADIUS, 'pose', HEADING_TOL)
    turn = info['piece_times'][-1]

    assert t_pos < t_pose <= ref.theoretical_time
    # the whole trailing turn is inside the pose-mode baseline again
    assert t_pose - t_pos == pytest.approx(turn, abs=2 * DT)
    # ... minus only the sliver in which the plan is already within tolerance
    sliver = ref.theoretical_time - t_pose
    assert 0.0 <= sliver < 2 * DT
    assert abs(float(np.arctan2(np.sin(theta_pose - ref.goal_pose[2]),
                                np.cos(theta_pose - ref.goal_pose[2])))) <= HEADING_TOL + 1e-9


def test_pose_mode_end_lands_after_the_measured_turn(info, ref):
    rec = RunRecorder(ref)
    simulate(rec, ref)
    t_end_pos = rec.goal_reached_time(GOAL_RADIUS)
    t_end_pose = rec.goal_reached_time(GOAL_RADIUS, 'pose', HEADING_TOL)
    assert t_end_pose is not None and t_end_pose > t_end_pos

    s_pose = rec.summarize(rec.first_nonzero_cmd_time(), t_end_pose, GOAL_RADIUS,
                           'pose', HEADING_TOL)
    s_pos = rec.summarize(rec.first_nonzero_cmd_time(), t_end_pos, GOAL_RADIUS)
    assert s_pose['end_criterion'] == 'pose'
    assert s_pose['heading_tolerance'] == HEADING_TOL
    assert s_pos['end_criterion'] == 'position'

    # the run now covers the rotation, so both the executed time and the
    # baseline grow, and the final heading error is inside the tolerance
    assert s_pose['execution_time'] > s_pos['execution_time']
    assert s_pose['theoretical_time_to_radius'] > s_pos['theoretical_time_to_radius']
    assert s_pose['final_heading_error'] <= HEADING_TOL
    # still a fair comparison: a lagging but otherwise perfect tracker stays ~1
    assert 0.9 < s_pose['time_ratio_to_radius'] < 1.3


def test_pose_mode_can_leave_a_run_unfinished(ref):
    """Reaching the circle without turning is not the end in pose mode."""
    rec = RunRecorder(ref)
    T = 1000.0
    gx, gy, gth = ref.goal_pose
    for i in range(50):                       # parked on the goal, wrong heading
        rec.add_pose(T + 0.02 * i, gx, gy, gth + 0.5)
        rec.add_cmd(T + 0.02 * i, 0.0, 0.0)
    assert rec.goal_reached_time(GOAL_RADIUS) == pytest.approx(T)
    assert rec.goal_reached_time(GOAL_RADIUS, 'pose', HEADING_TOL) is None


def test_position_stays_the_default_of_the_pure_functions():
    """The new argument must not change what an existing caller gets."""
    import inspect
    assert END_CRITERIA == ('position', 'pose')
    for fn in (RunRecorder.goal_reached_time, RunRecorder.summarize,
               Reference.entry_into_goal_radius):
        params = inspect.signature(fn).parameters
        assert params['end_criterion'].default == 'position', fn.__name__
        assert params['heading_tolerance'].default == pytest.approx(HEADING_TOL), fn.__name__


def test_append_csv_row_rotates_a_stale_header(tmp_path, info, ref):
    """A session CSV written by an older schema must not be appended to: the
    values would land one column off (as happened when end_criterion and the
    _to_radius fields were added)."""
    import csv as csv_mod
    rec = RunRecorder(ref)
    simulate(rec, ref)
    row = csv_row(info, rec.summarize(rec.first_nonzero_cmd_time(),
                                      rec.goal_reached_time(GOAL_RADIUS), GOAL_RADIUS),
                  'goal_reached')
    path = tmp_path / 'session.csv'

    # a file from an older version: same rows, fewer columns
    old_fields = [f for f in CSV_FIELDS if f not in ('end_criterion', 'time_ratio_to_radius')]
    with open(path, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=old_fields)
        w.writeheader()
        w.writerow({k: row[k] for k in old_fields})

    rotated = append_csv_row(str(path), row)
    assert rotated is not None and os.path.exists(rotated)
    with open(path, newline='') as f:
        rows = list(csv_mod.DictReader(f))
    assert list(rows[0]) == CSV_FIELDS
    assert rows[0]['end_criterion'] == 'position' and rows[0]['end_reason'] == 'goal_reached'

    # a matching header is appended to, not rotated
    assert append_csv_row(str(path), row) is None
    with open(path, newline='') as f:
        assert len(list(csv_mod.DictReader(f))) == 2
