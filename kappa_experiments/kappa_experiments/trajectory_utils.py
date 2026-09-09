"""Conversion of kappa analytical trajectories to a time-sampled reference.

Everything that touches attributes of kappa trajectory pieces lives here, so
that API differences only need to be fixed in one place:

- ``piece_to_arrays``   : one maneuver piece -> (t, x, y, theta, v, omega)
- ``resample_trajectory``: list of pieces -> single global time grid
- ``piece_circles``      : arc-construction circles for visualisation

Attribute names used for the unicycle pieces are the ones already relied on by
corridor_planner/core/plan_motion.py:
``maneuver_time``, ``resample(new_samples_number=)``, ``path_coordinates``,
``theta_trajectory``, ``forward_velocity``, ``angular_velocity``; they are
checked against kappa_planner/trajectory.py (``CurvilinearArcUnicycle``,
``LinearSegmentUnicycle``, ``TurnOnTheSpot``, ``BackwardArc``), which all
expose them plus ``time_grid`` and ``label``.
"""
from dataclasses import dataclass, field
from math import atan2, ceil, pi
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class SampledTrajectory:
    t: np.ndarray                      # (N,) seconds from the start of motion
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray                  # unwrapped, radians
    v: np.ndarray                      # forward velocity reference
    omega: np.ndarray                  # angular velocity reference
    dt: float
    total_time: float                  # sum of maneuver times (theoretical duration)
    piece_times: List[float]           # maneuver_time per piece
    piece_types: List[str]
    piece_boundaries: np.ndarray       # (n_pieces + 1,) cumulative times
    circles: List[Dict[str, Any]] = field(default_factory=list)

    def poses(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.theta])


def wrap_angle(a: float) -> float:
    return atan2(np.sin(a), np.cos(a))


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Piece adapter
# ---------------------------------------------------------------------------
def _as_1d(a, n: int) -> np.ndarray:
    """Return ``a`` as a float array of length ``n`` (broadcast scalars)."""
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size == 1 and n > 1:
        arr = np.full(n, float(arr[0]))
    return arr


def _local_time_grid(piece, n: int, T: float) -> np.ndarray:
    """Local time axis of a piece: its own ``time_grid`` shifted to start at 0.

    Every unicycle piece in kappa rebuilds ``time_grid`` as
    ``linspace(t0, tf, samples_number)`` with ``tf = t0 + maneuver_time``, both
    in ``__init__`` and in ``resample()``, so the grid is uniform in time and
    spans exactly ``maneuver_time``. It is used here instead of assuming that,
    and the assumption is verified: a grid of the wrong length or span (a piece
    whose ``time_grid`` was not refreshed together with ``path_coordinates``)
    falls back to a uniform axis.
    """
    tg = np.asarray(getattr(piece, 'time_grid', ()), dtype=float).reshape(-1)
    if tg.size == n and abs((tg[-1] - tg[0]) - T) <= 1e-9 + 1e-9 * T:
        return tg - tg[0]
    return np.linspace(0.0, T, n)


def piece_to_arrays(piece, dt_fine: float = 0.01):
    """Densely sample one maneuver piece on its own local time axis.

    Returns ``(t_local, x, y, theta, v, omega)`` where ``t_local`` runs from 0
    to ``piece.maneuver_time``, taken from the piece's own ``time_grid``
    (see :func:`_local_time_grid`).
    """
    T = float(piece.maneuver_time)
    n = max(int(ceil(T / dt_fine)) + 1, 2)
    if hasattr(piece, 'resample'):
        piece.resample(new_samples_number=n)

    xy = np.asarray(piece.path_coordinates, dtype=float)[:, :2]
    n = xy.shape[0]
    theta = _as_1d(piece.theta_trajectory, n)

    v = _as_1d(getattr(piece, 'forward_velocity', 0.0), n) if hasattr(piece, 'forward_velocity') \
        else np.zeros(n)
    omega = _as_1d(getattr(piece, 'angular_velocity', 0.0), n) if hasattr(piece, 'angular_velocity') \
        else np.zeros(n)

    # Velocity arrays are sometimes one shorter than the pose arrays (N-1 intervals).
    if v.size == n - 1:
        v = np.append(v, v[-1])
    if omega.size == n - 1:
        omega = np.append(omega, omega[-1])

    t_local = _local_time_grid(piece, n, T)
    return t_local, xy[:, 0], xy[:, 1], theta, v, omega


# Pieces that carry a construction circle. kappa draws a circle for exactly
# these two types in helpers.plot_helpers.plot_analytical_trajectory
# (plot_circles=True) -> Trajectory.plot_circle -> plot_helpers.plot_circle,
# which uses ``xc``, ``yc`` and ``radius``. ``label`` is set in the __init__ of
# every piece type and identifies the class ('segment', 'turn on-the-spot' and
# 'backward arc' being the other values); LinearSegmentUnicycle also has a
# ``radius`` (-100000, "for consistency"), so it must not be matched on that.
_CIRCLE_PIECE_LABELS = ('arc', 'backward arc')


def piece_circles(piece) -> List[Dict[str, Any]]:
    """Construction circle of an arc piece, in the same form kappa plots it.

    Returns ``[{'center': [xc, yc], 'radius': r, 'piece': label}]`` for a
    ``CurvilinearArcUnicycle`` or ``BackwardArc``, and ``[]`` for segments and
    turns on the spot.
    """
    if getattr(piece, 'label', None) not in _CIRCLE_PIECE_LABELS:
        return []
    try:
        return [{'center': [float(piece.xc), float(piece.yc)],
                 'radius': float(piece.radius),
                 'piece': piece.label}]
    except (AttributeError, TypeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Global resampling
# ---------------------------------------------------------------------------
def _zoh(t_query: np.ndarray, t_src: np.ndarray, y_src: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_src, t_query, side='right') - 1
    idx = np.clip(idx, 0, len(y_src) - 1)
    return y_src[idx]


def resample_trajectory(pieces: Sequence, dt: float, dt_fine: float = 0.01) -> SampledTrajectory:
    """Concatenate maneuver pieces and sample them on one global time grid.

    The grid is ``t_k = k * dt`` for ``k = 0 .. floor(T_total / dt)`` plus the
    exact end time ``T_total`` if it is not already on the grid. Poses are
    interpolated linearly (theta unwrapped first); v and omega are
    zero-order-held, since the analytical profiles are piecewise constant.

    With this construction ``len(t) - 1`` samples of ``dt`` equal the
    theoretical duration up to one ``dt``, independent of the number of
    pieces (unlike per-piece truncation).
    """
    t_all, x_all, y_all, th_all, v_all, w_all = [], [], [], [], [], []
    piece_times, piece_types, circles = [], [], []
    boundaries = [0.0]
    t_offset = 0.0

    for piece in pieces:
        T = float(piece.maneuver_time)
        piece_times.append(T)
        piece_types.append(type(piece).__name__)
        circles.extend(piece_circles(piece))
        if T <= 0.0:
            boundaries.append(t_offset)
            continue
        tl, x, y, th, v, w = piece_to_arrays(piece, dt_fine)
        t_all.append(tl + t_offset)
        x_all.append(x)
        y_all.append(y)
        th_all.append(th)
        v_all.append(v)
        w_all.append(w)
        t_offset += T
        boundaries.append(t_offset)

    if not t_all:
        raise ValueError('Trajectory has no pieces with positive duration.')

    t_src = np.concatenate(t_all)
    x_src = np.concatenate(x_all)
    y_src = np.concatenate(y_all)
    th_src = np.unwrap(np.concatenate(th_all))
    v_src = np.concatenate(v_all)
    w_src = np.concatenate(w_all)

    # Concatenation duplicates the junction sample of consecutive pieces
    # (identical time); keep monotone time for interpolation.
    order = np.argsort(t_src, kind='stable')
    t_src, x_src, y_src, th_src, v_src, w_src = (a[order] for a in
                                                 (t_src, x_src, y_src, th_src, v_src, w_src))
    # For duplicated times keep the *later* sample (start of the next piece),
    # so ZOH velocities switch exactly at the boundary.
    keep = np.concatenate([np.diff(t_src) > 1e-12, [True]])
    t_src, x_src, y_src, th_src, v_src, w_src = (a[keep] for a in
                                                 (t_src, x_src, y_src, th_src, v_src, w_src))

    total_time = t_offset
    n_steps = int(np.floor(total_time / dt + 1e-9))
    t = np.arange(n_steps + 1) * dt
    if total_time - t[-1] > 1e-6:
        t = np.append(t, total_time)

    return SampledTrajectory(
        t=t,
        x=np.interp(t, t_src, x_src),
        y=np.interp(t, t_src, y_src),
        theta=np.interp(t, t_src, th_src),
        v=_zoh(t, t_src, v_src),
        omega=_zoh(t, t_src, w_src),
        dt=dt,
        total_time=total_time,
        piece_times=piece_times,
        piece_types=piece_types,
        piece_boundaries=np.asarray(boundaries),
        circles=circles,
    )


def placement_error(measured, target):
    """(distance [m], |wrapped yaw error| [rad]) between two (x, y, yaw) poses."""
    d = float(np.hypot(measured[0] - target[0], measured[1] - target[1]))
    dyaw = abs(wrap_angle(measured[2] - target[2]))
    return d, dyaw


def deg(rad: float) -> float:
    return rad * 180.0 / pi
