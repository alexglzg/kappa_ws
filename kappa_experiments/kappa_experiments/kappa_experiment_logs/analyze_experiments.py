#!/usr/bin/env python3
"""Analyze Kappa real-robot trajectory-tracking experiment logs.

The script deliberately distinguishes:

* reference controls: analytical planner ``v`` and ``omega``;
* commanded controls: controller output observed on ``/rosbot3/cmd_vel``;
* measured poses: Vive/localization ``x``, ``y``, and ``theta``.

It never interprets commanded controls as measured robot velocities.  Runs are
eligible for normal statistics only when their end reason is ``goal_reached``
and they contain finite, ordered ``summary.t_start`` and ``summary.t_end``.
Reference time is relative; measured and command timestamps are absolute.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SUCCESS_REASON = "goal_reached"
POSE_KEYS = ("t", "x", "y", "theta")
REFERENCE_KEYS = ("t", "x", "y", "theta", "v", "omega")
CONTROL_KEYS = ("t", "v", "omega")
MAIN_METRICS = (
    "position_error_rms",
    "position_error_p95",
    "position_error_max",
    "path_error_rms",
    "path_error_p95",
    "path_error_max",
    "heading_abs_error_rms",
    "final_position_error",
    "final_heading_abs_error",
    "execution_time",
    "execution_time_increase_percent",
)


@dataclass(frozen=True)
class Run:
    path: Path
    relative_path: str
    batch: str
    info: dict[str, Any]
    summary: dict[str, Any]
    reference: dict[str, np.ndarray]
    measured: dict[str, np.ndarray]
    commands: dict[str, np.ndarray]

    @property
    def run_id(self) -> str:
        return f"{self.info['session']}_run{int(self.info['run_index']):03d}"

    @property
    def normally_completed(self) -> bool:
        start = self.summary.get("t_start")
        end = self.summary.get("t_end")
        return (
            self.summary.get("end_reason") == SUCCESS_REASON
            and finite_scalar(start)
            and finite_scalar(end)
            and float(end) >= float(start)
        )


def finite_scalar(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def require_mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key!r} is missing or is not an object")
    return value


def numeric_series(
    obj: dict[str, Any], keys: Sequence[str], label: str, path: Path
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in keys:
        value = obj.get(key)
        if not isinstance(value, list):
            raise ValueError(f"{path}: {label}.{key} is missing or is not an array")
        result[key] = np.asarray(value, dtype=float)
    lengths = {key: len(value) for key, value in result.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path}: unequal {label} array lengths: {lengths}")
    if not lengths or next(iter(lengths.values())) == 0:
        raise ValueError(f"{path}: empty {label} arrays")
    for key, value in result.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{path}: {label}.{key} contains non-finite values")
    dt = np.diff(result["t"])
    if np.any(dt <= 0.0):
        duplicates = int(np.count_nonzero(dt == 0.0))
        backwards = int(np.count_nonzero(dt < 0.0))
        raise ValueError(
            f"{path}: {label}.t is not strictly increasing "
            f"({duplicates} duplicates, {backwards} backwards steps)"
        )
    return result


def load_run(path: Path, root: Path) -> Run:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value is not an object")
    info = require_mapping(data, "info", path)
    summary = require_mapping(data, "summary", path)
    raw = require_mapping(data, "raw", path)
    reference = numeric_series(
        require_mapping(info, "reference", path), REFERENCE_KEYS, "info.reference", path
    )
    measured = numeric_series(
        require_mapping(raw, "measured", path), POSE_KEYS, "raw.measured", path
    )
    commands = numeric_series(
        require_mapping(raw, "commands", path), CONTROL_KEYS, "raw.commands", path
    )
    for key in (
        "session",
        "run_index",
        "scenario",
        "title",
        "start_pose_predefined",
        "start_pose_used",
        "goal_pose",
    ):
        if key not in info:
            raise ValueError(f"{path}: info.{key} is missing")
    if not (finite_scalar(info.get("theoretical_time")) or
            finite_scalar(summary.get("theoretical_time"))):
        raise ValueError(f"{path}: no finite theoretical time")
    relative = path.relative_to(root)
    # The input batch is the directory below the numbered scenario directory.
    batch = "/".join(relative.parts[:2])
    return Run(
        path=path,
        relative_path=str(relative),
        batch=batch,
        info=info,
        summary=summary,
        reference=reference,
        measured=measured,
        commands=commands,
    )


def discover_runs(root: Path) -> list[Path]:
    """Find real-robot JSON runs while excluding auxiliary directories."""
    return sorted(root.glob("[0-9]*/input_*/*.json"))


def interpolate_reference_pose(run: Run) -> dict[str, np.ndarray]:
    """Synchronize reference pose to measured samples during execution.

    The analytical pose is linearly interpolated. Heading is unwrapped before
    interpolation. If execution continues after the analytical trajectory has
    ended, NumPy's documented endpoint behavior holds the terminal reference
    pose; the number of such samples is recorded in the output metrics.
    """
    start = float(run.summary["t_start"])
    end = float(run.summary["t_end"])
    mask = (run.measured["t"] >= start) & (run.measured["t"] <= end)
    if not np.any(mask):
        raise ValueError(f"{run.relative_path}: no measured samples in execution interval")
    absolute_t = run.measured["t"][mask]
    t_rel = absolute_t - start
    ref_t = run.reference["t"]
    theta_unwrapped = np.unwrap(run.reference["theta"])
    return {
        "t_absolute": absolute_t,
        "t_rel": t_rel,
        "x_measured": run.measured["x"][mask],
        "y_measured": run.measured["y"][mask],
        "theta_measured": run.measured["theta"][mask],
        "x_reference": np.interp(t_rel, ref_t, run.reference["x"]),
        "y_reference": np.interp(t_rel, ref_t, run.reference["y"]),
        "theta_reference": np.interp(t_rel, ref_t, theta_unwrapped),
        "samples_after_reference_end": np.asarray(
            [np.count_nonzero(t_rel > ref_t[-1])], dtype=int
        ),
    }


def synchronize_controls(run: Run) -> dict[str, np.ndarray]:
    """Interpolate reference controls at commanded-control timestamps.

    Only timestamps within the execution interval and analytical reference
    support are used. No control extrapolation is performed.
    """
    start = float(run.summary["t_start"])
    end = float(run.summary["t_end"])
    t_rel_all = run.commands["t"] - start
    ref_t = run.reference["t"]
    mask = (
        (run.commands["t"] >= start)
        & (run.commands["t"] <= end)
        & (t_rel_all >= ref_t[0])
        & (t_rel_all <= ref_t[-1])
    )
    t_rel = t_rel_all[mask]
    return {
        "t_rel": t_rel,
        "v_commanded": run.commands["v"][mask],
        "omega_commanded": run.commands["omega"][mask],
        "v_reference": np.interp(t_rel, ref_t, run.reference["v"]),
        "omega_reference": np.interp(t_rel, ref_t, run.reference["omega"]),
    }


def descriptive(values: np.ndarray, prefix: str, include_mean: bool = True) -> dict[str, float]:
    if values.size == 0:
        result = {
            f"{prefix}_median": math.nan,
            f"{prefix}_rms": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }
        if include_mean:
            result[f"{prefix}_mean"] = math.nan
        return result
    result = {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }
    if include_mean:
        result[f"{prefix}_mean"] = float(np.mean(values))
    return result


def point_to_polyline_distances(
    points: np.ndarray, vertices: np.ndarray, chunk_size: int = 1000
) -> np.ndarray:
    """Minimum Euclidean distance from each point to polyline line segments."""
    if len(vertices) == 1:
        return np.linalg.norm(points - vertices[0], axis=1)
    starts = vertices[:-1]
    segments = vertices[1:] - starts
    length_squared = np.einsum("ij,ij->i", segments, segments)
    output = np.empty(len(points), dtype=float)
    for begin in range(0, len(points), chunk_size):
        block = points[begin:begin + chunk_size]
        relative = block[:, None, :] - starts[None, :, :]
        projection = np.zeros((len(block), len(starts)), dtype=float)
        nonzero = length_squared > 0.0
        projection[:, nonzero] = (
            np.einsum("bsi,si->bs", relative[:, nonzero, :], segments[nonzero])
            / length_squared[nonzero]
        )
        projection = np.clip(projection, 0.0, 1.0)
        closest = starts[None, :, :] + projection[:, :, None] * segments[None, :, :]
        distances = np.linalg.norm(block[:, None, :] - closest, axis=2)
        output[begin:begin + len(block)] = np.min(distances, axis=1)
    return output


def nearest_final_pose(run: Run) -> tuple[np.ndarray, float]:
    end = float(run.summary["t_end"])
    index = int(np.argmin(np.abs(run.measured["t"] - end)))
    pose = np.array([
        run.measured["x"][index],
        run.measured["y"][index],
        run.measured["theta"][index],
    ])
    return pose, float(run.measured["t"][index] - end)


def base_row(run: Run) -> dict[str, Any]:
    info = run.info
    summary = run.summary
    start = summary.get("t_start")
    end = summary.get("t_end")
    return {
        "run_id": run.run_id,
        "relative_path": run.relative_path,
        "batch": run.batch,
        "scenario": info.get("scenario"),
        "title": info.get("title"),
        "session": info.get("session"),
        "run_index": info.get("run_index"),
        "end_reason": summary.get("end_reason"),
        "normally_completed": run.normally_completed,
        "exclusion_reason": "" if run.normally_completed else exclusion_reason(run),
        "t_start": start,
        "t_end": end,
        "stored_execution_time": summary.get("execution_time"),
        "theoretical_time": theoretical_time(run),
        "start_pose_predefined": json.dumps(info.get("start_pose_predefined")),
        "start_pose_used": json.dumps(info.get("start_pose_used")),
        "goal_pose": json.dumps(info.get("goal_pose")),
        "n_reference_samples": len(run.reference["t"]),
        "n_measured_samples_total": len(run.measured["t"]),
        "n_command_samples_total": len(run.commands["t"]),
    }


def exclusion_reason(run: Run) -> str:
    if run.summary.get("end_reason") != SUCCESS_REASON:
        return f"end_reason={run.summary.get('end_reason')}"
    if not finite_scalar(run.summary.get("t_start")):
        return "missing_or_invalid_t_start"
    if not finite_scalar(run.summary.get("t_end")):
        return "missing_or_invalid_t_end"
    if float(run.summary["t_end"]) < float(run.summary["t_start"]):
        return "t_end_before_t_start"
    return "not_normally_completed"


def theoretical_time(run: Run) -> float:
    value = run.info.get("theoretical_time")
    if not finite_scalar(value):
        value = run.summary.get("theoretical_time")
    return float(value)


def analyze_successful_run(run: Run) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    synchronized = interpolate_reference_pose(run)
    controls = synchronize_controls(run)
    dx = synchronized["x_measured"] - synchronized["x_reference"]
    dy = synchronized["y_measured"] - synchronized["y_reference"]
    position_error = np.hypot(dx, dy)
    heading_error = wrap_to_pi(
        synchronized["theta_measured"] - synchronized["theta_reference"]
    )
    heading_abs_error = np.abs(heading_error)
    measured_points = np.column_stack(
        (synchronized["x_measured"], synchronized["y_measured"])
    )
    reference_vertices = np.column_stack((run.reference["x"], run.reference["y"]))
    path_error = point_to_polyline_distances(measured_points, reference_vertices)
    final_pose, final_sample_offset = nearest_final_pose(run)
    goal = np.asarray(run.info["goal_pose"], dtype=float)
    final_position_error = float(np.linalg.norm(final_pose[:2] - goal[:2]))
    final_heading_error = float(abs(wrap_to_pi(final_pose[2] - goal[2])))
    execution_time = float(run.summary["t_end"] - run.summary["t_start"])
    theory = theoretical_time(run)

    metrics: dict[str, Any] = {
        "n_measured_samples_execution": len(position_error),
        "n_command_samples_compared": len(controls["t_rel"]),
        "n_pose_samples_after_reference_end": int(
            synchronized["samples_after_reference_end"][0]
        ),
        "final_sample_time_offset": final_sample_offset,
        "final_position_error": final_position_error,
        "final_heading_abs_error": final_heading_error,
        "execution_time": execution_time,
        "execution_time_increase": execution_time - theory,
        "execution_time_increase_percent": 100.0 * (execution_time - theory) / theory,
    }
    metrics.update(descriptive(position_error, "position_error"))
    metrics.update(descriptive(heading_abs_error, "heading_abs_error"))
    metrics.update(descriptive(path_error, "path_error", include_mean=False))
    if len(controls["t_rel"]):
        metrics["reference_command_v_rms_difference"] = float(np.sqrt(np.mean(
            np.square(controls["v_commanded"] - controls["v_reference"])
        )))
        metrics["reference_command_omega_rms_difference"] = float(np.sqrt(np.mean(
            np.square(controls["omega_commanded"] - controls["omega_reference"])
        )))
    else:
        metrics["reference_command_v_rms_difference"] = math.nan
        metrics["reference_command_omega_rms_difference"] = math.nan
    series = {
        **synchronized,
        **{f"control_{key}": value for key, value in controls.items()},
        "position_error": position_error,
        "heading_error": heading_error,
        "heading_abs_error": heading_abs_error,
        "path_error": path_error,
    }
    return metrics, series


def diagnostic_series_for_excluded_run(run: Run) -> dict[str, np.ndarray]:
    """Build plot-only series for an excluded run without making it successful.

    Missing end boundaries are not reconstructed. For visualization only, the
    window starts at the saved start boundary (or first pose if absent) and
    stops at the last measured pose. These series never enter CSV metrics or
    scenario aggregates.
    """
    start = (float(run.summary["t_start"])
             if finite_scalar(run.summary.get("t_start"))
             else float(run.measured["t"][0]))
    end = (float(run.summary["t_end"])
           if finite_scalar(run.summary.get("t_end"))
           else float(run.measured["t"][-1]))
    pose_mask = (run.measured["t"] >= start) & (run.measured["t"] <= end)
    pose_t = run.measured["t"][pose_mask] - start
    theta_reference = np.interp(
        pose_t, run.reference["t"], np.unwrap(run.reference["theta"])
    )
    x_reference = np.interp(pose_t, run.reference["t"], run.reference["x"])
    y_reference = np.interp(pose_t, run.reference["t"], run.reference["y"])
    position_error = np.hypot(
        run.measured["x"][pose_mask] - x_reference,
        run.measured["y"][pose_mask] - y_reference,
    )
    heading_error = wrap_to_pi(run.measured["theta"][pose_mask] - theta_reference)

    command_t_all = run.commands["t"] - start
    command_mask = (
        (run.commands["t"] >= start)
        & (run.commands["t"] <= end)
        & (command_t_all >= run.reference["t"][0])
        & (command_t_all <= run.reference["t"][-1])
    )
    command_t = command_t_all[command_mask]
    return {
        "t_rel": pose_t,
        "x_measured": run.measured["x"][pose_mask],
        "y_measured": run.measured["y"][pose_mask],
        "theta_measured": run.measured["theta"][pose_mask],
        "x_reference": x_reference,
        "y_reference": y_reference,
        "theta_reference": theta_reference,
        "position_error": position_error,
        "heading_error": heading_error,
        "control_t_rel": command_t,
        "control_v_commanded": run.commands["v"][command_mask],
        "control_omega_commanded": run.commands["omega"][command_mask],
        "control_v_reference": np.interp(
            command_t, run.reference["t"], run.reference["v"]
        ),
        "control_omega_reference": np.interp(
            command_t, run.reference["t"], run.reference["omega"]
        ),
    }


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def save_figure(fig: plt.Figure, target: Path, save_png: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    if save_png:
        fig.savefig(target.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_run(
    run: Run,
    series: dict[str, np.ndarray],
    output_dir: Path,
    save_png: bool,
    diagnostic: bool = False,
) -> None:
    run_dir = output_dir / "runs" / slug(run.batch) / run.run_id
    start = np.asarray(run.info["start_pose_used"], dtype=float)
    goal = np.asarray(run.info["goal_pose"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    ax.plot(run.reference["x"], run.reference["y"], color="black", lw=1.8,
            label="Analytical reference")
    ax.plot(series["x_measured"], series["y_measured"], color="#0072B2", lw=1.2,
            label="Measured trajectory")
    ax.scatter(start[0], start[1], marker="o", s=35, color="#009E73", label="Start", zorder=3)
    ax.scatter(goal[0], goal[1], marker="*", s=80, color="#D55E00", label="Goal", zorder=3)
    title = run.run_id + (" — diagnostic excluded run" if diagnostic else "")
    ax.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, run_dir / "trajectory_xy", save_png)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True, constrained_layout=True)
    axes[0].plot(series["t_rel"], series["position_error"], color="#0072B2", lw=1.1)
    axes[0].set_ylabel(r"$e_p$ [m]")
    axes[1].plot(series["t_rel"], np.abs(series["heading_error"]), color="#D55E00", lw=1.1)
    axes[1].set(xlabel="Execution time [s]", ylabel=r"$|e_\theta|$ [rad]")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    save_figure(fig, run_dir / "tracking_errors", save_png)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True, constrained_layout=True)
    control_t = series["control_t_rel"]
    axes[0].plot(control_t, series["control_v_reference"], color="black", lw=1.5,
                 label="Reference control")
    axes[0].plot(control_t, series["control_v_commanded"], color="#0072B2", lw=1.0,
                 label="Commanded control")
    axes[0].set_ylabel(r"$v$ [m/s]")
    axes[1].plot(control_t, series["control_omega_reference"], color="black", lw=1.5,
                 label="Reference control")
    axes[1].plot(control_t, series["control_omega_commanded"], color="#D55E00", lw=1.0,
                 label="Commanded control")
    axes[1].set(xlabel="Execution time [s]", ylabel=r"$\omega$ [rad/s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    save_figure(fig, run_dir / "reference_and_commanded_controls", save_png)


def references_identical(runs: Sequence[Run]) -> bool:
    if not runs:
        return True
    first = runs[0].reference
    for run in runs[1:]:
        for key in REFERENCE_KEYS:
            if first[key].shape != run.reference[key].shape or not np.array_equal(
                first[key], run.reference[key]
            ):
                return False
    return True


def representative_index(rows: Sequence[dict[str, Any]]) -> int:
    values = np.asarray([float(row["position_error_rms"]) for row in rows])
    return int(np.argmin(np.abs(values - np.median(values))))


def plot_group_paths(
    group_name: str,
    runs: Sequence[Run],
    rows: Sequence[dict[str, Any]],
    series_by_id: dict[str, dict[str, np.ndarray]],
    target: Path,
    save_png: bool,
) -> None:
    completed = [(run, row) for run, row in zip(runs, rows) if run.normally_completed]
    if not completed:
        return
    successful_runs = [item[0] for item in completed]
    successful_rows = [item[1] for item in completed]
    common_reference = references_identical(successful_runs)
    representative = representative_index(successful_rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)

    # Draw every distinct corridor set represented in the group. Input-batch
    # plots normally have one set; a scenario-number plot may combine revised
    # inputs and therefore contain more than one.
    corridor_sets: list[list[Any]] = []
    corridor_signatures: set[str] = set()
    for run in successful_runs:
        corridors = run.info.get("corridors", [])
        signature = json.dumps(corridors, sort_keys=True, separators=(",", ":"))
        if signature not in corridor_signatures:
            corridor_signatures.add(signature)
            corridor_sets.append(corridors)
    corridor_label_used = False
    for corridors in corridor_sets:
        for corridor in corridors:
            polygon = np.asarray(corridor, dtype=float)
            if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
                continue
            closed = np.vstack((polygon[:, :2], polygon[0, :2]))
            ax.fill(polygon[:, 0], polygon[:, 1], color="0.75", alpha=0.10, zorder=0)
            ax.plot(closed[:, 0], closed[:, 1], color="0.45", ls="--", lw=0.8,
                    alpha=0.65, zorder=0,
                    label="Input corridors" if not corridor_label_used else None)
            corridor_label_used = True
    if common_reference:
        ref = successful_runs[0].reference
        ax.plot(ref["x"], ref["y"], color="black", lw=2.0, label="Analytical reference")
    else:
        for index, run in enumerate(successful_runs):
            ax.plot(run.reference["x"], run.reference["y"], color="black", alpha=0.13,
                    lw=0.65, label="Run-specific references" if index == 0 else None)
    measured_label_used = False
    for index, run in enumerate(successful_runs):
        series = series_by_id[run.run_id]
        highlighted = index == representative
        if highlighted:
            line_label = "Representative measured run"
        elif not measured_label_used:
            line_label = "Measured trajectories"
            measured_label_used = True
        else:
            line_label = None
        ax.plot(series["x_measured"], series["y_measured"],
                color="#0072B2" if highlighted else "#56B4E9",
                alpha=0.95 if highlighted else 0.35,
                lw=1.8 if highlighted else 0.7,
                label=line_label)

    # Frame the trajectories, not the full corridor polygons. Corridors remain
    # useful spatial context and are intentionally clipped when they extend far
    # beyond the paths of interest.
    path_x = np.concatenate([
        *(run.reference["x"] for run in successful_runs),
        *(series_by_id[run.run_id]["x_measured"] for run in successful_runs),
    ])
    path_y = np.concatenate([
        *(run.reference["y"] for run in successful_runs),
        *(series_by_id[run.run_id]["y_measured"] for run in successful_runs),
    ])
    x_span = max(float(np.ptp(path_x)), 0.2)
    y_span = max(float(np.ptp(path_y)), 0.2)
    ax.set_xlim(float(np.min(path_x) - 0.06 * x_span),
                float(np.max(path_x) + 0.06 * x_span))
    ax.set_ylim(float(np.min(path_y) - 0.06 * y_span),
                float(np.max(path_y) + 0.06 * y_span))
    ax.set(xlabel=r"$x$ [m]", ylabel=r"$y$ [m]", title=group_name)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8)
    note = "Common reference" if common_reference else "Run-specific references (not identical)"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=7, color="0.35")
    save_figure(fig, target, save_png)


def plot_paper_three_panel_paths(
    grouped: dict[str, list[tuple[Run, dict[str, Any]]]],
    series_by_id: dict[str, dict[str, np.ndarray]],
    target: Path,
    save_png: bool,
    font_style: str,
) -> None:
    """Create the paper figure for retained experiments 1, 3, and 4."""
    panels = (
        ("01_right_angle_turn/input_01_original", r"(a) Right-angle turn"),
        ("03_multiple_corridors/input_01", r"(b) Multiple corridors"),
        ("04_multiple_narrow_corridors/input_01", r"(c) Multiple narrow corridors"),
    )
    missing = [batch for batch, _ in panels if batch not in grouped]
    if missing:
        print(f"Paper path figure skipped; missing batches: {missing}", file=sys.stderr)
        return

    if font_style == "stix":
        serif_fonts = ["STIXGeneral", "DejaVu Serif"]
        math_fontset = "stix"
    elif font_style == "latex":
        serif_fonts = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
        math_fontset = "cm"
    else:
        raise ValueError(f"Unknown paper figure font style: {font_style}")

    figure_style = {
        "font.family": "serif",
        "font.serif": serif_fonts,
        "mathtext.fontset": math_fontset,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    }

    # Equal data scale and equal-height axes need unequal panel widths because
    # the three paths have different aspect ratios. Width ratios prevent wide
    # empty subplot slots around the tall multiple-corridor trajectory.
    width_ratios = []
    for batch, _ in panels:
        runs = [run for run, _ in grouped[batch] if run.normally_completed]
        path_x = np.concatenate([
            *(run.reference["x"] for run in runs),
            *(series_by_id[run.run_id]["x_measured"] for run in runs),
        ])
        path_y = np.concatenate([
            *(run.reference["y"] for run in runs),
            *(series_by_id[run.run_id]["y_measured"] for run in runs),
        ])
        width_ratios.append(max(float(np.ptp(path_x)), 0.2) /
                            max(float(np.ptp(path_y)), 0.2))

    with plt.rc_context(figure_style):
        fig, axes = plt.subplots(
            1, 3, figsize=(8.2, 4.9),
            gridspec_kw={"width_ratios": width_ratios},
        )
        for panel_index, (ax, (batch, title)) in enumerate(zip(axes, panels)):
            items = [(run, row) for run, row in grouped[batch]
                     if run.normally_completed]
            runs = [item[0] for item in items]
            rows = [item[1] for item in items]
            representative = representative_index(rows)

            corridors = runs[0].info.get("corridors", [])
            for corridor in corridors:
                polygon = np.asarray(corridor, dtype=float)
                if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
                    continue
                closed = np.vstack((polygon[:, :2], polygon[0, :2]))
                ax.fill(polygon[:, 0], polygon[:, 1], color="0.78", alpha=0.10,
                        zorder=0)
                ax.plot(closed[:, 0], closed[:, 1], color="0.48", ls="--",
                        lw=0.8, alpha=0.75, zorder=0)

            for run in runs:
                ax.plot(run.reference["x"], run.reference["y"], color="0.25",
                        alpha=0.24, lw=0.75, zorder=1)
            for index, run in enumerate(runs):
                series = series_by_id[run.run_id]
                highlighted = index == representative
                ax.plot(
                    series["x_measured"], series["y_measured"],
                    color="#0072B2" if highlighted else "#56B4E9",
                    alpha=1.0 if highlighted else 0.38,
                    lw=1.8 if highlighted else 0.75,
                    zorder=3 if highlighted else 2,
                )

            start = np.asarray(runs[representative].info["start_pose_used"], dtype=float)
            goal = np.asarray(runs[representative].info["goal_pose"], dtype=float)
            ax.scatter(start[0], start[1], marker="o", s=24, color="#009E73",
                       edgecolor="white", linewidth=0.4, zorder=5)
            ax.scatter(goal[0], goal[1], marker="*", s=62, color="#D55E00",
                       edgecolor="white", linewidth=0.4, zorder=5)

            path_x = np.concatenate([
                *(run.reference["x"] for run in runs),
                *(series_by_id[run.run_id]["x_measured"] for run in runs),
            ])
            path_y = np.concatenate([
                *(run.reference["y"] for run in runs),
                *(series_by_id[run.run_id]["y_measured"] for run in runs),
            ])
            x_span = max(float(np.ptp(path_x)), 0.2)
            y_span = max(float(np.ptp(path_y)), 0.2)
            ax.set_xlim(float(np.min(path_x) - 0.06 * x_span),
                        float(np.max(path_x) + 0.06 * x_span))
            ax.set_ylim(float(np.min(path_y) - 0.06 * y_span),
                        float(np.max(path_y) + 0.06 * y_span))
            ax.set_title(title, pad=7)
            ax.set_xlabel(r"$x\;[\mathrm{m}]$")
            if panel_index == 0:
                ax.set_ylabel(r"$y\;[\mathrm{m}]$")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="0.88", lw=0.55)
            ax.set_axisbelow(True)

        legend_handles = [
            Line2D([0], [0], color="0.48", ls="--", lw=0.9,
                   label=r"Corridor boundaries"),
            Line2D([0], [0], color="0.25", alpha=0.55, lw=1.0,
                   label=r"Analytical references"),
            Line2D([0], [0], color="#56B4E9", alpha=0.65, lw=1.0,
                   label=r"Measured trajectories"),
            Line2D([0], [0], color="#0072B2", lw=1.9,
                   label=r"Representative run"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#009E73",
                   markeredgecolor="white", markersize=6, label=r"Start"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#D55E00",
                   markeredgecolor="white", markersize=9, label=r"Goal"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   frameon=False, fontsize=10.5, handlelength=2.3,
                   columnspacing=1.8, labelspacing=0.8,
                   bbox_to_anchor=(0.5, 0.012))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.255,
                            wspace=0.0)
        save_figure(fig, target, save_png)


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"median": math.nan, "q1": math.nan, "q3": math.nan,
                "iqr": math.nan, "maximum": math.nan}
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "maximum": float(np.max(finite)),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_specs: list[tuple[str, dict[str, list[dict[str, Any]]]]] = []
    by_batch: dict[str, list[dict[str, Any]]] = {}
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_batch.setdefault(str(row["batch"]), []).append(row)
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    group_specs.extend([
        ("input_batch", by_batch),
        ("scenario_number", by_scenario),
    ])
    for group_type, groups in group_specs:
        for group_name, group_rows in sorted(groups.items()):
            completed = [row for row in group_rows if row["normally_completed"]]
            excluded_reasons = sorted({
                str(row["exclusion_reason"]) for row in group_rows
                if not row["normally_completed"]
            })
            for metric in MAIN_METRICS:
                values = np.asarray([float(row.get(metric, math.nan)) for row in completed])
                output.append({
                    "group_type": group_type,
                    "group_name": group_name,
                    "scenario": group_rows[0]["scenario"] if group_type == "input_batch" else group_name,
                    "metric": metric,
                    "n_attempted": len(group_rows),
                    "n_normally_completed": len(completed),
                    "n_excluded": len(group_rows) - len(completed),
                    "excluded_reasons": ";".join(excluded_reasons),
                    **quantiles(values),
                })
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_compact_summary(rows: Sequence[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["batch"]), []).append(row)
    header = f"{'Input batch':58} {'done/try':>8} {'RMS pos [m]':>13} {'RMS path [m]':>14} {'final pos [m]':>14} {'time +[%]':>11}"
    print(header)
    print("-" * len(header))
    for name, group in sorted(groups.items()):
        complete = [row for row in group if row["normally_completed"]]
        def med(key: str) -> float:
            return float(np.median([float(row[key]) for row in complete])) if complete else math.nan
        print(
            f"{name[:58]:58} {len(complete):>3}/{len(group):<4} "
            f"{med('position_error_rms'):13.4f} {med('path_error_rms'):14.4f} "
            f"{med('final_position_error'):14.4f} "
            f"{med('execution_time_increase_percent'):11.2f}"
        )


def print_detailed_batch_summary(rows: Sequence[dict[str, Any]]) -> None:
    """Print publication-oriented distributions for every input batch."""
    metric_specs = (
        ("path_error_rms", "RMS geometric path error", "m", 4),
        ("path_error_p95", "95th-percentile geometric path error", "m", 4),
        ("path_error_max", "Maximum geometric path error", "m", 4),
        ("final_position_error", "Final position error", "m", 4),
        ("final_heading_abs_error", "Final heading error", "rad", 4),
        ("heading_abs_error_rms", "RMS heading tracking error", "rad", 4),
        ("execution_time_increase_percent", "Relative execution-time increase", "%", 2),
        ("execution_time", "Actual execution time", "s", 3),
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["batch"]), []).append(row)

    print("\nDetailed input-batch statistics (median [Q1, Q3], maximum)")
    print("Excluded runs are not included.")
    for batch, attempted in sorted(groups.items()):
        completed = [row for row in attempted if row["normally_completed"]]
        print(f"\n{batch}  ({len(completed)}/{len(attempted)} normally completed)")
        for key, label, unit, digits in metric_specs:
            values = np.asarray([float(row[key]) for row in completed], dtype=float)
            stats = quantiles(values)
            number = f"{{:.{digits}f}}"
            print(
                f"  {label:40} "
                f"{number.format(stats['median'])} "
                f"[{number.format(stats['q1'])}, {number.format(stats['q3'])}], "
                f"max {number.format(stats['maximum'])} {unit}"
            )
        if completed:
            worst_path = max(completed, key=lambda row: float(row["path_error_max"]))
            worst_final = max(completed, key=lambda row: float(row["final_position_error"]))
            print(
                "  Worst geometric path-error run "
                f"(by per-run maximum): {worst_path['run_id']} "
                f"({float(worst_path['path_error_max']):.4f} m)"
            )
            print(
                "  Worst final-position run:             "
                f"{worst_final['run_id']} "
                f"({float(worst_final['final_position_error']):.4f} m)"
            )


def print_audit_notes(runs: Sequence[Run]) -> None:
    """Print data properties that affect interpretation but are not repaired."""
    reasons: dict[str, int] = {}
    for run in runs:
        reason = str(run.summary.get("end_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"Input audit: {len(runs)} real-robot runs; end reasons {reasons}.")

    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.batch, []).append(run)
    noncommon = [name for name, members in grouped.items() if not references_identical(members)]
    if noncommon:
        print(
            "Input audit: references are not exactly identical within "
            f"{len(noncommon)}/{len(grouped)} input batches; group figures draw each "
            "run-specific reference."
        )

    command_gaps = []
    measured_boundary_offsets = []
    for run in runs:
        if not run.normally_completed:
            continue
        end = float(run.summary["t_end"])
        command_gaps.append(end - float(run.commands["t"][-1]))
        measured_boundary_offsets.append(float(np.min(np.abs(run.measured["t"] - end))))
    positive_gaps = np.asarray([gap for gap in command_gaps if gap > 0.0])
    if len(positive_gaps):
        print(
            "Input audit: commanded-control streams end before t_end in "
            f"{len(positive_gaps)}/{len(command_gaps)} completed runs "
            f"(median gap {np.median(positive_gaps):.3f} s, max {np.max(positive_gaps):.3f} s); "
            "control comparisons use only recorded timestamps within reference support."
        )
    print(
        "Input audit: final pose uses the measured sample nearest t_end; "
        f"maximum absolute sample offset is {max(measured_boundary_offsets):.6f} s."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=default_root,
                        help="Root containing numbered scenario/input folders")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: INPUT_ROOT/analysis_output)")
    parser.add_argument("--png", action="store_true",
                        help="Also save PNG copies of all PDF figures")
    parser.add_argument("--no-run-figures", action="store_true",
                        help="Skip the three figures generated for every successful run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input_root.resolve()
    output_dir = (args.output_dir or root / "analysis_output").resolve()
    paths = discover_runs(root)
    if not paths:
        print(f"No real-robot JSON runs found below {root}", file=sys.stderr)
        return 2

    runs: list[Run] = []
    load_errors: list[str] = []
    for path in paths:
        try:
            runs.append(load_run(path, root))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            load_errors.append(str(error))
    if load_errors:
        print("Input validation failed; no questionable files were repaired:", file=sys.stderr)
        for error in load_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print_audit_notes(runs)

    rows: list[dict[str, Any]] = []
    series_by_id: dict[str, dict[str, np.ndarray]] = {}
    for run in runs:
        row = base_row(run)
        if run.normally_completed:
            try:
                metrics, series = analyze_successful_run(run)
            except ValueError as error:
                row["normally_completed"] = False
                row["exclusion_reason"] = f"analysis_error={error}"
            else:
                row.update(metrics)
                series_by_id[run.run_id] = series
                if not args.no_run_figures:
                    plot_run(run, series, output_dir, args.png)
        elif not args.no_run_figures:
            diagnostic = diagnostic_series_for_excluded_run(run)
            plot_run(run, diagnostic, output_dir, args.png, diagnostic=True)
        rows.append(row)

    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "run_metrics.csv", rows)
    write_csv(output_dir / "scenario_statistics.csv", aggregate)

    by_batch: dict[str, list[tuple[Run, dict[str, Any]]]] = {}
    by_scenario: dict[str, list[tuple[Run, dict[str, Any]]]] = {}
    for run, row in zip(runs, rows):
        by_batch.setdefault(run.batch, []).append((run, row))
        by_scenario.setdefault(str(run.info["scenario"]), []).append((run, row))
    for name, items in sorted(by_batch.items()):
        plot_group_paths(
            name, [item[0] for item in items], [item[1] for item in items], series_by_id,
            output_dir / "groups" / "input_batches" / slug(name), args.png,
        )
    for name, items in sorted(by_scenario.items()):
        plot_group_paths(
            f"Scenario {name}", [item[0] for item in items], [item[1] for item in items],
            series_by_id, output_dir / "groups" / "scenario_numbers" / f"scenario_{name}",
            args.png,
        )
    plot_paper_three_panel_paths(
        by_batch,
        series_by_id,
        output_dir / "groups" / "paper_experiments_1_3_4_latex",
        args.png,
        font_style="latex",
    )
    plot_paper_three_panel_paths(
        by_batch,
        series_by_id,
        output_dir / "groups" / "paper_experiments_1_3_4_stix",
        args.png,
        font_style="stix",
    )

    print_compact_summary(rows)
    print_detailed_batch_summary(rows)
    excluded = [row for row in rows if not row["normally_completed"]]
    print(f"\nAnalyzed {len(rows)} runs: {len(rows) - len(excluded)} normally completed, "
          f"{len(excluded)} excluded from successful-run statistics.")
    for row in excluded:
        print(f"  excluded {row['run_id']}: {row['exclusion_reason']}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
