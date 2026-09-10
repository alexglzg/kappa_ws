"""experiment_node: predefined corridor scenarios + on-demand analytical planning.

Workflow (per run):
  1. ``ros2 param set /experiment_node scenario N``  -> corridors, start/goal
     footprints and the projection area are (re)drawn on the floor. Every
     redraw starts with a DELETEALL marker, so nothing from a previous
     scenario stays on screen.
  2. Place the robot on the projected start footprint. A live ring is drawn
     around the measured pose so placement (and Vive calibration) can be
     checked visually.
  3. ``ros2 service call /experiment_node/plan std_srvs/srv/Trigger`` ->
     trajectory computed from the measured pose (or the predefined one, see
     ``start_pose_source``), resampled on a global ``sampling_dt`` grid,
     published as nav_msgs/Path (+ reference v, omega) and drawn on the
     floor. A JSON summary (theoretical time, pieces, placement error, ...)
     is logged and published on ``~/plan_info`` for the rosbag.
  4. ``.../clear_plan`` removes only the trajectory drawing; corridors stay.

Services:  ~/plan, ~/clear_plan, ~/show_scenario, ~/teleport_to_start
"""
import json
import math
import time
from math import cos, sin
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, Float64MultiArray, MultiArrayDimension, MultiArrayLayout, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from kappa_planner.corridor import CorridorWorld
from kappa_planner.helpers.corridor_geometry import shrink_corridor_list
from kappa_planner.motion_planner import MotionPlanner

from . import scenarios as sc
from .trajectory_utils import (SampledTrajectory, deg, placement_error,
                               resample_trajectory, wrap_angle, yaw_from_quaternion)

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST)


def _color(r, g, b, a=1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _quat(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw))


class ExperimentNode(Node):

    def __init__(self):
        super().__init__('experiment_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                # --- experiment selection / protocol ---
                ('scenario', 1),
                ('session_name', 'session'),
                ('start_pose_source', 'measured'),   # 'measured' | 'predefined'
                ('placement_tol_xy', 0.05),          # m
                ('placement_tol_yaw', 0.10),         # rad (~6 deg)
                ('enforce_placement_tolerance', False),
                ('assumptions', 'standing'),
                ('sampling_dt', 0.1),
                # --- robot (unicycle) ---
                ('v_max', sc.ROBOT_V_MAX),
                ('v_min', sc.ROBOT_V_MIN),
                ('omega_max', sc.ROBOT_OMEGA_MAX),
                ('omega_min', sc.ROBOT_OMEGA_MIN),
                ('robot_width', sc.ROBOT_WIDTH),
                ('robot_length', sc.ROBOT_LENGTH),
                # --- topics / frames ---
                ('map_frame', 'map'),
                ('pose_topic', '/robot_pose'),
                ('path_topic', '/rosbot2pro/planned_path'),
                ('initialpose_topic', '/initialpose'),
                # --- floor drawing ---
                ('plot_projection_area', True),
                ('plot_shrunken_corridors', True),
                ('plot_circles', True),
                ('plot_piece_boundaries', True),
                ('show_measured_pose', True),
                ('line_width', 0.03),
                ('trajectory_line_width', 0.04),
                ('marker_z', 0.01),
                ('text_height', 0.15),
            ],
        )
        self.dt = float(self.get_parameter('sampling_dt').value)
        self.map_frame = self.get_parameter('map_frame').value

        self.unicycle = sc.build_unicycle(
            v_max=self.get_parameter('v_max').value,
            v_min=self.get_parameter('v_min').value,
            omega_max=self.get_parameter('omega_max').value,
            omega_min=self.get_parameter('omega_min').value,
            width=self.get_parameter('robot_width').value,
            length=self.get_parameter('robot_length').value,
        )
        self.r = sc.robot_radius(self.unicycle)
        self.R = sc.turning_radius(self.unicycle)
        # Same safety margin MotionPlanner uses internally to build
        # ``shrunken_corridor_list`` (0.5 * vehicle.width); equal to ``r`` only
        # while the footprint is square, which is why it is not reused here.
        self.safety_margin = 0.5 * self.unicycle.width
        self.get_logger().info(f'Unicycle: r={self.r:.3f} m, R={self.R:.3f} m, '
                               f'planner safety margin={self.safety_margin:.3f} m, '
                               f'v_max={self.unicycle.v_max}, omega_max={self.unicycle.omega_max}')

        self.scenario: Optional[sc.Scenario] = None
        self.shrunken_corridor_list: List[CorridorWorld] = []
        self.sampled: Optional[SampledTrajectory] = None
        self.scenario_markers: List[Marker] = []
        self.trajectory_markers: List[Marker] = []
        self.measured_pose: Optional[List[float]] = None
        self.measured_stamp = None
        self.run_index = 0
        self._redraw_timer = None

        # --- publishers ---
        self.marker_pub = self.create_publisher(MarkerArray, '~/experiment_markers', LATCHED)
        self.robot_marker_pub = self.create_publisher(Marker, '~/robot_marker', 1)
        self.path_pub = self.create_publisher(Path, self.get_parameter('path_topic').value, LATCHED)
        self.controls_pub = self.create_publisher(Float64MultiArray, '~/planned_controls', LATCHED)
        self.plan_info_pub = self.create_publisher(String, '~/plan_info', LATCHED)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.get_parameter('initialpose_topic').value, 1)

        # --- subscriptions ---
        self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value,
                                 self._pose_cb, 10)

        # --- services ---
        self.create_service(Trigger, '~/plan', self._srv_plan)
        self.create_service(Trigger, '~/clear_plan', self._srv_clear_plan)
        self.create_service(Trigger, '~/show_scenario', self._srv_show_scenario)
        self.create_service(Trigger, '~/teleport_to_start', self._srv_teleport)

        self.add_on_set_parameters_callback(self._on_params)
        self.create_timer(0.1, self._publish_robot_marker)

        self.load_scenario(int(self.get_parameter('scenario').value))

    # ------------------------------------------------------------------
    # Parameters / scenario
    # ------------------------------------------------------------------
    def _on_params(self, params):
        redraw = False
        for p in params:
            if p.name == 'scenario':
                try:
                    self.load_scenario(int(p.value))
                except Exception as exc:
                    self.get_logger().error(f'Could not load scenario {p.value}: {exc}')
                    return SetParametersResult(successful=False, reason=str(exc))
            elif p.name in ('plot_projection_area', 'plot_shrunken_corridors', 'plot_circles',
                            'plot_piece_boundaries', 'line_width', 'trajectory_line_width',
                            'marker_z', 'text_height'):
                redraw = True
        if redraw and self.scenario is not None:
            # Values are only applied after this callback returns; redraw shortly after.
            self._redraw_timer = self.create_timer(0.05, self._redraw_once)
        return SetParametersResult(successful=True)

    def _redraw_once(self):
        # one-shot timer: rebuild static markers with the new drawing parameters
        self.scenario_markers = self._build_scenario_markers()
        if self.sampled is not None:
            self.trajectory_markers = self._build_trajectory_markers(self.sampled)
        self._publish_markers()
        # one-shot: rclpy has no single-shot timers
        if self._redraw_timer is not None:
            self.destroy_timer(self._redraw_timer)
            self._redraw_timer = None

    def load_scenario(self, number: int):
        self.scenario = sc.build_scenario(number, self.unicycle)
        # Same corridors MotionPlanner will build internally (see safety_margin);
        # kept here so both the floor drawing and plan_info use one list.
        self.shrunken_corridor_list = shrink_corridor_list(
            self.scenario.corridor_list, self.safety_margin)
        self.sampled = None
        self.trajectory_markers = []
        self.scenario_markers = self._build_scenario_markers()
        self._publish_markers()
        s = self.scenario
        self.get_logger().info(
            f'Loaded scenario {s.number}: {s.title} | {len(s.corridor_list)} corridors | '
            f'start={np.round(s.start_pose, 3).tolist()} goal={np.round(s.goal_pose, 3).tolist()}'
            + (f' | {s.notes}' if s.notes else ''))

    # ------------------------------------------------------------------
    # Pose input
    # ------------------------------------------------------------------
    def _pose_cb(self, msg: PoseStamped):
        self.measured_pose = [msg.pose.position.x, msg.pose.position.y,
                              yaw_from_quaternion(msg.pose.orientation)]
        self.measured_stamp = msg.header.stamp

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------
    def _srv_show_scenario(self, _req, res):
        self.scenario_markers = self._build_scenario_markers()
        self._publish_markers()
        res.success = True
        res.message = f'Scenario {self.scenario.number} redrawn.'
        return res

    def _srv_clear_plan(self, _req, res):
        self.trajectory_markers = []
        self.sampled = None
        self._publish_markers()
        res.success = True
        res.message = 'Trajectory markers cleared; corridors kept.'
        return res

    def _srv_teleport(self, _req, res):
        """Publish the predefined start pose on /initialpose (box_sim teleport).

        Only meant for simulation; on the real robot nothing should subscribe
        to this topic (check before using it with a running localisation).
        """
        s = self.scenario
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(s.start_pose[0])
        msg.pose.pose.position.y = float(s.start_pose[1])
        msg.pose.pose.orientation = _quat(float(s.start_pose[2]))
        self.initialpose_pub.publish(msg)
        res.success = True
        res.message = f'Published start pose of scenario {s.number} on {self.get_parameter("initialpose_topic").value}.'
        return res

    def _srv_plan(self, _req, res):
        try:
            info = self.plan()
        except Exception as exc:
            self.get_logger().error(f'Planning failed: {exc}')
            res.success = False
            res.message = f'Planning failed: {exc}'
            return res
        res.success = info['success']
        res.message = info['message']
        return res

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self) -> dict:
        s = self.scenario
        source = self.get_parameter('start_pose_source').value
        tol_xy = float(self.get_parameter('placement_tol_xy').value)
        tol_yaw = float(self.get_parameter('placement_tol_yaw').value)
        enforce = bool(self.get_parameter('enforce_placement_tolerance').value)

        if source == 'measured':
            if self.measured_pose is None:
                return {'success': False,
                        'message': f'No pose received on {self.get_parameter("pose_topic").value}.'}
            start_pose = list(self.measured_pose)
        elif source == 'predefined':
            start_pose = list(s.start_pose)
        else:
            return {'success': False, 'message': f'Unknown start_pose_source "{source}".'}

        d_xy, d_yaw = (placement_error(self.measured_pose, s.start_pose)
                       if self.measured_pose is not None else (float('nan'), float('nan')))
        within_tol = (d_xy <= tol_xy) and (d_yaw <= tol_yaw)
        if self.measured_pose is not None:
            self.get_logger().info(
                f'Placement error vs predefined start: {d_xy * 100:.1f} cm, {deg(d_yaw):.1f} deg '
                f'({"OK" if within_tol else "outside tolerance"})')
        if enforce and source == 'measured' and not within_tol:
            return {'success': False,
                    'message': f'Robot is {d_xy * 100:.1f} cm / {deg(d_yaw):.1f} deg from the start pose; '
                               f'tolerance {tol_xy * 100:.0f} cm / {deg(tol_yaw):.0f} deg.'}

        # Fresh planner per run, exactly as in maps_for_robot_experiments.py
        planner = MotionPlanner(
            self.unicycle,
            s.corridor_list,
            start_pose=start_pose,
            end_pose=list(s.goal_pose),
            assumptions=self.get_parameter('assumptions').value,
        )
        t0 = time.perf_counter()
        trajectory = planner.compute_trajectory_analytical()
        wall_ms = (time.perf_counter() - t0) * 1e3
        comp_time = getattr(planner, 'comp_time_analytical_sol', None)

        if trajectory is None:
            return {'success': False, 'message': 'compute_trajectory_analytical returned None.'}
        if not isinstance(trajectory, (list, tuple)):
            # e.g. UnicycleTrajectoryOptimal: handled in plan_motion.py via
            # trajectory.sampler(trajectory.gist, time_grid); not expected here.
            return {'success': False,
                    'message': f'Unexpected trajectory type {type(trajectory).__name__}.'}

        sampled = resample_trajectory(trajectory, self.dt)
        self.sampled = sampled
        self.run_index += 1
        stamp = self.get_clock().now()

        self._publish_path(sampled, stamp)
        self._publish_controls(sampled)
        self.trajectory_markers = self._build_trajectory_markers(sampled)
        self._publish_markers()

        info = {
            'success': True,
            'session': self.get_parameter('session_name').value,
            'run_index': self.run_index,
            'scenario': s.number,
            'title': s.title,
            'stamp': stamp.nanoseconds * 1e-9,
            'start_pose_source': source,
            'start_pose_used': [float(v) for v in start_pose],
            'start_pose_predefined': [float(v) for v in s.start_pose],
            'goal_pose': [float(v) for v in s.goal_pose],
            'measured_pose_at_plan': ([float(v) for v in self.measured_pose]
                                      if self.measured_pose is not None else None),
            'placement_error_xy': d_xy,
            'placement_error_yaw': d_yaw,
            'placement_within_tolerance': within_tol,
            'theoretical_time': sampled.total_time,
            'piece_times': sampled.piece_times,
            'piece_types': sampled.piece_types,
            'piece_boundaries': sampled.piece_boundaries.tolist(),
            'computation_time_planner': comp_time,
            'computation_time_wall_ms': wall_ms,
            'sampling_dt': self.dt,
            'n_samples': int(len(sampled.t)),
            'robot': {'r': self.r, 'R': self.R, 'safety_margin': self.safety_margin,
                      'v_max': self.unicycle.v_max, 'omega_max': self.unicycle.omega_max},
            'reference': {'t': sampled.t.tolist(), 'x': sampled.x.tolist(), 'y': sampled.y.tolist(),
                          'theta': sampled.theta.tolist(), 'v': sampled.v.tolist(),
                          'omega': sampled.omega.tolist()},
            'circles': sampled.circles,
            # Corridor polygons so that postprocess can draw the map offline
            # without importing kappa_planner.
            'corridors': [np.asarray(c.corners)[:, :2].tolist() for c in s.corridor_list],
            'shrunken_corridors': [np.asarray(c.corners)[:, :2].tolist()
                                   for c in self.shrunken_corridor_list],
        }
        self.plan_info_pub.publish(String(data=json.dumps(info)))

        pieces = ', '.join(f'{typ}:{T:.2f}s' for typ, T in zip(sampled.piece_types, sampled.piece_times))
        info['message'] = (f'Run {self.run_index} | scenario {s.number} | theoretical time '
                           f'{sampled.total_time:.2f} s | {len(sampled.t)} samples @ {self.dt} s | '
                           f'planner {wall_ms:.1f} ms | pieces: {pieces}')
        self.get_logger().info(info['message'])
        return info

    # ------------------------------------------------------------------
    # Publishing: path / controls
    # ------------------------------------------------------------------
    def _publish_path(self, tr: SampledTrajectory, stamp):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = stamp.to_msg()
        for k in range(len(tr.t)):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            # stamp encodes the reference time: t_plan + t_k
            ps.header.stamp = (stamp + Duration(seconds=float(tr.t[k]))).to_msg()
            ps.pose.position.x = float(tr.x[k])
            ps.pose.position.y = float(tr.y[k])
            ps.pose.orientation = _quat(float(wrap_angle(tr.theta[k])))
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_controls(self, tr: SampledTrajectory):
        n = len(tr.t)
        msg = Float64MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[MultiArrayDimension(label='sample', size=n, stride=3 * n),
                 MultiArrayDimension(label='t_v_omega', size=3, stride=3)],
            data_offset=0)
        msg.data = np.column_stack([tr.t, tr.v, tr.omega]).reshape(-1).tolist()
        self.controls_pub.publish(msg)

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------
    def _publish_markers(self):
        """Always DELETEALL first, then the full current set (clean cache)."""
        arr = MarkerArray()
        wipe = Marker()
        wipe.header.frame_id = self.map_frame
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        for i, m in enumerate(self.scenario_markers + self.trajectory_markers):
            m.id = i
            m.header.stamp = self.get_clock().now().to_msg()
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def _base_marker(self, ns: str, mtype: int) -> Marker:
        m = Marker()
        m.header.frame_id = self.map_frame
        m.ns = ns
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime.sec = 0
        return m

    def _line_strip(self, ns, points_xy, color, width, closed=False) -> Marker:
        m = self._base_marker(ns, Marker.LINE_STRIP)
        m.scale.x = float(width)
        m.color = color
        z = float(self.get_parameter('marker_z').value)
        pts = list(points_xy)
        if closed:
            pts.append(pts[0])
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in pts]
        return m

    def _circle(self, ns, center, radius, color, width, n=48) -> Marker:
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.column_stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)])
        return self._line_strip(ns, pts, color, width, closed=True)

    def _text(self, ns, xy, text, color) -> Marker:
        m = self._base_marker(ns, Marker.TEXT_VIEW_FACING)
        m.text = text
        m.scale.z = float(self.get_parameter('text_height').value)
        m.color = color
        m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1])
        m.pose.position.z = float(self.get_parameter('marker_z').value) + 0.02
        return m

    def _pose_footprint(self, ns, pose, color, width, label) -> List[Marker]:
        x, y, th = pose
        z = float(self.get_parameter('marker_z').value)
        ring = self._circle(ns, (x, y), self.r, color, width)
        arrow = self._base_marker(ns, Marker.ARROW)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = width, 2.5 * width, 0.0
        arrow.color = color
        arrow.points = [Point(x=float(x), y=float(y), z=z),
                        Point(x=float(x + self.r * cos(th)), y=float(y + self.r * sin(th)), z=z)]
        out = [ring, arrow]
        if label:
            out.append(self._text(ns, (x, y - 1.4 * self.r), label, color))
        return out

    def _build_scenario_markers(self) -> List[Marker]:
        s = self.scenario
        lw = float(self.get_parameter('line_width').value)
        markers: List[Marker] = []

        if self.get_parameter('plot_projection_area').value:
            box = [(sc.X_MIN, sc.Y_MIN), (sc.X_MAX, sc.Y_MIN), (sc.X_MAX, sc.Y_MAX), (sc.X_MIN, sc.Y_MAX)]
            markers.append(self._line_strip('projection_area', box, _color(0.5, 0.5, 0.5), lw * 0.7, closed=True))

        # Drawn exactly as the planner sees them: kappa's own shrink helper with
        # the margin MotionPlanner applies (CorridorWorld.shrink keeps the
        # centre and tilt and takes 2 * margin off width and height).
        n = len(s.corridor_list)
        for i, (c, cs) in enumerate(zip(s.corridor_list, self.shrunken_corridor_list)):
            t = i / max(n - 1, 1)
            col = _color(0.2 + 0.8 * t, 0.9 - 0.5 * t, 1.0 - 0.8 * t)
            markers.append(self._line_strip('corridors', np.asarray(c.corners)[:, :2], col, lw, closed=True))
            markers.append(self._text('corridor_labels', c.center, f'C{i + 1}', col))
            if self.get_parameter('plot_shrunken_corridors').value:
                if cs.width > 0.0 and cs.height > 0.0:
                    markers.append(self._line_strip('corridors_shrunk', np.asarray(cs.corners)[:, :2],
                                                    _color(col.r, col.g, col.b, 0.6), lw * 0.5, closed=True))
                else:
                    self.get_logger().warn(
                        f'Corridor C{i + 1} ({c.width:.2f} x {c.height:.2f} m) is not wider than '
                        f'2 x {self.safety_margin:.3f} m; the planner has no free space in it.')

        markers += self._pose_footprint('start', s.start_pose, _color(0.2, 1.0, 0.2), lw, 'START')
        markers += self._pose_footprint('goal', s.goal_pose, _color(1.0, 0.3, 0.3), lw, 'GOAL')
        markers.append(self._text('title', (sc.X_MIN + 0.1, sc.Y_MAX - 0.1), s.title, _color(1, 1, 1)))
        return markers

    def _build_trajectory_markers(self, tr: SampledTrajectory) -> List[Marker]:
        lw = float(self.get_parameter('trajectory_line_width').value)
        markers: List[Marker] = []
        pts = np.column_stack([tr.x, tr.y])
        markers.append(self._line_strip('trajectory', pts, _color(1.0, 0.9, 0.1), lw))

        if self.get_parameter('plot_circles').value:
            for circ in tr.circles:
                markers.append(self._circle('circles', circ['center'], circ['radius'],
                                            _color(0.3, 0.8, 1.0, 0.8), lw * 0.5))

        if self.get_parameter('plot_piece_boundaries').value:
            z = float(self.get_parameter('marker_z').value)
            m = self._base_marker('piece_boundaries', Marker.SPHERE_LIST)
            m.scale.x = m.scale.y = m.scale.z = 2.0 * lw
            m.color = _color(1.0, 1.0, 1.0)
            for tb in tr.piece_boundaries[1:-1]:
                m.points.append(Point(x=float(np.interp(tb, tr.t, tr.x)),
                                      y=float(np.interp(tb, tr.t, tr.y)), z=z))
            if m.points:
                markers.append(m)
        return markers

    def _publish_robot_marker(self):
        if not self.get_parameter('show_measured_pose').value or self.measured_pose is None:
            return
        x, y, th = self.measured_pose
        # True footprint radius, like the start/goal rings: when the robot is on
        # the start pose the two circles coincide, so the live one is told apart
        # by style (white, 1.5x line width) rather than by being drawn larger.
        m = self._circle('robot', (x, y), self.r, _color(1.0, 1.0, 1.0, 0.9),
                         1.5 * float(self.get_parameter('line_width').value))
        m.points.append(Point(x=float(x), y=float(y), z=m.points[0].z))  # centre spoke
        m.points.append(Point(x=float(x + self.r * cos(th)), y=float(y + self.r * sin(th)), z=m.points[0].z))
        m.id = 0
        m.header.stamp = self.get_clock().now().to_msg()
        self.robot_marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
