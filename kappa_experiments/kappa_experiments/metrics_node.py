"""metrics_node: live run timing and error metrics.

Lifecycle:
  - a message on ``plan_info_topic`` ARMS a new run (the JSON carries the
    reference trajectory and the goal pose);
  - the run STARTS at the configured start event (default: first non-zero
    command after arming);
  - the run ENDS when the measured pose satisfies ``end_criterion`` (own
    check, independent of the MPC): within ``goal_radius`` of the goal
    position, and under ``end_criterion: pose`` also within
    ``heading_tolerance`` of the goal heading. Keep both equal to the MPC's
    ``tolerance_radius`` / ``heading_tolerance``. A run also ends on
    ``~/abort_run`` or on timeout.

While a run is active a one-line status is logged every ``live_period`` s and
published on ``~/live``. At run end the full summary is logged, published
latched on ``~/run_summary`` (so a rosbag records it), appended as one row to
``<output_dir>/<session>.csv`` and written with the full error time series to
``<output_dir>/<session>_runNNN.json``. Postprocessing of the same rosbag with
``postprocess`` reproduces these numbers from the raw data.
"""
import json
import math
import os
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .metrics import (END_CRITERIA, Reference, RunRecorder, append_csv_row,
                      csv_row, summary_line)
from .trajectory_utils import wrap_angle, yaw_from_quaternion

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class MetricsNode(Node):

    def __init__(self):
        super().__init__('metrics_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('pose_topic', '/robot_pose'),
                ('plan_info_topic', '/experiment_node/plan_info'),
                ('cmd_topic', '/rosbot3/cmd_vel'),
                ('cmd_type', 'TwistStamped'),        # 'TwistStamped' | 'Twist'
                ('finished_topic', '/finished_tracking'),
                ('goal_radius', 0.05),               # = MPC tolerance_radius
                ('end_criterion', 'pose'),           # 'position' | 'pose'
                ('heading_tolerance', 0.10),         # rad, = MPC heading_tolerance
                ('start_event', 'first_nonzero_cmd'),  # 'first_cmd'|'first_nonzero_cmd'|'first_motion'
                ('v_eps', 0.01),
                ('w_eps', 0.05),
                ('motion_eps_xy', 0.01),
                ('motion_eps_yaw', 0.035),
                ('run_timeout_factor', 3.0),         # timeout = factor * theoretical + margin
                ('run_timeout_margin', 10.0),
                ('live_period', 2.0),
                ('output_dir', '~/kappa_experiment_logs'),
            ],
        )
        self.goal_radius = float(self.get_parameter('goal_radius').value)
        self.end_criterion = str(self.get_parameter('end_criterion').value)
        if self.end_criterion not in END_CRITERIA:
            raise ValueError(f'Unknown end_criterion {self.end_criterion}; '
                             f'expected one of {END_CRITERIA}')
        self.heading_tolerance = float(self.get_parameter('heading_tolerance').value)
        self.output_dir = os.path.expanduser(self.get_parameter('output_dir').value)
        os.makedirs(self.output_dir, exist_ok=True)

        self.recorder: Optional[RunRecorder] = None
        self.info: Optional[dict] = None
        self.armed = False
        self.t_start: Optional[float] = None
        self.finished_flag_time: Optional[float] = None
        self.timeout: Optional[float] = None
        self.arm_time: Optional[float] = None

        self.create_subscription(PoseStamped, self.get_parameter('pose_topic').value,
                                 self._pose_cb, 50)
        self.create_subscription(String, self.get_parameter('plan_info_topic').value,
                                 self._plan_info_cb, LATCHED)
        cmd_type = self.get_parameter('cmd_type').value
        if cmd_type == 'TwistStamped':
            self.create_subscription(TwistStamped, self.get_parameter('cmd_topic').value,
                                     self._cmd_stamped_cb, 20)
        elif cmd_type == 'Twist':
            self.create_subscription(Twist, self.get_parameter('cmd_topic').value,
                                     self._cmd_cb, 20)
        else:
            raise ValueError(f'Unknown cmd_type {cmd_type}')
        self.create_subscription(Bool, self.get_parameter('finished_topic').value,
                                 self._finished_cb, 10)

        self.live_pub = self.create_publisher(String, '~/live', 10)
        self.summary_pub = self.create_publisher(String, '~/run_summary', LATCHED)
        self.create_service(Trigger, '~/abort_run', self._srv_abort)
        self.create_timer(float(self.get_parameter('live_period').value), self._live_tick)

        self.get_logger().info(
            f'metrics_node ready; end criterion {self.end_criterion} '
            f'({self.goal_radius:.3f} m'
            + (f', {self.heading_tolerance:.3f} rad)' if self.end_criterion == 'pose' else ')')
            + f'; logs in {self.output_dir}')

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _plan_info_cb(self, msg: String):
        try:
            info = json.loads(msg.data)
            reference = Reference.from_plan_info(info)
        except (ValueError, KeyError) as exc:
            self.get_logger().error(f'Bad plan_info: {exc}')
            return
        if self.armed and self.recorder is not None and self.t_start is not None:
            # New plan while a run was active: close the old one first.
            self._end_run('replanned')
        self.info = info
        self.recorder = RunRecorder(reference)
        self.armed = True
        self.t_start = None
        self.finished_flag_time = None
        self.arm_time = self._now()
        factor = float(self.get_parameter('run_timeout_factor').value)
        margin = float(self.get_parameter('run_timeout_margin').value)
        self.timeout = factor * reference.theoretical_time + margin
        self.get_logger().info(
            f"ARMED run {info.get('run_index')} scenario {info.get('scenario')} | "
            f"theoretical {reference.theoretical_time:.2f} s | timeout {self.timeout:.0f} s | "
            f"waiting for start event '{self.get_parameter('start_event').value}'")

    def _pose_cb(self, msg: PoseStamped):
        if not self.armed or self.recorder is None:
            return
        t = self._now()
        self.recorder.add_pose(t, msg.pose.position.x, msg.pose.position.y,
                               yaw_from_quaternion(msg.pose.orientation))
        self._maybe_start()
        self._maybe_end(t)

    def _cmd_stamped_cb(self, msg: TwistStamped):
        if not self.armed or self.recorder is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.recorder.add_cmd(stamp if stamp > 0 else self._now(),
                              msg.twist.linear.x, msg.twist.angular.z)
        self._maybe_start()

    def _cmd_cb(self, msg: Twist):
        if not self.armed or self.recorder is None:
            return
        self.recorder.add_cmd(self._now(), msg.linear.x, msg.angular.z)
        self._maybe_start()

    def _finished_cb(self, msg: Bool):
        if msg.data and self.armed:
            self.finished_flag_time = self._now()
            # /finished_tracking is informative; the own goal-radius check ends
            # the run. If the MPC stops but the robot is outside the radius,
            # the timeout will close the run and the flag time is reported.

    # ------------------------------------------------------------------
    def _maybe_start(self):
        if self.t_start is not None or self.recorder is None:
            return
        event = self.get_parameter('start_event').value
        if event == 'first_cmd':
            t = self.recorder.first_cmd_time()
        elif event == 'first_motion':
            t = self.recorder.first_motion_time(
                float(self.get_parameter('motion_eps_xy').value),
                float(self.get_parameter('motion_eps_yaw').value))
        else:
            t = self.recorder.first_nonzero_cmd_time(
                float(self.get_parameter('v_eps').value),
                float(self.get_parameter('w_eps').value))
        if t is not None:
            self.t_start = t
            self.get_logger().info(f'Run STARTED (event {event})')

    def _maybe_end(self, t_now: float):
        if self.recorder is None:
            return
        if self.t_start is not None:
            t_goal = self.recorder.goal_reached_time(
                self.goal_radius, self.end_criterion, self.heading_tolerance)
            if t_goal is not None and t_goal >= self.t_start:
                self._end_run('goal_reached', t_end=t_goal)
                return
        if self.arm_time is not None and self.timeout is not None \
                and (t_now - self.arm_time) > self.timeout:
            self._end_run('timeout')

    def _srv_abort(self, _req, res):
        if self.armed and self.recorder is not None:
            self._end_run('aborted')
            res.success = True
            res.message = 'Run aborted and summarized.'
        else:
            res.success = False
            res.message = 'No active run.'
        return res

    # ------------------------------------------------------------------
    def _end_run(self, reason: str, t_end: Optional[float] = None):
        recorder, info = self.recorder, self.info
        self.armed = False
        summary = recorder.summarize(self.t_start, t_end, self.goal_radius,
                                     self.end_criterion, self.heading_tolerance)
        summary['end_reason'] = reason
        summary['finished_tracking_time'] = self.finished_flag_time
        summary['start_event'] = self.get_parameter('start_event').value
        summary['first_cmd_time'] = recorder.first_cmd_time()
        summary['first_nonzero_cmd_time'] = recorder.first_nonzero_cmd_time(
            float(self.get_parameter('v_eps').value), float(self.get_parameter('w_eps').value))
        summary['first_motion_time'] = recorder.first_motion_time(
            float(self.get_parameter('motion_eps_xy').value),
            float(self.get_parameter('motion_eps_yaw').value))

        run_idx = info.get('run_index', 0)
        session = info.get('session', 'session')
        self.get_logger().info(f'Run {run_idx} ENDED ({reason}): {summary_line(summary)}')

        payload = {'info': {k: v for k, v in info.items() if k != 'reference'},
                   'summary': summary}
        self.summary_pub.publish(String(data=json.dumps(payload)))

        # per-run JSON (with series) + one CSV row per run
        json_path = os.path.join(self.output_dir, f'{session}_run{run_idx:03d}.json')
        with open(json_path, 'w') as f:
            json.dump({'info': info, 'summary': summary,
                       'raw': recorder.raw_dict()}, f)
        csv_path = os.path.join(self.output_dir, f'{session}.csv')
        rotated = append_csv_row(csv_path, csv_row(info, summary, reason))
        if rotated is not None:
            self.get_logger().warn(
                f'{csv_path} had a different column set (an older schema); '
                f'moved it to {rotated} and started a new file.')
        self.get_logger().info(f'Saved {json_path} and appended to {csv_path}')

    def _live_tick(self):
        if not self.armed or self.recorder is None or not self.recorder.t:
            return
        d_goal = self.recorder.distance_to_goal()
        cross = self.recorder.cross_track_now()
        elapsed = (self.recorder.t[-1] - self.t_start) if self.t_start is not None else 0.0
        state = 'running' if self.t_start is not None else 'armed'
        line = (f'[{state}] t={elapsed:5.1f} s | cross-track {cross * 100:5.1f} cm | '
                f'to goal {d_goal * 100:5.1f} cm')
        if self.end_criterion == 'pose':
            d_yaw = abs(float(wrap_angle(self.recorder.theta[-1]
                                         - self.recorder.reference.goal_pose[2])))
            line += f' / {math.degrees(d_yaw):5.1f} deg'
        self.get_logger().info(line)
        self.live_pub.publish(String(data=json.dumps(
            {'state': state, 'elapsed': elapsed, 'cross_track': cross,
             'distance_to_goal': d_goal})))


def main(args=None):
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
