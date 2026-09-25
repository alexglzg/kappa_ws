import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, PoseStamped, TwistStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf_transformations

from rockit import *
from casadi import *

import numpy as np

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_test')

        # Control rate and horizon. The reference spacing (experiment_node's
        # sampling_dt) must equal control_period; the launch files derive both
        # from a single control_rate argument. Defaults are the historical
        # 10 Hz / 10 intervals = 1.0 s horizon.
        self.declare_parameter('control_period', 0.1)     # s, timer and OCP dt
        self.declare_parameter('n_horizon', 10)           # control intervals
        self.control_period = float(self.get_parameter('control_period').value)
        self.Nhor = int(self.get_parameter('n_horizon').value)

        # Cost weights. They are baked into the compiled OCP function, so a
        # change needs a node restart. The (v, omega) tracking terms are only
        # active while a planned_controls array is paired with the path
        # (ff_scale = 1); without one the cost is exactly the pose-only one.
        self.declare_parameter('w_v_track', 0.05)
        self.declare_parameter('w_w_track', 0.05)
        self.declare_parameter('w_pos', 10.0)
        self.declare_parameter('w_heading', 0.1)
        w_v_track = float(self.get_parameter('w_v_track').value)
        w_w_track = float(self.get_parameter('w_w_track').value)
        w_pos = float(self.get_parameter('w_pos').value)
        w_heading = float(self.get_parameter('w_heading').value)

        self.state_subscriber = self.create_subscription(Float64MultiArray, '/rosbot2pro/state', self.state_listener_callback, 10)
        self.current_state = None
        self.path_subscriber = self.create_subscription(Path, '/rosbot2pro/planned_path', self.path_listener_callback, 10)
        # Reference (v, omega) on the same grid as the path, rows [t, v, omega].
        # The publisher is latched, so subscribe transient-local; it may arrive
        # before or after the path it belongs to.
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.controls_subscriber = self.create_subscription(
            Float64MultiArray, '/experiment_node/planned_controls',
            self.controls_listener_callback, latched)
        self.PATH_RECEIVED = False
        self.poses = []
        self.traj_len = 0
        self.latest_controls = None          # (M, 2) v_ref, omega_ref, last received
        self.ref_controls = None             # (N, 2) paired with the current path
        self.ff_scale = 0.0                  # 1.0 while ref_controls is paired
        self.ff_warned = False
        self.FINISHED = False
        self.tolerance_radius = 0.05
        self.heading_tolerance = 0.10        # rad, checked together with the radius
        self.settle_timeout = 3.0            # s to settle after the reference ends
        self.settle_start = None             # when the reference was first exhausted

        while (not self.current_state) or (not self.PATH_RECEIVED):
            rclpy.spin_once(self)
            self.get_logger().info(f'Waiting for current state and first path')

        self.control_signal_publisher = self.create_publisher(TwistStamped, '/rosbot3/cmd_vel', 10)
        self.finished_tracking_publisher = self.create_publisher(Bool, '/finished_tracking', 10)
        #self.planned_path_publisher = self.create_publisher(Path, '/rosbot2pro/planned_path', 10)
        self.timer = self.create_timer(self.control_period, self.timer_callback)

        # -------------------------------
        # Problem parameters
        # -------------------------------
        max_vel = 0.6
        min_vel = -0.02
        max_omega = 2.5
        max_accel = 1.0
        max_omega_dot = 5.0

        nx    = 5                   # the system is composed of 5 states
        nu    = 2                   # the system has 2 inputs
        self.dt    = self.control_period      # sample time = control period
        Tf    = self.dt * self.Nhor           # control horizon [s]
        self.get_logger().info(
            f'MPC at {1.0/self.control_period:.1f} Hz: dt {self.dt:.3f} s, '
            f'N {self.Nhor}, horizon {Tf:.2f} s')

        self.current_X = vertcat(0.0, 0.0, 0.0, 0.0, 0.0)  # initial state

        self.ocp = Ocp(T=Tf)

        # Define states
        x = self.ocp.state()
        y = self.ocp.state()
        theta = self.ocp.state()
        self.v = self.ocp.state()
        self.w = self.ocp.state()

        # Defince controls
        self.u1 = self.ocp.control()
        self.u2 = self.ocp.control()

        # Define parameter
        self.X_0 = self.ocp.parameter(nx)

        # Specify ODE
        self.ocp.set_der(x, (self.v*cos(theta)))
        self.ocp.set_der(y, (self.v*sin(theta)))
        self.ocp.set_der(theta, self.w)
        self.ocp.set_der(self.v, self.u1)
        self.ocp.set_der(self.w, self.u2)

        # Define a placeholder for concrete waypoints to be defined on edges of the control grid
        # Rows: x, y, theta, v_ref, omega_ref
        self.trajectory = self.ocp.parameter(5, grid='control')
        # 1.0: track the reference (v, omega); 0.0: pure pose feedback
        self.ff_scale_p = self.ocp.parameter(1)
        ff = self.ff_scale_p

        # Lagrange objective
        # self.ocp.add_objective(self.ocp.sum(5*(x-self.trajectory[0])**2 + 5*(y-self.trajectory[1])**2 + 0.001*self.w**2 + 0.01*self.u1**2 + 0.001*self.u2**2))
        self.ocp.add_objective(self.ocp.sum(w_pos*(x-self.trajectory[0])**2 + w_pos*(y-self.trajectory[1])**2
                                            + ff*(w_v_track*(self.v-self.trajectory[3])**2 + w_w_track*(self.w-self.trajectory[4])**2) ))
                                            # + (1-ff)*0.001*self.w**2
                                            # + 0.01*self.u1**2 + 0.001*self.u2**2))
        self.ocp.add_objective(self.ocp.sum(w_heading*(sin(theta)-sin(self.trajectory[2]))**2 + w_heading*(cos(theta)-cos(self.trajectory[2]))**2))
        self.ocp.add_objective(self.ocp.at_tf(w_pos*(x-self.trajectory[0])**2 + w_pos*(y-self.trajectory[1])**2))
        self.ocp.add_objective(self.ocp.at_tf(w_heading*(sin(theta)-sin(self.trajectory[2]))**2 + w_heading*(cos(theta)-cos(self.trajectory[2]))**2))

        # Vehicle constraints
        self.ocp.subject_to( (min_vel <= self.v) <= max_vel )
        self.ocp.subject_to( (-max_omega <= self.w) <= max_omega )
        self.ocp.subject_to( (-max_accel <= self.u1) <= max_accel )
        self.ocp.subject_to( (-max_omega_dot <= self.u2) <= max_omega_dot )

        # Initial constraints
        self.X = vertcat(x, y, theta, self.v, self.w)
        self.ocp.subject_to(self.ocp.at_t0(self.X)==self.X_0)

        # # Pick a solution method
        # options = {"ipopt": {"print_level": 0, "tol":1e-3}}
        # options["expand"] = True
        # options["print_time"] = False
        # self.ocp.solver('ipopt',options)
        
        self.ocp.solver("fatrop", {"expand":True, "print_time": False, "fatrop": {"tol": 1e-3, "print_level": 0}})


        # Make it concrete for this ocp
        self.ocp.method(MultipleShooting(N=self.Nhor,M=1,intg='expl_euler'))

        self.trajectory_N = np.zeros([5,self.Nhor])
        x_multiplier = 0.1
        y_amplitude = 0.0
        y_freq = 0.0
        for i in range(self.Nhor):
            timer = i * self.dt
            x_d = x_multiplier * timer
            y_d = y_amplitude*sin((y_freq) * timer)
            self.trajectory_N[0,i] = x_d
            self.trajectory_N[1,i] = y_d
            self.trajectory_N[2,i] = 0.0

        self.ocp.set_value(self.trajectory, self.trajectory_N)
        self.ocp.set_value(self.X_0, self.current_X)
        self.ocp.set_value(self.ff_scale_p, 0.0)
        # Solve
        #self.sol = self.ocp.solve()

        #self.sim_dyn = self.ocp._method.discrete_system(self.ocp)

        self.i = 0

        self.last_v = 0.0
        self.last_w = 0.0

        ###########################################################
        # u1_samp = self.ocp.sample(self.u1, grid='control-')[1]
        # u2_samp = self.ocp.sample(self.u2, grid='control-')[1]

        trajectory_samp = self.ocp.sample(self.trajectory, grid='control-')[1]
        X_0_samp = self.ocp.value(self.X_0)

        ff_samp = self.ocp.value(self.ff_scale_p)

        input_vector = [trajectory_samp, X_0_samp, ff_samp]
        input_names = ['trajectory', 'X_0', 'ff_scale']

        v_res = self.ocp.sample(self.v, grid='control')[1]
        w_res = self.ocp.sample(self.w, grid='control')[1]

        output_vector = [v_res[1], w_res[1]]
        output_names = ['v', 'w']

        self.OCP_function = self.ocp.to_function('ocp_fun', input_vector, output_vector, input_names, output_names)
       


    def state_listener_callback(self, msg):
        self.current_state = msg.data
        #print(self.current_state)

    def path_listener_callback(self, msg):
        print("New trajectory")
        self.PATH_RECEIVED = True
        self.FINISHED = False
        self.settle_start = None
        self.i = 0
        self.poses = msg.poses 
        self.traj_len = len(self.poses)
        self.ff_warned = False
        self._pair_controls()

    def controls_listener_callback(self, msg):
        self.latest_controls = np.asarray(msg.data, dtype=float).reshape(-1, 3)[:, 1:3]
        if self.PATH_RECEIVED:
            self._pair_controls()

    def _pair_controls(self):
        """Pair the current path with the latest controls if they have its length."""
        if self.latest_controls is not None and len(self.latest_controls) == self.traj_len:
            self.ref_controls = self.latest_controls
            self.ff_scale = 1.0
        else:
            self.ref_controls = None
            self.ff_scale = 0.0

    def quat2eul(self, x, y, z, w):
        quat = [x, y, z, w]
        eul = tf_transformations.euler_from_quaternion(quat)
        return eul

    def timer_callback(self):
        if self.PATH_RECEIVED:
            self.i+=1
            #tsa, f1sol = self.sol.sample(self.u1, grid='control')
            #_, f2sol = self.sol.sample(self.u2, grid='control')
            #_, Vsol = self.sol.sample(self.v, grid='control')
            #_, Wsol = self.sol.sample(self.w, grid='control')
            
            #self.current_X = self.sim_dyn(x0=self.current_X, u=vertcat(f1sol[0],f2sol[0]), T=self.dt)["xf"]
            
            self.current_X = vertcat(self.current_state[0], self.current_state[1], self.current_state[2], self.last_v, self.last_w)

            # Warned here rather than in the path callback: the controls of a
            # new plan normally arrive just after its path.
            if self.ref_controls is None and not self.ff_warned:
                n_ctrl = None if self.latest_controls is None else len(self.latest_controls)
                self.get_logger().warn(
                    f'No planned controls for this path (received {n_ctrl} rows, '
                    f'path has {self.traj_len}): pure pose feedback')
                self.ff_warned = True

            # self.ocp.set_value(self.X_0, self.current_X)
            # Set the new trajectory
            #x_multiplier = 0.5
            #y_amplitude = 0.2
            #y_freq = 3*pi/40

            #planned_path = Path()
            #planned_path.header.frame_id = "odom"

            '''for j in range(self.Nhor):
                timer = ((j+1)*self.dt) + (self.i*self.dt)
                x_d = x_multiplier * timer
                y_d = y_amplitude*sin((y_freq) * timer)
                self.trajectory_N[0,j] = x_d
                self.trajectory_N[1,j] = y_d
                #pose = PoseStamped()
                #pose.pose.position.x = x_d
                #pose.pose.position.y = y_d
                #pose.pose.position.z = 0.0
                #planned_path.poses.append(pose)
            '''

            if ((self.traj_len - self.i) > self.Nhor):
                for j in range(self.Nhor):
                    self.trajectory_N[0,j] = self.poses[j + self.i].pose.position.x
                    self.trajectory_N[1,j] = self.poses[j + self.i].pose.position.y
                    quat = self.poses[j + self.i].pose.orientation
                    _, _, yaw = self.quat2eul(quat.x, quat.y, quat.z, quat.w)
                    self.trajectory_N[2,j] = yaw
                    # self.trajectory_N[2,j] = self.poses[j + self.i].pose.position.z
                    if self.ref_controls is not None:
                        self.trajectory_N[3:5,j] = self.ref_controls[j + self.i]
                    else:
                        self.trajectory_N[3:5,j] = 0.0
            else:
                # Past the end of the reference: hold the final pose for the rest
                # of the horizon instead of leaving the previous horizon's stale
                # entries in trajectory_N.
                for j in range(self.Nhor):
                    k = min(j + self.i, self.traj_len - 1)
                    self.trajectory_N[0,j] = self.poses[k].pose.position.x
                    self.trajectory_N[1,j] = self.poses[k].pose.position.y
                    quat = self.poses[k].pose.orientation
                    _, _, yaw = self.quat2eul(quat.x, quat.y, quat.z, quat.w)
                    self.trajectory_N[2,j] = yaw
                    # velocities stop with the reference
                    if self.ref_controls is not None and j + self.i < self.traj_len:
                        self.trajectory_N[3:5,j] = self.ref_controls[j + self.i]
                    else:
                        self.trajectory_N[3:5,j] = 0.0

            # The reference is exhausted once i walks past the last pose. The
            # index is clamped right away, so the flag carries that to the
            # finish check below.
            reference_exhausted = (self.i >= self.traj_len)
            if reference_exhausted:
                if self.settle_start is None:
                    self.settle_start = self.get_clock().now()
                self.i = self.traj_len - 1

            #self.planned_path_publisher.publish(planned_path)

            # self.ocp.set_value(self.trajectory, self.trajectory_N)

            # self.sol = self.ocp.solve()

            # tsa, Vsol = self.sol.sample(self.v, grid='control')
            # _, Wsol = self.sol.sample(self.w, grid='control')
            # _, f2sol = self.sol.sample(self.u2, grid='control')

            Vsol, Wsol = self.OCP_function(self.trajectory_N, self.current_X, self.ff_scale)
            Vsol = float(Vsol)
            Wsol = float(Wsol)

            # Only check for the end once the whole reference has been played
            # out: before that the robot can pass within the tolerance radius of
            # the goal (a trajectory coming back on itself) without being done.
            if reference_exhausted:
                goal_pose = self.poses[self.traj_len-1].pose
                x = float(self.current_X[0])
                y = float(self.current_X[1])
                theta = float(self.current_X[2])
                distance_to_end = np.hypot(goal_pose.position.x - x,
                                           goal_pose.position.y - y)
                _, _, theta_goal = self.quat2eul(goal_pose.orientation.x,
                                                 goal_pose.orientation.y,
                                                 goal_pose.orientation.z,
                                                 goal_pose.orientation.w)
                heading_error = abs(np.arctan2(np.sin(theta_goal - theta),
                                               np.cos(theta_goal - theta)))
                settling_time = (self.get_clock().now() - self.settle_start).nanoseconds * 1e-9

                on_pose = (distance_to_end <= self.tolerance_radius
                           and heading_error <= self.heading_tolerance)
                if on_pose or settling_time > self.settle_timeout:
                    if on_pose:
                        print(f"Finished: {distance_to_end:.3f} m, "
                              f"{heading_error:.3f} rad from the goal pose")
                    else:
                        print(f"Finished on settle timeout after {settling_time:.1f} s: "
                              f"{distance_to_end:.3f} m, {heading_error:.3f} rad remaining")
                    Vsol = 0.0
                    Wsol = 0.0
                    self.PATH_RECEIVED = False
                    self.FINISHED = True
                    finished_msg = Bool()
                    finished_msg.data = self.FINISHED
                    self.finished_tracking_publisher.publish(finished_msg)

            control_signal = TwistStamped()
            # control_signal.linear.x = Vsol
            # control_signal.angular.z = Wsol
            control_signal.twist.linear.x = Vsol
            control_signal.twist.angular.z = Wsol
            control_signal.header.stamp = self.get_clock().now().to_msg()
            self.control_signal_publisher.publish(control_signal)

            #print(Vsol)

            self.last_v = Vsol
            self.last_w = Wsol

def main(args=None):
    rclpy.init(args=args)

    mpc_node = MPCNode()

    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ =='__main__':
    main()