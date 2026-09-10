"""Lab launch: rosbot interface + MPC + experiment node + metrics node + RViz.

The full lab stack:

  rosbot_interface  /robot_pose             -> /rosbot2pro/state
  mpc_test_node     /rosbot2pro/state + /rosbot2pro/planned_path
                                            -> /rosbot3/cmd_vel (TwistStamped)
  experiment_node   scenarios, planning     -> /rosbot2pro/planned_path
  metrics_node      pose + commands         -> run summaries / CSV / JSON

Lab and sim are wired identically. The measured pose comes in on
``/robot_pose`` (PoseStamped in the ``map`` frame) in both cases — the lab
publisher and box_sim use that same topic, so sim.launch.py overrides nothing.
``/vive/pose`` is the raw tracker output in the tilted ``vive_world`` frame and
must never be used here. One ``pose_topic`` argument still covers all four
nodes: it is the parameter of experiment_node and metrics_node, and
rosbot_interface's hardcoded input is remapped onto it (an identity by
default). The command interface is likewise identical: box_sim subscribes to
TwistStamped on ``/rosbot3/cmd_vel``, the topic the MPC publishes.

Two more single-point-of-truth arguments:

``map_frame``    the frame everything RViz shows is published in — every
                 marker, the path and the robot ring come from experiment_node,
                 so this one argument moves the whole scene.
``control_rate`` the control frequency [Hz]. The MPC timer period, its horizon
                 discretisation (``n_horizon = horizon_time * rate``) and the
                 reference spacing (``sampling_dt = 1 / rate``) are all derived
                 from it, so they cannot drift apart.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('kappa_experiments')
    params = os.path.join(share, 'config', 'experiment_params.yaml')

    scenario = LaunchConfiguration('scenario')
    session = LaunchConfiguration('session_name')
    start_src = LaunchConfiguration('start_pose_source')
    use_rviz = LaunchConfiguration('rviz')
    use_mpc = LaunchConfiguration('mpc')
    use_metrics = LaunchConfiguration('metrics')
    pose_topic = LaunchConfiguration('pose_topic')
    cmd_topic = LaunchConfiguration('cmd_topic')
    cmd_type = LaunchConfiguration('cmd_type')
    goal_radius = LaunchConfiguration('goal_radius')
    output_dir = LaunchConfiguration('output_dir')
    map_frame = LaunchConfiguration('map_frame')
    rviz_config = LaunchConfiguration('rviz_config')
    control_rate = LaunchConfiguration('control_rate')
    horizon_time = LaunchConfiguration('horizon_time')

    # Derived from control_rate, so the three stay consistent by construction.
    control_period = ParameterValue(
        PythonExpression(['1.0 / ', control_rate]), value_type=float)
    n_horizon = ParameterValue(
        PythonExpression(['round(', horizon_time, ' * ', control_rate, ')']), value_type=int)
    sampling_dt = ParameterValue(
        PythonExpression(['1.0 / ', control_rate]), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='1'),
        DeclareLaunchArgument('session_name', default_value='session'),
        DeclareLaunchArgument('start_pose_source', default_value='measured'),
        DeclareLaunchArgument('rviz', default_value='true'),
        # rosbot_interface + mpc_test_node (the sim launch keeps them on too).
        DeclareLaunchArgument('mpc', default_value='true'),
        DeclareLaunchArgument('metrics', default_value='true'),
        # Measured pose in the map frame, from the lab publisher or box_sim.
        DeclareLaunchArgument('pose_topic', default_value='/robot_pose'),
        # Commands: same topic and type in both cases.
        DeclareLaunchArgument('cmd_topic', default_value='/rosbot3/cmd_vel'),
        DeclareLaunchArgument('cmd_type', default_value='TwistStamped'),
        # Keep equal to the MPC tolerance_radius (mpc_test_node: 0.05).
        DeclareLaunchArgument('goal_radius', default_value='0.05'),
        DeclareLaunchArgument('output_dir', default_value='~/kappa_experiment_logs'),
        # Frame of every marker, of the planned path and of the robot ring.
        DeclareLaunchArgument('map_frame', default_value='map'),
        # Calibrated projector view (angle, scale, offset, 3840x2123 at X=1920).
        # Never regenerate this file; sim.launch.py points at floor_projection.rviz.
        DeclareLaunchArgument('rviz_config',
                              default_value=os.path.join(share, 'config', 'projector_lab.rviz')),
        # One knob for the control frequency: MPC timer, MPC horizon and the
        # reference sample time are all derived from it.
        # DeclareLaunchArgument('control_rate', default_value='10.0'),
        DeclareLaunchArgument('control_rate', default_value='20.0'),
        DeclareLaunchArgument('horizon_time', default_value='1.0'),
        Node(
            package='demo_mpc',
            executable='rosbot_interface',
            name='rosbot_interface',
            output='screen',
            condition=IfCondition(use_mpc),
            # Identity at the default; follows pose_topic if it is overridden.
            remappings=[('/robot_pose', pose_topic)],
        ),
        Node(
            package='demo_mpc',
            executable='mpc_test_node',
            name='mpc_test',
            output='screen',
            condition=IfCondition(use_mpc),
            parameters=[{
                'control_period': control_period,
                'n_horizon': n_horizon,
            }],
        ),
        Node(
            package='kappa_experiments',
            executable='experiment_node',
            name='experiment_node',
            output='screen',
            parameters=[params, {
                'scenario': scenario,
                'session_name': session,
                'start_pose_source': start_src,
                'pose_topic': pose_topic,
                'map_frame': map_frame,
                'sampling_dt': sampling_dt,
            }],
        ),
        Node(
            package='kappa_experiments',
            executable='metrics_node',
            name='metrics_node',
            output='screen',
            condition=IfCondition(use_metrics),
            parameters=[params, {
                'pose_topic': pose_topic,
                'cmd_topic': cmd_topic,
                'cmd_type': cmd_type,
                'goal_radius': goal_radius,
                'output_dir': output_dir,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(use_rviz),
        ),
    ])
