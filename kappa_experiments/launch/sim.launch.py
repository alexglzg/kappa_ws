"""Simulation launch: box_sim + the same stack as the lab.

box_sim publishes the Vive pose on ``/vive/pose`` and — since it also
subscribes to TwistStamped on ``/rosbot3/cmd_vel`` — accepts the MPC's
commands unchanged. Two things differ from the lab launch:

``pose_topic``   ``/vive/pose`` (also remaps rosbot_interface's input);
``rviz_config``  ``floor_projection.rviz``, the laptop-screen view, instead of
                 the calibrated ``projector_lab.rviz`` (which carries the
                 projector's angle, scale, offset and 3840x2123 geometry).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory('kappa_experiments')
    box_share = get_package_share_directory('box_sim')

    return LaunchDescription([
        DeclareLaunchArgument('scenario', default_value='1'),
        DeclareLaunchArgument('session_name', default_value='sim'),
        DeclareLaunchArgument('start_pose_source', default_value='measured'),
        DeclareLaunchArgument('mpc', default_value='true'),
        DeclareLaunchArgument('metrics', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('control_rate', default_value='10.0'),
        # Laptop screen: no projector geometry. Same fixed frame (map).
        DeclareLaunchArgument('rviz_config',
                              default_value=os.path.join(share, 'config',
                                                         'floor_projection.rviz')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(box_share, 'launch', 'box_sim.launch.py')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'experiment.launch.py')),
            launch_arguments={
                'scenario': LaunchConfiguration('scenario'),
                'session_name': LaunchConfiguration('session_name'),
                'start_pose_source': LaunchConfiguration('start_pose_source'),
                'mpc': LaunchConfiguration('mpc'),
                'metrics': LaunchConfiguration('metrics'),
                # box_sim's pose topic; also remaps rosbot_interface's input.
                'pose_topic': '/vive/pose',
                'map_frame': LaunchConfiguration('map_frame'),
                'control_rate': LaunchConfiguration('control_rate'),
                'rviz': LaunchConfiguration('rviz'),
                'rviz_config': LaunchConfiguration('rviz_config'),
            }.items(),
        ),
    ])
