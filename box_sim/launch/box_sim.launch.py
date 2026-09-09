#!/usr/bin/env python3
"""
Simple launch file for box robot simulation
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    
    # Read the URDF file directly (no xacro processing needed!)
    urdf_file = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'box_robot.urdf')
    
    # If the URDF doesn't exist at that path, try current directory
    if not os.path.exists(urdf_file):
        urdf_file = 'box_robot.urdf'
    
    # Read URDF content
    with open(urdf_file, 'r') as file:
        robot_description_content = file.read()
    
    # Robot State Publisher - publishes the robot model to RViz
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': False
        }],
        output='screen'
    )
    
    # Box Robot Simulator
    simulator = Node(
        package='box_sim',
        executable='box_robot_simulator',
        name='box_robot_simulator',
        output='screen'
    )
    
    # Static transform: map -> odom
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_broadcaster',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    
    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )
    
    return LaunchDescription([
        robot_state_publisher,
        simulator,
        # static_tf,
        # rviz
    ])