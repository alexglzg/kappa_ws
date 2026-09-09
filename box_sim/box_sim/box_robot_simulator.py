#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import (Twist, TwistStamped, PoseStamped, PoseWithCovarianceStamped,
                              TransformStamped, Quaternion)
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf_transformations
import math

class BoxRobotSimulator(Node):
    def __init__(self):
        super().__init__('box_robot_simulator')
        self.get_logger().info('Box Robot Simulator Started')
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.wheel_angles = [0.0, 0.0]
        
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/vive/pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.cmd_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        # Same command interface as the real robot, so that the MPC and the
        # metrics node need no topic/type changes between sim and lab.
        self.cmd_stamped_sub = self.create_subscription(
            TwistStamped, '/rosbot3/cmd_vel', self.cmd_stamped_callback, 10)
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)
        self.timer = self.create_timer(0.05, self.update)
        self.last_time = self.get_clock().now()
    
    def cmd_callback(self, msg):
        self.linear_vel = max(-1.0, min(1.0, msg.linear.x))
        self.angular_vel = max(-2.0, min(2.0, msg.angular.z))

    def cmd_stamped_callback(self, msg):
        self.cmd_callback(msg.twist)

    def initial_pose_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
    
    def euler_to_quaternion(self, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.w = cy
        q.x = 0.0
        q.y = 0.0
        q.z = sy
        return q
    
    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        
        self.x += self.linear_vel * math.cos(self.theta) * dt
        self.y += self.linear_vel * math.sin(self.theta) * dt
        self.theta += self.angular_vel * dt
        
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
        
        self.wheel_angles[0] += (self.linear_vel / 0.05) * dt
        self.wheel_angles[1] += (self.linear_vel / 0.05) * dt
        
        # Publish joint states
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = ['left_wheel_joint', 'right_wheel_joint']
        js.position = self.wheel_angles
        self.joint_pub.publish(js)
        
        # Publish odometry
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = self.euler_to_quaternion(self.theta)
        odom.twist.twist.linear.x = self.linear_vel
        odom.twist.twist.angular.z = self.angular_vel
        self.odom_pub.publish(odom)
        
        # Publish TF
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = self.euler_to_quaternion(self.theta)
        self.tf_broadcaster.sendTransform(t)
        
        # Publish vive pose
        pose = PoseStamped()
        pose.header.stamp = now.to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.orientation = self.euler_to_quaternion(self.theta)
        self.pose_pub.publish(pose)
        
        self.last_time = now

def main():
    rclpy.init()
    node = BoxRobotSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
