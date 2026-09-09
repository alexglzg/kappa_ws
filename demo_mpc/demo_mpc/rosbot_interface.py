import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Empty
from geometry_msgs.msg import PoseStamped
from math import atan2, pi, sin, cos, ceil, floor

def yaw_from_quaternion(quaternion):
    siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
    cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
    yaw = atan2(siny_cosp, cosy_cosp)
    return yaw

class RosbotInterface(Node):
    def __init__(self):
        super().__init__('rosbot_interface')

        self.vive_2D_pose = [0.0]*3
        self.state_measurement = [0.0]*4

        # Create a publisher for the /truck_trailer/state topic
        self.state_publisher = self.create_publisher(Float64MultiArray, '/rosbot2pro/state', 1)
        self.rosbot_pose_publisher = self.create_publisher(PoseStamped, '/rosbot2pro/pose', 1)

        self.TURN_COUNTER = 0

        self.reset_counter_subscriber = self.create_subscription(Empty, '/rosbot2pro/reset_turn_counter', self.reset_counter_callback, 10)

        self.previous_yaw = None

        # Create a subscriber for the /vive_pose topic
        self.vive_pose_subscription = self.create_subscription(
            PoseStamped,
            '/rosbot3/vive/pose',
            self.vive_pose_callback,
            1
        )
        self.vive_pose_subscription  # prevent unused variable warning

    def efficient_sign(self, number):
        return ceil(number/abs(number)) if number > 0 else floor(number/abs(number)) if number < 0 else 0

    def reset_counter_callback(self, msg):
        self.TURN_COUNTER = 0

    def vive_pose_callback(self, msg):

        yaw_offset = 0 #(13*pi/180)
               
        if self.previous_yaw is not None:
            yaw = yaw_from_quaternion(msg.pose.orientation)

            yaw_diff = self.previous_yaw - yaw
            self.previous_yaw = yaw

            if abs(yaw_diff) > 1.57:
                self.TURN_COUNTER += self.efficient_sign(yaw_diff)

            corrected_yaw = yaw + self.TURN_COUNTER*2*pi
            
            self.vive_2D_pose = [msg.pose.position.x, msg.pose.position.y, corrected_yaw-yaw_offset]      

            self.set_state_measurement(msg, self.vive_2D_pose)

        else:
            yaw = yaw_from_quaternion(msg.pose.orientation)
            self.previous_yaw = yaw
            self.vive_2D_pose = [msg.pose.position.x, msg.pose.position.y, yaw-yaw_offset]      

            self.set_state_measurement(msg, self.vive_2D_pose)


    def set_state_measurement(self, msg, vive_2D_pose):
        x = vive_2D_pose[0]
        y = vive_2D_pose[1]
        theta = vive_2D_pose[2]
  
        self.state_measurement = [x, y, theta]

        # Create a Float64MultiArray message
        state_msg = Float64MultiArray()
        state_msg.data = self.state_measurement

        # self.get_logger().info(f'[TruckTrailerInterface] State message: {state_msg.data}')

        # Publish the state message
        self.state_publisher.publish(state_msg)

        # Create a PoseStamped message
        rosbot_pose_msg = PoseStamped()
        rosbot_pose_msg.header = msg.header
        rosbot_pose_msg.pose.position.x = vive_2D_pose[0]
        rosbot_pose_msg.pose.position.y = vive_2D_pose[1]
        rosbot_pose_msg.pose.position.z = 0.00

        # convert from euler angles to quaternion and set orientation of tracker_pose_msg
        rosbot_pose_msg.pose.orientation.x = 0.0
        rosbot_pose_msg.pose.orientation.y = 0.0
        rosbot_pose_msg.pose.orientation.z = sin(theta/2)
        rosbot_pose_msg.pose.orientation.w = cos(theta/2)

        # Publish the tracker_pose message
        self.rosbot_pose_publisher.publish(rosbot_pose_msg)


def main(args=None):
    rclpy.init(args=args)

    rosbot_interface = RosbotInterface()

    rclpy.spin(rosbot_interface)

    rosbot_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
