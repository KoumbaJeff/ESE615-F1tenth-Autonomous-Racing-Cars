#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        # Declare parameters (with defaults)
        self.declare_parameter('v', 0.0)
        self.declare_parameter('d', 0.0)

        # Publisher to "drive"
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)

        # Timer to publish at ~100 Hz (fast)
        self.timer_ = self.create_timer(0.01, self.publish_drive)

    def publish_drive(self):
        # Read the parameters
        v = self.get_parameter('v').get_parameter_value().double_value
        d = self.get_parameter('d').get_parameter_value().double_value

        # Create and populate the message
        msg = AckermannDriveStamped()
        msg.drive.speed = v
        msg.drive.steering_angle = d

        # Publish
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing speed={v}, steering_angle={d}')

def main(args=None):
    rclpy.init(args=args)
    node = Talker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()

