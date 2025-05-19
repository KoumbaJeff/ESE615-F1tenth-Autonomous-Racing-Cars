#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped

class Relay(Node):
    def __init__(self):
        super().__init__('relay')
        # Subscribe to the "drive" topic
        self.subscriber_ = self.create_subscription(
            AckermannDriveStamped,
            'drive',
            self.drive_callback,
            10
        )
        # Publisher to "drive_relay"
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive_relay', 10)

    def drive_callback(self, msg_in):
        # Multiply speed and steering angle by 3
        msg_out = AckermannDriveStamped()
        msg_out.drive.speed = msg_in.drive.speed * 3
        msg_out.drive.steering_angle = msg_in.drive.steering_angle * 3

        # Publish the updated message
        self.publisher_.publish(msg_out)
        self.get_logger().info(f'Relaying speed={msg_out.drive.speed}, angle={msg_out.drive.steering_angle}')

def main(args=None):
    rclpy.init(args=args)
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
