#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
import time
from numpy.linalg import norm
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped

class WaypointLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')
        
        # Create log directory if it doesn't exist
        home = os.path.expanduser('~')
        log_dir = os.path.join(home, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped file
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
        self.file_path = os.path.join(log_dir, f'wp-{timestamp}.csv')
        self.file = open(self.file_path, 'w')
        
        # Subscribe to odometry topic
        self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.save_waypoint, 10)
        
        self.get_logger().info(f'Saving waypoints to {self.file_path}...')

    def save_waypoint(self, data):
        # Extract position and orientation
        x, y = data.pose.position.x, data.pose.position.y
        quaternion = [
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        ]
        
        # Convert quaternion to yaw (Euler angles)
        _, _, yaw = euler_from_quaternion(quaternion)

        # Compute speed
        # speed = norm([
        #     data.twist.linear.x,
        #     data.twist.linear.y,
        #     data.twist.linear.z
        # ], 2)

        # Log data if the robot is moving
        # if data.twist.linear.x > 0.:
        #     self.get_logger().info(f'Moving: x={x}, y={y}, yaw={yaw:.3f}, speed={speed:.3f}')
        
        # Write to file
        self.file.write(f'{x}, {y}, {yaw}\n')

    def shutdown(self):
        self.file.close()
        self.get_logger().info('Waypoint logging stopped.')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
