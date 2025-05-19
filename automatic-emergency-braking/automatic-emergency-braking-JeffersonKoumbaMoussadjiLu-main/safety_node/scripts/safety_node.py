#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.0 #Store current speed

        self.TTC = 1.1 #Threshold for TTC

        # TODO: create ROS subscribers and publishers.

        # One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.
        self.publisher_ = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

        # You should also subscribe to the /scan topic to get the LaserScan messages and
        # the /ego_racecar/odom topic to get the current speed of the vehicle.

        # LaserScan messages
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Odometry 
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, odom_msg):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x # x component of the linear velocity in odom is the speed (Forward direction)

    def scan_callback(self, scan_msg):
        # TODO: calculate TTC

        # LaserScan parameters
        ranges = scan_msg.ranges
        num_ranges = len(ranges)
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        range_min = scan_msg.range_min
        range_max = scan_msg.range_max

        # Calculate TTC for each laser scan (laser beam)
        min_ttc = float('inf')

        for i in range(num_ranges):

            # Calculate distance to object (range)
            distance = ranges[i]

            # Skip invalid measurements
            if math.isnan(distance) or math.isinf(distance) or distance < range_min or distance > range_max:
                continue

            # Calculate angle laser beam
            angle = angle_min + i * angle_inc

            # Calculate relative velocity (vx * cos(theta) = r_dot)
            relative_speed = self.speed * np.cos(angle)

            # Calculate TTC 
            #
            #if relative_speed is negative, set it to 0 (skip it)
            if relative_speed <= 0:
                continue
                
            
            ttc = distance / relative_speed
            
            #find the minimum ttc
            if ttc < min_ttc:
                min_ttc = ttc

        #self.get_logger().info(f"min_ttc: {min_ttc:.2f}s")
    
        # TODO: publish command to brake

        #Check if min_ttc is less than the threshold
        if min_ttc < self.TTC:

            # Emergency brake
            brake_msg = AckermannDriveStamped()

            brake_msg.drive.speed = 0.0 #set speed to 0
            brake_msg.drive.steering_angle = 0.0 #set steering angle to 0

            self.publisher_.publish(brake_msg)

            self.get_logger().warn(f"EMERGENCY BRAKE! iTTC: {min_ttc:.2f}s")


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

