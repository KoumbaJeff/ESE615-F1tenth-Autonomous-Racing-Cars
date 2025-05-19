#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# TODO CHECK: include needed ROS msg type headers and libraries
from nav_msgs.msg import Odometry
from scipy.spatial import KDTree, transform
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sim = True

        # TODO: create ROS subscribers and publishers
        if self.sim:
            odom_topic = "/ego_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.L = 1.3
        self.P = 0.5
        csv_data = np.loadtxt("./src/slam-and-pure-pursuit-team10/pure_pursuit/scripts/levine.csv", delimiter=",", skiprows=0)
        self.waypoints = csv_data[:, 0:2]
        self.kd_tree = KDTree(self.waypoints)

        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints_markers)

    def publish_waypoints_markers(self):
        marker_array = MarkerArray()
        
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0 
            marker.color.r = 1.0 
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array) 

    def pose_callback(self, pose_msg):
        # TODO: find the current waypoint to track using methods mentioned in lecture
        if self.sim:
            car_x = pose_msg.pose.pose.position.x
            car_y = pose_msg.pose.pose.position.y
            quat = pose_msg.pose.pose.orientation
        else:
            car_x = pose_msg.pose.position.x
            car_y = pose_msg.pose.position.y
            quat = pose_msg.pose.orientation

        
        quat = [quat.x, quat.y, quat.z, quat.w]
        R = transform.Rotation.from_quat(quat)
        self.rot = R.as_matrix()

        _, idx = self.kd_tree.query([car_x, car_y])
        for i in range(idx, len(self.waypoints)):
            dist = np.linalg.norm(self.waypoints[i] - np.array([car_x, car_y]))
            if dist >= self.L:
                goal_x, goal_y = self.waypoints[i]
                break
        else:
            # self.get_logger().warn("No valid lookahead point found.")
            return
        
        print(goal_x, goal_y)

        # TODO: transform goal point to vehicle frame of reference
        goal_y_vehicle = self.translatePoint(np.array([car_x, car_y]), np.array([goal_x, goal_y]))[1]

        # TODO: calculate curvature/steering angle
        curvature = 2 * goal_y_vehicle / (self.L ** 2)
        steering_angle = self.P * curvature

        # TODO: publish drive message, don't forget to limit the steering angle.
        steering_angle = np.clip(steering_angle, -0.35, 0.35)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.5
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)

    def get_yaw_from_pose(self, pose_msg):
        q = pose_msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y ** 2 + q.z ** 2)
        return np.arctan2(siny_cosp, cosy_cosp)
    
    def translatePoint(self, currPoint, targetPoint):
        H = np.zeros((4, 4))
        H[0:3, 0:3] = np.linalg.inv(self.rot)
        H[0, 3] = currPoint[0]
        H[1, 3] = currPoint[1]
        H[3, 3] = 1.0
        dir = targetPoint - currPoint
        translated_point = (H @ np.array((dir[0], dir[1], 0, 0))).reshape((4))
        
        return translated_point

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
