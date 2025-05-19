#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from scipy.spatial import KDTree, transform

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car with extra utility functions
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.sim = False

        # Subscribers
        if self.sim:
            self.create_subscription(Odometry, "/ego_racecar/odom", self.pose_callback, 10)
        else:
            self.create_subscription(PoseStamped, "/pf/viz/inferred_pose", self.pose_callback, 10)

        # Publisher
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Load CSV waypoints
        csv_data = np.loadtxt("./src/slam-and-pure-pursuit-team10/pure_pursuit/scripts/final_smoothed_points.csv", delimiter=",", skiprows=0)
        self.waypoints = csv_data[:, :2]  # each row: [x, y]
        self.kd_tree = KDTree(self.waypoints)

        # Basic parameters
        self.L = 1.3    # dynamic lookahead will be updated
        self.P = 0.35    # "proportional" gain for curvature → steering
        
        # Speed control parameters
        self.max_speed = 3.75
        self.min_speed = 1.5
        self.curvature_factor = 2.0  # Adjust this to control how much curvature affects speed
        self.look_ahead_distance = 5.0  # Distance to look ahead for curvature calculation

        # Visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints_markers)

        # State
        self.target_index = 0  # index of current target waypoint
        self.loop_waypoints = False  # set True if you want to wrap around at the end
        self.prev_speed = 0.0  # For speed smoothing

        self.get_logger().info("PurePursuit Node Initialized.")

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
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array) 

    def pose_callback(self, msg):
        """Main callback for pose updates."""
        if self.sim:
            car_x = msg.pose.pose.position.x
            car_y = msg.pose.pose.position.y
            quat = msg.pose.pose.orientation
            speed = msg.twist.twist.linear.x
        else:
            car_x = msg.pose.position.x
            car_y = msg.pose.position.y
            quat = msg.pose.orientation
            speed = 0.0

        # Convert quaternion to rotation matrix
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w])
        R_obj = transform.Rotation.from_quat(quat_np)
        self.rot = R_obj.as_matrix()

        # 1) Find the target waypoint at a distance >= self.L from current position
        _, nearest_idx = self.kd_tree.query([car_x, car_y])
        goal_x, goal_y, found_goal = self.find_lookahead_waypoint(car_x, car_y, nearest_idx, self.L)
        if not found_goal:
            # No valid waypoint was found
            return

        # 2) Transform goal point to vehicle frame
        car_pos = np.array([car_x, car_y])
        goal_y_vehicle = self.translatePoint(car_pos, np.array([goal_x, goal_y]))[1]

        # 3) Calculate curvature -> steering angle
        curvature = 2.0 * goal_y_vehicle / (self.L ** 2)
        steering_angle = self.P * curvature
        steering_angle = self.constrain(steering_angle, -0.4, 0.4)

        # 4) Calculate path curvature ahead for speed adjustment
        avg_curvature = self.calculate_path_curvature(car_x, car_y, nearest_idx)
        
        # 5) Adjust speed based on curvature
        target_speed = self.calculate_speed_from_curvature(avg_curvature, abs(steering_angle))
        
        # 6) Smooth speed transitions
        speed = self.smooth_speed(target_speed)

        # 7) Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed

        self.drive_pub.publish(drive_msg)

    def calculate_path_curvature(self, car_x, car_y, nearest_idx):
        """Calculate the average curvature of the path ahead."""
        # Get indices of waypoints ahead within look_ahead_distance
        waypoints_ahead = []
        current_distance = 0.0
        idx = nearest_idx
        
        while current_distance < self.look_ahead_distance and idx < len(self.waypoints) - 1:
            next_idx = (idx + 1) % len(self.waypoints)
            segment_distance = self.distance_2d(
                self.waypoints[idx, 0], self.waypoints[idx, 1],
                self.waypoints[next_idx, 0], self.waypoints[next_idx, 1]
            )
            current_distance += segment_distance
            waypoints_ahead.append(idx)
            idx = next_idx
            if idx == nearest_idx:  # We've looped around
                break
        
        # Calculate curvature for segments
        curvatures = []
        for i in range(len(waypoints_ahead) - 2):
            p1 = self.waypoints[waypoints_ahead[i]]
            p2 = self.waypoints[waypoints_ahead[i+1]]
            p3 = self.waypoints[waypoints_ahead[i+2]]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle between vectors (approximation of curvature)
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 * norm_v2 == 0:
                continue
                
            cos_angle = dot_product / (norm_v1 * norm_v2)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Ensure within valid range
            angle = math.acos(cos_angle)
            
            # Higher angle means sharper turn
            curvatures.append(angle)
        
        if not curvatures:
            return 0.0
            
        # Return average curvature
        return sum(curvatures) / len(curvatures)

    def calculate_speed_from_curvature(self, curvature, steering_angle):
        """Calculate target speed based on path curvature and current steering angle."""
        # Combine both curvature and steering angle to determine speed
        combined_factor = curvature + abs(steering_angle) * 2.0
        
        # Exponential decay of speed based on combined factor
        speed = self.max_speed * math.exp(-self.curvature_factor * combined_factor)
        
        # Ensure speed is within bounds
        return max(self.min_speed, min(self.max_speed, speed))

    def smooth_speed(self, target_speed):
        """Apply smoothing to speed changes to prevent jerky acceleration/deceleration."""
        max_accel = 0.5  # Maximum acceleration per callback
        max_decel = 0.7  # Maximum deceleration per callback
        
        if target_speed > self.prev_speed:
            # Accelerating
            self.prev_speed = min(target_speed, self.prev_speed + max_accel)
        else:
            # Decelerating
            self.prev_speed = max(target_speed, self.prev_speed - max_decel)
            
        return self.prev_speed

    def translatePoint(self, currPoint, targetPoint):
        """Transform targetPoint into the vehicle frame using the current rotation matrix."""
        H = np.zeros((4, 4))
        H[0:3, 0:3] = np.linalg.inv(self.rot)  # from world to car
        H[0, 3] = currPoint[0]
        H[1, 3] = currPoint[1]
        H[3, 3] = 1.0
        direction = targetPoint - currPoint
        # Homogeneous coords
        translated_point = (H @ np.array((direction[0], direction[1], 0.0, 0.0))).reshape((4))
        return translated_point

    def find_lookahead_waypoint(self, cx, cy, start_idx, lookahead_dist):
        """
        Finds a waypoint from self.waypoints that is at least lookahead_dist away
        from (cx,cy), starting at index 'start_idx'.
        """
        for i in range(start_idx, len(self.waypoints)):
            d = self.distance_2d(self.waypoints[i,0], self.waypoints[i,1], cx, cy)
            if d >= lookahead_dist:
                self.target_index = i
                return self.waypoints[i,0], self.waypoints[i,1], True

        # If we want to loop, wrap around to 0
        if self.loop_waypoints and len(self.waypoints) > 0:
            self.target_index = 0
            return self.waypoints[0,0], self.waypoints[0,1], True

        return 0.0, 0.0, False

    def distance_2d(self, x1, y1, x2, y2):
        """
        Simple 2D Euclidean distance function
        """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def constrain(self, val, min_val, max_val):
        """Constrain a value between min and max."""
        return min(max_val, max(min_val, val))

    def get_x_offset(self, pose_x, pose_y, yaw, waypoint_x, waypoint_y):
        """
        Returns the 'sideways' offset in the vehicle frame.
        """
        dx = waypoint_x - pose_x
        dy = waypoint_y - pose_y
        distance = math.hypot(dx, dy)
        beta = math.atan2(dx, dy)
        gamma = math.pi/2 - yaw - beta
        x_car = -distance * math.sin(gamma)
        return x_car

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
