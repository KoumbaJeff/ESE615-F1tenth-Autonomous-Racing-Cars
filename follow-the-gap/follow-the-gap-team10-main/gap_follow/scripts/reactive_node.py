#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('reactive_node')
        
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        
        self.pub_drive = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        self.drive_msg = AckermannDriveStamped()
        self.lidar_start = 180
        self.lidar_end   = 899
        self.downsample_gap = 10   
        self.max_sight = 4.0       
        self.bubble_radius = 2    
        self.max_gap_safe_dist = 1.2  

        self.extender_thres = 0.5   

        self.get_logger().info("ReactiveFollowGap Node Initialized.")
        
    def preprocess_lidar(self, ranges):
        n_total = len(ranges)
        n_downsampled = n_total // self.downsample_gap
        proc_ranges = np.zeros(n_downsampled, dtype=np.float32)

        for i in range(n_downsampled):
            start = i * self.downsample_gap
            end   = start + self.downsample_gap
            chunk = ranges[start:end]
            proc_ranges[i] = np.mean(chunk)
        proc_ranges = np.clip(proc_ranges, 0.0, self.max_sight)
        return proc_ranges

    def zero_out_bubble(self, proc_ranges, closest_idx):
        start = max(0, closest_idx - self.bubble_radius)
        end   = min(len(proc_ranges), closest_idx + self.bubble_radius)
        proc_ranges[start:end] = 0.0
        return proc_ranges

    def disparity_extender(self, proc_ranges):
        i = 0
        while i < len(proc_ranges) - 1:
            # If next beam is much larger => extend current beam forward
            if proc_ranges[i+1] - proc_ranges[i] >= self.extender_thres:
                proc_ranges[i : min(i + self.bubble_radius + 1, len(proc_ranges))] = proc_ranges[i]
                i += self.bubble_radius + 1
            # If current beam is much larger => extend next beam backward
            elif proc_ranges[i] - proc_ranges[i+1] >= self.extender_thres:
                start = max(0, i - self.bubble_radius + 1)
                proc_ranges[start : i+1] = proc_ranges[i+1]
                i += self.bubble_radius + 1
            else:
                i += 1
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        longest_streak = 0
        streak = 0
        end_index = 0
        start_index = 0

        for i in range(len(free_space_ranges)):
            if free_space_ranges[i] > self.max_gap_safe_dist:
                streak += 1
                if streak > longest_streak:
                    longest_streak = streak
                    end_index = i
                    start_index = end_index - longest_streak + 1
            else:
                streak = 0
        return start_index, end_index + 1

    def find_best_point(self, start_i, end_i, ranges):
        return (start_i + end_i) // 2

    def lidar_callback(self, data: LaserScan):
        """
        Follow-the-Gap main pipeline:
          1. Slice forward-facing LiDAR
          2. Downsample + clip
          3. Find closest obstacle + zero out bubble
          4. (Optional) Apply disparity extender
          5. Find max gap + best point
          6. Compute steering + dynamic speed
          7. Publish drive message
        """
        forward_lidar = np.array(data.ranges[self.lidar_start : self.lidar_end], dtype=np.float32)
        proc_ranges = self.preprocess_lidar(forward_lidar)
        closest_idx = np.argmin(proc_ranges)
        proc_ranges = self.zero_out_bubble(proc_ranges, closest_idx)
        proc_ranges = self.disparity_extender(proc_ranges)
        start_max_gap, end_max_gap = self.find_max_gap(proc_ranges)
        if start_max_gap == end_max_gap:
            self.get_logger().warn("No valid gap found! Driving straight slowly.")
            self.drive_msg.drive.steering_angle = 0.0
            self.drive_msg.drive.speed = 1.0
            self.pub_drive.publish(self.drive_msg)
            return
        
        best_i = self.find_best_point(start_max_gap, end_max_gap, proc_ranges)

        # 6. Compute steering angle
        total_degs = 180.0  
        deg_per_index = total_degs / len(proc_ranges)
        angle_deg = (best_i * deg_per_index) - 90.0  
        steering_angle = np.deg2rad(angle_deg)

        center_start = 530
        center_end   = 549
        center_ranges = np.array(data.ranges[center_start:center_end], dtype=np.float32)
        min_distance_ahead = np.min(center_ranges) if len(center_ranges) > 0 else 0.0

        if min_distance_ahead < 1.0:
            velocity = 1.0
        elif min_distance_ahead < 2.0:
            velocity = 2.5
        elif min_distance_ahead < 3.0:
            velocity = 4.0
        else:
            velocity = 5.0

        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)

        self.get_logger().info(f"Steering={steering_angle:.2f} rad, Speed={velocity:.1f}")

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveFollowGap()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
