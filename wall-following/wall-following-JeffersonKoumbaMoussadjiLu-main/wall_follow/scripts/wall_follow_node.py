
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # TODO: create subscribers and publishers

        # Subscriber (LaserScan)
        self.subscription_scan = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10
        )

        # Publisher (Ackermann Drive)
        self.drive_publication = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        # TODO: set PID gains (Through PID simulator online free)
        '''
        #Previous Value
        self.kp = 0.27
        self.kd = 0.1
        self.ki = 0.0002

        '''
        self.kp = 0.3 
        self.kd = 0.1 
        self.ki = 0.0001

        # TODO: store history
        self.integral = 0.0
        self.prev_error = 0.0
        self.error = 0.0


        # TODO: store any necessary values you think you'll need

        # Store LaserScan info 
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.angle_increment = 0.0
        self.range_min = 0.0
        self.range_max = 0.0
        

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        #TODO: implement

        '''
        When you receive a LaserScan message in ROS, the LiDAR’s readings are laid out in a fixed angular order from angle_min to angle_max in equal increments (angle_increment).
        Each index i in the array corresponds to the angle

        ranges is an array which gives you the distance measured for each angle bin.
        '''

        # Check if constants are uninitialized
        if self.angle_increment == 0.0 and self.angle_min == 0.0:
            return -1

        # Calculate index corresponding to given angle
        # angle = angle_min + index * angle_increment
        # index = (angle - angle_min)/(angle_increment)
        idx = int((angle - self.angle_min) / self.angle_increment)

        # Ensure index is within bounds
        if idx < 0 or idx >= len(range_data):
            return -1

        # Get range value at the calculated index
        range_val = range_data[idx]

        # Check for NaNs or infs
        if np.isnan(range_val) or np.isinf(range_val):
            return -1

        # Check if range is out of bounds
        if range_val < self.range_min or range_val > self.range_max:
            return -1

        return range_val

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """

        #TODO:implement

        theta_min = 30
        theta_max = 60
        theta_step = 5

        alphas = []
        
        a = 0

        b = self.get_range(range_data, np.pi/2) # beam b (90 degrees to the right of the car)

        L = 2.5 # Lookahead distance of the car (hard coded)

        # Loop Through all the angles from theta_min to theta_max to Compute alphas 
        for i in range(theta_min, theta_max + 1, theta_step):

            theta_rad = np.radians(i) #Convert theta into radians

            # If theta is greater than 70, skip this angle (0<Theta)
            if theta_rad > 70:
                continue

            # a is at (pi/2 - theta) => a = pi - (theta+pi/2) => a = pi/2 - theta
            a = self.get_range(range_data, np.pi / 2 - theta_rad)

            # If a/sin(theta) is  or too close to zero, skip this angle
            if abs(a * np.sin(theta_rad)) < 1e-6:
                continue

            # alpha = atan2( b - a cos(theta), a sin(theta) ) for left 
            alpha = np.arctan2(b - a * np.cos(theta_rad), a * np.sin(theta_rad))

            alphas.append(alpha) #Add alpha to the list

        # If alphas is empty, return 0
        if len(alphas) == 0:
            return 0.0
        
        # Calculate the average of alphas
        alpha = np.mean(alphas)

        # Current distance between car and wall Dt = b * cos(alpha)
        curr_distance = b * np.cos(alpha) 

        # Next Distance D_t+1 = Dt - L * sin(alpha) (opposite of Dt+1 = Dt + L * sin(alpha))
        d_next = curr_distance - L * np.sin(alpha)

        #Error = Next_Distance - Desired_Distance
        error = d_next - dist 

        return error

    def pid_control(self, error, velocity):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error
            velocity: desired velocity

        Returns:
            None
        """
        angle = 0.0

        # TODO: Use kp, ki & kd to implement a PID controller
        drive_msg = AckermannDriveStamped()

        # TODO: fill in drive message and publish

        # Compute derivative and update previous error
        d_error = error - self.prev_error
        self.prev_error = error

        # Accumulate integral Only if it is within "reasonable" range
        if abs(error) <= 0.3:
            self.integral += error

        # PID formula => Compute the steering angle u(t) we want the car to drive at
        # kp * e + ki * integral + kd * de/dt
        angle = (self.kp * error) + (self.ki * self.integral) + (self.kd * d_error)

        # Speed based on steering angle
        #angle_deg = abs(np.degrees(angle))  # convert to degrees (absolute)

        #if angle_deg < 10.0:
        #    velocity = 3

        #elif angle_deg < 20.0:
        #    velocity = 2

        #else:
        #    velocity = 1

        # Build and publish the Ackermann Drive message
        drive_msg.drive.steering_angle = float(angle)
        drive_msg.drive.speed = float(velocity)
        self.drive_publication.publish(drive_msg)




    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """

        # Update LIDAR parameters
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.range_min = msg.range_min
        self.range_max = msg.range_max

        # Convert ranges to a float array
        range_data = np.array(msg.ranges, dtype=float)

        #Desired distance from wall
        dist = 1.35 #1.2

        # Calculate Error
        error = self.get_error(range_data, dist) # TODO: replace with error calculated by get_error()

        #velocity = 0.0 # TODO: calculate desired car velocity based on error
        abs_err = abs(error)
        if abs_err <= 0.15:
            velocity = 1.5

        elif abs_err <= 0.3:
            velocity = 1
            
        else:
            velocity = 0.5

        self.pid_control(error, velocity) # TODO: actuate the car with PID
        


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
