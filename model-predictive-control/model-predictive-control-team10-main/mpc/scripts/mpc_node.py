#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from utils import nearest_point
from tf_transformations import euler_from_quaternion

import csv

# TODO CHECK: include needed ROS msg type headers and libraries

@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering speed, acceleration]
    TK: int = 8  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        # default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, delta, v, yaw, yaw-rate, beta]
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 5.5, 13.0])
    )  # final state error matrix, penalty for the final state constraints: [x, y, delta, v, yaw, yaw-rate, beta]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.25  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 2.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0

class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)
        self.sim = False

        if self.sim:
            odom_topic = "/ego_racecar/odom"
            self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        else:
            odom_topic = "/pf/viz/inferred_pose"
            self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)
            #self.create_subscription(Odometry, "/odom", self.pose_callback, 10)




        #self.pose_sub_ = self.create_subscription(Odometry, odom_topic, self.pose_callback, 10)
        self.pose_sub_ = self.create_subscription(PoseStamped, odom_topic, self.pose_callback, 10)
        

        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.drive = AckermannDriveStamped()
        self.current_speed = 0.0  # Add this line to track the current speed as a float

        self.waypoint_pub_ = self.create_publisher(MarkerArray, '/waypoints', 10)
        self.ref_path_pub_ = self.create_publisher(Path, "/ref_path", 10)
        self.mpc_path_pub_ = self.create_publisher(Path, "/mpc_path", 10)
        self.marker_pub_ = self.create_publisher(Marker, "/mpc_marker", 10)

        #self.twist_sub = self.create_subscription(
        #    Odometry, '/odom', self.twist_callback, 10
        #)

        # TODO: get waypoints here
        csv_data = np.loadtxt("/home/nvidia/f1tenth_ws/src/model-predictive-control-team10/mpc/scripts/mpc_levine_1000.csv", delimiter=",", skiprows=0)
        self.waypoints = csv_data[::4,:]
        self.visualize_waypoints() #visualize waypoints

        self.config = mpc_config()
        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        self.mpc_prob_init()

    def pose_callback(self, pose_msg):
        pass
        # TODO: extract pose from ROS msg
        vehicle_state = State()
        vehicle_state.x = pose_msg.pose.pose.position.x if self.sim else pose_msg.pose.position.x
        vehicle_state.y = pose_msg.pose.pose.position.y if self.sim else pose_msg.pose.position.y

        #vehicle_state.v = self.drive  #pose_msg.twist.twist.linear.x

        # Use the current_speed variable instead of the AckermannDriveStamped object
        vehicle_state.v = self.current_speed #added

        #if self.sim == False:
        #    vehicle_state.v = -1 * vehicle_state.v  # negate the monitoring speed

        print(vehicle_state.x)
        print(vehicle_state.y)
        print(vehicle_state.v)

        ori = pose_msg.pose.pose.orientation if self.sim else pose_msg.pose.orientation
        quaternion = np.array([ori.x, 
                               ori.y, 
                               ori.z, 
                               ori.w])

        euler = euler_from_quaternion(quaternion)
        vehicle_state.yaw = np.arctan2(2.0 * (ori.w * ori.z + ori.x * ori.y), 1 - 2.0 * (ori.y**2 + ori.z**2))

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        
        ref_x = self.waypoints[:, 0]
        ref_y = self.waypoints[:, 1]
        ref_yaw = self.waypoints[:, 2]
        ref_v = self.waypoints[:, 3]* 4

        ref_path = self.calc_ref_trajectory(vehicle_state, cx=ref_x, cy=ref_y, cyaw=ref_yaw, sp=ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        #########################################################################
        # Publish the reference path and the mpc path

        if ox is not None:
            ref_path_msg = Path()
            ref_path_msg.header.frame_id = "/map"
            ref_path_msg.header.stamp = self.get_clock().now().to_msg()

            for i in range(len(ref_path[0])):
                ref_pose_ = PoseStamped()
                ref_pose_.pose.position.x = ref_path[0, i]
                ref_pose_.pose.position.y = ref_path[1, i]
                ref_pose_.pose.orientation.w = 1.0
                ref_path_msg.poses.append(ref_pose_)

            self.ref_path_pub_.publish(ref_path_msg)

            mpc_path_msg = Path()
            mpc_path_msg.header.frame_id = "/map"
            mpc_path_msg.header.stamp = self.get_clock().now().to_msg()

            for i in range(len(ox)):
                mpc_pose_ = PoseStamped()
                mpc_pose_.pose.position.x = ox[i]
                mpc_pose_.pose.position.y = oy[i]
                mpc_pose_.pose.orientation.w = 1.0
                mpc_path_msg.poses.append(mpc_pose_)

            self.mpc_path_pub_.publish(mpc_path_msg)

        if self.odelta_v is None:
            return

        ##########################################################################

        # TODO: publish drive message.
        steer_output = self.odelta_v[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

        # Update the current_speed with the new calculated speed at the end of the function
        self.current_speed = speed_output #added

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer_output
        msg.drive.speed = speed_output

        print("steering ={}, speed ={}".format(steer_output, speed_output))

        self.drive_pub_.publish(msg)

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector (x y v yaw)
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input Vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # # --------------------------------------------------------
        # # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere
       
        # # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        
        # This penalizes large control inputs (acceleration and steering) using the R matrix
        # Higher values in R will lead to smaller control inputs (more conservative control)
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)
        
        # # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        
        # This penalizes deviations from the reference trajectory, weighted by Q matrix
        # The final state deviation is weighted by Qf (included in Q_block)
        # Higher values in Q will force the vehicle to follow the reference more closely
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)
        
        # # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd
        
        # This penalizes rapid changes in control inputs between consecutive timesteps
        # It encourages smooth control actions (smooth acceleration and steering)
        # Higher values in Rd will result in more gradual control changes
        objective += cvxpy.quad_form(cvxpy.vec(self.uk[:, 1:] - self.uk[:, :-1]), Rd_block)

        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            # A: 4x4 B:4x2 
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")
        

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_

        next_states = self.xk[:, 1:]  # States from t=1 to t=T
        current_states = self.xk[:, :-1]  # States from t=0 to t=T-1
        control_inputs = self.uk  # Control inputs from t=0 to t=T-1

        # Vectorize the matrices for the constraint equation
        vectorized_next_states = cvxpy.vec(next_states)
        vectorized_current_states = cvxpy.vec(current_states)
        vectorized_control_inputs = cvxpy.vec(control_inputs)

        # Apply the state transition equation: x_{t+1} = A*x_t + B*u_t + C
        dynamics_constraint = vectorized_next_states == self.Ak_ @ vectorized_current_states + self.Bk_ @ vectorized_control_inputs + self.Ck_

        # Add the constraint to the constraint list
        constraints += [dynamics_constraint]

        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        
        # Extract steering angle inputs for easier understanding
        # uk[1, :] represents steering angles for all timesteps
        steering_angles = self.uk[1, :]  

        # Calculate the change in steering angle between consecutive timesteps
        steering_changes = steering_angles[1:] - steering_angles[:-1]  # Differences between t+1 and t
        steering_change_rates = steering_changes * self.config.DTK  # Convert to rate by dividing by timestep

        # Vectorize for easier constraint definition
        vectorized_steering_change_rates = cvxpy.vec(steering_change_rates)

        # Set the upper bound on steering rate change
        max_steering_rate_constraint = vectorized_steering_change_rates <= self.config.MAX_DSTEER

        # Add a lower bound on steering rate change (this was missing in the original)
        min_steering_rate_constraint = vectorized_steering_change_rates >= -self.config.MAX_DSTEER

        # Add the constraints to the constraint list
        constraints += [max_steering_rate_constraint, min_steering_rate_constraint]

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        
        # Initial state constraint - first state must match current vehicle state
        initial_state_constraint = self.xk[:, 0] == self.x0k
        constraints += [initial_state_constraint]

        # Speed constraints - extract the speed component (index 2) from all states
        speeds = self.xk[2, :]  # Speed at all timesteps

        # Define upper and lower bounds on speed
        max_speed_constraint = speeds <= self.config.MAX_SPEED
        min_speed_constraint = speeds >= self.config.MIN_SPEED

        # Add the speed constraints to the constraint list
        constraints += [max_speed_constraint, min_speed_constraint]

        #Input constraints

        # Extract acceleration and steering inputs for all timesteps
        accelerations = self.uk[0, :]
        steering_angles = self.uk[1, :]

        # Define upper and lower bounds on acceleration
        max_accel_constraint = accelerations <= self.config.MAX_ACCEL
        min_accel_constraint = accelerations >= -self.config.MAX_ACCEL  # Allow for braking

        # Define upper and lower bounds on steering angle
        max_steering_constraint = steering_angles <= self.config.MAX_STEER
        min_steering_constraint = steering_angles >= self.config.MIN_STEER

        # Add the input constraints to the constraint list
        constraints += [max_accel_constraint, min_accel_constraint, 
                        max_steering_constraint, min_steering_constraint]

        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        self.publish_marker(ref_traj[0, 0], ref_traj[1, 0])

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        # cyaw[cyaw - state.yaw > 4.5] = np.abs(
        #     cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        # )
        # cyaw[cyaw - state.yaw < -4.5] = np.abs(
        #     cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        # )

        # This is the current implementation that properly handles angle wrapping
        # Apply the normalized headings to the reference trajectory
        cyaw[cyaw - state.yaw > 4.5] = cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        cyaw[cyaw - state.yaw < -4.5] = cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict
    
########################################################################################################

    #Helper Function

    #Visualize waypoints

    def visualize_waypoints(self):
        self.get_logger().info("Visualizing Waypoints")
        markers = MarkerArray()

        t = self.get_clock().now()

        self.idx = 0
        for pt in self.waypoints:
            marker = Marker()

            marker.header.frame_id = "/map"
            marker.header.stamp = rclpy.clock.Clock().now().to_msg()
            marker.id = self.idx
            self.idx += 1

            marker.type = Marker.SPHERE
            marker.action = 0
            marker.pose.position.x = pt[0]
            marker.pose.position.y = pt[1]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            markers.markers.append(marker)
        self.markers = markers
        self.waypoint_pub_.publish(markers)
    
    def publish_marker(self, x, y):
        # self.get_logger().info("Publishing Marker")
        marker = Marker()

        marker.header.frame_id = "/map"
        marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        marker.id = 10000
        marker.type = Marker.SPHERE
        marker.action = 0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.marker_pub_.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
