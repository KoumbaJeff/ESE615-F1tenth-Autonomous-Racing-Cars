#!/usr/bin/env python3

#pip install --upgrade jax jaxlib
import os
import math
import yaml
import cv2
import rclpy
import jax
import jax.numpy as jnp
from jax import jit, lax
from dataclasses import dataclass, field
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.node import Node

from utils import nearest_point  # keep your existing numba‐accelerated util

#―┒ config & state ┑───────────────────────────────────────────────────────────
@dataclass
class MPCConfig:
    NX: int = 4     # [x, y, v, yaw]
    NU: int = 2     # [accel, steering]
    TK: int = 8     # horizon length

    Q: jnp.ndarray = field(
        default_factory=lambda: jnp.diag(jnp.array([13.5,13.5,5.5,13.0]))
    )
    Qf: jnp.ndarray = field(
        default_factory=lambda: jnp.diag(jnp.array([13.5,13.5,5.5,13.0]))
    )
    R: jnp.ndarray = field(
        default_factory=lambda: jnp.diag(jnp.array([0.01,100.0]))
    )
    DT: float = 0.1
    WB: float = 0.33
    MAX_ACCEL: float = 3.0
    MAX_STEER: float = 0.4189
    MAX_SPEED: float = 2.0
    MIN_SPEED: float = 0.0

@dataclass
class State:
    x: float; y: float; v: float; yaw: float

#―┒ helper – dynamics linearization in JAX ┑───────────────────────────────────
@jit
def linearize_dynamics(v, yaw, steering, cfg: MPCConfig):
    """
    Linearize x_{t+1} ≈ A x_t + B u_t + c  around (v, yaw, steering)
    """
    cos_y = jnp.cos(yaw)
    sin_y = jnp.sin(yaw)
    DT = cfg.DT; WB = cfg.WB

    A = jnp.array([
        [1.0, 0.0, DT*cos_y, -DT*v*sin_y],
        [0.0, 1.0, DT*sin_y,  DT*v*cos_y],
        [0.0, 0.0, 1.0,       0.0       ],
        [0.0, 0.0, DT*jnp.tan(steering)/WB, 1.0]
    ])
    B = jnp.array([
        [0.0,                 0.0],
        [0.0,                 0.0],
        [DT,                  0.0],
        [0.0,   DT*v/(WB*(jnp.cos(steering)**2))]
    ])
    c = jnp.zeros((4,))  # we ignore affine term for LQR
    return A, B, c

#―┒ LQR Riccati recursion ┑────────────────────────────────────────────────────
def make_lqr_fn(cfg: MPCConfig):
    Q, R, Qf, TK = cfg.Q, cfg.R, cfg.Qf, cfg.TK

    @jit
    def riccati(A_seq, B_seq):
        # backward recursion to compute P_t
        def step(P_next, AB):
            A, B = AB
            # S = R + B^T P_next B
            S = R + B.T @ P_next @ B
            Kt = jnp.linalg.solve(S, B.T @ P_next @ A)  # K = S^{-1} B^T P A
            P = Q + A.T @ P_next @ (A - B @ Kt)
            return P, Kt

        P0 = Qf
        # scan backwards
        (_, K_rev), _ = lax.scan(
            step,
            P0,
            (A_seq[::-1], B_seq[::-1]),
            length=TK
        )
        # reverse back to forward order
        K_seq = K_rev[::-1]
        return K_seq  # shape (TK, NU, NX)
    return riccati

#―┒ main node ┑────────────────────────────────────────────────────────────────
class MPCJax(Node):
    def __init__(self):
        super().__init__('mpc_jax_node')
        # load map & race‐line here if you want to follow a path
        # ... (omitted to focus on MPC speedup)

        # ROS interfaces
        self.create_subscription(PoseStamped, '/pf/viz/inferred_pose', self.pose_cb, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.path_pub  = self.create_publisher(Path, '/mpc_path', 1)

        # load waypoints once
        self.waypoints = jnp.array(
            # Nx4 array: [x, y, yaw, speed]
            jnp.loadtxt("mpc_levine_1000.csv", delimiter=",", skiprows=0)[::4]
        )

        # config & pre‐compiled LQR
        self.cfg = MPCConfig()
        self._riccati = make_lqr_fn(self.cfg)

    def pose_cb(self, msg: PoseStamped):
        # extract vehicle state
        px = msg.pose.position.x
        py = msg.pose.position.y
        ori = msg.pose.orientation
        yaw = math.atan2(2*(ori.w*ori.z+ori.x*ori.y),
                         1-2*(ori.y*ori.y+ori.z*ori.z))
        v   = 0.0  # you'd get from IMU/odom
        x0  = jnp.array([px, py, v, yaw])

        # build reference trajectory over TK
        # find nearest waypoint
        wp_xy = jnp.array(self.waypoints[:,:2])
        _, _, _, idx = nearest_point(jnp.array([px,py]), wp_xy)
        idx = int(idx)
        refs = jnp.roll(self.waypoints, -idx, axis=0)[:self.cfg.TK+1]
        x_ref = refs[:,:4].T               # (NX, TK+1)
        u_ref = jnp.zeros((self.cfg.NU, self.cfg.TK))  # assume zero‐steer/ref accel

        # linearizations along horizon
        def make_seq(i):
            v_r = x_ref[2,i]
            yaw_r = x_ref[3,i]
            return linearize_dynamics(v_r, yaw_r, 0.0, self.cfg)[:2]  # A,B
        A_seq, B_seq = jax.vmap(make_seq)(jnp.arange(self.cfg.TK))
        A_seq = A_seq.reshape(self.cfg.TK, self.cfg.NX, self.cfg.NX)
        B_seq = B_seq.reshape(self.cfg.TK, self.cfg.NX, self.cfg.NU)

        # compute time‐varying feedback gains
        K_seq = self._riccati(A_seq, B_seq)  # (TK, NU, NX)

        # compute first control: u0 = u_ref0  –  K0 (x0 – x_ref0)
        dx0 = x0 - x_ref[:,0]
        u0  = u_ref[:,0] - K_seq[0] @ dx0
        # saturate
        accel = float(jnp.clip(u0[0], -self.cfg.MAX_ACCEL, self.cfg.MAX_ACCEL))
        steer = float(jnp.clip(u0[1], -self.cfg.MAX_STEER, self.cfg.MAX_STEER))

        # publish
        cmd = AckermannDriveStamped()
        cmd.drive.speed = float(x_ref[2,0])
        cmd.drive.steering_angle = steer
        self.drive_pub.publish(cmd)

        # (optionally) publish the MPC predicted path
        path = Path()
        path.header = msg.header
        for t in range(self.cfg.TK+1):
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x_ref[0,t])
            p.pose.position.y = float(x_ref[1,t])
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self.path_pub.publish(path)

def main():
    rclpy.init()
    node = MPCJax()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
