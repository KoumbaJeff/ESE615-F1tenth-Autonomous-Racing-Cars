// mpc_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <nav_msgs/msg/path.hpp>

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <memory>

// OSQP headers
#include "osqp/osqp.h"

// ——— Configuration ———
struct MPCConfig {
  static constexpr int NXK = 4;   // [x,y,v,yaw]
  static constexpr int NU  = 2;   // [accel, steer_rate]
  static constexpr int TK  = 8;   // horizon length
  double DTK = 0.1;               // timestep
  double dlk = 0.25;              // dist step
  double WB  = 0.33;              // wheelbase
  double MAX_STEER = 0.4189;
  double MAX_DSTEER = M_PI;       // rad/s
  double MAX_SPEED  = 2.0;
  double MAX_ACCEL  = 3.0;

  // cost mats (flattened)
  Eigen::Matrix<double,NU,NU> Rk  = (Eigen::Matrix<double,NU,NU>() << 0.01, 0, 0, 100).finished();
  Eigen::Matrix<double,NU,NU> Rd  = (Eigen::Matrix<double,NU,NU>() << 0.01, 0, 0, 100).finished();
  Eigen::Matrix<double,NXK,NXK> Qk  = (Eigen::Matrix<double,NXK,NXK>() 
    << 13.5,0,0,0, 0,13.5,0,0, 0,0,5.5,0, 0,0,0,13.0).finished();
  Eigen::Matrix<double,NXK,NXK> Qf  = (Eigen::Matrix<double,NXK,NXK>() 
    << 13.5,0,0,0, 0,13.5,0,0, 0,0,5.5,0, 0,0,0,13.0).finished();
};

// simple 2D pose + heading
struct State {
  double x,y,v,yaw;
};

// nearest‐point on polyline (from utils.py)
static inline void nearest_point(
  const Eigen::Vector2d &pt, 
  const std::vector<Eigen::Vector2d> &traj,
  Eigen::Vector2d &out_proj, 
  int &out_idx
){
  double best_d = 1e9;
  Eigen::Vector2d best_p; int best_i=0;
  for(int i=0;i+1<(int)traj.size();++i){
    const auto &A = traj[i];
    const auto &B = traj[i+1];
    Eigen::Vector2d d = B-A;
    double l2 = d.squaredNorm();
    double t  = std::clamp((pt-A).dot(d)/l2,0.0,1.0);
    Eigen::Vector2d proj = A + t*d;
    double dist = (pt-proj).squaredNorm();
    if(dist<best_d){ best_d=dist; best_p=proj; best_i=i; }
  }
  out_proj=best_p; out_idx=best_i;
}

// read comma‐separated waypoints [x,y,yaw,speed]
static std::vector<Eigen::Vector4d> load_waypoints(const std::string &csv){
  std::ifstream in(csv);
  std::string line;
  std::vector<Eigen::Vector4d> wps;
  while(std::getline(in,line)){
    std::stringstream ss(line);
    Eigen::Vector4d v;
    char c;
    ss>>v[0]>>c>>v[1]>>c>>v[2]>>c>>v[3];
    wps.push_back(v);
  }
  return wps;
}

class MPCNode : public rclcpp::Node {
public:
  MPCNode(): Node("mpc_node"){
    // load waypoints
    waypoints_ = load_waypoints(
      "/home/nvidia/.../mpc_levine_1000.csv"
    );
    // setup ROS pubs/subs
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/pf/viz/inferred_pose", 10,
      std::bind(&MPCNode::pose_cb,this,std::placeholders::_1)
    );
    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "/drive", 10
    );
    // pre‐allocate QP structures
    setupQP();
  }

private:
  // ROS
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

  // data
  std::vector<Eigen::Vector4d> waypoints_;
  State current_state_{0,0,0,0};

  // QP
  OSQPData   *data_{nullptr};
  OSQPSettings *settings_{nullptr};
  OSQPWorkspace *work_{nullptr};
  Eigen::SparseMatrix<double> P_; // Hessian
  Eigen::VectorXd q_;             // gradient
  Eigen::SparseMatrix<double> Aeq_;
  Eigen::VectorXd lb_, ub_;

  // config
  MPCConfig cfg_;

  // build constant blocks for P = blkdiag(R,R,..., Rd) etc.
  void setupQP(){
    // allocate OSQPData with dims
    int n = cfg_.NU*cfg_.TK + cfg_.NXK*(cfg_.TK+1);
    // placeholder: you need to fill P_, q_, Aeq_, lb_/ub_ here
    // then:
    data_ = (OSQPData*)c_malloc(sizeof(OSQPData));
    data_->n = n;
    data_->m = Aeq_.rows();
    // TODO: convert P_, Aeq_ to csc and assign data_->P, data_->A
    osqp_set_default_settings(&*settings_);
    settings_ = (OSQPSettings*)c_malloc(sizeof(OSQPSettings));
    osqp_settings_default(settings_);
    settings_->alpha = 1.0;
    settings_->verbose = false;
    osqp_setup(&work_, data_, settings_);
  }

  // callback
  void pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg){
    // 1) unpack pose
    auto &p = msg->pose.position;
    auto &o = msg->pose.orientation;
    current_state_.x = p.x;
    current_state_.y = p.y;
    // keep last speed: (could subscribe odom for this)
    // current_state_.v unchanged
    // yaw
    double siny = 2*(o.w*o.z + o.x*o.y);
    double cosy = 1 - 2*(o.y*o.y + o.z*o.z);
    current_state_.yaw = std::atan2(siny,cosy);

    // 2) build reference trajectory
    Eigen::Matrix<double,MPCConfig::NXK,MPCConfig::TK+1> ref;
    build_ref_traj(current_state_, ref);

    // 3) solve MPC
    std::vector<double> u_opt = solve_mpc(ref);

    // 4) publish
    ackermann_msgs::msg::AckermannDriveStamped out;
    out.drive.steering_angle    = u_opt[1];
    out.drive.speed             = current_state_.v + u_opt[0]*cfg_.DTK;
    drive_pub_->publish(out);
  }

  // builds ref[x,y,v,yaw]
  void build_ref_traj(
    const State &st,
    Eigen::Matrix<double,MPCConfig::NXK,MPCConfig::TK+1> &ref
  ){
    // find nearest waypoint
    Eigen::Vector2d pt{st.x,st.y}, proj;
    int idx;
    std::vector<Eigen::Vector2d> path2d;
    for(auto &w: waypoints_) path2d.emplace_back(w[0],w[1]);
    nearest_point(pt,path2d,proj,idx);
    // fill ref for each k=0..TK
    int N = waypoints_.size();
    for(int k=0;k<=MPCConfig::TK;++k){
      int ii = (idx + int(std::round(k*st.v*cfg_.DTK/cfg_.dlk))) % N;
      auto &w = waypoints_[ii];
      ref(0,k)=w[0];
      ref(1,k)=w[1];
      ref(2,k)=w[3]*4.0;  // same scaling as Python
      // wrap yaw
      double yaw = w[2];
      double dy = yaw - st.yaw;
      if(dy> M_PI)  yaw-=2*M_PI;
      if(dy<-M_PI) yaw+=2*M_PI;
      ref(3,k)=yaw;
    }
  }

  // solves the QP around “ref” and returns [accel_0, steerRate_0]
  std::vector<double> solve_mpc(
    const Eigen::Matrix<double,MPCConfig::NXK,MPCConfig::TK+1> &ref
  ){
    // 1) update linear dynamics blocks (A,B,C) into data_->A
    // 2) update q_ = ∇[(x–ref)ᵀQ(x–ref)] + …   
    // 3) osqp_update_P_A_q etc.
    osqp_solve(work_);
    // read control
    std::vector<double> u(2,0.0);
    if(work_->info->status_val==OSQP_SOLVED){
      // first two of work_->solution->x
      u[0] = work_->solution->x[0];
      u[1] = work_->solution->x[1];
    }
    return u;
  }
};

int main(int argc, char **argv){
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<MPCNode>());
  rclcpp::shutdown();
  return 0;
}
