/**
 * @file rigid_spindle.cpp
 */

#include "meshbrane/rigid_spindle.hpp"
#include "meshbrane/math_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace meshbrane {
//////////////////////////////////////////////////////////
// RigidMesh /////////////////////////////////////////////
//////////////////////////////////////////////////////////
void RigidMesh::update_coords_from_frame() {
  //   printf("RigidMesh::update_coords_from_frame\n");
  int num_vertices = xyz_relative_V_.rows();
  xyz_coord_V_.resize(num_vertices, 3);
  for (int v = 0; v < num_vertices; v++) {
    Vec3d xyz = xyz_relative_V_.row(v);
    Vec3d xyz_rotated = rotation_matrix_frame_ * xyz;
    xyz_coord_V_.row(v) = xyz_frame_ + xyz_rotated;
  }
};

void RigidMesh::sum_force_V() {
  //   printf("RigidMesh::sum_force_V\n");
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    Vec3d force = force_V_.row(v);
    force_ += force;
    // Vec3d r = xyz_coord_v(v) - xyz_frame_;
    // Vec3d torque = math::cross(r, force);
    // torque_ += torque;
  }
};

//////////////////////////////////////////////////////////
// SphericalSPB //////////////////////////////////////////
//////////////////////////////////////////////////////////
void SphericalSPB::set_attributes_from_parameters() {
  printf("SphericalSPB::set_attributes_from_parameters\n");
  RigidMesh::set_attributes_from_parameters();
  // set object specific parameters
  if (parameters_["contact_radius"]) {
    contact_radius_ = parameters_["contact_radius"].as<double>();
  }
  if (parameters_["wca_epsilon"]) {
    wca_epsilon_total_ = parameters_["wca_epsilon"].as<double>();
  }
  if (parameters_["wca_sigma"]) {
    wca_sigma_ = parameters_["wca_sigma"].as<double>();
  }
  if (parameters_["num_refinements"]) {
    num_refinements_ = parameters_["num_refinements"].as<int>();
  }
  if (parameters_["enable_fluctuations"]) {
    enable_fluctuations_ = parameters_["enable_fluctuations"].as<bool>();
  }
  // set global parameters
  if ((*sim_parameters_)["kBT"]) {
    kBT_ = (*sim_parameters_)["kBT"].as<double>();
  }
  double wca_r_cutoff = std::pow(2.0, 1.0 / 6.0) * wca_sigma_;
  interaction_radius_ = contact_radius_ + wca_r_cutoff;
  int num_vertices = 12 * 4 * num_refinements_;
  wca_epsilon_ = wca_epsilon_total_ / num_vertices;
  // set global parameters
  if ((*sim_parameters_)["bulk_viscosity"]) {
    double mu = (*sim_parameters_)["bulk_viscosity"].as<double>();
    linear_drag_coefficient_ = 6.0 * M_PI * mu * contact_radius_;
    angular_drag_coefficient_ =
        8.0 * M_PI * mu * contact_radius_ * contact_radius_ * contact_radius_;
  }
  //
}

void SphericalSPB::init_state(Vec3d &center, Eigen::Matrix3d &rotation_matrix) {
  MatrixMesh::init_icosohedron();
  // double r = contact_radius_;
  // int N = get_num_vertices();
  // double rho = N / (4 * M_PI * r * r);
  // double rho_min = 400.0;
  // int ref = 0;
  // while (rho <= rho_min) {
  //   refine_icososphere();
  //   N = get_num_vertices();
  //   rho = N / (4 * M_PI * r * r);
  //   ++ref;
  // }
  // printf("SphericalSPB::init_state - r = %f, N = %d, rho = %f\n", r, N, rho);

  for (int refinement = 0; refinement < num_refinements_; refinement++) {
    refine_icososphere();
  }
  double r = contact_radius_;
  int N = get_num_vertices();
  double rho = N / (4 * M_PI * r * r);
  printf("SphericalSPB::init_state - r = %f, N = %d, rho = %f\n", r, N, rho);
  // throw std::runtime_error("SphericalSPB::init_state");

  xyz_relative_V_ = contact_radius_ * xyz_coord_V_;
  xyz_frame_ = center;
  rotation_matrix_frame_ = rotation_matrix;
  update_coords_from_frame();
  init_from_he_mats();
}

void SphericalSPB::sum_envelope_force_V() {
  int num_vertices = get_num_vertices();
  for (int v = 0; v < num_vertices; v++) {
    Vec3d force = force_V_.row(v);
    force_envelope_ += force;
    Vec3d r = xyz_coord_v(v) - xyz_frame_;
    Vec3d torque = math::cross(r, force);
    torque_envelope_ += torque;
  }
}

Vec3d SphericalSPB::get_linear_fluctuations(double dt) {
  Vec3d dx = Vec3d::Zero();
  for (int i = 0; i < 3; i++) {
    dx[i] = std::sqrt(2 * kBT_ * dt / linear_drag_coefficient_) *
            randng_.standard_normal();
  }
  return dx;
}

Vec3d SphericalSPB::get_rotational_fluctuations(double dt) {
  Vec3d dtheta = Vec3d::Zero();
  for (int i = 0; i < 3; i++) {
    dtheta[i] = std::sqrt(2 * kBT_ * dt / angular_drag_coefficient_) *
                randng_.standard_normal();
  }
  return dtheta;
}

void SphericalSPB::compute_velocities() {
  //   printf("SphericalSPB::compute_velocities\n");
  Vec3d F = force_envelope_ + force_mt_bundle_;
  Vec3d T = torque_envelope_ + couple_mt_bundle_;
  xyz_dot_ = F / linear_drag_coefficient_;
  angular_velocity_ = T / angular_drag_coefficient_;
}

void SphericalSPB::update_state_variables(double dt) {

  Vec3d delta_x = dt * xyz_dot_;
  Vec3d delta_theta = angular_velocity_ * dt;
  if (enable_fluctuations_) {
    delta_x += get_linear_fluctuations(dt);
    delta_theta += get_rotational_fluctuations(dt);
  }

  xyz_frame_ += delta_x;
  Eigen::Matrix3d R = lie::exp_so3(delta_theta);
  rotation_matrix_frame_ = R * rotation_matrix_frame_;
  // Vec3d angular_velocity = torque_ / angular_drag_coefficient_;
  // Eigen::AngleAxisd rotation(angular_velocity.norm() * dt,
  //                            angular_velocity.normalized());
  // Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
  // rotation_matrix_frame_ = rotation_matrix * rotation_matrix_frame_;
  update_coords_from_frame();
}

///////////////////////////////////////////////////////////
//// RigidMTBundle ////////////////////////////////////////
///////////////////////////////////////////////////////////
void RigidMTBundle::set_attributes_from_parameters() {
  // set object specific parameters
  if (parameters_["radius"]) {
    radius_ = parameters_["radius"].as<double>();
  }
  if (parameters_["v_grow"]) {
    v_grow_ = parameters_["v_grow"].as<double>();
  }
  if (parameters_["max_length"]) {
    max_length_ = parameters_["max_length"].as<double>();
  }
  if (parameters_["max_force"]) {
    max_force_ = parameters_["max_force"].as<double>();
  }
  if (parameters_["motor_force_per_length"]) {
    motor_force_per_length_ =
        parameters_["motor_force_per_length"].as<double>();
  }
  if (parameters_["wca_epsilon"]) {
    wca_epsilon_ = parameters_["wca_epsilon"].as<double>();
  }
  if (parameters_["wca_sigma"]) {
    wca_sigma_ = parameters_["wca_sigma"].as<double>();
  }
  if (parameters_["overlap_length"]) {
    overlap_length_ = parameters_["overlap_length"].as<double>();
  }
  if (parameters_["length"]) {
    length_ = parameters_["length"].as<double>();
  }
  if (parameters_["xyz_center"]) {
    std::vector<double> xyz_center =
        parameters_["xyz_center"].as<std::vector<double>>();
    xyz_center_ = Vec3d(xyz_center.data());
  }
  if (parameters_["axis"]) {
    std::vector<double> axis = parameters_["axis"].as<std::vector<double>>();
    axis_ = Vec3d(axis.data());
    // axis_ = axis_ / math::L2norm(axis_);
    axis_.normalize();
  }

  if (parameters_["enable_fluctuations"]) {
    enable_fluctuations_ = parameters_["enable_fluctuations"].as<bool>();
  }

  double wca_r_cutoff = std::pow(2.0, 1.0 / 6.0) * wca_sigma_;
  interaction_radius_ = radius_ + wca_r_cutoff;

  // visualization
  if (parameters_["num_mts"]) {
    num_mts_ = parameters_["num_mts"].as<int>();
  }
  if (parameters_["rgb_mt1"]) {
    std::vector<double> rgb = parameters_["rgb_mt1"].as<std::vector<double>>();
    rgb_mt1_ = Vec3d(rgb.data());
  }
  if (parameters_["rgb_mt2"]) {
    std::vector<double> rgb = parameters_["rgb_mt2"].as<std::vector<double>>();
    rgb_mt2_ = Vec3d(rgb.data());
  }

  // set global parameters
  if ((*sim_parameters_)["kBT"]) {
    kBT_ = (*sim_parameters_)["kBT"].as<double>();
  }
  if ((*sim_parameters_)["dt"]) {
    dt0_ = (*sim_parameters_)["dt"].as<double>();
  }
  if ((*sim_parameters_)["bulk_viscosity"]) {
    bulk_viscosity_ = (*sim_parameters_)["bulk_viscosity"].as<double>();
    double mu = bulk_viscosity_;
    // double L = 0.5 * (length_ + overlap_length_);
    double a = radius_;
    // double c_par = c_drag_constant_par_;
    // double c_perp = c_drag_constant_perp_;
    // double log_term = std::log(0.5 * L / a);
    // lin_drag_per_len_par_ = 2.0 * M_PI * mu / (log_term + c_par);
    // lin_drag_per_len_perp_ = 4.0 * M_PI * mu / (log_term + c_perp);
    ang_drag_per_len_par_ = 4.0 * M_PI * mu * a * a;
    update_linear_drag_coefficients();
  }
}

void RigidMTBundle::update_linear_drag_coefficients() {
  // update drag coefficients
  // double mu = bulk_viscosity_;
  // double L = 0.5 * (length_ + overlap_length_);
  // double a = radius_;
  // double c_par = c_drag_constant_par_;
  // double c_perp = c_drag_constant_perp_;
  // double log_term = std::log(0.5 * L / a);
  // lin_drag_per_len_par_ = 2.0 * M_PI * mu / (log_term + c_par);
  // lin_drag_per_len_perp_ = 4.0 * M_PI * mu / (log_term + c_perp);

  double mu = bulk_viscosity_;
  double L = 0.5 * (length_ + overlap_length_);
  double a = radius_;
  double aspect_ratio = std::max(L / a, 2.25);
  double c_par = c_drag_constant_par_;
  double c_perp = c_par + 1;
  double log_term = std::log(aspect_ratio);
  lin_drag_per_len_par_ = 2.0 * M_PI * mu / (log_term + c_par);
  lin_drag_per_len_perp_ = 4.0 * M_PI * mu / (log_term + c_perp);
  if (lin_drag_per_len_par_ < 0.0) {
    throw std::runtime_error(
        "RigidMTBundle::update_linear_drag_coefficients - Negative drag "
        "coefficient! Aspect ratio of half bundle must be larger than 2.4ish.");
  }
}

void RigidMTBundle::init_state() {
  // set initial state
  // xyz_center_ = Vec3d(0.0, 0.0, 0.0);
  rotation_matrix_center_ = Eigen::Matrix3d::Identity();
  Vec3d ez = axis_;
  Vec3d ex = math::cross(ez, Vec3d(1.0, 0.0, 0.0));
  if (ex.norm() < 1e-6) {
    ex = math::cross(ez, Vec3d(0.0, 1.0, 0.0));
  }
  ex.normalize();
  Vec3d ey = math::cross(ez, ex);
  ey.normalize();
  rotation_matrix_center_.col(0) = ex;
  rotation_matrix_center_.col(1) = ey;
  rotation_matrix_center_.col(2) = ez;

  force_spb1_.setZero();
  force_spb2_.setZero();
  torque_spb1_.setZero();
  torque_spb2_.setZero();
}

void RigidMTBundle::compute_velocities() {
  double L = length_;
  double l = overlap_length_;
  double F_active = motor_force_per_length_ * l;
  double g_par = lin_drag_per_len_par_;
  double g_perp = lin_drag_per_len_perp_;
  double G_par = (L + l) * g_par;
  double G_perp = (L + l) * g_perp;
  Vec3d u = get_axis();
  double F_compress = u.dot(force_spb1_ - force_spb2_);
  F_compress_ = F_compress;
  double v_grow = get_grow_velocity();

  length_dot_ = (4 * F_active + 2 * F_compress) / (g_par * (L + l));
  // overlap_length_dot_ = -length_dot_;
  // if (length_ < max_length_ && overlap_length_ < length_ &&
  //     motor_force_per_length_ * overlap_length_ < max_force_) {
  //   overlap_length_dot_ += 2 * v_grow_;
  // }
  overlap_length_dot_ = -length_dot_ + 2 * v_grow;

  Vec3d F = force_spb1_ + force_spb2_ + force_envelope_;
  Vec3d F_par = F.dot(u) * u;
  Vec3d F_perp = F - F_par;
  xyz_center_dot_ = F_par / G_par + F_perp / G_perp;

  double h_par = ang_drag_per_len_par_;
  double H_par = (L + l) * h_par;
  // double H_perp = g_perp * (L * L + l * l) / 24.0;
  double H_perp = g_perp * (L * L * L + l * l * l) / 12.0;
  Vec3d T = torque_spb1_ + torque_spb2_ + torque_envelope_;
  Vec3d T_par = T.dot(u) * u;
  Vec3d T_perp = T - T_par;
  angular_velocity_center_ = T_par / H_par + T_perp / H_perp;
}

Vec3d RigidMTBundle::get_linear_fluctuations(double dt) {
  double L = length_;
  double l = overlap_length_;
  double g_par = lin_drag_per_len_par_;
  double g_perp = lin_drag_per_len_perp_;
  double G_par = (L + l) * g_par;
  double G_perp = (L + l) * g_perp;

  Vec3d u_par = rotation_matrix_center_.col(2);
  Vec3d u_perp1 = rotation_matrix_center_.col(0);
  Vec3d u_perp2 = rotation_matrix_center_.col(1);

  double dx_par = std::sqrt(2 * kBT_ * dt / G_par) * randng_.standard_normal();
  double dx_perp1 =
      std::sqrt(2 * kBT_ * dt / G_perp) * randng_.standard_normal();
  double dx_perp2 =
      std::sqrt(2 * kBT_ * dt / G_perp) * randng_.standard_normal();

  return dx_par * u_par + dx_perp1 * u_perp1 + dx_perp2 * u_perp2;
}

Vec3d RigidMTBundle::get_rotational_fluctuations(double dt) {
  double L = length_;
  double l = overlap_length_;
  double g_perp = lin_drag_per_len_perp_;
  double h_par = ang_drag_per_len_par_;
  double H_par = (L + l) * h_par;
  double H_perp = g_perp * (L * L * L + l * l * l) / 12.0;

  Vec3d u_par = rotation_matrix_center_.col(2);
  Vec3d u_perp1 = rotation_matrix_center_.col(0);
  Vec3d u_perp2 = rotation_matrix_center_.col(1);

  double dtheta_par =
      std::sqrt(2 * kBT_ * dt / H_par) * randng_.standard_normal();
  double dtheta_perp1 =
      std::sqrt(2 * kBT_ * dt / H_perp) * randng_.standard_normal();
  double dtheta_perp2 =
      std::sqrt(2 * kBT_ * dt / H_perp) * randng_.standard_normal();

  return dtheta_par * u_par + dtheta_perp1 * u_perp1 + dtheta_perp2 * u_perp2;
}

void RigidMTBundle::update_state_variables(double dt) {
  // update state variables
  length_ += length_dot_ * dt;
  if (overlap_length_ < length_) {
    overlap_length_ += overlap_length_dot_ * dt;
  }

  Vec3d delta_xyz = xyz_center_dot_ * dt;
  Vec3d delta_theta = angular_velocity_center_ * dt;

  if (enable_fluctuations_) {
    // Vec3d ddx = get_linear_fluctuations(dt);
    // Vec3d ddt = get_rotational_fluctuations(dt);
    // printf("    fluc dx,dy,dz: %.10f, %.10f, %.10f\n", ddx[0], ddx[1],
    // ddx[2]); printf("    fluc dt,dt,dt: %.10f, %.10f, %.10f\n", ddt[0],
    // ddt[1], ddt[2]);
    delta_xyz += get_linear_fluctuations(dt);
    delta_theta += get_rotational_fluctuations(dt);
  }

  xyz_center_ += delta_xyz;
  Eigen::Matrix3d R = lie::exp_so3(delta_theta);
  rotation_matrix_center_ = R * rotation_matrix_center_;

  update_linear_drag_coefficients();
}

//////////////////////////////////////////////////////////
// RigidSpindle //////////////////////////////////////////
//////////////////////////////////////////////////////////
void RigidSpindle::set_attributes_from_parameters() {
  // set object specific parameters
  if (parameters_["mt_spb_stretch_stiffness"]) {
    mt_spb_stretch_stiffness_ =
        parameters_["mt_spb_stretch_stiffness"].as<double>();
  }
  if (parameters_["mt_spb_rotation_stiffness"]) {
    mt_spb_rotation_stiffness_ =
        parameters_["mt_spb_rotation_stiffness"].as<double>();
  }
  if (parameters_["draw_axes"]) {
    draw_axes_ = parameters_["draw_axes"].as<bool>();
  }
  if (parameters_["symmetric"]) {
    bool symmetric = parameters_["symmetric"].as<bool>();
    if (symmetric) {
      double rad = parameters_["spb1"]["contact_radius"].as<double>();
      parameters_["spb2"]["contact_radius"] = rad;
    }
  }
  // if (parameters_["xyz_center"]) {
  //   std::vector<double> xyz_center =
  //       parameters_["xyz_center"].as<std::vector<double>>();
  //   xyz_center0_ = Vec3d(xyz_center.data());
  // }
  // if (parameters_["xyz_center"]) {
  //   std::vector<double> axis = parameters_["axis"].as<std::vector<double>>();
  //   axis0_ = Vec3d(axis.data());
  // }

  // if (parameters_["v_grow"]) {
  //   v_grow_ = parameters_["v_grow"].as<double>();
  // }
  // if (parameters_["overlap_length"]) {
  //   overlap_length_ = parameters_["overlap_length"].as<double>();
  // }
  // if (parameters_["length"]) {
  //   length_ = parameters_["length"].as<double>();
  // }
  // if (parameters_["motor_force_per_length"]) {
  //   motor_force_per_length_ =
  //       parameters_["motor_force_per_length"].as<double>();
  // }
  // if (parameters_["axis"]) {
  //   std::vector<double> axis = parameters_["axis"].as<std::vector<double>>();
  //   axis_ = Vec3d(axis.data());
  // }
  // set global parameters
  if ((*sim_parameters_)["kBT"]) {
    kBT_ = (*sim_parameters_)["kBT"].as<double>();
  }
  if ((*sim_parameters_)["dt"]) {
    dt0_ = (*sim_parameters_)["dt"].as<double>();
  }
  // assign parameters to all components
  spb1_.sim_parameters_ = sim_parameters_;
  spb1_.parameters_ = parameters_["spb1"];
  spb2_.sim_parameters_ = sim_parameters_;
  spb2_.parameters_ = parameters_["spb2"];
  mt_bundle_.sim_parameters_ = sim_parameters_;
  mt_bundle_.parameters_ = parameters_["mt_bundle"];

  // set component attributes from parameters
  spb1_.set_attributes_from_parameters();
  spb1_.name_ = "spb1";
  spb2_.set_attributes_from_parameters();
  spb2_.name_ = "spb2";
  mt_bundle_.set_attributes_from_parameters();
  mt_bundle_.name_ = "mt_bundle";
  // adjust components
}

void RigidSpindle::init_state() {
  printf("RigidSpindle::init_state\n");

  mt_bundle_.init_state();
  Eigen::Matrix3d R = mt_bundle_.rotation_matrix_center_;
  Vec3d xyz1 = mt_bundle_.get_xyz1();
  Vec3d xyz2 = mt_bundle_.get_xyz2();
  spb1_.init_state(xyz1, R);
  spb2_.init_state(xyz2, R);
  // Vec3d xyz1 = 0.5 * length_ * axis_;
  // Vec3d xyz2 = -0.5 * length_ * axis_;
  // spb1_.init_spb(xyz1);
  // spb2_.init_spb(xyz2);
}

void RigidSpindle::zero_forces() {
  //   printf("RigidSpindle::zero_forces\n");
  int Nspb1 = spb1_.get_num_vertices();
  int Nspb2 = spb2_.get_num_vertices();
  spb1_.force_V_.resize(Nspb1, 3);
  spb1_.force_V_.setZero();
  spb2_.force_V_.resize(Nspb2, 3);
  spb2_.force_V_.setZero();

  spb1_.force_.setZero();
  spb2_.force_.setZero();
  spb1_.torque_.setZero();
  spb2_.torque_.setZero();
  spb1_.force_envelope_.setZero();
  spb2_.force_envelope_.setZero();
  spb1_.torque_envelope_.setZero();
  spb2_.torque_envelope_.setZero();
  spb1_.force_mt_bundle_.setZero();
  spb2_.force_mt_bundle_.setZero();
  spb1_.couple_mt_bundle_.setZero();
  spb2_.couple_mt_bundle_.setZero();

  // mt_bundle_.force_center_.setZero();
  // mt_bundle_.torque_center_.setZero();
  mt_bundle_.force_spb1_.setZero();
  mt_bundle_.force_spb2_.setZero();
  mt_bundle_.torque_spb1_.setZero();
  mt_bundle_.torque_spb2_.setZero();
  mt_bundle_.force_envelope_.setZero();
  mt_bundle_.torque_envelope_.setZero();
}

void RigidSpindle::apply_internal_forces() {
  Vec3d x_mt_center = mt_bundle_.xyz_center_;
  Vec3d x_spb1 = spb1_.xyz_frame_;
  Vec3d x_spb2 = spb2_.xyz_frame_;
  Vec3d x_mt1 = mt_bundle_.get_xyz1();
  Vec3d x_mt2 = mt_bundle_.get_xyz2();
  Vec3d F_spb1 = -mt_spb_stretch_stiffness_ * (x_spb1 - x_mt1);
  Vec3d F_spb2 = -mt_spb_stretch_stiffness_ * (x_spb2 - x_mt2);
  spb1_.force_mt_bundle_ += F_spb1;
  spb2_.force_mt_bundle_ += F_spb2;
  mt_bundle_.force_spb1_ -= F_spb1;
  mt_bundle_.force_spb2_ -= F_spb2;

  // l_vec1 = 0.5 * mt_bundle_.length_ * mt_bundle_.get_axis();
  Vec3d l_vec1 = x_mt1 - x_mt_center;
  // l_vec2 = -0.5 * mt_bundle_.length_ * mt_bundle_.get_axis();
  Vec3d l_vec2 = x_mt2 - x_mt_center;
  mt_bundle_.torque_spb1_ += -math::cross(l_vec1, F_spb1);
  mt_bundle_.torque_spb2_ += -math::cross(l_vec2, F_spb2);

  Eigen::Matrix3d R_mt = mt_bundle_.rotation_matrix_center_;
  Eigen::Matrix3d R_mt_inv = R_mt.transpose();
  Eigen::Matrix3d R_spb1 = spb1_.rotation_matrix_frame_;
  Eigen::Matrix3d R_spb2 = spb2_.rotation_matrix_frame_;

  Eigen::Matrix3d delta_R1 = R_spb1 * R_mt_inv;
  Eigen::Matrix3d delta_R2 = R_spb2 * R_mt_inv;
  Vec3d delta_theta1 = lie::log_so3(delta_R1);
  Vec3d delta_theta2 = lie::log_so3(delta_R2);
  Vec3d couple1 = -mt_spb_rotation_stiffness_ * delta_theta1;
  Vec3d couple2 = -mt_spb_rotation_stiffness_ * delta_theta2;
  spb1_.couple_mt_bundle_ += couple1;
  spb2_.couple_mt_bundle_ += couple2;
  mt_bundle_.torque_spb1_ -= couple1;
  mt_bundle_.torque_spb2_ -= couple2;
}

void RigidSpindle::compute_velocities() {
  //   printf("RigidSpindle::compute_velocities\n");
  mt_bundle_.compute_velocities();
  spb1_.compute_velocities();
  spb2_.compute_velocities();
}

void RigidSpindle::update_state_variables(double dt) {
  //   printf("RigidSpindle::update_state_variables\n");
  mt_bundle_.update_state_variables(dt);
  spb1_.update_state_variables(dt);
  spb2_.update_state_variables(dt);
}

void RigidSpindle::print_info() {
  double v_grow = mt_bundle_.get_grow_velocity();
  double length = mt_bundle_.length_;
  double max_length = mt_bundle_.max_length_;
  double overlap_length = mt_bundle_.overlap_length_;
  double extensile_force =
      mt_bundle_.motor_force_per_length_ * mt_bundle_.overlap_length_;
  double compressive_force = mt_bundle_.F_compress_;
  Vec3d axis = mt_bundle_.get_axis();
  double envelope_compressive_force =
      spb1_.force_envelope_.dot(axis) - spb2_.force_envelope_.dot(axis);
  Eigen::Matrix3d R_mt = mt_bundle_.rotation_matrix_center_;
  Eigen::Matrix3d R_mt_inv = R_mt.transpose();
  // double det_R_mt = R_mt.determinant();
  Vec3d x_mt1 = mt_bundle_.get_xyz1();
  Vec3d x_mt2 = mt_bundle_.get_xyz2();
  Eigen::Matrix3d R_spb1 = spb1_.rotation_matrix_frame_;
  // Eigen::Matrix3d R_spb1_inv = R_spb1.transpose();
  // double det_R_spb1 = R_spb1.determinant();
  Vec3d x_spb1 = spb1_.xyz_frame_;
  Eigen::Matrix3d R_spb2 = spb2_.rotation_matrix_frame_;
  // Eigen::Matrix3d R_spb2_inv = R_spb2.transpose();
  // double det_R_spb2 = R_spb2.determinant();
  Vec3d x_spb2 = spb2_.xyz_frame_;

  Eigen::Matrix3d R1 = R_spb1 * R_mt_inv;
  Eigen::Matrix3d R2 = R_spb2 * R_mt_inv;

  double delta_x1 = (x_spb1 - x_mt1).norm();
  double delta_x2 = (x_spb2 - x_mt2).norm();
  double delta_theta1 = lie::log_so3(R1).norm();
  double delta_theta2 = lie::log_so3(R2).norm();

  // double one_plus_2cos_theta1 = R1(0, 0) + R1(1, 1) + R1(2, 2);
  // double one_plus_2cos_theta2 = R2(0, 0) + R2(1, 1) + R2(2, 2);
  // double cos_theta1 = (one_plus_2cos_theta1 - 1.0) / 2.0;
  // double cos_theta2 = (one_plus_2cos_theta2 - 1.0) / 2.0;
  // double delta_theta11 = std::acos(cos_theta1);
  // double delta_theta22 = std::acos(cos_theta2);

  // print name_
  // std::cout << "  " << name_ << std::endl;
  // printf("    det_R_mt: %.10f\n", det_R_mt);
  // printf("    det_R_spb1: %.10f\n", det_R_spb1);
  // printf("    det_R_spb2: %.10f\n", det_R_spb2);
  // printf("    delta_x1: %.10f\n", delta_x1);
  // printf("    delta_x2: %.10f\n", delta_x2);
  // printf("    delta_theta1: %.10f\n", delta_theta1);
  // printf("    delta_theta11: %.10f\n", delta_theta11);
  // printf("    delta_theta2: %.10f\n", delta_theta2);
  // printf("    delta_theta22: %.10f\n", delta_theta22);

  ///////////////////////////////////
  printf("  spb1:\n");
  printf("    stretch: %.10f\n", delta_x1);
  printf("    angle: %.10f\n", delta_theta1);
  // printf("    xyz_frame: %.10f %.10f %.10f\n", spb1_.xyz_frame_(0),
  //        spb1_.xyz_frame_(1), spb1_.xyz_frame_(2));
  // printf("    xyz_dot: %.10f %.10f %.10f\n", spb1_.xyz_dot_(0),
  //        spb1_.xyz_dot_(1), spb1_.xyz_dot_(2));
  // printf("    angular_velocity: %.10f %.10f %.10f\n",
  //        spb1_.angular_velocity_(0), spb1_.angular_velocity_(1),
  //        spb1_.angular_velocity_(2));
  // printf("    force_mt_bundle: %.10f %.10f %.10f\n",
  // spb1_.force_mt_bundle_(0),
  //        spb1_.force_mt_bundle_(1), spb1_.force_mt_bundle_(2));
  // printf("    couple_mt_bundle: %.10f %.10f %.10f\n",
  //        spb1_.couple_mt_bundle_(0), spb1_.couple_mt_bundle_(1),
  //        spb1_.couple_mt_bundle_(2));
  // printf("    force_envelope: %.10f %.10f %.10f\n", spb1_.force_envelope_(0),
  //        spb1_.force_envelope_(1), spb1_.force_envelope_(2));
  // printf("    torque_envelope: %.10f %.10f %.10f\n",
  // spb1_.torque_envelope_(0),
  //        spb1_.torque_envelope_(1), spb1_.torque_envelope_(2));
  ///////////////////////////////////
  printf("  spb2:\n");
  printf("    stretch: %.10f\n", delta_x2);
  printf("    angle: %.10f\n", delta_theta2);
  // printf("    xyz_frame: %.10f %.10f %.10f\n", spb2_.xyz_frame_(0),
  //        spb2_.xyz_frame_(1), spb2_.xyz_frame_(2));
  // printf("    xyz_dot: %.10f %.10f %.10f\n", spb2_.xyz_dot_(0),
  //        spb2_.xyz_dot_(1), spb2_.xyz_dot_(2));
  // printf("    angular_velocity: %.10f %.10f %.10f\n",
  //        spb2_.angular_velocity_(0), spb2_.angular_velocity_(1),
  //        spb2_.angular_velocity_(2));
  // printf("    force_mt_bundle: %.10f %.10f %.10f\n",
  // spb2_.force_mt_bundle_(0),
  //        spb2_.force_mt_bundle_(1), spb2_.force_mt_bundle_(2));
  // printf("    couple_mt_bundle: %.10f %.10f %.10f\n",
  //        spb2_.couple_mt_bundle_(0), spb2_.couple_mt_bundle_(1),
  //        spb2_.couple_mt_bundle_(2));
  // printf("    force_envelope: %.10f %.10f %.10f\n", spb2_.force_envelope_(0),
  //        spb2_.force_envelope_(1), spb2_.force_envelope_(2));
  // printf("    torque_envelope: %.10f %.10f %.10f\n",
  // spb2_.torque_envelope_(0),
  //        spb2_.torque_envelope_(1), spb2_.torque_envelope_(2));
  ///////////////////////////////////
  printf("  mt_bundle:\n");
  // printf("    xyz_center: %.10f %.10f %.10f\n", mt_bundle_.xyz_center_(0),
  //        mt_bundle_.xyz_center_(1), mt_bundle_.xyz_center_(2));
  if (v_grow > 0.0) {
    printf("    growing: true\n");
  } else {
    printf("    growing: false\n");
  }
  printf("    length: %.10f\n", mt_bundle_.length_);
  printf("    overlap_length: %.10f\n", mt_bundle_.overlap_length_);
  // Vec3d axis = mt_bundle_.get_axis();
  // printf("    axis: %.10f %.10f %.10f\n", axis(0), axis(1), axis(2));
  // printf("    xyz_center_dot: %.10f %.10f %.10f\n",
  //        mt_bundle_.xyz_center_dot_(0), mt_bundle_.xyz_center_dot_(1),
  //        mt_bundle_.xyz_center_dot_(2));
  // printf("    angular_velocity_center: %.10f %.10f %.10f\n",
  //        mt_bundle_.angular_velocity_center_(0),
  //        mt_bundle_.angular_velocity_center_(1),
  //        mt_bundle_.angular_velocity_center_(2));
  printf("    length_dot: %.10f\n", mt_bundle_.length_dot_);
  printf("    overlap_length_dot: %.10f\n", mt_bundle_.overlap_length_dot_);
  printf("    extensile force: %.10f\n", extensile_force);
  printf("    compressive_force: %.10f\n", compressive_force);
  printf("    envelope_compressive_force: %.10f\n", envelope_compressive_force);
  // printf("    force_spb1: %.10f %.10f %.10f\n", mt_bundle_.force_spb1_(0),
  //        mt_bundle_.force_spb1_(1), mt_bundle_.force_spb1_(2));
  // printf("    force_spb2: %.10f %.10f %.10f\n", mt_bundle_.force_spb2_(0),
  //        mt_bundle_.force_spb2_(1), mt_bundle_.force_spb2_(2));
  // printf("    torque_spb1: %.10f %.10f %.10f\n", mt_bundle_.torque_spb1_(0),
  //        mt_bundle_.torque_spb1_(1), mt_bundle_.torque_spb1_(2));
  // printf("    torque_spb2: %.10f %.10f %.10f\n", mt_bundle_.torque_spb2_(0),
  //        mt_bundle_.torque_spb2_(1), mt_bundle_.torque_spb2_(2));
  // printf("    force_envelope: %.10f %.10f %.10f\n",
  //        mt_bundle_.force_envelope_(0), mt_bundle_.force_envelope_(1),
  //        mt_bundle_.force_envelope_(2));
  // printf("    torque_envelope: %.10f %.10f %.10f\n",
  //        mt_bundle_.torque_envelope_(0), mt_bundle_.torque_envelope_(1),
  //        mt_bundle_.torque_envelope_(2));
  // double normF = mt_bundle_.force_envelope_.norm();
  // double normT = mt_bundle_.torque_envelope_.norm();
  // printf("    normF: %.20f\n", normF);
  // printf("    normT: %.20f\n", normT);
  // if (normF == 0.0) {
  //   printf("    force_envelope is zero\n");
  // } else {
  //   printf("    force_envelope is not zero\n");
  // }
}

} // namespace meshbrane
