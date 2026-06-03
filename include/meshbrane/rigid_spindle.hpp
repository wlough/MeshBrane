#pragma once

/**
 * @file rigid_spindle.hpp
 * @brief Defines objects used in rigid spindle sim
 */

#include "meshbrane/kmc.hpp"
#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane_data_types.hpp"
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class RigidMesh : public MatrixMesh {
public:
  Vec3d force_;
  Vec3d torque_;
  Vec3d xyz_frame_{0.0, 0.0, 0.0};
  Eigen::Matrix3d rotation_matrix_frame_ = Eigen::Matrix3d::Identity();
  Samples3d xyz_relative_V_;
  RigidMesh() = default;
  ~RigidMesh() = default;
  RigidMesh(MatrixMesh &mesh) : MatrixMesh(mesh) {}
  RigidMesh(const std::string &ply_path) : MatrixMesh(ply_path) {}
  RigidMesh(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
            const Samplesi &v_origin_H, const Samplesi &h_next_H,
            const Samplesi &h_twin_H, const Samplesi &f_left_H,
            const Samplesi &h_right_F, const Samplesi &h_negative_B)
      : MatrixMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                   f_left_H, h_right_F, h_negative_B) {}
  // void init_sphere(Vec3d &xyz, double radius, int num_refinements = 0);
  void update_coords_from_frame();
  void sum_force_V();
  void set_parameters() {
    // printf("RigidMesh::set_parameters\n");
    MatrixMesh::set_parameters();
  }
  void set_attributes_from_parameters() {
    printf("RigidMesh::set_attributes_from_parameters\n");
    MatrixMesh::set_attributes_from_parameters();
  }
};

class SphericalSPB : public RigidMesh {
public:
  YAML::Node *sim_parameters_{nullptr};
  // YAML::Node parameters_;
  ////////////////////////////
  // state variables /////////
  ////////////////////////////
  // Vec3d xyz_frame_; // from RigidMesh
  // Eigen::Matrix3d rotation_matrix_frame_; // from RigidMesh
  ///////////////////////////
  // Velocities /////////////
  ///////////////////////////
  Vec3d xyz_dot_;
  Vec3d angular_velocity_;
  ////////////////
  // forces //////
  ////////////////
  Vec3d force_mt_bundle_;
  Vec3d couple_mt_bundle_;
  Vec3d force_envelope_;
  Vec3d torque_envelope_;
  ////////////////////////////
  // free parameters /////////
  ////////////////////////////
  double contact_radius_{0.25};
  double wca_epsilon_total_{192.0};
  double wca_sigma_{1.0};
  int num_refinements_{0};
  ////////////////////////
  // global parameters ///
  ////////////////////////
  double kBT_{1.0};
  double dt0_{0.01};
  double bulk_viscosity_{0.0};
  ////////////////////////
  // derived parameters //
  ////////////////////////
  double wca_epsilon_{1.0};
  double linear_drag_coefficient_{500.0};
  double angular_drag_coefficient_{1.0};
  /////////////////
  // cached data //
  /////////////////
  double interaction_radius_{0.5};
  //////////
  // misc //
  //////////
  kmc::RandomNumberGenerator randng_;
  bool enable_fluctuations_{false};
  /////////////////
  // methods //////
  /////////////////
  SphericalSPB() = default;
  ~SphericalSPB() = default;
  void set_attributes_from_parameters();
  void init_state(Vec3d &center, Eigen::Matrix3d &rotation_matrix);
  void sum_envelope_force_V();
  void compute_velocities();
  void update_state_variables(double dt);
  Vec3d get_linear_fluctuations(double dt);
  Vec3d get_rotational_fluctuations(double dt);
  /////////////////////////////////////////////
  // void init_sphere(Vec3d &xyz, double radius, int num_refinements = 0);
  // void init_spb(Vec3d &center);
  // SphericalSPB(const YAML::Node &parameters) : parameters_(parameters) {
  //   set_parameters();
  // }

  // void set_parameters();
  // void set_global_sim_parameters();
  // void time_step(double dt);
};

class RigidMTBundle : public MatrixMesh {
public:
  YAML::Node *sim_parameters_{nullptr};
  YAML::Node parameters_;
  ////////////////////////////
  // state variables /////////
  ////////////////////////////
  Vec3d xyz_center_;
  Eigen::Matrix3d rotation_matrix_center_ = Eigen::Matrix3d::Identity();
  double length_;
  double overlap_length_;
  ///////////////////////////
  // Velocities /////////////
  ///////////////////////////
  double length_dot_;
  double overlap_length_dot_;
  Vec3d xyz_center_dot_;
  Vec3d angular_velocity_center_;
  ////////////////
  // forces //////
  ////////////////
  Vec3d force_spb1_;
  Vec3d force_spb2_;
  Vec3d torque_spb1_;
  Vec3d torque_spb2_;
  Vec3d force_envelope_;
  Vec3d torque_envelope_;
  //////////////////////////
  // Misc parameters ///////
  //////////////////////////
  double radius_;
  double v_grow_;
  double max_length_{std::numeric_limits<double>::max()};
  double max_force_{std::numeric_limits<double>::max()};
  double motor_force_per_length_;
  double wca_epsilon_;
  double wca_sigma_;
  // MT aspect ratio (L+ell)/2a must be larger than exp(3/2)/2~2.24ish!
  double c_drag_constant_par_{-0.8068528194400547}; // -3/2+log(2) for cylinder
  // double c_drag_constant_perp_{1.0}; = c_drag_constant_par_+1
  // global parameters
  double kBT_;
  double dt0_;
  double bulk_viscosity_;
  // visualization
  double Nphi_{20};
  Vec3d rgb_{0.0, 0.63335, 0.05295};
  Vec3d rgba_overlap_;
  ////////////////////////////
  // derived parameters //////
  ////////////////////////////
  double lin_drag_per_len_par_;
  double lin_drag_per_len_perp_;
  double ang_drag_per_len_par_;
  /////////////////
  // cached data //
  /////////////////
  double interaction_radius_{0.5};
  Vec3d axis_;
  double F_compress_;
  ///////////////////
  // visualization //
  //////////////////
  int num_mts_{20};
  Vec3d rgb_mt1_{0.0, 0.63335, 0.05295};
  Vec3d rgb_mt2_{1.0, 0.498, 0.0};
  //////////
  // misc //
  //////////
  kmc::RandomNumberGenerator randng_;
  bool enable_fluctuations_{false};
  /////////////////
  // methods //////
  /////////////////
  RigidMTBundle() = default;
  ~RigidMTBundle() = default;
  Vec3d get_axis() { return rotation_matrix_center_.col(2); }
  Vec3d get_xyz1() {
    return xyz_center_ + 0.5 * length_ * rotation_matrix_center_.col(2);
  }
  Vec3d get_xyz2() {
    return xyz_center_ - 0.5 * length_ * rotation_matrix_center_.col(2);
  }
  double get_grow_velocity() {
    // double Lmt = 0.5 * (length_ + overlap_length_);
    if (0.5 * (length_ + overlap_length_) < max_length_ &&
        overlap_length_ < length_ &&
        motor_force_per_length_ * overlap_length_ < max_force_) {
      return v_grow_;
    } else {
      return 0.0;
    }
  }
  void set_attributes_from_parameters();
  void init_state();
  void compute_velocities();
  void update_linear_drag_coefficients();
  void update_state_variables(double dt);
  Vec3d get_linear_fluctuations(double dt);
  Vec3d get_rotational_fluctuations(double dt);
};

class RigidSpindle : public MeshBraneObject {
public:
  YAML::Node *sim_parameters_{nullptr};
  YAML::Node parameters_;
  ////////////////////////////
  // state variables /////////
  ////////////////////////////
  RigidMTBundle mt_bundle_;
  SphericalSPB spb1_;
  SphericalSPB spb2_;
  /////////////////////////////
  // Free parameters //////////
  /////////////////////////////
  double mt_spb_stretch_stiffness_{1000.0};
  double mt_spb_rotation_stiffness_{1000.0};
  // global parameters
  double dt0_{0.01};
  double kBT_{0.0};
  ///////////////////////////
  // Cached data ////////////
  ///////////////////////////
  double envelope_force_{0.0};
  // std::vector<double> envelope_force_vec_;
  ///////////////////
  // Misc ///////////
  ///////////////////
  std::string name_;
  bool draw_axes_{false};
  // Vec3d xyz_center0_{0.0, 0.0, 0.0};
  // Vec3d axis0_{0.0, 0.0, 1.0};
  ///////////////////////////
  // Methods ////////////////
  ///////////////////////////
  RigidSpindle() = default;
  ~RigidSpindle() = default;
  RigidSpindle(YAML::Node *sim_parameters, std::string name) {
    // assign parameters
    sim_parameters_ = sim_parameters;
    parameters_ = (*sim_parameters)[name];
    set_attributes_from_parameters();
    init_state();
    name_ = name;
  }

  void set_attributes_from_parameters();
  void init_state();
  void zero_forces();
  void apply_internal_forces();
  void compute_velocities();
  void update_state_variables(double dt);
  void print_info();
  ///////////////////////////////////////////////////////
  // double contact_length_{0.5};
  // double overlap_length_{0.25};
  // double length_{0.5};
  // double motor_force_per_length_{300.0};
  // Vec3d axis_{0.0, 0.0, 1.0};
  // double v_grow_{0.0};

  // void grow(double dt);
  // void update_lengths();
  // void apply_forces_to_spbs();
  // void init_spherical_spbs();

  // void set_parameters();
  // void set_global_sim_parameters();
  // void time_step(double dt);

  //////////////////////////////
  //////////////////////////////
};
} // namespace meshbrane
