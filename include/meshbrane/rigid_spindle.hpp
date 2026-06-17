#pragma once

/**
 * @file rigid_spindle.hpp
 * @brief Defines objects used in rigid spindle sim
 */

#include "meshbrane/kmc.hpp"
#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane_data_types.hpp"
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class SphericalSPBChromatinComplex : public MeshBraneObject {
public:
  // state variables
  Vec3d xyz_center_{0.0, 0.0, 0.0};
  Eigen::Matrix3d rotation_matrix_center_ = Eigen::Matrix3d::Identity();
  // force accumulators
  Vec3d force_{0.0, 0.0, 0.0};
  Vec3d torque_{0.0, 0.0, 0.0};
  // own parameters
  double radius_{0.0};
  double wca_sigma_{0.0};
  double wca_epsilon_{0.0};
  bool enable_fluctuations_{false};
  // global parameters
  double bulk_viscosity_{0.0};
  // set in after_init()
  double linear_drag_coefficient_{0.0};
  double angular_drag_coefficient_{0.0};

  void set_own_parameters(const YAML::Node &sim_node,
                          const YAML::Node &own_node) override {
    // global sim parameters
    if (sim_node["bulk_viscosity"]) {
      bulk_viscosity_ = sim_node["bulk_viscosity"].as<double>();
    } else {
      throw std::runtime_error("No bulk_viscosity provided in sim parameters");
    }
    // object specific parameters
    if (own_node["radius"]) {
      radius_ = own_node["radius"].as<double>();
    } else {
      throw std::runtime_error("No radius provided in parameters");
    }
    if (own_node["wca_sigma"]) {
      wca_sigma_ = own_node["wca_sigma"].as<double>();
    } else {
      throw std::runtime_error("No wca_sigma provided in parameters");
    }
    if (own_node["wca_epsilon"]) {
      wca_epsilon_ = own_node["wca_epsilon"].as<double>();
    } else {
      throw std::runtime_error("No wca_epsilon provided in parameters");
    }
    if (own_node["enable_fluctuations"]) {
      enable_fluctuations_ = own_node["enable_fluctuations"].as<bool>();
    } else {
      throw std::runtime_error("No enable_fluctuations provided in parameters");
    }
  }

  void after_init() override {
    linear_drag_coefficient_ = 6.0 * M_PI * bulk_viscosity_ * radius_;
    angular_drag_coefficient_ =
        8.0 * M_PI * bulk_viscosity_ * radius_ * radius_ * radius_;
  }

  void
  apply_thermal_fluctuations_to_self(double dt, double kBT,
                                     kmc::RandomNumberGenerator &rng) override {
    if (!enable_fluctuations_) {
      return;
    }
    for (int i = 0; i < 3; i++) {
      force_[i] += std::sqrt(2 * kBT * linear_drag_coefficient_ / dt) *
                   rng.standard_normal();
      torque_[i] += std::sqrt(2 * kBT * angular_drag_coefficient_ / dt) *
                    rng.standard_normal();
    }
  }

  void clear_own_interactions() override {
    force_.setZero();
    torque_.setZero();
  }

  void update_own_state_variables(double dt) override {
    xyz_center_ += dt * force_ / linear_drag_coefficient_;
    rotation_matrix_center_ =
        lie::exp_so3(dt * torque_ / angular_drag_coefficient_) *
        rotation_matrix_center_;
  }
};

class RigidMTBundle0 : public MeshBraneObject {
public:
  // state variables
  Vec3d xyz_center_{0.0, 0.0, 0.0};
  Eigen::Matrix3d rotation_matrix_center_ = Eigen::Matrix3d::Identity();
  double total_length_{0.0};
  double overlap_length_{0.0};
  // force accumulators
  Vec3d force_center_{0.0, 0.0, 0.0};
  Vec3d torque_center_{0.0, 0.0, 0.0};
  double force_compress_{0.0};
  // own parameters
  double radius_{0.0};
  double wca_epsilon_{0.0};
  double wca_sigma_{0.0};
  bool enable_fluctuations_{false};
  double v_grow_{0.0};
  double motor_force_per_length_{0.0};
  double max_length_{std::numeric_limits<double>::max()};
  double max_force_{std::numeric_limits<double>::max()};
  // global parameters
  double bulk_viscosity_{0.0};
  // set in after_init()
  double lin_drag_per_len_par_{0.0};
  double lin_drag_per_len_perp_{0.0};
  double ang_drag_per_len_par_{0.0};

  // fixed parameters
  // MT aspect ratio (L+ell)/2a must be larger than exp(3/2)/2~2.24ish!
  double c_drag_constant_par_{-0.8068528194400547}; // -3/2+log(2) for cylinder
  // double c_drag_constant_perp_{1.0}; = c_drag_constant_par_+1

  // visualization
  int num_mts_{20};
  Vec3d rgb_mt1_{0.0, 0.63335, 0.05295};
  Vec3d rgb_mt2_{1.0, 0.498, 0.0};

  void set_own_parameters(const YAML::Node &sim_node,
                          const YAML::Node &own_node) override {
    // global sim parameters
    if (sim_node["bulk_viscosity"]) {
      bulk_viscosity_ = sim_node["bulk_viscosity"].as<double>();
    } else {
      throw std::runtime_error("No bulk_viscosity provided in sim parameters");
    }
    // object specific parameters
    if (own_node["radius"]) {
      radius_ = own_node["radius"].as<double>();
    } else {
      throw std::runtime_error("No radius provided in parameters");
    }
    if (own_node["wca_sigma"]) {
      wca_sigma_ = own_node["wca_sigma"].as<double>();
    } else {
      throw std::runtime_error("No wca_sigma provided in parameters");
    }
    if (own_node["wca_epsilon"]) {
      wca_epsilon_ = own_node["wca_epsilon"].as<double>();
    } else {
      throw std::runtime_error("No wca_epsilon provided in parameters");
    }
    if (own_node["enable_fluctuations"]) {
      enable_fluctuations_ = own_node["enable_fluctuations"].as<bool>();
    } else {
      throw std::runtime_error("No enable_fluctuations provided in parameters");
    }
    if (own_node["v_grow"]) {
      v_grow_ = own_node["v_grow"].as<double>();
    } else {
      throw std::runtime_error("No v_grow provided in parameters");
    }
    if (own_node["motor_force_per_length"]) {
      motor_force_per_length_ = own_node["motor_force_per_length"].as<double>();
    } else {
      throw std::runtime_error(
          "No motor_force_per_length provided in parameters");
    }
    if (own_node["max_length"]) {
      max_length_ = own_node["max_length"].as<double>();
    } else {
      throw std::runtime_error("No max_length provided in parameters");
    }
    if (own_node["max_force"]) {
      max_force_ = own_node["max_force"].as<double>();
    } else {
      throw std::runtime_error("No max_force provided in parameters");
    }
  }

  void after_init() override {
    ang_drag_per_len_par_ = 4.0 * M_PI * bulk_viscosity_ * radius_ * radius_;
    update_linear_drag_coefficients();
  }

  void
  apply_thermal_fluctuations_to_self(double dt, double kBT,
                                     kmc::RandomNumberGenerator &rng) override {
    double L = total_length_;
    double l = overlap_length_;
    double g_par = lin_drag_per_len_par_;
    double g_perp = lin_drag_per_len_perp_;
    double G_par = (L + l) * g_par;
    double G_perp = (L + l) * g_perp;

    Vec3d u_par = rotation_matrix_center_.col(2);
    Vec3d u_perp1 = rotation_matrix_center_.col(0);
    Vec3d u_perp2 = rotation_matrix_center_.col(1);

    double F_par = std::sqrt(2 * kBT * G_par / dt) * rng.standard_normal();
    double F_perp1 = std::sqrt(2 * kBT * G_perp / dt) * rng.standard_normal();
    double F_perp2 = std::sqrt(2 * kBT * G_perp / dt) * rng.standard_normal();
    force_center_ += F_par * u_par + F_perp1 * u_perp1 + F_perp2 * u_perp2;
    // angular fluctuations
    double h_par = ang_drag_per_len_par_;
    double H_par = (L + l) * h_par;
    double H_perp = g_perp * (L * L * L + l * l * l) / 12.0;

    double T_par = std::sqrt(2 * kBT * H_par / dt) * rng.standard_normal();
    double T_perp1 = std::sqrt(2 * kBT * H_perp / dt) * rng.standard_normal();
    double T_perp2 = std::sqrt(2 * kBT * H_perp / dt) * rng.standard_normal();

    torque_center_ += T_par * u_par + T_perp1 * u_perp1 + T_perp2 * u_perp2;
  }
  void clear_own_interactions() override {
    force_center_.setZero();
    torque_center_.setZero();
    force_compress_ = 0.0;
  }
  void update_own_state_variables(double dt) override {
    double L = total_length_;
    double l = overlap_length_;
    double F_active = motor_force_per_length_ * l;
    double g_par = lin_drag_per_len_par_;
    double g_perp = lin_drag_per_len_perp_;
    double G_par = (L + l) * g_par;
    double G_perp = (L + l) * g_perp;

    Vec3d u_par = rotation_matrix_center_.col(2);
    Vec3d u_perp1 = rotation_matrix_center_.col(0);
    Vec3d u_perp2 = rotation_matrix_center_.col(1);

    double v_grow = get_grow_velocity();

    double total_length_dot =
        (4 * F_active + 2 * force_compress_) / (g_par * (L + l));
    double overlap_length_dot = -total_length_dot + 2 * v_grow;

    Vec3d F_par = force_center_.dot(u_par) * u_par;
    Vec3d F_perp = F_par - F_par;

    xyz_center_ += dt * (F_par / G_par + F_perp / G_perp);

    double h_par = ang_drag_per_len_par_;
    double H_par = (L + l) * h_par;
    double H_perp = g_perp * (L * L * L + l * l * l) / 12.0;
    Vec3d T_par = torque_center_.dot(u_par) * u_par;
    Vec3d T_perp = torque_center_ - T_par;

    rotation_matrix_center_ =
        lie::exp_so3(dt * (T_par / H_par + T_perp / H_perp)) *
        rotation_matrix_center_;
  }

  void update_own_cached_data() override {
    //
    update_linear_drag_coefficients();
  }
  // helpers
  void update_linear_drag_coefficients() {
    double L = 0.5 * (total_length_ + overlap_length_);
    double aspect_ratio = std::max(L / radius_, 2.25);
    double c_perp = c_drag_constant_par_ + 1;
    double log_term = std::log(aspect_ratio);
    lin_drag_per_len_par_ =
        2.0 * M_PI * bulk_viscosity_ / (log_term + c_drag_constant_par_);
    lin_drag_per_len_perp_ = 4.0 * M_PI * bulk_viscosity_ / (log_term + c_perp);
    if (lin_drag_per_len_par_ < 0.0) {
      throw std::runtime_error(
          "RigidMTBundle::update_linear_drag_coefficients - Negative drag "
          "coefficient! Aspect ratio of half bundle must be larger than "
          "2.4ish.");
    }
  }
  double get_grow_velocity() {
    if (0.5 * (total_length_ + overlap_length_) < max_length_ &&
        overlap_length_ < total_length_ &&
        motor_force_per_length_ * overlap_length_ < max_force_) {
      return v_grow_;
    } else {
      return 0.0;
    }
  }
};

class RigidSpindle0 : public MeshBraneObject {
public:
  SphericalSPBChromatinComplex *spb_chrom_complex1_{nullptr};
  SphericalSPBChromatinComplex *spb_chrom_complex2_{nullptr};
  RigidMTBundle0 *mt_bundle_{nullptr};

  double mt_spb_stretch_stiffness_{0.0};
  double mt_spb_rotation_stiffness_{0.0};
  bool draw_axes_{false};
  //
  //
  //
  class MTSPB1Interaction : public PairInteraction {
  public:
    RigidMTBundle0 &mt_bundle_;
    SphericalSPBChromatinComplex &spb_;
    double stretch_stiffness_{0.0};
    double rotation_stiffness_{0.0};

    MTSPB1Interaction(RigidMTBundle0 &mt_bundle,
                      SphericalSPBChromatinComplex &spb)
        : mt_bundle_(mt_bundle), spb_(spb) {}

    void interact() override {
      Vec3d x_mt_center = mt_bundle_.xyz_center_;
      double L_center_to_end =
          (mt_bundle_.total_length_ + mt_bundle_.overlap_length_) / 2;
      Vec3d u = mt_bundle_.rotation_matrix_center_.col(2); // ***
      Vec3d vec_center_to_end = L_center_to_end * u;
      Vec3d x_spb = spb_.xyz_center_;
      Vec3d x_mt_end = x_mt_center + vec_center_to_end;

      // force exerted on SPB by MT bundle
      Vec3d F_spb = -stretch_stiffness_ * (x_spb - x_mt_end);

      spb_.force_ += F_spb;

      mt_bundle_.force_center_ += -F_spb;

      mt_bundle_.torque_center_ += -math::cross(vec_center_to_end, F_spb);

      mt_bundle_.force_compress_ += -F_spb.dot(u);

      Eigen::Matrix3d R_mt = mt_bundle_.rotation_matrix_center_;
      Eigen::Matrix3d R_mt_inv = R_mt.transpose();
      Eigen::Matrix3d R_spb = spb_.rotation_matrix_center_;

      Eigen::Matrix3d delta_R = R_spb * R_mt_inv;

      Vec3d delta_theta = lie::log_so3(delta_R);

      // torque exerted on SPB by MT bundle
      Vec3d T_spb = -rotation_stiffness_ * delta_theta;

      spb_.torque_ += T_spb;

      mt_bundle_.torque_center_ -= T_spb;
    }
  };
  //
  //
  //
  class MTSPB2Interaction : public PairInteraction {
  public:
    RigidMTBundle0 &mt_bundle_;
    SphericalSPBChromatinComplex &spb_;
    double stretch_stiffness_{0.0};
    double rotation_stiffness_{0.0};

    MTSPB2Interaction(RigidMTBundle0 &mt_bundle,
                      SphericalSPBChromatinComplex &spb)
        : mt_bundle_(mt_bundle), spb_(spb) {}

    void interact() override {
      Vec3d x_mt_center = mt_bundle_.xyz_center_;
      double L_center_to_end =
          (mt_bundle_.total_length_ + mt_bundle_.overlap_length_) / 2;
      Vec3d u = -mt_bundle_.rotation_matrix_center_.col(2); // ***
      Vec3d vec_center_to_end = L_center_to_end * u;
      Vec3d x_spb = spb_.xyz_center_;
      Vec3d x_mt_end = x_mt_center + vec_center_to_end;

      // force exerted on SPB by MT bundle
      Vec3d F_spb = -stretch_stiffness_ * (x_spb - x_mt_end);

      spb_.force_ += F_spb;

      mt_bundle_.force_center_ += -F_spb;

      mt_bundle_.torque_center_ += -math::cross(vec_center_to_end, F_spb);

      mt_bundle_.force_compress_ += -F_spb.dot(u);

      Eigen::Matrix3d R_mt = mt_bundle_.rotation_matrix_center_;
      Eigen::Matrix3d R_mt_inv = R_mt.transpose();
      Eigen::Matrix3d R_spb = spb_.rotation_matrix_center_;

      Eigen::Matrix3d delta_R = R_spb * R_mt_inv;

      Vec3d delta_theta = lie::log_so3(delta_R);

      // torque exerted on SPB by MT bundle
      Vec3d T_spb = -rotation_stiffness_ * delta_theta;

      spb_.torque_ += T_spb;

      mt_bundle_.torque_center_ -= T_spb;
    }
  };
  //
  //
  //
  void set_own_parameters(const YAML::Node &sim_node,
                          const YAML::Node &own_node) override {} // ***
  void set_initial_conditions(const YAML::Node &sim_node,
                              const YAML::Node &own_node) override {} // ***
  //
  //
  //
  void init_subcomponents(const YAML::Node &sim_node,
                          const YAML::Node &own_node) override {
    mt_bundle_ =
        &init_subcomponent<RigidMTBundle0>(sim_node, own_node, "mt_bundle");
    spb_chrom_complex1_ = &init_subcomponent<SphericalSPBChromatinComplex>(
        sim_node, own_node, "spb_chrom_complex1");
    spb_chrom_complex2_ = &init_subcomponent<SphericalSPBChromatinComplex>(
        sim_node, own_node, "spb_chrom_complex2");
  }
  void init_interactions() override {
    MTSPB1Interaction &mtspb1_int =
        init_interaction<MTSPB1Interaction>(*mt_bundle_, *spb_chrom_complex1_);
    mtspb1_int.stretch_stiffness_ = mt_spb_stretch_stiffness_;
    mtspb1_int.rotation_stiffness_ = mt_spb_rotation_stiffness_;
    MTSPB2Interaction &mtspb2_int =
        init_interaction<MTSPB2Interaction>(*mt_bundle_, *spb_chrom_complex2_);
    mtspb2_int.stretch_stiffness_ = mt_spb_stretch_stiffness_;
    mtspb2_int.rotation_stiffness_ = mt_spb_rotation_stiffness_;
  }
};

class SphericalSPB : public MatrixMesh {
public:
  YAML::Node *sim_parameters_{nullptr};
  ////////////////////////////
  // from RigidMesh /////////
  ////////////////////////////
  Vec3d force_;
  Vec3d torque_;
  Vec3d xyz_frame_{0.0, 0.0, 0.0};
  Eigen::Matrix3d rotation_matrix_frame_ = Eigen::Matrix3d::Identity();
  Samples3d xyz_relative_V_;
  void update_coords_from_frame();
  void sum_force_V();
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
  // ~SphericalSPB() = default;
  void set_attributes_from_parameters();
  void init_state(Vec3d &center, Eigen::Matrix3d &rotation_matrix);
  void sum_envelope_force_V();
  void compute_velocities();
  void update_state_variables(double dt);
  Vec3d get_linear_fluctuations(double dt);
  Vec3d get_rotational_fluctuations(double dt);
  void apply_thermal_fluctuations(double dt, kmc::RandomNumberGenerator &rng);
};

class RigidMTBundle {
  // class RigidMTBundle : public MatrixMesh {
public:
  YAML::Node *sim_parameters_{nullptr};
  YAML::Node parameters_;
  std::string name_;
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
  // void set_attributes_from_yaml_node(const YAML::Node &node) override;
  // void init(const YAML::Node &node) override;
  RigidMTBundle() = default;
  // ~RigidMTBundle() = default;
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
  void apply_thermal_fluctuations(double dt, kmc::RandomNumberGenerator &rng);
};

class RigidSpindle {
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
  // double envelope_force_{0.0};
  // std::vector<double> envelope_force_vec_;
  ///////////////////
  // Misc ///////////
  ///////////////////
  std::string name_;
  bool draw_axes_{false};
  ///////////////////////////
  // Methods ////////////////
  ///////////////////////////
  RigidSpindle() = default;
  // ~RigidSpindle() = default;
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
  void apply_thermal_fluctuations(double dt, kmc::RandomNumberGenerator &rng);
};
} // namespace meshbrane
