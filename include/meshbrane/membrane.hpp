#pragma once

/**
 * @file membrane.hpp
 * @brief Defines the Membrane class.
 */

#include "meshbrane/kmc.hpp"
#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/patch.hpp"
#include "meshbrane/simple_vector_field.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <filesystem>
#include <iostream>
#include <yaml-cpp/yaml.h>

/**
 * @addtogroup CoreStructures Core MeshBrane structures
 */

namespace meshbrane {

/**
 * @brief Nuclear envelope
 */
class Membrane : public MatrixMesh {
public:
  /////////////////////////
  // fixed parameters /////
  /////////////////////////

  ////////////////////
  // Cached data /////
  ////////////////////

  ///////////////////////////////////////////////////////
  // Data members ///////////////////////////////////////
  ///////////////////////////////////////////////////////
  double dx_max_{1.0};
  double wca_sigma_{0.1};
  double wca_epsilon_{12.5};
  double wca_epsilon_total_{192};

  kmc::RandomNumberGenerator randng_;
  YAML::Node *sim_parameters_{nullptr};
  // Samples3d force_V_;
  // Samples3d external_force_V_;
  // Samples3d internal_force_V_;
  // Samples3d contact_force_V_;

  double initial_area_{0.0};
  double initial_volume_{0.0};
  // Bending force
  double bending_modulus_{1.0};
  double spontaneous_curvature_{0.0};
  double splay_modulus_{0.0};
  // Tether force
  double target_edge_length_{0.0};
  double dimensionless_tether_repulsive_singularity_{0.4};
  double dimensionless_tether_repulsive_onset_{0.8};
  double dimensionless_tether_attractive_onset_{1.2};
  double dimensionless_tether_attractive_singularity_{1.6};
  double tether_stiffness_{0.16};
  // Surface tension
  bool fix_target_face_area_{true};
  double area_stiffness_{100.0};
  double target_face_area_{0.0};
  Samples1d surface_tension_F_;
  // Pressure
  bool fix_target_volume_{true};
  double target_volume_{0.0};
  double volume_stiffness_{10000.0};
  double pressure_{0.0};
  // Viscous force
  double node_drag_coefficient_{0.0};
  double bulk_viscosity_{0.0};
  bool use_local_drag_coefficient_{true};
  double local_drag_coefficient_{0.0};
  // Fluctuations
  double kBT_{0.0};
  bool enable_flipping_{false};
  bool enable_fluctuations_{false};

  // Visualization
  bool show_force_field_{false};
  bool show_mcvec_field_{false};
  double vector_field_scale_{0.01};
  void update_force_arrows();
  void update_mcvec_arrows();
  SimpleVectorField mcvec_arrows_;
  SimpleVectorField force_arrows_;
  double t_{0.0};
  // double dt0_{0.0};
  double dt_{0.0};
  double dt_flip_{0.0};
  double t_flip_{0.0};
  double flipping_probability_{0.0};
  Patch spb_patch_plus_;
  Patch spb_patch_minus_;
  bool show_contact_patches_{false};

  int num_flips_{0};
  int total_edge_flips_{0};

  bool use_surface_tension_constant_{false};
  bool use_surface_tension_penalty_local_{true};
  double surface_tension_constant_{10.0};
  ///////////////////////////////////////////////////////
  // Initialization /////////////////////////////////////
  ///////////////////////////////////////////////////////

  // void set_attributes_from_yaml_node(const YAML::Node &node) override;
  // void init(const YAML::Node &node) override;
  void init_membrane_from_attributes();

  Membrane() = default;
  // ~Membrane() = default;
  // Membrane(const YAML::Node &parameters) {
  //   throw std::runtime_error("Membrane(const YAML::Node &parameters)");
  //   parameters_ = parameters;
  //   set_attributes_from_parameters();
  //   // init();
  // }
  Membrane(YAML::Node *sim_parameters, std::string name) {
    sim_parameters_ = sim_parameters;
    parameters_ = (*sim_parameters_)[name];
    set_attributes_from_parameters();
    init_from_ply();
    name_ = name;
    integration_patch_ = Patch(this);
    spb_patch_plus_ = Patch(this);
    spb_patch_minus_ = Patch(this);
  }

  ////////////////////////////////////////
  // stuff that actually gets used

  void apply_internal_interactions();
  void clear_interactions();
  void apply_thermal_fluctuations_to_self(double dt, double kBT,
                                          kmc::RandomNumberGenerator &rng);
  void update_cached_data();
  void update_own_state_variables(double dt);
  /**
   * @brief Get the maximum time step for the current applied forces.
   *
   * @return double
   */
  double dt_max() const;

  void print_info();

  void update_membrane_visuals();

  ///////////////////////////////////////////////////////
  // Prototyping ////////////////////////////////////////
  ///////////////////////////////////////////////////////

  void init_from_ply();

  void set_attributes_from_parameters();

  /**
   * @brief Update target edge length, area, and volume for constraint forces.
   *
   */
  void update_geotargets();

  ///////////////////////////////////////////////////////
  // Membrane physics ///////////////////////////////////
  ///////////////////////////////////////////////////////

  void update_pressure_soft_penalty();
  void update_surface_tension_soft_penalty();
  void update_surface_tension_constant();
  void update_surface_tension_and_pressure();

  Vec3d get_tether_force_v(int v) const;
  void apply_tether_force_V();
  Vec3d get_bending_force_v(int v) const;
  void apply_bending_force_V();
  Vec3d get_area_force_v(int v) const;
  void apply_area_force_V();
  Vec3d get_volume_force_v(int v) const;
  void apply_volume_force_V();

  int monte_flip_sweep();
  double monte_flip_probability(int e) const;

  void test_surface_tension() {
    printf("test_surface_tension");
    double surface_tension_mean = 0.0;
    size_t Nf = get_num_faces();
    // Samples1d gamma_F(Nf);
    double A0 = target_face_area_;
    double k = area_stiffness_;
    double gamma0 = surface_tension_constant_;
    for (size_t f = 0; f < Nf; f++) {
      double gamma_f = surface_tension_F_[f];
      double A = area_F_[f];
      double gamma = gamma0 + k * (A - A0) / A0;
      if (std::abs(gamma - gamma_f) > 1e-6) {
        throw std::runtime_error("bad surface tension");
      }
    }
  }

  // void apply_thermal_drift_force_V() {
  //   //
  //   if (use_local_drag_coefficient_) {
  //     force_V_ += kBT_ * mcvec_V_ / 3.0;
  //   }
  // }

  void run_force_mag_test(double dt, double kBT,
                          kmc::RandomNumberGenerator &rng) {

    size_t Nv = get_num_vertices();

    Samples3d F_bend(Nv, 3);
    Samples3d F_tension(Nv, 3);
    Samples3d F_pressure(Nv, 3);
    Samples3d F_tether(Nv, 3);
    Samples3d F_thermal(Nv, 3);
    Samples3d F_drift = kBT_ * mcvec_V_ / 3.0;

    for (int v = 0; v < Nv; v++) {
      F_bend.row(v) = get_bending_force_v(v);
      F_tension.row(v) = get_area_force_v(v);
      F_pressure.row(v) = get_volume_force_v(v);
      F_tether.row(v) = get_tether_force_v(v);
      for (int i = 0; i < 3; i++) {
        F_thermal(v, i) =
            std::sqrt(2 * kBT * area_V_[v] * local_drag_coefficient_ / dt) *
            rng.standard_normal();
      }
    }

    Samples1d normF_bend = F_bend.rowwise().norm();
    Samples1d normF_tension = F_tension.rowwise().norm();
    Samples1d normF_pressure = F_pressure.rowwise().norm();
    Samples1d normF_tether = F_tether.rowwise().norm();
    Samples1d normF_thermal = F_thermal.rowwise().norm();
    Samples1d normF_drift = F_drift.rowwise().norm();

    double meanF_bend = normF_bend.mean();
    double meanF_tension = normF_tension.mean();
    double meanF_pressure = normF_pressure.mean();
    double meanF_tether = normF_tether.mean();
    double meanF_thermal = normF_thermal.mean();
    double meanF_drift = normF_drift.mean();

    double maxF_bend = normF_bend.maxCoeff();
    double maxF_tension = normF_tension.maxCoeff();
    double maxF_pressure = normF_pressure.maxCoeff();
    double maxF_tether = normF_tether.maxCoeff();
    double maxF_thermal = normF_thermal.maxCoeff();
    double maxF_drift = normF_drift.maxCoeff();

    printf("*** maxF_bend = %.10f\n", maxF_bend);
    printf("*** maxF_tension = %.10f\n", maxF_tension);
    printf("*** maxF_pressure = %.10f\n", maxF_pressure);
    printf("*** maxF_tether = %.10f\n", maxF_tether);
    printf("*** maxF_thermal = %.10f\n", maxF_thermal);
    printf("*** maxF_drift = %.10f\n", maxF_drift);
    printf("*** *** *** ***\n");
    printf("*** meanF_bend = %.10f\n", meanF_bend);
    printf("*** meanF_tension = %.10f\n", meanF_tension);
    printf("*** meanF_pressure = %.10f\n", meanF_pressure);
    printf("*** meanF_tether = %.10f\n", meanF_tether);
    printf("*** meanF_thermal = %.10f\n", meanF_thermal);
    printf("*** meanF_drift = %.10f\n", meanF_drift);
  }
  //
  //
};

} // namespace meshbrane
