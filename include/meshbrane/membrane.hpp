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
  // std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3> force_arrows_;
  void update_mcvec_arrows();
  // std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3> mcvec_arrows_;
  SimpleVectorField mcvec_arrows_;
  SimpleVectorField force_arrows_;
  double t_{0.0};
  double dt0_{0.0};
  double dt_{0.0};
  double dt_flip_{0.0};
  double t_flip_{0.0};
  double flipping_probability_{0.0};
  Patch spb_patch_plus_;
  Patch spb_patch_minus_;
  bool show_contact_patches_{false};

  int num_flips_{0};
  int total_edge_flips_{0};

  std::string timestep_type_{"euler_adaptive"}; //"euler"

  std::string pressure_type_{"penalty"};
  // std::string surface_tension_type_{"penalty_local"};
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
  void apply_thermal_fluctuations(double dt, double kBT,
                                  kmc::RandomNumberGenerator &rng);
  void update_cached_data();
  void update_state_variables(double dt);
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

  // void set_node_drag_coefficient_from_bulk_viscosity() {
  //   // Vec3d origin;
  //   // origin.setZero();
  //   // int num_vertices = get_num_vertices();
  //   // double rms_radius = 0.0;
  //   // for (int v = 0; v < num_vertices; v++) {
  //   //   origin += xyz_coord_v(v);
  //   // }
  //   // origin /= num_vertices;
  //   // for (int v = 0; v < num_vertices; v++) {
  //   //   Vec3d dx = xyz_coord_v(v) - origin;
  //   //   double r = dx.norm();
  //   //   rms_radius += r * r;
  //   // }
  //   // rms_radius = std::sqrt(rms_radius / num_vertices);
  //   // node_drag_coefficient_ =
  //   //     6.0 * M_PI * bulk_viscosity_ * rms_radius / num_vertices;
  //   double mu = bulk_viscosity_;
  //   double A = initial_area_;
  //   int num_vertices = get_num_vertices();
  //   double Av = A / num_vertices;
  //   double Rv = std::sqrt(Av / M_PI);
  //   node_drag_coefficient_ = 6.0 * M_PI * mu * Rv;
  // }

  void set_attributes_from_parameters();

  /**
   * @brief Update target edge length, area, and volume for constraint forces.
   *
   */
  void update_geotargets();
  // void update_membrane();

  // void update_internal_forces();

  // void euler_step(double dt);

  // void evolve_until(double t_end);
  // void evolve_until(double t_end, double dt0);

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

  // Vec3d get_fluctuations_v(int v, double dt);
  // void apply_fluctuations_V(double dt);

  // void apply_contact_force_V();
  // void apply_external_force_V();

  int flip_sweep();
  // int tether_flip_sweep();
  // bool tether_wants_flip(int e) const;
  int monte_flip_sweep();
  double monte_flip_probability(int e) const;
};

} // namespace meshbrane
