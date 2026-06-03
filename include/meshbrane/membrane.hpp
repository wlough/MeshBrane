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
  Samples3d external_force_V_;
  Samples3d internal_force_V_;
  Samples3d contact_force_V_;

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

  Membrane() = default;
  ~Membrane() = default;
  Membrane(const YAML::Node &parameters) {
    parameters_ = parameters;
    set_parameters();
    init();
  }
  Membrane(YAML::Node *sim_parameters, std::string name) {
    sim_parameters_ = sim_parameters;
    parameters_ = (*sim_parameters_)[name];
    set_attributes_from_parameters();
    init_from_ply();
    name_ = name;
  }
  Membrane(MatrixMesh &mesh) : MatrixMesh(mesh) {}
  Membrane(const std::string &ply_path) : MatrixMesh(ply_path) {}
  Membrane(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
           const Samplesi &v_origin_H, const Samplesi &h_next_H,
           const Samplesi &h_twin_H, const Samplesi &f_left_H,
           const Samplesi &h_right_F, const Samplesi &h_negative_B)
      : MatrixMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                   f_left_H, h_right_F, h_negative_B) {}

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

  // void set_parameters() override;
  void set_attributes_from_parameters() {
    MatrixMesh::set_attributes_from_parameters();
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    if (parameters_["timestep_type"]) {
      timestep_type_ = parameters_["timestep_type"].as<std::string>();
    }
    // Bending force
    if (parameters_["bending_modulus"]) {
      bending_modulus_ = parameters_["bending_modulus"].as<double>();
    }
    if (parameters_["spontaneous_curvature"]) {
      spontaneous_curvature_ =
          parameters_["spontaneous_curvature"].as<double>();
    }
    if (parameters_["splay_modulus"]) {
      splay_modulus_ = parameters_["splay_modulus"].as<double>();
    }
    ///////////////////////////////////////////////////////
    // Tether force
    if (parameters_["dimensionless_tether_repulsive_singularity"]) {
      dimensionless_tether_repulsive_singularity_ =
          parameters_["dimensionless_tether_repulsive_singularity"]
              .as<double>();
    }
    if (parameters_["dimensionless_tether_repulsive_onset"]) {
      dimensionless_tether_repulsive_onset_ =
          parameters_["dimensionless_tether_repulsive_onset"].as<double>();
    }
    if (parameters_["dimensionless_tether_attractive_onset"]) {
      dimensionless_tether_attractive_onset_ =
          parameters_["dimensionless_tether_attractive_onset"].as<double>();
    }
    if (parameters_["dimensionless_tether_attractive_singularity"]) {
      dimensionless_tether_attractive_singularity_ =
          parameters_["dimensionless_tether_attractive_singularity"]
              .as<double>();
    }
    if (parameters_["tether_stiffness"]) {
      tether_stiffness_ = parameters_["tether_stiffness"].as<double>();
    }
    // Area conservation force
    if (parameters_["area_stiffness"]) {
      area_stiffness_ = parameters_["area_stiffness"].as<double>();
    }
    if (parameters_["fix_target_face_area"]) {
      fix_target_face_area_ = parameters_["fix_target_face_area"].as<bool>();
    }
    // Volume conservation force
    if (parameters_["fix_target_volume"]) {
      fix_target_volume_ = parameters_["fix_target_volume"].as<bool>();
    }
    if (parameters_["volume_stiffness"]) {
      volume_stiffness_ = parameters_["volume_stiffness"].as<double>();
    }
    if (parameters_["node_drag_coefficient"]) {
      node_drag_coefficient_ =
          parameters_["node_drag_coefficient"].as<double>();
    }
    ///////////////////////////////////////////////////////
    if (parameters_["vector_field_scale"]) {
      vector_field_scale_ = parameters_["vector_field_scale"].as<double>();
    }
    if (parameters_["show_force_field"]) {
      show_force_field_ = parameters_["show_force_field"].as<bool>();
    }
    if (parameters_["show_mcvec_field"]) {
      show_mcvec_field_ = parameters_["show_mcvec_field"].as<bool>();
    }
    if (parameters_["enable_flipping"]) {
      enable_flipping_ = parameters_["enable_flipping"].as<bool>();
    }
    if (parameters_["enable_fluctuations"]) {
      enable_fluctuations_ = parameters_["enable_fluctuations"].as<bool>();
    }
    if (parameters_["dt_flip"]) {
      dt_flip_ = parameters_["dt_flip"].as<double>();
    }
    if (parameters_["flipping_probability"]) {
      flipping_probability_ = parameters_["flipping_probability"].as<double>();
    }
    ///////////////////////////////////////////////////////
    if (parameters_["show_contact_patches"]) {
      show_contact_patches_ = parameters_["show_contact_patches"].as<bool>();
    }
    if (parameters_["pressure_type"]) {
      pressure_type_ = parameters_["pressure_type"].as<std::string>();
    }
    if (parameters_["use_surface_tension_constant"]) {
      use_surface_tension_constant_ =
          parameters_["use_surface_tension_constant"].as<bool>();
    }
    if (parameters_["use_surface_tension_penalty_local"]) {
      use_surface_tension_penalty_local_ =
          parameters_["use_surface_tension_penalty_local"].as<bool>();
    }
    if (parameters_["surface_tension_constant"]) {
      surface_tension_constant_ =
          parameters_["surface_tension_constant"].as<double>();
    } else if (use_surface_tension_constant_) {
      printf(
          "Warning: use_surface_tension_constant_ = true but no constant value "
          "found\n");
    }
    if (parameters_["wca_sigma"]) {
      wca_sigma_ = parameters_["wca_sigma"].as<double>();
      // 0.1*r_cutoff=0.1*2^(1/6)*sigma
      dx_max_ = 0.1122462048309373 * wca_sigma_;
    }
    if (parameters_["wca_epsilon"]) {
      wca_epsilon_total_ = parameters_["wca_epsilon"].as<double>();
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
    }
    if (parameters_["use_local_drag_coefficient"]) {
      use_local_drag_coefficient_ =
          parameters_["use_local_drag_coefficient"].as<bool>();
      // local_drag_coefficient_ = 4 * M_PI * bulk_viscosity_;
      local_drag_coefficient_ = 1.5 * bulk_viscosity_; // * 1/R but R=1
    }
  }
  // void set_global_sim_parameters();
  void sync_with_sim_parameters();
  void init();
  /**
   * @brief Update target edge length, area, and volume for constraint forces.
   *
   */
  void update_geotargets();
  void update_membrane();
  void update_cached_data();
  void update_internal_forces();
  void zero_forces() {
    force_V_.resize(get_num_vertices(), 3);
    force_V_.setZero();
    // external_force_V_.setZero();
    // contact_force_V_.setZero();
    // internal_force_V_.setZero();
  }
  void update_membrane_visuals();

  /**
   * @brief Get the maximum time step for the current applied forces.
   *
   * @return double
   */
  double dt_max() const;
  void euler_step(double dt);
  void predictor_corrector_step(double dt);
  void time_step(double dt);
  void evolve_until(double t_end);
  void evolve_until(double t_end, double dt0);
  void apply_internal_forces() {
    apply_tether_force_V();
    apply_bending_force_V();
    apply_area_force_V();
    apply_volume_force_V();
  }

  void save_state(const std::string &filename) const;

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

  Vec3d get_fluctuations_v(int v, double dt);
  void apply_fluctuations_V(double dt);

  void apply_contact_force_V();
  void apply_external_force_V();

  int flip_sweep();
  int tether_flip_sweep();
  bool tether_wants_flip(int e) const;
  int monte_flip_sweep();
  double monte_flip_probability(int e) const;
  int flip_edges_monte();

  //////////////////////////////////////////////
  // Visualization /////////////////////////////
  //////////////////////////////////////////////

  void apply_forces() {
    force_V_.resize(get_num_vertices(), 3);
    force_V_.setZero();
    apply_tether_force_V();
    apply_bending_force_V();
    apply_area_force_V();
    apply_volume_force_V();
    apply_contact_force_V();
    apply_external_force_V();
  }

  void print_info() {
    double A = area_F_.sum();
    double A0 = initial_area_;
    double V = total_volume_;
    double V0 = initial_volume_;
    double P = pressure_;
    std::cout << "  " << name_ << std::endl;
    printf("    (A-A0)/A0=%.10f\n", (A - A0) / A0);
    printf("    (V-V0)/V0=%.10f\n", (V - V0) / V0);
    printf("    pressure=%.10f\n", pressure_);
    printf("    surface tension mean=%.10f\n", surface_tension_F_.mean());
    printf("    surface tension max=%.10f\n", surface_tension_F_.maxCoeff());
    printf("    surface tension min=%.10f\n", surface_tension_F_.minCoeff());
    // printf("    num_flips: %d\n", num_flips_);
    printf("    num_flips: %d\n", total_edge_flips_);
    // printf("    num_flips/num_edges: %.10f\n",
    //        static_cast<double>(num_flips_) /
    //            static_cast<double>(get_num_edges()));
    // printf("  dt=%.20f\n", dt_);
  }

  //////////////////////
  // to be deprecated //
  //////////////////////
  void update_curvature_data();
};

} // namespace meshbrane
