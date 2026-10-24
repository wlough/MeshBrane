#pragma once

/**
 * @file chromatin_chain.hpp
 * @brief Defines the ChromatinChain class.
 */

#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simplicial_complex_1d.hpp"
#include <filesystem>
#include <iostream>
#include <limits>
#include <string> // std::string
#include <tuple>  // std::tuple
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class ChromatinChain : public SimplicialComplex1d {
public:
  std::string name_{"chromatin"};

  size_t num_beads_{10};
  double bead_radius_{0.05};
  bool enable_fluctuations_{true};
  double wca_epsilon_{1.25e1};
  double wca_sigma_{3.25e-2};
  double preferred_bond_length_{0.1};
  double bond_stiffness_{1.0};
  double bead_drag_coefficient_{1.0};

  double bulk_viscosity_{0.0};

  Vec3d loop_center_{0, 0, 0};
  Vec3d loop_normal_{0, 0, 1};
  double loop_radius_{0.35};
  double dt_new_bead_{std::numeric_limits<double>::infinity()};
  double t_since_new_bead_{0.0};
  kmc::RandomNumberGenerator randng_;

  YAML::Node *sim_parameters_{nullptr};

  ////////////////////////
  // Initialization //////
  ////////////////////////
  ChromatinChain() = default;

  ChromatinChain(YAML::Node *sim_parameters, std::string name) {
    sim_parameters_ = sim_parameters;
    parameters_ = (*sim_parameters_)[name];
    set_attributes_from_parameters();
    name_ = name;

    init_chromatin_from_attributes();
  }

  void set_attributes_from_parameters() {
    SimplicialComplex1d::set_attributes_from_parameters();
    printf("ChromatinChain::set_attributes_from_parameters\n");
    if (parameters_["num_beads"]) {
      num_beads_ = parameters_["num_beads"].as<size_t>();
    }
    if (parameters_["bead_radius"]) {
      bead_radius_ = parameters_["bead_radius"].as<double>();
    }
    if (parameters_["enable_fluctuations"]) {
      enable_fluctuations_ = parameters_["enable_fluctuations"].as<bool>();
    }
    if (parameters_["wca_epsilon"]) {
      wca_epsilon_ = parameters_["wca_epsilon"].as<double>();
    }
    if (parameters_["wca_sigma"]) {
      wca_sigma_ = parameters_["wca_sigma"].as<double>();
    }
    if (parameters_["preferred_bond_length"]) {
      preferred_bond_length_ =
          parameters_["preferred_bond_length"].as<double>();
    }
    if (parameters_["bond_stiffness"]) {
      bond_stiffness_ = parameters_["bond_stiffness"].as<double>();
    }
    // if (parameters_["bead_drag_coefficient"]) {
    //   bead_drag_coefficient_ =
    //       parameters_["bead_drag_coefficient"].as<double>();
    // }

    if (parameters_["loop_center"]) {
      loop_center_ = Eigen::Map<Eigen::Vector3d>(
          parameters_["loop_center"].as<std::vector<double>>().data());
    }
    if (parameters_["loop_normal"]) {
      loop_normal_ = Eigen::Map<Eigen::Vector3d>(
          parameters_["loop_normal"].as<std::vector<double>>().data());
    }
    if (parameters_["loop_radius"]) {
      loop_radius_ = parameters_["loop_radius"].as<double>();
    }

    if (parameters_["dt_new_bead"]) {
      dt_new_bead_ = parameters_["dt_new_bead"].as<double>();
      t_since_new_bead_ = 0.0;
    }

    if ((*sim_parameters_)["bulk_viscosity"]) {
      bulk_viscosity_ = (*sim_parameters_)["bulk_viscosity"].as<double>();
    }
    printf("ChromatinChain::set_attributes_from_parameters - done\n");
  }

  void init_chromatin_from_attributes() {
    printf("ChromatinChain::init_chromatin_from_attributes\n");
    init_single_loop(num_beads_, loop_center_, loop_normal_, loop_radius_);

    bead_drag_coefficient_ = 6.0 * M_PI * bulk_viscosity_ * bead_radius_;

    update_complex_geometric_data();
    set_visual_defaults();
  }

  Vec3d get_bond_force_v(int v) {
    double L0 = preferred_bond_length_;
    double k = bond_stiffness_;
    Vec3d F = Vec3d::Zero();
    int h_start = h_out_V_[v];
    int h = h_start;
    do {
      int e = e_undirected_H_[h];
      double L = length_E_[e];
      Vec3d vec_h = vec_H_.row(h);
      F += k * (L - L0) * vec_h / L;
      h = h_rot_H_[h];
    } while (h != h_start);
    return F;
  }
  void apply_bond_force_V() {
    int Nv = get_num_vertices();
    for (int v = 0; v < Nv; v++) {
      force_V_.row(v) += get_bond_force_v(v);
    }
  }
  void apply_internal_interactions() { apply_bond_force_V(); }
  void clear_interactions() {
    force_V_.resize(get_num_vertices(), 3);
    force_V_.setZero();
  }
  void apply_thermal_fluctuations_to_self(double dt, double kBT,
                                          kmc::RandomNumberGenerator &rng) {

    if (!enable_fluctuations_) {
      return;
    }
    size_t Nv = get_num_vertices();
    for (int v = 0; v < Nv; v++) {
      for (int i = 0; i < 3; i++) {
        force_V_(v, i) += std::sqrt(2 * kBT * bead_drag_coefficient_ / dt) *
                          rng.standard_normal();
      }
    }
  }
  void update_cached_data() { update_complex_geometric_data(); }
  void update_own_state_variables(double dt) {
    for (int v = 0; v < get_num_vertices(); v++) {
      xyz_coord_V_.row(v) += dt * force_V_.row(v) / bead_drag_coefficient_;
    }

    if (t_since_new_bead_ >= dt_new_bead_) {
      add_random_bead();
      t_since_new_bead_ = 0.0;
    } else {
      t_since_new_bead_ += dt;
    }
  }

  void print_info() {
    //
    int Nv = get_num_vertices();
    double Fmean{0};
    double Fmax{0};
    for (int v = 0; v < Nv; v++) {
      double F = force_V_.row(v).norm();
      Fmean += F;
      Fmax = std::max(Fmax, F);
    }
    Fmean /= Nv;

    std::cout << "  " << name_ << std::endl;

    printf("    num_beads = %lu\n", num_beads_);
    printf("    t_since_new_bead = %.10f\n", t_since_new_bead_);
    printf("    <L_e> = %.10f\n", average_edge_length_);
    printf("    <|F_v|> = %.10f\n", Fmean);
    printf("    max|F_v| = %.10f\n", Fmax);
  }

  void add_random_bead() {
    int Ne = get_num_edges();
    int e = randng_.randint(0, Ne);
    split_edge(e);
    num_beads_++;
  }
};

} // namespace meshbrane
