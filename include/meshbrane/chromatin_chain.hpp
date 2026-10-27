#pragma once

/**
 * @file chromatin_chain.hpp
 * @brief Defines the ChromatinChain class.
 */

#include "meshbrane/meshbrane_data_types.hpp"
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

  size_t num_initial_beads_{10};
  size_t num_max_beads_{std::numeric_limits<size_t>::infinity()};

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

  Vec3d initial_center_{0, 0, 0};
  double initial_radius_{0.35};

  double dt_new_bead_{std::numeric_limits<double>::infinity()};
  double t_since_new_bead_{0.0};
  kmc::RandomNumberGenerator randng_;

  YAML::Node *sim_parameters_{nullptr};

  bool split_edges_{false};
  bool grow_ends_{false};
  bool seed_new_edges_{false};
  bool seed_isolated_vertices_{false};
  bool seed_vertex_{false};
  bool seed_ring_{false};
  Vec3d xyz_seed_{0, 0, 0};

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

  // void do_swap_test() {
  //   printf("ChromatinChain::do_swap_test 0");
  //   int Nv = get_num_vertices();
  //   int v0 = Nv, v1 = Nv + 1;
  //   add_random_isolated_vertex();
  //   add_random_isolated_vertex();
  //   // add_random_isolated_vertex();
  //   // add_random_isolated_vertex();
  //   // add_random_isolated_vertex();
  //   add_edge_between_vertices(v0, v1);
  //   printf("ChromatinChain::do_swap_test 1");
  //   swap_boundary_indices(0, 1);
  //   printf("ChromatinChain::do_swap_test 2");
  //   swap_vertex_indices(0, 1);
  //   printf("ChromatinChain::do_swap_test 3");
  //   swap_edge_indices(0, 1); //***
  //   printf("ChromatinChain::do_swap_test 4");
  //   // swap_halfedge_indices(0, 1);
  //   // printf("ChromatinChain::do_swap_test 5");
  //   //
  //   // printf("ChromatinChain::do_swap_test 6");
  // }

  void do_swap_test() {
    printf("ChromatinChain::do_swap_test 0");
    int Nv = get_num_vertices();
    int v0 = Nv, v1 = Nv + 1;
    // add_random_isolated_vertex();
    // add_random_isolated_vertex();
    // add_random_isolated_vertex();
    // add_random_isolated_vertex();
    // add_random_isolated_vertex();
    // add_edge_between_vertices(v0, v1);
    printf("ChromatinChain::do_swap_test 1");
    // swap_boundary_indices(0, 1);
    printf("ChromatinChain::do_swap_test 2");
    swap_vertex_indices(0, 1);
    printf("ChromatinChain::do_swap_test 3");
    swap_edge_indices(0, 1);
    printf("ChromatinChain::do_swap_test 4");
    // swap_halfedge_indices(0, 1);
    // printf("ChromatinChain::do_swap_test 5");
    //
    // printf("ChromatinChain::do_swap_test 6");

    for (int v = 0; v < Nv; v++) {
      int vv = (v + Nv / 2) % Nv;
      add_edge_between_vertices(v, vv);
    }
  }

  void set_attributes_from_parameters() {
    SimplicialComplex1d::set_attributes_from_parameters();
    printf("ChromatinChain::set_attributes_from_parameters\n");
    if (parameters_["num_initial_beads"]) {
      num_initial_beads_ = parameters_["num_initial_beads"].as<size_t>();
    }
    if (parameters_["num_max_beads"]) {
      num_max_beads_ = parameters_["num_max_beads"].as<size_t>();
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
    if (parameters_["initial_radius"]) {
      initial_radius_ = parameters_["initial_radius"].as<double>();
    }
    if (parameters_["initial_center"]) {
      initial_center_ = Eigen::Map<Eigen::Vector3d>(
          parameters_["initial_center"].as<std::vector<double>>().data());
    }

    if (parameters_["dt_new_bead"]) {
      dt_new_bead_ = parameters_["dt_new_bead"].as<double>();
      t_since_new_bead_ = 0.0;
    }

    if (parameters_["split_edges"]) {
      split_edges_ = parameters_["split_edges"].as<bool>();
    }
    if (parameters_["grow_ends"]) {
      grow_ends_ = parameters_["grow_ends"].as<bool>();
    }
    if (parameters_["seed_new_edges"]) {
      seed_new_edges_ = parameters_["seed_new_edges"].as<bool>();
    }
    if (parameters_["seed_isolated_vertices"]) {
      seed_isolated_vertices_ =
          parameters_["seed_isolated_vertices"].as<bool>();
    }

    if (parameters_["seed_vertex"]) {
      seed_vertex_ = parameters_["seed_vertex"].as<bool>();
    }
    if (parameters_["seed_ring"]) {
      seed_ring_ = parameters_["seed_ring"].as<bool>();
    }

    if (parameters_["xyz_seed"]) {
      xyz_seed_ = Eigen::Map<Eigen::Vector3d>(
          parameters_["xyz_seed"].as<std::vector<double>>().data());
    }

    if ((*sim_parameters_)["bulk_viscosity"]) {
      bulk_viscosity_ = (*sim_parameters_)["bulk_viscosity"].as<double>();
    }
    printf("ChromatinChain::set_attributes_from_parameters - done\n");
  }

  void init_chromatin_from_attributes() {
    printf("ChromatinChain::init_chromatin_from_attributes\n");
    if (seed_vertex_) {
      init_point_topo(xyz_seed_);
    } else if (seed_ring_) {
      init_single_loop(num_initial_beads_, loop_center_, loop_normal_,
                       loop_radius_);
      // init_fib_sphere(num_initial_beads_, initial_center_, initial_radius_);
    } else {
      init_fib_sphere(num_initial_beads_, initial_center_, initial_radius_);
    }
    bead_drag_coefficient_ = 6.0 * M_PI * bulk_viscosity_ * bead_radius_;
    update_complex_geometric_data();
    set_visual_defaults();
    do_swap_test();
  }

  Vec3d get_bond_force_v(int v) {
    double L0 = preferred_bond_length_;
    double k = bond_stiffness_;
    Vec3d F = Vec3d::Zero();
    int h_start = h_out_V_[v];
    int h = h_start;
    do {
      int e = e_undirected_H_[h];
      if (e >= 0) {
        double L = length_E_[e];
        Vec3d vec_h = vec_H_.row(h);
        F += k * (L - L0) * vec_h / L;
      }
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

  // Vec3d get_bending_force_v(int v) {
  //   double k = bond_stiffness_;
  //   Vec3d F = Vec3d::Zero();
  //   if (is_boundary_v(v)) {
  //     return F;
  //   }
  //   int h_start = h_out_V_[v];
  //   int h = h_start;
  //   int hr = h_rot_H_[h];
  //   Vec3d x0 = xyz_coord_v(v);
  //   do {
  //
  //     int ht = h_twin_H_[h];
  //     int hrt = h_twin_H_[hr];
  //     int v = v_origin_H_[ht];
  //     int vr = v_origin_H_[hrt];
  //     // Vec3d x = xyz_coord_v(v);
  //     // Vec3d xr = xyz_coord_v(vr);
  //     // Vec3d dx0 = x0 - xr;
  //     // Vec3d dx1 = x - x0;
  //     // double ds0 = dx0.norm();
  //     // double ds1 = dx1.norm();
  //     // Vec3d B = dx0.cross(dx1);
  //     // double sin_theta = B.norm() / (ds0 * ds1);
  //     // double theta = std::asin(sin_theta);
  //     // B /= B.norm();
  //     // Vec3d T = x - xr;
  //     // double norm_T = T.norm();
  //     // T /= norm_T;
  //     // Vec3d N = B.cross(T);
  //     // F += k * theta * N;
  //     Vec3d xp = xyz_coord_v(v);
  //     Vec3d xm = xyz_coord_v(vr);
  //     Vec3d dxp = xp - x0;
  //     Vec3d dxm = x0 - xm;
  //     double dsp = dxp.norm();
  //     double dsm = dxm.norm();
  //
  //     Vec3d x_ss = (2.0 / (dsm + dsp)) * (dxp / dsp - dxm / dsm);
  //     F += k * x_ss;
  //
  //     h = hr;
  //     hr = h_rot_H_[h];
  //   } while (hr != h_start);
  //   return F;
  // }

  Vec3d get_bending_force_v(int v) {
    double k = bond_stiffness_;
    Vec3d F = Vec3d::Zero();
    // if (is_boundary_v(v)) {
    //   return F;
    // }
    // int h_start = h_out_V_[v];
    // int h = h_start;
    // int hr = h_rot_H_[h];
    // Vec3d x0 = xyz_coord_v(v);
    // do {
    //   int ht = h_twin_H_[h];
    //   int hrt = h_twin_H_[hr];
    //   int v = v_origin_H_[ht];
    //   int vr = v_origin_H_[hrt];
    //   Vec3d xp = xyz_coord_v(v);
    //   Vec3d xm = xyz_coord_v(vr);
    //
    //   ...
    //
    //   F += ???;
    //
    //   h = hr;
    //   hr = h_rot_H_[h];
    // } while (hr != h_start);
    return F;
  }

  void apply_bending_force_V() {
    int Nv = get_num_vertices();
    for (int v = 0; v < Nv; v++) {
      force_V_.row(v) += get_bending_force_v(v);
    }
  }

  void apply_internal_interactions() {
    apply_bond_force_V();
    apply_bending_force_V();
  }
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
    int Ne = get_num_edges();
    int Nh = get_num_half_edges();
    int Nb = get_num_boundaries();
    double Fmean{0};
    double Fmax{0};
    for (int v = 0; v < Nv; v++) {
      double F = force_V_.row(v).norm();
      Fmean += F;
      Fmax = std::max(Fmax, F);
    }
    Fmean /= Nv;

    std::cout << "  " << name_ << std::endl;

    printf("    vertices = %d\n", Nv);
    printf("    edges = %d\n", Ne);
    printf("    boundaries = %d\n", Nb);
    printf("    half-edges = %d\n", Nh);
    printf("    t_since_new_bead = %.10f\n", t_since_new_bead_);
    printf("    <L_e> = %.10f\n", average_edge_length_);
    printf("    <|F_v|> = %.10f\n", Fmean);
    printf("    max|F_v| = %.10f\n", Fmax);
  }

  void split_random_edge() {
    int Ne = get_num_edges();
    int e = randng_.randint(0, Ne);
    split_edge(e);
  }

  void grow_random_end() {

    int Nb = get_num_boundaries();
    if (Nb < 1) {
      return;
    }
    int b = randng_.randint(0, Nb);
    int h = h_ghost_B_[b];
    int v = v_origin_H_[h];

    // int hh = h_rot_H_[h];
    // if (h == hh) {
    //   return;
    // }
    // hh = h_twin_H_[hh];
    // int vv = v_origin_H_[hh];
    //
    // Vec3d x = xyz_coord_v(v);
    // Vec3d xx = xyz_coord_v(vv);
    //
    // Vec3d direction = 1e0 * (x - xx);

    Vec3d direction{randng_.standard_uniform() - 0.5,
                    randng_.standard_uniform() - 0.5,
                    randng_.standard_uniform() - 0.5};
    direction *= 1e-2 / direction.norm();
    seed_edge(v, direction);
  }

  void seed_random_edge() {
    int Nv = get_num_vertices();
    int v = randng_.randint(0, Nv);
    Vec3d direction{randng_.standard_uniform(), randng_.standard_uniform(),
                    randng_.standard_uniform()};
    direction *= 1e-2 / direction.norm();
    seed_edge(v, direction);
  }

  void add_random_isolated_vertex() {
    Vec3d xyz{randng_.standard_uniform() - 0.5,
              randng_.standard_uniform() - 0.5,
              randng_.standard_uniform() - 0.5};
    xyz *= .8 * randng_.standard_uniform() / xyz.norm();
    // double r =  randng_.standard_uniform();
    add_isolated_vertex(xyz);
  }

  void add_random_bead() {
    if (get_num_vertices() >= num_max_beads_) {
      return;
    }
    if (split_edges_ && get_num_vertices() < 1000) {
      split_random_edge();
    }
    int num_iters = get_num_boundaries();
    if (grow_ends_) {
      for (int iter = 0; iter < num_iters; iter++) {
        grow_random_end();
      }
    }
    if (seed_new_edges_ && get_num_vertices() < 1000) {
      seed_random_edge();
    }
    if (seed_isolated_vertices_ && get_num_vertices() < 1000) {
      add_random_isolated_vertex();
    }
    // check_he_matrices();
  }
};

} // namespace meshbrane
