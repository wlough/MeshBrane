#pragma once

/**
 * @file chromatin_test_sim.hpp
 * @brief Defines ChromatinTestSim class and associated classes
 */

#include "meshbrane/chromatin_chain.hpp"
#include "meshbrane/kmc.hpp"
#include "meshbrane/membrane.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/simulation_base.hpp"
#include "meshbrane/viewer.hpp"
#include <cassert>
#include <cstdint>
#include <filesystem>

namespace meshbrane {

class ChromatinTestSim : public SimulationBase {
public:
  // core
  kmc::RandomNumberGenerator rng_;
  double kBT_;
  double bulk_viscosity_;
  double dt_save_{1.0};
  Viewer viewer_;

  bool save_sim_data_{true};

  Membrane envelope_;
  ChromatinChain chromatin_;

  double dt_mean_{0.0};
  double midpoint_radius_{0.0};
  Samples2d zr_coords_V_;
  Vec3d envelope_xyz_center_;
  Samples1d envelope_moments_;

  bool no_graph_{false};

  ChromatinTestSim(const std::filesystem::path &path_to_parameters);

  double dt_max();
  void evolve_until_next_frame();
  void evolve_until(double t_end);
  void draw_scene();
  void run(int argc, char *argv[]);
  void run_no_graph();

  void save_frame();
  void write_outputs();
  void save_enevelope_state();
  std::filesystem::path get_envelope_ply_path();

  // void find_contact_patch1();
  // void find_contact_patch2();
  // void record_envelope_data();
  // void record_spindle_data();

  //
  //
  //
  //
  //
  //
  //
  void print_info();
  void clear_interactions();
  void update_cached_data();
  void apply_pair_interactions();
  void apply_internal_interactions();
  void apply_thermal_fluctuations(double dt);
  void update_state_variables(double dt);
};

} // namespace meshbrane
