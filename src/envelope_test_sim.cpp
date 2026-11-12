/**
 * @file envelope_test_sim.cpp
 */

#include "meshbrane/envelope_test_sim.hpp"
#include "igl/opengl/glfw/Viewer.h" // igl::opengl::glfw::Viewer
#include "meshbrane/lennard_jones.hpp"
#include "meshbrane/math_utils.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

namespace meshbrane {

EnvelopeTestSim::EnvelopeTestSim(const fs::path &path_to_parameters)
    : SimulationBase(path_to_parameters) {

  printf("EnvelopeTestSim::EnvelopeTestSim\n");
  if (parameters_["dt_save"]) {
    printf("set dt_save_\n");
    dt_save_ = parameters_["dt_save"].as<double>();
  } else {
    throw std::runtime_error("No dt_save provided in parameters file");
  }
  if (parameters_["kBT"]) {
    printf("set kBT_\n");
    kBT_ = parameters_["kBT"].as<double>();
  } else {
    throw std::runtime_error("No kBT provided in parameters file");
  }

  if (parameters_["save_sim_data"]) {
    printf("set save_sim_data_\n");
    save_sim_data_ = parameters_["save_sim_data"].as<bool>();
  }
  if (parameters_["no_graph"]) {
    printf("set no_graph_\n");
    no_graph_ = parameters_["no_graph"].as<bool>();
  }

  if (parameters_["envelope"]) {
    printf("set envelope_\n");
    envelope_ = Membrane(&parameters_, "envelope");
    printf("Initialized envelope_\n");
  } else {
    throw std::runtime_error("Fatal error: no parameters found for Membrane");
  }

  init_concentric_spheres();
  // init_with_npcs();

  viewer_.sim_parameters_ = &parameters_;
  viewer_.sync_parameters();
  viewer_.init();
  printf("Initialized viewer_\n");

  // compute stuff for first time series sample
  envelope_.clear_interactions();
  printf("Zeroed envelope forces\n");
  apply_pair_interactions();
  printf("Applied interaction forces\n");

  envelope_.check_he_matrices();

  ////////////////////////////////////
  envelope_.update_cached_data();
  printf("Updated envelope cached data\n");
  envelope_.apply_internal_interactions();
  printf("Applied envelope internal forces\n");
}

// Core methods

void EnvelopeTestSim::print_info() {
  // printf("EnvelopeTestSim::print_info\n");

  printf(run_name_.c_str());
  printf("\n");
  printf("  t=%.10f\n", t_);
  printf("  dt=%.10f\n", dt_mean_);
  // printf("  Midpoint radius: %.10f\n", midpoint_radius_);
  envelope_.print_info();
  // printf("  Spindle length: %.10f\n", spindle_.length_);
  // viewer_.print_info();

  // double dt = envelope_.dt_;
  // double kBT = kBT_;
  // envelope_.run_force_mag_test(dt, kBT, rng_);
}

void EnvelopeTestSim::apply_pair_interactions() {

  //
  //
}

void EnvelopeTestSim::apply_internal_interactions() {
  envelope_.apply_internal_interactions();
}

void EnvelopeTestSim::clear_interactions() { envelope_.clear_interactions(); }

void EnvelopeTestSim::update_cached_data() {
  //
  envelope_.update_cached_data();
}

void EnvelopeTestSim::apply_thermal_fluctuations(double dt) {
  //
  envelope_.apply_thermal_fluctuations_to_self(dt, kBT_, rng_);
}

void EnvelopeTestSim::update_state_variables(double dt) {

  envelope_.update_own_state_variables(dt);
}

double EnvelopeTestSim::dt_max() {
  // printf("EnvelopeTestSim::dt_max\n");
  double dt_max = dt_max_;
  dt_max = std::min(dt_max, envelope_.dt_max());
  // dt_max = std::min(dt_max, spindle_.dt_max());
  return dt_max;
}

void EnvelopeTestSim::evolve_until(double t_end) {
  // printf("EnvelopeTestSim::evolve_until\n");
  int step = 0;
  dt_mean_ = 0.0;
  envelope_.total_edge_flips_ = 0;
  while (t_ < t_end) {
    /////////////////////
    /////////////////////
    clear_interactions();
    update_cached_data();
    apply_pair_interactions();
    apply_internal_interactions();
    double dt = dt_max();
    double dt_end = t_end - t_;
    dt = std::min(dt, dt_end);
    apply_thermal_fluctuations(dt);
    update_state_variables(dt);
    t_ += dt;
    /////////////////////
    /////////////////////

    dt_mean_ += dt;
    envelope_.total_edge_flips_ += envelope_.num_flips_;
    ++step;
  }
  dt_mean_ /= step;
}

void EnvelopeTestSim::evolve_until_next_frame() {
  // printf("EnvelopeTestSim::timestep\n");
  evolve_until(t_ + dt_frame_);

  envelope_.update_membrane_visuals();
}

void EnvelopeTestSim::save_frame() {
  fs::path frame_path = get_frame_path();
  viewer_.save_frame(frame_path);
  frame_count_++;
}

void EnvelopeTestSim::draw_scene() {
  viewer_.iglviewer_.data().clear();
  // std::vector<MatrixMesh *> meshes;
  viewer_.draw_wireframe(envelope_);
  // viewer_.draw_mesh(envelope_);

  // MatrixMesh m0 = MatrixMesh::icosohedron();
  // m0.init_cache_from_core_data();
  // m0.show_half_edges_ = true;
  //
  // MatrixMesh m1 = MatrixMesh::icosohedron();
  // m1.xyz_coord_V_ = m1.xyz_coord_V_.array() + 1.5;
  // m1.reverse_orientation();
  // m1.init_cache_from_core_data();
  // m1.show_half_edges_ = true;
  //
  // viewer_.draw_wireframe(m0);
  // viewer_.draw_wireframe(m1);
}

void EnvelopeTestSim::run_no_graph() {
  printf("Running EnvelopeTestSim::run_no_graph\n");
  while (true) {
    print_info();
    write_outputs();
    if (t_ >= T_run_) {
      return;
    }
    evolve_until_next_frame();
    // add_data_samples();
  }
}

void EnvelopeTestSim::run(int argc, char *argv[]) {
  printf("Running EnvelopeTestSim::run\n");

  if (no_graph_) {
    run_no_graph();
    return;
  }

  viewer_.iglviewer_.core().is_animating = true;
  viewer_.iglviewer_.callback_pre_draw =
      [&](igl::opengl::glfw::Viewer &viewer) {
        draw_scene();
        return false;
      };
  viewer_.iglviewer_.callback_post_draw =
      [&](igl::opengl::glfw::Viewer &viewer) {
        print_info();
        write_outputs();
        if (t_ >= T_run_) {
          viewer_.iglviewer_.core().is_animating = false;
          glfwSetWindowShouldClose(viewer_.iglviewer_.window, GL_TRUE);
          return true;
        }

        evolve_until_next_frame();
        // add_data_samples();
        return false;
      };
  viewer_.iglviewer_.launch(false, run_name_, viewer_.width_, viewer_.height_);
  make_a_movie();
}

fs::path EnvelopeTestSim::get_envelope_ply_path() {
  std::string frame_count_str = std::to_string(frame_count_);
  // pad with zeros so the index is always 6 digits
  frame_count_str =
      std::string(6 - frame_count_str.size(), '0') + frame_count_str;
  return raw_data_dir_ / ("envelope_" + frame_count_str + ".ply");
}

void EnvelopeTestSim::write_outputs() {

  if (!no_graph_) {
    fs::path frame_path = get_frame_path();
    viewer_.save_frame(frame_path);
  }
  // if (save_sim_data_) {
  //   double t0 = data_.t_.first();
  //   double t1 = data_.t_.last();
  //   double Dt = t1 - t0;
  //   if (dt_save_ < Dt) {
  //     data_.append_file();
  //   }
  // }

  // fs::path ply_path = get_envelope_ply_path();
  // envelope_.write_he_ply(ply_path);

  frame_count_++;
}

} // namespace meshbrane
