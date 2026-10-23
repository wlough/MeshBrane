/**
 * @file chromatin_test_sim.cpp
 */

#include "meshbrane/chromatin_test_sim.hpp"
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

ChromatinTestSim::ChromatinTestSim(const fs::path &path_to_parameters)
    : SimulationBase(path_to_parameters) {

  printf("ChromatinTestSim::ChromatinTestSim\n");
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

  if (parameters_["chromatin"]) {
    printf("set spindle_\n");
    chromatin_ = ChromatinChain(&parameters_, "chromatin");
    printf("Initialized spindle_\n");
  } else {
    throw std::runtime_error(
        "Fatal error: no parameters found for ChromatinChain");
  }
  if (parameters_["envelope"]) {
    printf("set envelope_\n");
    envelope_ = Membrane(&parameters_, "envelope");
    printf("Initialized envelope_\n");
  } else {
    throw std::runtime_error("Fatal error: no parameters found for Membrane");
  }

  viewer_.sim_parameters_ = &parameters_;
  viewer_.sync_parameters();
  viewer_.init();
  printf("Initialized viewer_\n");

  // compute stuff for first time series sample
  chromatin_.clear_interactions();
  printf("Zeroed chromatin forces\n");
  envelope_.clear_interactions();
  printf("Zeroed envelope forces\n");
  apply_pair_interactions();
  printf("Applied interaction forces\n");

  envelope_.check_he_matrices();
  chromatin_.check_he_matrices();

  ////////////////////////////////////
  envelope_.update_cached_data();
  printf("Updated envelope cached data\n");
  envelope_.apply_internal_interactions();
  printf("Applied envelope internal forces\n");
  chromatin_.apply_internal_interactions();
  printf("Applied chromatin internal forces\n");
  // chromatin_.compute_velocities();
  // printf("Computed chromatin velocities\n");
}

// Core methods

void ChromatinTestSim::print_info() {
  // printf("ChromatinTestSim::print_info\n");

  printf(run_name_.c_str());
  printf("\n");
  printf("  t=%.10f\n", t_);
  printf("  dt=%.10f\n", dt_mean_);
  // printf("  Midpoint radius: %.10f\n", midpoint_radius_);
  envelope_.print_info();
  // printf("  Spindle length: %.10f\n", spindle_.length_);
  chromatin_.print_info();
  // viewer_.print_info();

  // double dt = envelope_.dt_;
  // double kBT = kBT_;
  // envelope_.run_force_mag_test(dt, kBT, rng_);
}

void ChromatinTestSim::apply_pair_interactions() { return; }

void ChromatinTestSim::apply_internal_interactions() {
  // envelope_.test_surface_tension();
  envelope_.apply_internal_interactions();
  chromatin_.apply_internal_interactions();
}

void ChromatinTestSim::clear_interactions() {
  chromatin_.clear_interactions();
  envelope_.clear_interactions();
}

void ChromatinTestSim::update_cached_data() {
  //
  envelope_.update_cached_data();
  chromatin_.update_cached_data();
}

void ChromatinTestSim::apply_thermal_fluctuations(double dt) {
  //
  envelope_.apply_thermal_fluctuations_to_self(dt, kBT_, rng_);
  chromatin_.apply_thermal_fluctuations_to_self(dt, kBT_, rng_);
}

void ChromatinTestSim::update_state_variables(double dt) {
  chromatin_.update_own_state_variables(dt);
  envelope_.update_own_state_variables(dt);
}

// Helpers

// void ChromatinTestSim::record_envelope_data() {
//
//   int Nv = envelope_.get_num_vertices();
//   envelope_moments_.resize(3, 1);
//   envelope_moments_.setZero();
//   double A = envelope_.get_num_faces() * envelope_.average_face_area_;
//   // Samples1d area_V = envelope_.get_area_V();
//   double variance_z = 0.0;
//   double skewness_z = 0.0;
//   double kurtosis_z = 0.0;
//
//   Samples1d z_coord_V(Nv, 1);
//
//   envelope_xyz_center_.setZero();
//
//
//   Vec3d Ez = spindle_.mt_bundle_.rotation_matrix_center_.col(2);
//   for (int v = 0; v < Nv; v++) {
//     Vec3d r_NE = envelope_.xyz_coord_v(v);
//     Vec3d r_Spindle = spindle_.mt_bundle_.xyz_center_;
//     double dA = envelope_.area_V_[v];
//     Vec3d dr = r_NE - r_Spindle;
//     double z = math::dot(dr, Ez);
//     // throw std::runtime_error(std::to_string(z));
//     z_coord_V[v] = z;
//     envelope_xyz_center_ += r_NE * dA / A;
//   }
//
//   for (int v = 0; v < Nv; v++) {
//     double dA = envelope_.area_V_[v];
//     double z = z_coord_V[v];
//     variance_z += z * z * dA / A;
//     skewness_z += z * z * z * dA / A;
//     kurtosis_z += z * z * z * z * dA / A;
//   }
//   double sdev = std::sqrt(variance_z);
//   skewness_z /= variance_z * sdev;
//   kurtosis_z /= variance_z * variance_z;
//   envelope_moments_ << variance_z, skewness_z, kurtosis_z;
//
//   // // // // // //
//   // midpoint_radius_
//   // zr_coords_V_
//   zr_coords_V_.resize(Nv, 2);
//   double r_midpt;
//   double abs_z_midpt = std::numeric_limits<double>::max();
//   int v_midpt;
//   for (int v = 0; v < Nv; v++) {
//     Vec3d p1 = envelope_.xyz_coord_v(v);
//     Vec3d xyz = p1 - spindle_.mt_bundle_.xyz_center_;
//     double z = xyz.dot(Ez);
//     Vec3d r_vec = xyz - z * Ez;
//     double r = r_vec.norm();
//     zr_coords_V_.row(v) << z, r; // **
//     // // get midpoint stuff
//     double abs_z = std::abs(z);
//     if (abs_z < abs_z_midpt) {
//       abs_z_midpt = abs_z;
//       r_midpt = r;
//       v_midpt = v;
//     }
//   }
//   midpoint_radius_ = r_midpt;
//
//   // // // // // //
//   // spb_antipodality_
//   Vec3d r_plus = spindle_.spb1_.xyz_frame_ - envelope_xyz_center_;
//   Vec3d r_minus = spindle_.spb2_.xyz_frame_ - envelope_xyz_center_;
//   double mag_r_plus = math::L2norm(r_plus);
//   double mag_r_minus = math::L2norm(r_minus);
//   spb_antipodality_ =
//       0.5 * (1 - math::dot(r_plus, r_minus) / (mag_r_plus * mag_r_minus));
// }

// void ChromatinTestSim::record_chromatin_data() {}

double ChromatinTestSim::dt_max() {
  // printf("ChromatinTestSim::dt_max\n");
  double dt_max = dt_max_;
  dt_max = std::min(dt_max, envelope_.dt_max());
  // dt_max = std::min(dt_max, spindle_.dt_max());
  return dt_max;
}

void ChromatinTestSim::evolve_until(double t_end) {
  // printf("ChromatinTestSim::evolve_until\n");
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

void ChromatinTestSim::evolve_until_next_frame() {
  // printf("ChromatinTestSim::timestep\n");
  evolve_until(t_ + dt_frame_);

  envelope_.update_membrane_visuals();
}

void ChromatinTestSim::save_frame() {
  fs::path frame_path = get_frame_path();
  viewer_.save_frame(frame_path);
  frame_count_++;
}

void ChromatinTestSim::draw_scene() {
  viewer_.iglviewer_.data().clear();
  std::vector<MatrixMesh *> meshes;
  viewer_.draw_wireframe(envelope_);

  viewer_.draw_chromatin_chain(chromatin_);
}

void ChromatinTestSim::run_no_graph() {
  printf("Running ChromatinTestSim::run_no_graph\n");
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

void ChromatinTestSim::run(int argc, char *argv[]) {
  printf("Running ChromatinTestSim::run\n");

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

fs::path ChromatinTestSim::get_envelope_ply_path() {
  std::string frame_count_str = std::to_string(frame_count_);
  // pad with zeros so the index is always 6 digits
  frame_count_str =
      std::string(6 - frame_count_str.size(), '0') + frame_count_str;
  return raw_data_dir_ / ("envelope_" + frame_count_str + ".ply");
}

void ChromatinTestSim::write_outputs() {

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
