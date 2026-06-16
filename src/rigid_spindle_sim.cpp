/**
 * @file rigid_spindle_sim.cpp
 */

#include "meshbrane/rigid_spindle_sim.hpp"
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

RigidSpindleSim::RigidSpindleSim(const fs::path &path_to_parameters)
    : SimulationBase(path_to_parameters) {
  printf("RigidSpindleSim::RigidSpindleSim\n");
  // moved to base class...
  // printf("set dt0_\n");
  // dt0_ = parameters_["dt"].as<double>();
  // printf("set dt_frame_\n");
  // dt_frame_ = parameters_["dt_frame"].as<double>();
  // printf("set T_run_\n");
  // T_run_ = parameters_["T_run"].as<double>();
  printf("set dt_save_\n");
  dt_save_ = parameters_["dt_save"].as<double>();
  printf("set kBT_\n");
  kBT_ = parameters_["kBT"].as<double>();

  if (parameters_["spindle"]) {
    printf("set spindle_\n");
    spindle_ = RigidSpindle(&parameters_, "spindle");
    printf("Initialized spindle_\n");
  } else {
    throw std::runtime_error(
        "Fatal error: no parameters found for RigidSpindle");
  }
  if (parameters_["envelope"]) {
    printf("set envelope_\n");
    envelope_ = Membrane(&parameters_, "envelope");
    printf("Initialized envelope_\n");
  } else {
    throw std::runtime_error("Fatal error: no parameters found for Membrane");
  }

  find_contact_patch1();
  printf("Found contact_patch1\n");
  find_contact_patch2();
  printf("Found contact_patch2\n");

  viewer_.sim_parameters_ = &parameters_;
  viewer_.sync_parameters();
  viewer_.init();
  printf("Initialized viewer_\n");

  // compute stuff for first time series sample
  spindle_.zero_forces();
  printf("Zeroed spindle forces\n");
  envelope_.clear_interactions();
  printf("Zeroed envelope forces\n");
  apply_pair_interactions();
  printf("Applied interaction forces\n");
  ////////////////////////////////////
  // size_t Nh = envelope_.get_num_half_edges();
  // std::cout << "Nh=" << Nh << '\n';
  // for (size_t h = 0; h < Nh; h++) {
  //   int ht = envelope_.h_twin_h(h);
  //   int hn = envelope_.h_next_h(h);

  //   if (ht < 0 || ht >= Nh) {
  //     std::cout << "h=" << h << '\n';
  //     std::cout << "ht=" << ht << '\n';
  //     throw std::out_of_range("twin index out of range");
  //   }
  //   if (hn < 0 || hn >= Nh) {
  //     std::cout << "h=" << h << '\n';
  //     std::cout << "hn=" << hn << '\n';
  //     throw std::out_of_range("next index out of range");
  //   }
  // }

  // size_t Nv = envelope_.get_num_vertices();
  // std::cout << "Nv=" << Nv << '\n';
  // for (size_t v = 0; v < Nv; v++) {
  //   int h_out = envelope_.h_out_v(v);
  //   if (h_out < 0 || h_out >= Nh) {
  //     std::cout << "v=" << v << '\n';
  //     std::cout << "h_out=" << h_out << '\n';
  //     throw std::out_of_range("h_out index out of range");
  //   }
  // }
  envelope_.check_he_matrices();

  ////////////////////////////////////
  envelope_.update_cached_data();
  printf("Updated envelope cached data\n");
  envelope_.apply_internal_interactions();
  printf("Applied envelope internal forces\n");
  spindle_.apply_internal_forces();
  printf("Applied internal spindle forces\n");
  spindle_.compute_velocities();
  printf("Computed spindle velocities\n");
  //
  data_ = RigidSpindleSimData(raw_data_dir_);
  printf("Initialized data_\n");
  add_data_samples();
  printf("Added data samples\n");
  data_.save_file();
  data_.clear();
}

// Core methods

void RigidSpindleSim::print_info() {
  // printf("RigidSpindleSim::print_info\n");

  printf(run_name_.c_str());
  printf("\n");
  printf("  t=%.10f\n", t_);
  printf("  dt=%.10f\n", dt_mean_);
  printf("  Midpoint radius: %.10f\n", midpoint_radius_);
  envelope_.print_info();
  // printf("  Spindle length: %.10f\n", spindle_.length_);
  spindle_.print_info();
  // viewer_.print_info();
}

void RigidSpindleSim::apply_pair_interactions() {
  // printf("RigidSpindleSim::apply_pair_interactions\n");
  // spindle_.envelope_force_ = 0.0;
  ///////////////////////////
  if (!spindle_force_on_) {
    return;
  }
  //////////////////////////
  //////////////////////////
  //////////////////////////
  int Nplus = envelope_.spb_patch_plus_.V_.size();
  int Nminus = envelope_.spb_patch_minus_.V_.size();
  if (Nplus == 0) {
    find_contact_patch1();
  }
  if (Nminus == 0) {
    find_contact_patch2();
  }
  int Nspb1 = spindle_.spb1_.get_num_vertices();
  int Nspb2 = spindle_.spb2_.get_num_vertices();
  double sigma1 = 0.5 * (spindle_.spb1_.wca_sigma_ + envelope_.wca_sigma_);
  double epsilon1 = spindle_.spb1_.wca_epsilon_;
  for (int v1 : envelope_.spb_patch_plus_.V_) {
    Vec3d x1 = envelope_.xyz_coord_v(v1);
    for (int v2{0}; v2 < Nspb1; v2++) {
      Vec3d x2 = spindle_.spb1_.xyz_coord_v(v2);
      Vec3d F = wca_force(x1, x2, epsilon1, sigma1);
      envelope_.force_V_.row(v1) += F;
      spindle_.spb1_.force_V_.row(v2) -= F;
    }
  }
  double sigma2 = 0.5 * (spindle_.spb2_.wca_sigma_ + envelope_.wca_sigma_);
  double epsilon2 = spindle_.spb2_.wca_epsilon_;
  for (int v1 : envelope_.spb_patch_minus_.V_) {
    Vec3d x1 = envelope_.xyz_coord_v(v1);
    for (int v2{0}; v2 < Nspb2; v2++) {
      Vec3d x2 = spindle_.spb2_.xyz_coord_v(v2);
      Vec3d F = wca_force(x1, x2, epsilon2, sigma2);
      envelope_.force_V_.row(v1) += F;
      spindle_.spb2_.force_V_.row(v2) -= F;
    }
  }
  // ***
  // int Nspb1 = spindle_.spb1_.get_num_vertices();
  // int Nspb2 = spindle_.spb2_.get_num_vertices();
  // double sigma1 = 0.5 * (spindle_.spb1_.wca_sigma_ + envelope_.wca_sigma_);
  // double epsilon1 = spindle_.spb1_.wca_epsilon_;
  // for (int v1 = 0; v1 < envelope_.get_num_vertices(); v1++) {
  //   Vec3d x1 = envelope_.xyz_coord_v(v1);
  //   for (int v2{0}; v2 < Nspb1; v2++) {
  //     Vec3d x2 = spindle_.spb1_.xyz_coord_v(v2);
  //     Vec3d F = wca_force(x1, x2, epsilon1, sigma1);
  //     envelope_.force_V_.row(v1) += F;
  //     spindle_.spb1_.force_V_.row(v2) -= F;
  //   }
  // }
  // double sigma2 = 0.5 * (spindle_.spb2_.wca_sigma_ + envelope_.wca_sigma_);
  // double epsilon2 = spindle_.spb2_.wca_epsilon_;
  // for (int v1 = 0; v1 < envelope_.get_num_vertices(); v1++) {
  //   Vec3d x1 = envelope_.xyz_coord_v(v1);
  //   for (int v2{0}; v2 < Nspb2; v2++) {
  //     Vec3d x2 = spindle_.spb2_.xyz_coord_v(v2);
  //     Vec3d F = wca_force(x1, x2, epsilon2, sigma2);
  //     envelope_.force_V_.row(v1) += F;
  //     spindle_.spb2_.force_V_.row(v2) -= F;
  //   }
  // }
  //////////////////////////
  //////////////////////////
  //////////////////////////

  spindle_.spb1_.sum_envelope_force_V();
  spindle_.spb2_.sum_envelope_force_V();

  ///////////////////////////////////////////////
  ///////////////////////////////////////////////
  ///////////////////////////////////////////////
  double sigma = 0.5 * (spindle_.mt_bundle_.wca_sigma_ + envelope_.wca_sigma_);
  double epsilon = spindle_.mt_bundle_.wca_epsilon_;
  int Nv = envelope_.get_num_vertices();
  Vec3d o = spindle_.mt_bundle_.xyz_center_;
  Vec3d ex = spindle_.mt_bundle_.rotation_matrix_center_.col(0);
  Vec3d ey = spindle_.mt_bundle_.rotation_matrix_center_.col(1);
  Vec3d ez = spindle_.mt_bundle_.rotation_matrix_center_.col(2);
  double z_min = -0.5 * spindle_.mt_bundle_.length_;
  double z_max = 0.5 * spindle_.mt_bundle_.length_;
  double r_max = spindle_.mt_bundle_.interaction_radius_;

  for (int v = 0; v < Nv; v++) {
    Vec3d p1 = envelope_.xyz_coord_v(v);
    Vec3d xyz = p1 - o;
    double z = xyz.dot(ez);
    Vec3d r_vec = xyz - z * ez;
    double r = r_vec.norm();
    if (z < z_min || z > z_max || r > r_max) {
      continue;
    }
    Vec3d r_unit = r_vec / r;
    Vec3d p2 = o + z * ez + spindle_.mt_bundle_.radius_ * r_unit;
    Vec3d F = wca_force(p1, p2, epsilon, sigma);
    envelope_.force_V_.row(v) += F;
    spindle_.mt_bundle_.force_envelope_ -= F;
    Vec3d l_vec = p2 - o;
    Vec3d T = -math::cross(l_vec, F);
    spindle_.mt_bundle_.torque_envelope_ += T;
  }
}

void RigidSpindleSim::apply_internal_interactions() {
  envelope_.apply_internal_interactions();
  spindle_.apply_internal_forces();
  spindle_.compute_velocities();
}

void RigidSpindleSim::clear_interactions() {
  spindle_.zero_forces();
  envelope_.clear_interactions();
}

void RigidSpindleSim::update_cached_data() {
  //
  envelope_.update_cached_data();
}

void RigidSpindleSim::apply_thermal_fluctuations(double dt) {
  //
  envelope_.apply_thermal_fluctuations(dt, kBT_, rng_);
  // spindle_.apply_thermal_fluctuations(dt, kBT_, rng_);
}

void RigidSpindleSim::update_state_variables(double dt) {
  spindle_.update_state_variables(dt_);
  envelope_.update_state_variables(dt_);
}

// Helpers

void RigidSpindleSim::find_contact_patch1() {
  Vec3d p = spindle_.spb1_.xyz_frame_; // point on spindle axis
  // Vec3d u = spindle_.axis_;
  double r_max = 1.25 * spindle_.spb1_.interaction_radius_;
  // double contact_length = 2 * r_max;
  // envelope_.spb_patch_plus_ =
  //     Patch::from_cylinder(&envelope_, p, u, r_max, contact_length);
  envelope_.spb_patch_plus_ = Patch::from_ball(&envelope_, p, r_max);

  // envelope_.spb_patch_plus_.rgba_face_ = RGBA_DICT.at("meshbrane_green");
  // envelope_.spb_patch_plus_.rgba_edge_ = RGBA_DICT.at("meshbrane_green");
  // printf("Found seed vertices for SPB contact patch 1\n");
}

void RigidSpindleSim::find_contact_patch2() {
  Vec3d p = spindle_.spb2_.xyz_frame_; // point on spindle axis
  // Vec3d u = -spindle_.axis_;
  double r_max = 1.25 * spindle_.spb2_.interaction_radius_;
  // double contact_length = 2 * r_max;
  // envelope_.spb_patch_minus_ =
  //     Patch::from_cylinder(&envelope_, p, u, r_max, contact_length);
  envelope_.spb_patch_minus_ = Patch::from_ball(&envelope_, p, r_max);

  // printf("Found seed vertices for SPB contact patch 2\n");
}

void RigidSpindleSim::record_envelope_data() {

  int Nv = envelope_.get_num_vertices();
  envelope_moments_.resize(3, 1);
  envelope_moments_.setZero();
  double A = envelope_.get_num_faces() * envelope_.average_face_area_;
  // Samples1d area_V = envelope_.get_area_V();
  double variance_z = 0.0;
  double skewness_z = 0.0;
  double kurtosis_z = 0.0;

  Samples1d z_coord_V(Nv, 1);

  envelope_xyz_center_.setZero();

  ////////////////////////////////////
  // compute z_coord_V
  ////////////////////////////////////
  // Vec3d ex = spindle_.mt_bundle_.rotation_matrix_center_.col(0);
  // Vec3d ey = spindle_.mt_bundle_.rotation_matrix_center_.col(1);
  // Vec3d ez = spindle_.mt_bundle_.rotation_matrix_center_.col(2);
  // Eigen::Vector3d mean_r{0, 0, 0};
  // Eigen::Matrix<double, 3, 3> var_r{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  //
  // for (int v = 0; v < Nv; v++) {
  //   Vec3d r = envelope_.xyz_coord_v(v);
  //   double dA = envelope_.area_V_[v];
  //   mean_r += r * dA / A;
  // }
  // for (int v = 0; v < Nv; v++) {
  //   Vec3d r = envelope_.xyz_coord_v(v);
  //   Vec3d dr = r - mean_r;
  //   double dA = envelope_.area_V_[v];
  //   var_r += dr * dr.transpose() * dA / A;
  // }
  // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(var_r);
  // if (solver.info() != Eigen::Success) {
  //   throw std::runtime_error("Eigen decomposition failed");
  // }
  // Eigen::Vector3d eigenvalues = solver.eigenvalues();
  // Eigen::Matrix3d eigenvectors = solver.eigenvectors();
  //
  // // compute angles between eigenvectors and existing vector `ez`
  // // sort eigenvectors by angle
  // // store column indices 0,1,2
  // std::array<int, 3> idx = {0, 1, 2};
  //
  // // store actual angles
  // std::array<double, 3> angles;
  //
  // // compute angles
  // for (int i = 0; i < 3; ++i) {
  //   double c = eigenvectors.col(i).dot(ez);
  //   if (c < 0) {
  //     c *= -1;
  //   }
  //   c = std::clamp(c, -1.0, 1.0);
  //   angles[i] = std::acos(c); // angle in radians
  // }
  //
  // // sort by increasing angle
  // std::sort(idx.begin(), idx.end(),
  //           [&](int a, int b) { return angles[a] < angles[b]; });
  //
  // // reorder eigenvalues/eigenvectors
  // Eigen::Vector3d eigenvalues_sorted;
  // Eigen::Matrix3d eigenvectors_sorted;
  // std::array<double, 3> angles_sorted;
  // for (int k = 0; k < 3; ++k) {
  //   eigenvalues_sorted(k) = eigenvalues(idx[k]);
  //   eigenvectors_sorted.col(k) = eigenvectors.col(idx[k]);
  //   angles_sorted[k] = angles[idx[k]];
  // }
  // Vec3d Ez = eigenvectors_sorted.col(0);
  // if (math::dot(ez, Ez) < 0) {
  //   Ez *= -1;
  // }
  // Vec3d Ex = eigenvectors_sorted.col(1);
  // Vec3d Ey = eigenvectors_sorted.col(2);
  // if (math::dot(Ez, math::cross(Ex, Ey)) < 0) {
  //   Ex = eigenvectors_sorted.col(2);
  //   Ey = eigenvectors_sorted.col(1);
  // }
  // for (int v = 0; v < Nv; v++) {
  //   Vec3d r = envelope_.xyz_coord_v(v);
  //   double dA = envelope_.area_V_[v];
  //   Vec3d dr = r - mean_r;
  //   double z = math::dot(dr, Ez);
  //   // throw std::runtime_error(std::to_string(z));
  //   z_coord_V[v] = z;
  // }
  ////////////////////////////////////
  ////////////////////////////////////
  Vec3d Ez = spindle_.mt_bundle_.rotation_matrix_center_.col(2);
  for (int v = 0; v < Nv; v++) {
    Vec3d r_NE = envelope_.xyz_coord_v(v);
    Vec3d r_Spindle = spindle_.mt_bundle_.xyz_center_;
    double dA = envelope_.area_V_[v];
    Vec3d dr = r_NE - r_Spindle;
    double z = math::dot(dr, Ez);
    // throw std::runtime_error(std::to_string(z));
    z_coord_V[v] = z;
    envelope_xyz_center_ += r_NE * dA / A;
  }

  for (int v = 0; v < Nv; v++) {
    double dA = envelope_.area_V_[v];
    double z = z_coord_V[v];
    variance_z += z * z * dA / A;
    skewness_z += z * z * z * dA / A;
    kurtosis_z += z * z * z * z * dA / A;
  }
  double sdev = std::sqrt(variance_z);
  skewness_z /= variance_z * sdev;
  kurtosis_z /= variance_z * variance_z;
  envelope_moments_ << variance_z, skewness_z, kurtosis_z;

  // // // // // //
  // midpoint_radius_
  // zr_coords_V_
  zr_coords_V_.resize(Nv, 2);
  double r_midpt;
  double abs_z_midpt = std::numeric_limits<double>::max();
  int v_midpt;
  for (int v = 0; v < Nv; v++) {
    Vec3d p1 = envelope_.xyz_coord_v(v);
    Vec3d xyz = p1 - spindle_.mt_bundle_.xyz_center_;
    double z = xyz.dot(Ez);
    Vec3d r_vec = xyz - z * Ez;
    double r = r_vec.norm();
    zr_coords_V_.row(v) << z, r; // **
    // // get midpoint stuff
    double abs_z = std::abs(z);
    if (abs_z < abs_z_midpt) {
      abs_z_midpt = abs_z;
      r_midpt = r;
      v_midpt = v;
    }
  }
  midpoint_radius_ = r_midpt;

  // // // // // //
  // spb_antipodality_
  Vec3d r_plus = spindle_.spb1_.xyz_frame_ - envelope_xyz_center_;
  Vec3d r_minus = spindle_.spb2_.xyz_frame_ - envelope_xyz_center_;
  double mag_r_plus = math::L2norm(r_plus);
  double mag_r_minus = math::L2norm(r_minus);
  spb_antipodality_ =
      0.5 * (1 - math::dot(r_plus, r_minus) / (mag_r_plus * mag_r_minus));
}

void RigidSpindleSim::record_spindle_data() {}

double RigidSpindleSim::dt_max() {
  // printf("RigidSpindleSim::dt_max\n");
  double dt_max = dt0_;
  dt_max = std::min(dt_max, envelope_.dt_max());
  // dt_max = std::min(dt_max, spindle_.dt_max());
  return dt_max;
}

void RigidSpindleSim::evolve_until(double t_end) {
  // printf("RigidSpindleSim::evolve_until\n");
  int step = 0;
  dt_mean_ = 0.0;
  envelope_.total_edge_flips_ = 0;
  while (t_ < t_end) {
    clear_interactions();
    update_cached_data();
    apply_pair_interactions();
    apply_internal_interactions();

    dt_ = dt_max();
    double dt_end = t_end - t_;
    dt_ = std::min(dt_, dt_end);

    apply_thermal_fluctuations(dt_);

    update_state_variables(dt_);
    t_ += dt_;

    dt_mean_ += dt_;
    envelope_.total_edge_flips_ += envelope_.num_flips_;
    ++step;
  }
  dt_mean_ /= step;
}

void RigidSpindleSim::evolve_until_next_frame() {
  // printf("RigidSpindleSim::timestep\n");
  evolve_until(t_ + dt_frame_);

  Vec3d x1 = spindle_.spb1_.xyz_frame_;
  double r1 = 1.25 * spindle_.spb1_.interaction_radius_;

  Vec3d x2 = spindle_.spb2_.xyz_frame_;
  double r2 = 1.25 * spindle_.spb2_.interaction_radius_;

  envelope_.spb_patch_plus_.move_towards_sphere(x1, r1);  // ***
  envelope_.spb_patch_minus_.move_towards_sphere(x2, r2); // ***

  envelope_.update_membrane_visuals();
}

void RigidSpindleSim::save_frame() {
  fs::path frame_path = get_frame_path();
  viewer_.save_frame(frame_path);
  frame_count_++;
}

void RigidSpindleSim::draw_scene() {
  viewer_.iglviewer_.data().clear();
  std::vector<MatrixMesh *> meshes;
  if (envelope_.draw_wireframe_) {
    viewer_.draw_wireframe(envelope_);
  } else {
    meshes.push_back(&envelope_);
  }
  if (spindle_.spb1_.draw_wireframe_) {
    viewer_.draw_wireframe(spindle_.spb1_);
  } else {
    meshes.push_back(&spindle_.spb1_);
  }
  if (spindle_.spb2_.draw_wireframe_) {
    viewer_.draw_wireframe(spindle_.spb2_);
  } else {
    meshes.push_back(&spindle_.spb2_);
  }
  if (meshes.size() > 0) {
    viewer_.draw_meshes(meshes);
  }
  viewer_.draw_rigid_mt_bundle(spindle_.mt_bundle_);
  if (spindle_.draw_axes_) {
    viewer_.draw_mt_bundle_spb_axes(spindle_.mt_bundle_, spindle_.spb1_,
                                    spindle_.spb2_);
  }
}

void RigidSpindleSim::run(int argc, char *argv[]) {
  printf("Running RigidSpindleSim::run\n");
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
        add_data_samples();
        return false;
      };
  viewer_.iglviewer_.launch(false, run_name_, viewer_.width_, viewer_.height_);
  make_a_movie();
}

fs::path RigidSpindleSim::get_envelope_ply_path() {
  std::string frame_count_str = std::to_string(frame_count_);
  // pad with zeros so the index is always 6 digits
  frame_count_str =
      std::string(6 - frame_count_str.size(), '0') + frame_count_str;
  return raw_data_dir_ / ("envelope_" + frame_count_str + ".ply");
}

void RigidSpindleSim::write_outputs() {
  fs::path frame_path = get_frame_path();
  viewer_.save_frame(frame_path);

  double t0 = data_.t_.first();
  double t1 = data_.t_.last();
  double Dt = t1 - t0;
  if (dt_save_ < Dt) {
    data_.append_file();
  }

  // fs::path ply_path = get_envelope_ply_path();
  // envelope_.write_he_ply(ply_path);

  frame_count_++;
}

} // namespace meshbrane
