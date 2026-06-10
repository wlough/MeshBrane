#pragma once

/**
 * @file rigid_spindle_sim.hpp
 * @brief Defines RigidSpindleSim class and associated classes
 */

#include "meshbrane/membrane.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/rigid_spindle.hpp"
#include "meshbrane/simulation_base.hpp"
#include "meshbrane/viewer.hpp"
#include <cassert>
#include <cstdint>
#include <filesystem>

namespace meshbrane {

//
template <typename T> class TimeSeries {
public:
  TimeSeries() = default;
  TimeSeries(const std::string &save_path) : save_path_(save_path) {}
  void add_sample(const T &sample) { samples_.push_back(sample); }
  std::vector<T> samples_;
  std::string save_path_;
  T first() { return samples_.front(); }
  T last() { return samples_.back(); }
  void save_file() {
    std::ofstream file(save_path_, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open file for writing");
    }
    size_t size = samples_.size();
    file.write(reinterpret_cast<const char *>(&size), sizeof(size));
    file.write(reinterpret_cast<const char *>(samples_.data()),
               size * sizeof(T));
  }
  // void append_file() {
  //   std::ofstream file(save_path_, std::ios::binary | std::ios::app);
  //   if (!file) {
  //     throw std::runtime_error("Could not open file for appending");
  //   }
  //   size_t size = samples_.size();
  //   file.write(reinterpret_cast<const char *>(&size), sizeof(size));
  //   file.write(reinterpret_cast<const char *>(samples_.data()), size *
  //   sizeof(T));

  //   samples_.clear(); // Clear the data after appending
  // }
  void append_file() {
    // Open the file in read-write mode
    std::fstream file(save_path_,
                      std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
      throw std::runtime_error("Could not open file for appending");
    }

    // Read the current size from the header
    size_t current_size;
    file.read(reinterpret_cast<char *>(&current_size), sizeof(current_size));

    // Update the size header
    current_size += samples_.size();
    file.seekp(0, std::ios::beg); // Move to the beginning of the file
    file.write(reinterpret_cast<const char *>(&current_size),
               sizeof(current_size));

    // Append the new data
    file.seekp(0, std::ios::end); // Move to the end of the file
    file.write(reinterpret_cast<const char *>(samples_.data()),
               samples_.size() * sizeof(T));

    samples_.clear(); // Clear the data after appending
  }
  void load_file() {
    std::ifstream file(save_path_, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open file for reading");
    }
    size_t size;
    file.read(reinterpret_cast<char *>(&size), sizeof(size));
    samples_.resize(size);
    file.read(reinterpret_cast<char *>(samples_.data()), size * sizeof(T));
  }
};

// Write samples as: uint64_t n_samples; int64_t rows; int64_t cols; then
// n_samples * rows*cols doubles (row-major)
template <> void TimeSeries<Samples2d>::save_file() {
  std::ofstream file(save_path_, std::ios::binary);
  if (!file)
    throw std::runtime_error("Could not open file for writing");
  uint64_t n = static_cast<uint64_t>(samples_.size());
  file.write(reinterpret_cast<const char *>(&n), sizeof(n));

  int64_t rows = n ? samples_[0].rows() : 0;
  int64_t cols = 2;
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (const auto &M : samples_) {
    assert(M.rows() == rows && M.cols() == cols);
    // write row-major for easy numpy reshape
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> tmp = M;
    file.write(reinterpret_cast<const char *>(tmp.data()),
               sizeof(double) * rows * cols);
  }
}

template <> void TimeSeries<Samples2d>::append_file() {
  std::fstream file(save_path_,
                    std::ios::binary | std::ios::in | std::ios::out);
  if (!file)
    throw std::runtime_error("Could not open file for appending");

  uint64_t current_size;
  file.read(reinterpret_cast<char *>(&current_size), sizeof(current_size));
  uint64_t add = static_cast<uint64_t>(samples_.size());
  uint64_t new_size = current_size + add;

  file.seekp(0, std::ios::beg);
  file.write(reinterpret_cast<const char *>(&new_size), sizeof(new_size));
  file.seekp(0, std::ios::end);

  // assume rows/cols header already present and equal for all frames
  for (const auto &M : samples_) {
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> tmp = M;
    file.write(reinterpret_cast<const char *>(tmp.data()),
               sizeof(double) * tmp.rows() * tmp.cols());
  }
  samples_.clear();
}

template <> void TimeSeries<Samples2d>::load_file() {
  std::ifstream file(save_path_, std::ios::binary);
  if (!file)
    throw std::runtime_error("Could not open file for reading");
  uint64_t n;
  file.read(reinterpret_cast<char *>(&n), sizeof(n));
  int64_t rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));
  samples_.clear();
  samples_.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    Eigen::Matrix<double, Eigen::Dynamic, 2> M(rows, cols);
    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> tmp(rows, cols);
    file.read(reinterpret_cast<char *>(tmp.data()),
              sizeof(double) * rows * cols);
    M = tmp; // copy to column-major Samples2d
    samples_.push_back(std::move(M));
  }
}

template <> void TimeSeries<Eigen::Vector3d>::save_file() {
  std::ofstream file(save_path_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file for writing");
  }

  uint64_t n = static_cast<uint64_t>(samples_.size());
  int64_t rows = 3;
  int64_t cols = 1;

  file.write(reinterpret_cast<const char *>(&n), sizeof(n));
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (const auto &v : samples_) {
    double data[3] = {v[0], v[1], v[2]};
    file.write(reinterpret_cast<const char *>(data), sizeof(data));
  }
}

template <> void TimeSeries<Eigen::Vector3d>::append_file() {
  std::fstream file(save_path_,
                    std::ios::binary | std::ios::in | std::ios::out);
  if (!file) {
    throw std::runtime_error("Could not open file for appending");
  }

  uint64_t current_size;
  file.read(reinterpret_cast<char *>(&current_size), sizeof(current_size));

  int64_t rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  if (rows != 3 || cols != 1) {
    throw std::runtime_error(
        "TimeSeries<Eigen::Vector3d>: file shape mismatch");
  }

  uint64_t add = static_cast<uint64_t>(samples_.size());
  uint64_t new_size = current_size + add;

  file.seekp(0, std::ios::beg);
  file.write(reinterpret_cast<const char *>(&new_size), sizeof(new_size));

  file.seekp(0, std::ios::end);
  for (const auto &v : samples_) {
    double data[3] = {v[0], v[1], v[2]};
    file.write(reinterpret_cast<const char *>(data), sizeof(data));
  }

  samples_.clear();
}

template <> void TimeSeries<Eigen::Vector3d>::load_file() {
  std::ifstream file(save_path_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file for reading");
  }

  uint64_t n;
  int64_t rows, cols;

  file.read(reinterpret_cast<char *>(&n), sizeof(n));
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  if (rows != 3 || cols != 1) {
    throw std::runtime_error(
        "TimeSeries<Eigen::Vector3d>: file shape mismatch");
  }

  samples_.clear();
  samples_.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    double data[3];
    file.read(reinterpret_cast<char *>(data), sizeof(data));
    samples_.emplace_back(data[0], data[1], data[2]);
  }
}

//
//
//
//
//
//
template <> void TimeSeries<Samples1d>::save_file() {
  std::ofstream file(save_path_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file for writing");
  }

  uint64_t n = static_cast<uint64_t>(samples_.size());
  int64_t rows = n ? samples_[0].rows() : 0;
  int64_t cols = 1;

  file.write(reinterpret_cast<const char *>(&n), sizeof(n));
  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (const auto &v : samples_) {
    if (v.rows() != rows || v.cols() != cols) {
      throw std::runtime_error(
          "TimeSeries<Samples1d>: inconsistent sample shape");
    }
    file.write(reinterpret_cast<const char *>(v.data()),
               sizeof(double) * rows * cols);
  }
}

template <> void TimeSeries<Samples1d>::append_file() {
  std::fstream file(save_path_,
                    std::ios::binary | std::ios::in | std::ios::out);
  if (!file) {
    throw std::runtime_error("Could not open file for appending");
  }

  uint64_t current_size;
  file.read(reinterpret_cast<char *>(&current_size), sizeof(current_size));

  int64_t rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  if (cols != 1) {
    throw std::runtime_error("TimeSeries<Samples1d>: file shape mismatch");
  }

  for (const auto &v : samples_) {
    if (v.rows() != rows || v.cols() != cols) {
      throw std::runtime_error("TimeSeries<Samples1d>: sample shape mismatch");
    }
  }

  uint64_t add = static_cast<uint64_t>(samples_.size());
  uint64_t new_size = current_size + add;

  file.seekp(0, std::ios::beg);
  file.write(reinterpret_cast<const char *>(&new_size), sizeof(new_size));

  file.seekp(0, std::ios::end);
  for (const auto &v : samples_) {
    file.write(reinterpret_cast<const char *>(v.data()),
               sizeof(double) * rows * cols);
  }

  samples_.clear();
}

template <> void TimeSeries<Samples1d>::load_file() {
  std::ifstream file(save_path_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file for reading");
  }

  uint64_t n;
  int64_t rows, cols;

  file.read(reinterpret_cast<char *>(&n), sizeof(n));
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  if (cols != 1) {
    throw std::runtime_error("TimeSeries<Samples1d>: file shape mismatch");
  }

  samples_.clear();
  samples_.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    Samples1d v(rows, cols);
    file.read(reinterpret_cast<char *>(v.data()), sizeof(double) * rows * cols);
    samples_.push_back(std::move(v));
  }
}

class RigidSpindleSimData {
public:
  TimeSeries<double> t_;
  TimeSeries<double> length_;
  TimeSeries<double> length_dot_;
  TimeSeries<double> overlap_length_;
  TimeSeries<double> overlap_length_dot_;
  TimeSeries<double> extensile_force_;
  TimeSeries<double> compressive_force_;
  TimeSeries<double> envelope_compressive_force_;
  TimeSeries<double> midpoint_radius_;
  TimeSeries<Samples2d> zr_coords_V_;

  TimeSeries<Vec3d> mt_xyz_center_;
  // TimeSeries<Samples2d> mt_rotation_matrix_center_;
  TimeSeries<Vec3d> envelope_xyz_center_;
  TimeSeries<Samples1d> envelope_moments_;
  TimeSeries<double> spb_antipodality_;

  RigidSpindleSimData() = default;
  RigidSpindleSimData(const std::string &data_dir) {
    t_ = TimeSeries<double>(data_dir + "/t.dat");
    length_ = TimeSeries<double>(data_dir + "/mt_bundle_length.dat");
    length_dot_ = TimeSeries<double>(data_dir + "/mt_bundle_length_dot.dat");
    overlap_length_ = TimeSeries<double>(data_dir + "/mt_overlap_length.dat");
    overlap_length_dot_ =
        TimeSeries<double>(data_dir + "/mt_overlap_length_dot.dat");
    extensile_force_ = TimeSeries<double>(data_dir + "/extensile_force.dat");
    compressive_force_ =
        TimeSeries<double>(data_dir + "/compressive_force.dat");
    envelope_compressive_force_ =
        TimeSeries<double>(data_dir + "/envelope_compressive_force.dat");
    midpoint_radius_ =
        TimeSeries<double>(data_dir + "/envelope_midpoint_radius.dat");
    zr_coords_V_ = TimeSeries<Samples2d>(data_dir + "/envelope_zr_coords.dat");
    mt_xyz_center_ = TimeSeries<Vec3d>(data_dir + "/mt_xyz_center.dat");
    // mt_rotation_matrix_center_ =
    //     TimeSeries<Samples2d>(data_dir + "/mt_rotation_matrix_center.dat");
    envelope_xyz_center_ =
        TimeSeries<Vec3d>(data_dir + "/envelope_xyz_center.dat");
    envelope_moments_ =
        TimeSeries<Samples1d>(data_dir + "/envelope_moments.dat");
    spb_antipodality_ = TimeSeries<double>(data_dir + "/spb_antipodality.dat");
  }
  void save_file() {
    t_.save_file();
    length_.save_file();
    length_dot_.save_file();
    overlap_length_.save_file();
    overlap_length_dot_.save_file();
    extensile_force_.save_file();
    compressive_force_.save_file();
    envelope_compressive_force_.save_file();
    midpoint_radius_.save_file();
    zr_coords_V_.save_file();
    mt_xyz_center_.save_file();
    // mt_rotation_matrix_center_.save_file();
    spb_antipodality_.save_file();
    envelope_xyz_center_.save_file();
    envelope_moments_.save_file();
  }
  void append_file() {
    t_.append_file();
    length_.append_file();
    length_dot_.append_file();
    overlap_length_.append_file();
    overlap_length_dot_.append_file();
    extensile_force_.append_file();
    compressive_force_.append_file();
    envelope_compressive_force_.append_file();
    midpoint_radius_.append_file();
    zr_coords_V_.append_file();
    mt_xyz_center_.append_file();
    // mt_rotation_matrix_center_.append_file();
    spb_antipodality_.append_file();
    envelope_xyz_center_.append_file();
    envelope_moments_.append_file();
  }
  void clear() {
    t_.samples_.clear();
    length_.samples_.clear();
    length_dot_.samples_.clear();
    overlap_length_.samples_.clear();
    overlap_length_dot_.samples_.clear();
    extensile_force_.samples_.clear();
    compressive_force_.samples_.clear();
    envelope_compressive_force_.samples_.clear();
    midpoint_radius_.samples_.clear();
    zr_coords_V_.samples_.clear();
    mt_xyz_center_.samples_.clear();
    // mt_rotation_matrix_center_.samples_.clear();
    spb_antipodality_.samples_.clear();
    envelope_xyz_center_.samples_.clear();
    envelope_moments_.samples_.clear();
  }

  void make_output_directory(const std::string &output_dir) {
    // create the directory if it doesn't exist
    if (!std::filesystem::exists(output_dir)) {
      std::filesystem::create_directories(output_dir);
    }
  }
};

class RigidSpindleSim : public SimulationBase {
public:
  double dt_mean_{0.0};
  double dt_save_{1.0};
  double kBT_;
  double bulk_viscosity_;
  Membrane envelope_;
  RigidSpindle spindle_;

  double midpoint_radius_{0.0};

  RigidSpindleSimData data_;
  Samples2d zr_coords_V_;
  Vec3d envelope_xyz_center_;
  Samples1d envelope_moments_;
  double spb_antipodality_{0.0};
  // RigidSpindleSimData data_save_;

  void add_data_samples() {

    record_spindle_data();
    record_envelope_data();

    data_.spb_antipodality_.add_sample(spb_antipodality_);
    data_.midpoint_radius_.add_sample(midpoint_radius_);         // ***
    data_.zr_coords_V_.add_sample(zr_coords_V_);                 // ***
    data_.envelope_xyz_center_.add_sample(envelope_xyz_center_); // ***
    data_.envelope_moments_.add_sample(envelope_moments_);       // ***
    data_.t_.add_sample(t_);                                     // ***

    data_.length_.add_sample(spindle_.mt_bundle_.length_);
    data_.length_dot_.add_sample(spindle_.mt_bundle_.length_dot_);
    data_.overlap_length_.add_sample(spindle_.mt_bundle_.overlap_length_);
    data_.overlap_length_dot_.add_sample(
        spindle_.mt_bundle_.overlap_length_dot_);
    data_.extensile_force_.add_sample(
        spindle_.mt_bundle_.overlap_length_ *
        spindle_.mt_bundle_.motor_force_per_length_);
    data_.compressive_force_.add_sample(spindle_.mt_bundle_.F_compress_);
    Vec3d axis = spindle_.mt_bundle_.get_axis();
    double envelope_compressive_force =
        spindle_.spb2_.force_envelope_.dot(axis) -
        spindle_.spb1_.force_envelope_.dot(axis);
    data_.envelope_compressive_force_.add_sample(envelope_compressive_force);
    data_.mt_xyz_center_.add_sample(spindle_.mt_bundle_.xyz_center_);

    // data_.mt_rotation_matrix_center_.add_sample(
    //     spindle_.mt_bundle_.rotation_matrix_center_);
  }

  Viewer viewer_;
  bool spindle_force_on_{true};

  RigidSpindleSim(const std::string &path_to_parameters);

  // void set_parameters() override;
  // void initialize_sim() override;
  double dt_max();
  void timestep();
  void evolve_until(double t_end);
  void draw_scene();
  void run(int argc, char *argv[]);

  void save_frame();
  void write_outputs();
  void save_enevelope_state();
  std::string get_envelope_ply_path();

  void find_contact_patch1();
  void find_contact_patch2();
  void apply_interaction_forces();

  void print_info();
  // double get_
  void record_envelope_data();
  void record_spindle_data();
};

} // namespace meshbrane
