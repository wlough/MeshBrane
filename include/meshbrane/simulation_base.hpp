#pragma once

/**
 * @file simulation_base.hpp
 * @brief Base class for simulations
 */

#include <filesystem>
#include <string>
#include <yaml-cpp/yaml.h>

namespace meshbrane {
/**
 * @brief Base class for simulations. Makes output directory and loads parameter
 * file.
 *
 */
class SimulationBase {
public:
  YAML::Node parameters_;

  std::string run_name_;
  std::filesystem::path output_dir_;

  std::filesystem::path logs_dir_;
  std::filesystem::path raw_data_dir_;
  std::filesystem::path visualizations_dir_;
  std::filesystem::path temp_images_dir_;

  std::filesystem::path log_path_;

  int frame_count_ = 0;
  int frame_index_length_ = 6;
  std::string frame_prefix_ = "frame";

  double t_ = 0.0;
  double dt_;
  double dt0_;
  double dt_frame_;
  double T_run_;

  SimulationBase(const std::filesystem::path &path_to_parameters);

  /**
   * @brief Make output directories for logs/checkpoints/temp_images/etc...
   *
   * @param overwrite
   */
  void make_output_directory(bool overwrite = false);

  /**
   * @brief Make a movie from images in temp_images_dir_
   *
   */
  void make_a_movie();

  // /**
  //  * @brief Set additional parameters not set by SimulationBase
  //  *
  //  */
  // virtual void set_parameters() = 0;

  // /**
  //  * @brief Initialize objects in the simulation
  //  *
  //  */
  // virtual void initialize_sim() = 0;

  // virtual void timestep() = 0;

  // virtual void run_sim() = 0;

  void configure_logging();

  std::filesystem::path get_frame_path();
};

} // namespace meshbrane
