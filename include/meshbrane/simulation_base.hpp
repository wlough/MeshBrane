#pragma once

/**
 * @file simulation_base.hpp
 * @brief Base class for simulations
 */

#include <filesystem>
#include <string>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace meshbrane {
/**
 * @brief Base class for simulations. Makes output directory and loads parameter
 * file.
 *
 */
class SimulationBase {
public:
  YAML::Node parameters_;
  std::string output_dir_;
  std::string run_name_;
  std::string input_dir_;
  std::string logs_dir_;
  std::string checkpoints_dir_;
  std::string raw_data_dir_;
  std::string processed_data_dir_;
  std::string visualizations_dir_;
  std::string temp_images_dir_;
  std::string input_path_;
  std::string log_path_;
  std::string checkpoint_path_;
  std::string raw_data_path_;
  std::string processed_data_path;

  int frame_count_ = 0;
  int frame_index_length_ = 6;
  std::string frame_prefix_ = "frame";
  int sample_count_ = 0;
  int sample_index_length_ = 6;
  std::string sample_prefix_ = "sample";
  double t_ = 0.0;
  double dt_;
  double dt_frame_;
  double T_run_;

  SimulationBase(const std::string &path_to_parameters);

  void ConfigureData();
  /**
   * @brief Load yaml
   *
   * @param file_path
   */
  void loadParameters(const std::string &file_path);

  /**
   * @brief Make output directories for logs/checkpoints/temp_images/etc...
   *
   * @param output_dir
   * @param overwrite
   */
  static void make_output_directory(const std::string &output_dir,
                                    bool overwrite = false);

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

  void configureLogging();

  std::string get_frame_path();
  std::string get_sample_path();
};

} // namespace meshbrane
