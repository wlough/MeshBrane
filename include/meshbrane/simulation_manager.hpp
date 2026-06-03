#pragma once

/**
 * @file simulation_manager.hpp
 * @brief SimulationManager class for RigidSpindleSim parameter sweeps
 */

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace meshbrane {
class SimulationManager {
public:
  std::string spindle_type_ = "simple";
  YAML::Node parameters_;

  bool make_six_six_movie = false;
  bool make_vstack_movie = false;
  bool make_four_three_movie = false;
  // bool make_grid_movie = false;

  std::vector<std::string> envelope_keys_;
  std::vector<std::string> spindle_keys_;

  std::vector<std::vector<double>> envelope_lists_;
  std::vector<std::vector<double>> spindle_lists_;
  std::string output_dir_;
  int frame_index_length_ = 2;
  std::string run_prefix_ = "run";
  std::vector<std::vector<int>> run_indices_;
  std::vector<std::string> run_names_;
  std::vector<YAML::Node> sim_parameters_;
  int num_lists_;
  std::vector<int> list_lengths_;
  int num_runs_;
  std::string exe_path_;
  std::vector<std::string> movie_paths_;

  void generate_index_combinations(const std::vector<int> &list_lengths);

  std::vector<std::string> param_paths_;

  //////////////////////////////
  SimulationManager(const std::string &path_to_parameters);

  /**
   * @brief Load yaml
   *
   * @param file_path
   */
  void loadParameters(const std::string &file_path);

  void save_parameters(const std::string &path);

  void get_run_names();
  void make_sim_parameters();
  void save_sim_parameters();
  void make_output_directory();
  void update_movie_paths();
  std::string grid_movie_command();
  std::string six_six_movie_command();
  std::string vstack_movie_command();
  std::string hstack_movie_command();

  YAML::Node get_run_parameters(std::vector<int> run_index);

  YAML::Node get_run_parameters_from_pair(int r, int c);

  void make_grid_movie();
};

} // namespace meshbrane
