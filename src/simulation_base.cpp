/**
 * @file simulation_base.cpp
 */

#include "meshbrane/simulation_base.hpp"
#include "meshbrane/meshbrane_config.hpp"
#include "meshbrane/system_utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace meshbrane {

SimulationBase::SimulationBase(const std::string &path_to_parameters) {
  loadParameters(path_to_parameters);
  run_name_ = parameters_["run_name"].as<std::string>();
  output_dir_ = parameters_["output_dir"].as<std::string>() + "/" + run_name_;

  make_output_directory(output_dir_, true);
  input_dir_ = output_dir_ + "/input";
  logs_dir_ = output_dir_ + "/logs";
  checkpoints_dir_ = output_dir_ + "/checkpoints";
  raw_data_dir_ = output_dir_ + "/raw_data";
  processed_data_dir_ = output_dir_ + "/processed_data";
  visualizations_dir_ = output_dir_ + "/visualizations";
  temp_images_dir_ = output_dir_ + "/temp_images";

  input_path_ = input_dir_ + "/" + run_name_ + ".yaml";
  log_path_ = logs_dir_ + "/" + run_name_ + ".log";
  checkpoint_path_ = checkpoints_dir_ + "/" + run_name_ + ".pkl";
  raw_data_path_ = raw_data_dir_ + "/data.h5";
  processed_data_path = processed_data_dir_ + "/" + run_name_ + ".h5";

  // Configure logging
  configureLogging();
  ConfigureData();
}

void SimulationBase::ConfigureData() {
  // raw_data_ = meshbrane::HDF5Group(raw_data_path_);
};

void SimulationBase::make_output_directory(const std::string &output_dir,
                                           bool overwrite) {
  std::vector<std::string> sub_dirs = {
      output_dir + "/input",          output_dir + "/logs",
      output_dir + "/checkpoints",    output_dir + "/raw_data",
      output_dir + "/processed_data", output_dir + "/visualizations",
      output_dir + "/temp_images"};

  if (!overwrite) {
    for (const auto &sub_dir : sub_dirs) {
      if (fs::exists(sub_dir)) {
        throw std::runtime_error(sub_dir +
                                 " already exists. Choose a different "
                                 "output_dir, or set overwrite=true");
      } else {
        fs::create_directories(sub_dir);
      }
    }
  } else {
    fs::remove_all(output_dir);
    for (const auto &sub_dir : sub_dirs) {
      fs::create_directories(sub_dir);
    }
  }
}

void SimulationBase::configureLogging() {
  std::ofstream log_file(log_path_, std::ios_base::app);
  if (!log_file.is_open()) {
    throw std::runtime_error("Unable to open log file: " + log_path_);
  }
  log_file << "Initialized simulation with parameters: " << std::endl;
  log_file << parameters_ << std::endl;
  log_file.close();
}

void SimulationBase::loadParameters(const std::string &file_path) {
  YAML::Node parameters = YAML::LoadFile(file_path);
  parameters_ = parameters;
}

std::string SimulationBase::get_frame_path() {
  std::string frame_count_str = std::to_string(frame_count_);
  // pad with zeros so the index is always 6 digits
  frame_count_str =
      std::string(6 - frame_count_str.size(), '0') + frame_count_str;
  return temp_images_dir_ + "/" + frame_prefix_ + "_" + frame_count_str +
         ".png";
}

void SimulationBase::make_a_movie() {
  std::cout << "Making a movie" << std::endl;

  std::string command = shell_quote(meshbrane::python_executable) + " " +
                        shell_quote(meshbrane::make_movie_script) + " " +
                        shell_quote(temp_images_dir_) + " " +
                        shell_quote(visualizations_dir_);

  int result = std::system(command.c_str());

  if (result != 0) {
    throw std::runtime_error("Failed to make movie");
  }

  std::ofstream log_file(log_path_, std::ios_base::app);
  if (!log_file.is_open()) {
    throw std::runtime_error("Unable to open log file: " + log_path_);
  }

  log_file << "Made a movie from images in " << temp_images_dir_ << '\n';
  log_file << "Movie saved to " << visualizations_dir_ << '\n';
}

} // namespace meshbrane
