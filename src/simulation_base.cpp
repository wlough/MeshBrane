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

SimulationBase::SimulationBase(const fs::path &path_to_parameters) {
  // loadParameters(path_to_parameters);
  parameters_ = YAML::LoadFile(path_to_parameters);
  if (parameters_["dt"]) {
    dt_max_ = parameters_["dt"].as<double>();
  } else {
    throw std::runtime_error("No dt provided in parameters file");
  }
  if (parameters_["dt_frame"]) {
    dt_frame_ = parameters_["dt_frame"].as<double>();
  } else {
    throw std::runtime_error("No dt_frame provided in parameters file");
  }
  if (parameters_["T_run"]) {
    T_run_ = parameters_["T_run"].as<double>();
  } else {
    throw std::runtime_error("No T_run provided in parameters file");
  }
  if (parameters_["run_name"]) {
    run_name_ = parameters_["run_name"].as<std::string>();
  } else {
    throw std::runtime_error("No run_name provided in parameters file");
  }
  if (parameters_["output_dir"]) {
    output_dir_ =
        fs::path(parameters_["output_dir"].as<std::string>()) / run_name_;
  } else {
    throw std::runtime_error("No output_dir provided in parameters file");
  }
  logs_dir_ = output_dir_ / "logs";
  raw_data_dir_ = output_dir_ / "raw_data";
  visualizations_dir_ = output_dir_ / "visualizations";
  temp_images_dir_ = output_dir_ / "temp_images";
  make_output_directory(true);

  log_path_ = logs_dir_ / "sim.log";

  std::ofstream fout(output_dir_ / "parameters.yaml");
  fout << parameters_;
  fout.close();

  configure_logging();
}

void SimulationBase::make_output_directory(bool overwrite) {
  std::vector<std::string> sub_dirs = {logs_dir_, raw_data_dir_,
                                       visualizations_dir_, temp_images_dir_};

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
    fs::remove_all(output_dir_);
    for (const auto &sub_dir : sub_dirs) {
      fs::create_directories(sub_dir);
    }
  }
}

void SimulationBase::configure_logging() {
  std::ofstream log_file(log_path_, std::ios_base::app);
  if (!log_file.is_open()) {
    throw std::runtime_error("Unable to open log file: " + log_path_.string());
  }
  log_file << "Initialized simulation with parameters: " << std::endl;
  log_file << parameters_ << std::endl;
  log_file.close();
}

fs::path SimulationBase::get_frame_path() {
  std::string frame_count_str = std::to_string(frame_count_);
  // pad with zeros so the index is always 6 digits
  frame_count_str =
      std::string(6 - frame_count_str.size(), '0') + frame_count_str;
  return temp_images_dir_ / (frame_prefix_ + "_" + frame_count_str + ".png");
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
    throw std::runtime_error("Unable to open log file: " + log_path_.string());
  }

  log_file << "Made a movie from images in " << temp_images_dir_.string()
           << '\n';
  log_file << "Movie saved to " << visualizations_dir_.string() << '\n';
}

} // namespace meshbrane
