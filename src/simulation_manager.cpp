/**
 * @file simulation_manager.cpp
 */

#include "meshbrane/simulation_manager.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

SimulationManager::SimulationManager(const std::string &path_to_parameters) {
  printf("SimulationManager\n");
  loadParameters(path_to_parameters);
  run_prefix_ = parameters_["run_name"].as<std::string>();
  output_dir_ = parameters_["output_dir"].as<std::string>();
  exe_path_ = parameters_["executable"].as<std::string>();
  printf("output_dir_ "
         "= %s\n",
         output_dir_.c_str());
  printf("run_prefix_ = %s\n", run_prefix_.c_str());
  make_output_directory();
  save_parameters(output_dir_ + "/param_sweep.yaml");
  generate_index_combinations(list_lengths_);
  make_sim_parameters();
}

void SimulationManager::generate_index_combinations(
    const std::vector<int> &list_lengths) {
  int num_lists = list_lengths.size();
  std::vector<int> indices(num_lists, 0);
  int run_count = 0;
  while (true) {
    // run_indices_[run_count] = indices;
    // Print the current combination
    for (int i = 0; i < num_lists; ++i) {
      // std::cout << indices[i] << " ";
      run_indices_[run_count][i] = indices[i];
    }
    run_count++;
    // std::cout << std::endl;

    // Find the rightmost list that has more elements left after the current
    // element in that list
    int next = num_lists - 1;
    while (next >= 0 && (indices[next] + 1 >= list_lengths[next])) {
      --next;
    }

    // No more combinations can be generated
    if (next < 0) {
      break;
    }

    // Move to the next element in that list
    ++indices[next];

    // Set all the lists to the right of this list to their first elements
    for (int i = next + 1; i < num_lists; ++i) {
      indices[i] = 0;
    }
  }
}

void SimulationManager::make_sim_parameters() {
  printf("make_sim_parameters\n");
  // std::unordered_map<> envelope_lists_
  // get run_names
  for (int i = 0; i < num_runs_; i++) {
    run_names_[i] = run_prefix_;
    for (int j = 0; j < num_lists_; j++) {
      run_names_[i] += "_";
      if (run_indices_[i][j] < 10) {
        run_names_[i] += "0";
      }
      run_names_[i] += std::to_string(run_indices_[i][j]);
    }
    // printf("run_names_[%d] = %s\n", i, run_names_[i].c_str());
  }
  printf("computed run names\n");
  param_paths_.resize(num_runs_);
  for (int i = 0; i < num_runs_; i++) {
    sim_parameters_.push_back(parameters_);
    // std::cout << "sim_parameters_[" << i << "] = " << sim_parameters_[i]
    //           << std::endl;
    std::vector<int> indices = run_indices_[i];
    printf("indices = ");
    for (int j = 0; j < num_lists_; j++) {
      printf("%d ", indices[j]);
    }
    printf("\n");

    for (int j = 0; j < envelope_keys_.size(); j++) {
      sim_parameters_[i]["envelope"][envelope_keys_[j]] =
          envelope_lists_[j][indices[j]];
    }

    if (spindle_type_ == "simple") {
      for (int j = 0; j < spindle_keys_.size(); j++) {
        sim_parameters_[i]["spindle"][spindle_keys_[j]] =
            spindle_lists_[j][indices[j + envelope_keys_.size()]];
      }
    } else if (spindle_type_ == "rigid_spindle") {
      for (int j = 0; j < spindle_keys_.size(); j++) {
        if (spindle_keys_[j] == "max_length" || spindle_keys_[j] == "v_grow" ||
            spindle_keys_[j] == "max_force") {
          sim_parameters_[i]["spindle"]["mt_bundle"][spindle_keys_[j]] =
              spindle_lists_[j][indices[j + envelope_keys_.size()]];
        }
        if (spindle_keys_[j] == "contact_radius") {
          sim_parameters_[i]["spindle"]["spb1"][spindle_keys_[j]] =
              spindle_lists_[j][indices[j + envelope_keys_.size()]];
        }
      }
    } else {
      throw std::runtime_error("Unknown spindle type");
    }

    // for (int j = 0; j < num_lists_; j++) {
    //   if (j < envelope_keys_.size()) {
    //     sim_parameters_[i]["envelope"][envelope_keys_[j]] =
    //         parameters_["envelope"][envelope_keys_[j]][run_indices_[i][j]];
    //   } else {
    //     sim_parameters_[i]["spindle"]
    //                    [spindle_keys_[j - envelope_keys_.size()]] =
    //                        parameters_["spindle"]
    //                                   [spindle_keys_[j -
    //                                   envelope_keys_.size()]]
    //                                   [run_indices_[i][j]];
    //   }
    // }
    sim_parameters_[i]["run_name"] = run_names_[i];
    sim_parameters_[i]["output_dir"] = output_dir_;
    // std::filesystem::create_directories(sim_parameters_[i]["output_dir"].as<std::string>());
    std::string param_path = output_dir_ + "/" + run_names_[i] + ".yaml";

    printf("param_path = %s\n", param_path.c_str());
    std::ofstream fout(param_path);
    fout << sim_parameters_[i];
    fout.close();
    // save_parameters(param_path);
    param_paths_[i] = param_path;
  }
}

void SimulationManager::loadParameters(const std::string &file_path) {
  printf("loadParameters\n");
  YAML::Node parameters = YAML::LoadFile(file_path);
  parameters_ = parameters;

  if (parameters_["spindle_type"]) {
    spindle_type_ = parameters_["spindle_type"].as<std::string>();
  }

  envelope_keys_ =
      parameters_["envelope"]["list_keys"].as<std::vector<std::string>>();
  printf("envelope_keys_\n");
  for (int _ = 0; _ < envelope_keys_.size(); _++) {
    printf("  %s\n", envelope_keys_[_].c_str());
  }
  spindle_keys_ =
      parameters_["spindle"]["list_keys"].as<std::vector<std::string>>();
  printf("spindle_keys_\n");
  for (int _ = 0; _ < spindle_keys_.size(); _++) {
    printf("   %s\n", spindle_keys_[_].c_str());
  }
  //////////////////////////////

  spindle_lists_.resize(spindle_keys_.size());
  if (spindle_type_ == "simple") {
    for (int i = 0; i < spindle_keys_.size(); i++) {
      spindle_lists_[i] =
          parameters_["spindle"][spindle_keys_[i]].as<std::vector<double>>();
    }
  } else if (spindle_type_ == "rigid_spindle") {
    for (int i = 0; i < spindle_keys_.size(); i++) {
      if (spindle_keys_[i] == "max_length" || spindle_keys_[i] == "v_grow" ||
          spindle_keys_[i] == "max_force") {
        spindle_lists_[i] =
            parameters_["spindle"]["mt_bundle"][spindle_keys_[i]]
                .as<std::vector<double>>();
      }
      if (spindle_keys_[i] == "contact_radius") {
        spindle_lists_[i] = parameters_["spindle"]["spb1"][spindle_keys_[i]]
                                .as<std::vector<double>>();
      }
    }
  } else {
    throw std::runtime_error("Unknown spindle type");
  }

  envelope_lists_.resize(envelope_keys_.size());
  for (int i = 0; i < envelope_keys_.size(); i++) {
    envelope_lists_[i] =
        parameters_["envelope"][envelope_keys_[i]].as<std::vector<double>>();
  }

  //////////////////////////////
  num_lists_ = envelope_keys_.size() + spindle_keys_.size();
  printf("num_lists_ = %d\n", num_lists_);
  list_lengths_.resize(num_lists_);
  for (int i = 0; i < envelope_keys_.size(); i++) {
    list_lengths_[i] = parameters_["envelope"][envelope_keys_[i]].size();
  }
  if (spindle_type_ == "simple") {
    for (int i = 0; i < spindle_keys_.size(); i++) {
      list_lengths_[i + envelope_keys_.size()] =
          parameters_["spindle"][spindle_keys_[i]].size();
    }
  } else if (spindle_type_ == "rigid_spindle") {
    for (int i = 0; i < spindle_keys_.size(); i++) {
      if (spindle_keys_[i] == "max_length" || spindle_keys_[i] == "v_grow" ||
          spindle_keys_[i] == "max_force") {
        list_lengths_[i + envelope_keys_.size()] =
            parameters_["spindle"]["mt_bundle"][spindle_keys_[i]].size();
      }
      if (spindle_keys_[i] == "contact_radius") {
        printf("adding spindle_keys_[i] == contact_radius ");
        list_lengths_[i + envelope_keys_.size()] =
            parameters_["spindle"]["spb1"][spindle_keys_[i]].size();
      }
    }
  } else {
    throw std::runtime_error("Unknown spindle type");
  }

  for (int i = 0; i < num_lists_; i++) {
    printf("list_lengths_[%d] = %d\n", i, list_lengths_[i]);
  }
  num_runs_ = 1;
  for (int i = 0; i < num_lists_; i++) {
    num_runs_ *= list_lengths_[i];
  }
  printf("num_runs_ = %d\n", num_runs_);
  run_indices_.resize(num_runs_);
  run_names_.resize(num_runs_);
  for (int i = 0; i < num_runs_; i++) {
    run_indices_[i].resize(num_lists_);
  }
}

void SimulationManager::save_parameters(const std::string &path) {
  printf("save_parameters\n");
  std::ofstream fout(path);
  fout << parameters_;
  fout.close();
}

void SimulationManager::make_output_directory() {
  printf("make_output_directory\n");
  std::filesystem::remove_all(output_dir_);
  std::filesystem::create_directories(output_dir_);
}

void SimulationManager::update_movie_paths() {
  printf("update_movie_paths\n");
  movie_paths_.clear();
  std::string movie_path;
  for (int i = 0; i < num_runs_; i++) {
    movie_path =
        output_dir_ + "/" + run_names_[i] + "/visualizations/movie.mp4";
    movie_paths_.push_back(movie_path);
  }
}

YAML::Node SimulationManager::get_run_parameters(std::vector<int> run_index) {
  printf("get_run_parameters\n");
  int run_num = -1;
  bool match = true;
  for (int i = 0; i < num_runs_; i++) {
    match = true;
    for (int j = 0; j < num_lists_; j++) {
      // printf("i, j = %d, %d\n", i, j);
      // printf("run_index[%d] = %d\n", j, run_index[j]);
      // printf("run_indices_[%d][%d] = %d\n", i, j, run_indices_[i][j]);
      if (run_index[j] != run_indices_[i][j]) {
        match = false;
        break;
      }
    }
    if (match) {
      run_num = i;
      break;
    }
  }
  if (!match) {
    throw std::invalid_argument("Run index not found");
  } else {
    // printf("******found run_num\n");
    printf("run_num = %d\n", run_num);
    printf("run_names_[run_num] = %s\n", run_names_[run_num].c_str());
  }
  return sim_parameters_[run_num];
}

std::string SimulationManager::vstack_movie_command() {
  printf("vstack_movie_command\n");
  // movie_paths_.clear();
  // for (int i = 0; i < num_runs_; i++) {
  //   std::string movie_path_i =
  //       output_dir_ + "/" + run_names_[i] + "/visualizations/movie.mp4";
  //   movie_paths_.push_back(movie_path_i);
  // }
  update_movie_paths();
  std::string command = "ffmpeg";
  for (int i = 0; i < num_runs_; i++) {
    command += " -i " + movie_paths_[i];
  }

  // arrange the movies in a vertical stack
  command += " -filter_complex \"";
  for (int i = 0; i < num_runs_; i++) {
    command += "[" + std::to_string(i) + ":v]";
  }
  command += "vstack=inputs=" + std::to_string(num_runs_) + "\" ";
  command += output_dir_ + "/vstack_movie.mp4";

  // std::system(command.c_str());
  return command;
}

std::string SimulationManager::grid_movie_command() {
  printf("grid_movie_command\n");
  std::vector<int> run_index = run_indices_[0];
  if (num_lists_ != 2) {
    return "";
  }
  // int num_rows = list_lengths_[0];
  // int num_cols = list_lengths_[1];

  // std::string command = "ffmpeg";
  // for (int r = 0; r < num_rows; r++) {
  //   for (int c = 0; c < num_cols; c++) {
  //     run_index[0] = r;
  //     run_index[1] = c;
  //     // int run_num = r * num_cols + c;
  //     YAML::Node run_parameters = get_run_parameters(run_index);
  //     // YAML::Node run_parameters = sim_parameters_[run_num];
  //     // printf("run_num = %d\n", run_num);
  //     printf("*****************************************************************"
  //            "*run_name = %s\n",
  //            run_parameters["run_name"].as<std::string>().c_str());
  //     std::string run_name = run_parameters["run_name"].as<std::string>();
  //     std::string movie_path =
  //         output_dir_ + "/" + run_name + "/visualizations/movie.mp4";
  //     command += " -i " + movie_path;
  //   }
  // }

  // // arrange the movies in a grid
  // command += " -filter_complex \"";
  // for (int r = 0; r < num_rows; r++) {
  //   for (int c = 0; c < num_cols; c++) {
  //     int i = r * num_cols + c;
  //     command += "[" + std::to_string(i) + ":v]";
  //   }
  //   command += "hstack=inputs=" + std::to_string(num_cols) + "[r" +
  //              std::to_string(r) + "];";
  // }
  // for (int r = 0; r < num_rows; r++) {
  //   command += "[r" + std::to_string(r) + "]";
  // }
  // command += "vstack=inputs=" + std::to_string(num_rows) + "\" ";
  // command += output_dir_ + "/grid_movie.mp4";

  /////////////////////////////////////////////////////////////////////
  //////////////////////////// grid_movie.sh //////////////////////////
  int num_rows = list_lengths_[0];
  int num_cols = list_lengths_[1];
  std::string command = "./scripts/grid_movie.sh " + std::to_string(num_rows) +
                        " " + std::to_string(num_cols) + " " + output_dir_;
  return command;
}

std::string SimulationManager::six_six_movie_command() {
  printf("six_six_movie_command\n");
  update_movie_paths();
  std::string command = "ffmpeg";
  for (int i = 0; i < num_runs_; i++) {
    command += " -i " + movie_paths_[i];
  }

  // arrange the movies in a vertical stack
  command += " -filter_complex \"";
  for (int i = 0; i < num_runs_; i++) {
    command += "[" + std::to_string(i) + ":v]";
  }
  command += "vstack=inputs=" + std::to_string(num_runs_) + "\" ";
  command += output_dir_ + "/vstack_movie.mp4";

  // std::system(command.c_str());
  return command;
}

void SimulationManager::make_grid_movie() {
  printf("make_grid_movie\n");
  // std::string command = grid_movie_command();
  // std::system(command.c_str());
  int num_rows = list_lengths_[0];
  int num_cols = list_lengths_[1];
  std::string command = "./scripts/grid_movie.sh " + std::to_string(num_rows) +
                        " " + std::to_string(num_cols) + " " + output_dir_;
  std::system(command.c_str());
}
} // namespace meshbrane
