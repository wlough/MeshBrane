/**
 * @file tests/param_sweep/main.cpp
 * @brief Test simulation
 */

#include "meshbrane/simulation_manager.hpp"
#include <iostream>

int init(int argc, char *argv[], const std::string &param_file) {
  meshbrane::SimulationManager m(param_file);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_param_file>" << std::endl;
    return 1;
  }
  std::string param_file = argv[1];

  std::cout << "--------------------" << std::endl;
  std::cout << "- Running ParamSweep -" << std::endl;
  int success = 0;
  meshbrane::SimulationManager m(param_file);
  std::vector<std::string> param_paths = m.param_paths_;
  std::string exe_path = m.exe_path_;

  std::string command = exe_path + " " + param_paths[0];

  for (int i = 1; i < param_paths.size(); i++) {
    command += " & " + exe_path + " " + param_paths[i];
  }
  command += " & wait";
  printf("command = %s\n", command.c_str());

  std::string vstack_movie_command = m.vstack_movie_command();
  std::string grid_movie_command = m.grid_movie_command();

  // command += " && " + vstack_movie_command;
  command += " && " + grid_movie_command;

  std::system(command.c_str());
  printf("vstack_movie_command = %s\n", vstack_movie_command.c_str());
  printf("grid_movie_command = %s\n", grid_movie_command.c_str());

  if (success == 0) {
    std::cout << "- ParamSweep completed successfully -" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
  } else {
    std::cout << "- Something went wrong -" << std::endl;
    std::cout << "------------------------" << std::endl;
  }

  return success;
}
