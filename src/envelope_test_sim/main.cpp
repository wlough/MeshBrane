/**
 * @file tests/envelope_test_sim/main.cpp
 * @brief sim for envelope prototyping
 */

#include "meshbrane/envelope_test_sim.hpp"

int run(int argc, char *argv[], const std::string &param_file) {
  meshbrane::EnvelopeTestSim sim(param_file);
  sim.run(argc, argv);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_param_file>" << std::endl;
    return 1;
  }
  std::string param_file = argv[1];

  std::cout << "--------------------" << std::endl;
  std::cout << "- Running EnvelopeTestSim -" << std::endl;
  int success = 0;
  success = run(argc, argv, param_file);

  if (success == 0) {
    std::cout << "- EnvelopeTestSim completed successfully -" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
  } else {
    std::cout << "- Something went wrong -" << std::endl;
    std::cout << "------------------------" << std::endl;
  }

  return success;
}
