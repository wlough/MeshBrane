#pragma once

/**
 * @file chromatin_chain.hpp
 * @brief Defines the ChromatinChain class.
 */

#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simplicial_complex_1d.hpp"
#include <filesystem>
#include <string> // std::string
#include <tuple>  // std::tuple
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class ChromatinChain : public SimplicialComplex1d {
public:
  std::string name_{"chromatin"};

  ////////////////////////
  // Initialization //////
  ////////////////////////
  ChromatinChain() = default;

  void set_attributes_from_parameters() {
    SimplicialComplex1d::set_attributes_from_parameters();
    printf("ChromatinChain::set_attributes_from_parameters\n");
  }

  //////////////////////////////////////////////
  // Visualization /////////////////////////////
  //////////////////////////////////////////////

  void apply_internal_interactions();
  void clear_interactions();
  void apply_thermal_fluctuations_to_self(double dt, double kBT,
                                          kmc::RandomNumberGenerator &rng);
  void update_cached_data();
  void update_own_state_variables(double dt);
};

} // namespace meshbrane
