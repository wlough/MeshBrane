#pragma once

/**
 * @file kmc.hpp
 * @brief Utilities for kinetic Monte Carlo simulations.
 */

#include <algorithm> // std::shuffle
#include <random>    // std::mt19937, std::random_device
#include <vector>    // std::vector
// #include <pcg_random.hpp>

namespace meshbrane {
namespace kmc {
class RandomNumberGenerator {
public:
  std::mt19937 rng;
  //   RandomNumberGenerator() = default;
  RandomNumberGenerator() : rng(std::random_device{}()) {};

  RandomNumberGenerator(uint64_t seed) : rng(seed) {}

  double standard_uniform() { return static_cast<double>(rng()) / rng.max(); }

  double standard_normal() {
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng);
  }

  std::vector<int> random_permutation(int n) {
    std::vector<int> perm(n);
    for (int i = 0; i < n; i++) {
      perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), rng);
    return perm;
  }
};

// class RandomNumberGenerator {
// public:
//   pcg32 rng;

//   RandomNumberGenerator()
//       : rng(pcg_extras::seed_seq_from<std::random_device>{}) {}

//   RandomNumberGenerator(uint64_t seed) : rng(seed) {}

//   double standard_uniform() {
//     std::uniform_real_distribution<double> dist(0.0, 1.0);
//     return dist(rng);
//   }

//   double standard_normal() {
//     std::normal_distribution<double> dist(0.0, 1.0);
//     return dist(rng);
//   }

//   std::vector<int> random_permutation(int n) {
//     std::vector<int> perm(n);
//     for (int i = 0; i < n; i++) {
//       perm[i] = i;
//     }
//     std::shuffle(perm.begin(), perm.end(), rng);
//     return perm;
//   }
// };

// std::mt19937 rng(std::random_device{}());

/**
 * @brief Sample a real number between 0.0 and 1.0 from a uniform distribution.
 */
// double standard_uniform() { return static_cast<double>(rand()) / RAND_MAX; }

// std::vector<int> random_permutation(int n) {
//   std::vector<int> perm(n);
//   for (int i = 0; i < n; i++) {
//     perm[i] = i;
//   }
//   std::shuffle(perm.begin(), perm.end(), rng);
//   return perm;
// }

} // namespace kmc
} // namespace meshbrane
