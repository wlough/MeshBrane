#pragma once

/**
 * @file lennard_jones.hpp
 * @brief Lennard-Jones, WCA potentials and derivatives
 */

#include "meshbrane/meshbrane_data_types.hpp"
// #include <cmath>

namespace meshbrane {

constexpr double TWO_POW_MINUS_ONE_THIRD =
    0.7937005259840998; // std::pow(2.0, -1.0 / 3.0);
constexpr double TWO_POW_ONE_SIXTH =
    1.122462048309373; // std::pow(2.0, 1.0 / 6.0);

inline Vec3d lennard_jones_force(Vec3d xa, Vec3d xb, double epsilon,
                                 double sigma) {
  Vec3d x = xa - xb;
  double r2 = 1 / x.squaredNorm();
  double sr2 = sigma * sigma * r2;
  double sr6 = sr2 * sr2 * sr2;
  double f_r = 48 * epsilon * (sr6 - 0.5) * sr6 * r2;
  return f_r * x;
}

inline Vec3d wca_force(Vec3d xa, Vec3d xb, double epsilon, double sigma) {
  Vec3d x = xa - xb;
  double r2 = 1 / x.squaredNorm();
  double s2 = sigma * sigma;
  // rcutoff = 2^(1/6) * sigma
  double rcutoff2 = TWO_POW_MINUS_ONE_THIRD / s2; // 1/rcutoff^2
  if (rcutoff2 < r2) {
    double sr2 = s2 * r2;
    double sr6 = sr2 * sr2 * sr2;
    double f_r = 48 * epsilon * (sr6 - 0.5) * sr6 * r2;
    return f_r * x;
  }
  return Vec3d::Zero();
}

} // namespace meshbrane
