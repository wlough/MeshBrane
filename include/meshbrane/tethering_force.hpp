#pragma once

/**
 * @file tethering_force.hpp
 * @brief Functions for computing forces that regulate mesh edge lengths
 */

#include "meshbrane/math_utils.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace meshbrane {

inline double Gtether(double z, double beta, double lam, double z_inf,
                      double z_on) {
  double dz = std::abs(z - 1);
  double dzi = std::abs(z_inf - 1);
  double dzo = std::abs(z_on - 1);
  double G = lam * std::exp(-beta / (dz / dzo - 1)) / (1 - dz / dzi);
  return G;
}

inline double normDGtether(double z, double beta, double lam, double z_inf,
                           double z_on) {
  double dz = std::abs(z - 1);
  double dzi = std::abs(z_inf - 1);
  double dzo = std::abs(z_on - 1);
  double G = lam * std::exp(-beta / (dz / dzo - 1)) / (1 - dz / dzi);

  double Di = 1 - dz / dzi;
  double Do = dz / dzo - 1;

  double normDG =
      (beta * Di * dzi + Do * Do * dzo) * G / (Di * Do * Do * dzi * dzo);

  return normDG;
}

inline double Utether(double z, double beta, double lam, double z_rep_inf,
                      double z_rep_on, double z_att_on, double z_att_inf) {
  if (z <= z_rep_inf) {
    // throw std::runtime_error("Fatal error: z = " + std::to_string(z) +
    //                          " reached repulsive singularity");
    return std::numeric_limits<double>::infinity();
  }
  if (z >= z_att_inf) {
    // throw std::runtime_error("Fatal error: z = " + std::to_string(z) +
    //                          " reached attractive singularity");
    return std::numeric_limits<double>::infinity();
  }
  if (z < z_rep_on) {
    return Gtether(z, beta, lam, z_rep_inf, z_rep_on);
  }
  if (z <= z_att_on) {
    return 0;
  }
  return Gtether(z, beta, lam, z_att_inf, z_att_on);
}

// use z=L/Ltarget and multiply by K/Ltarget for force magnitude
inline double normDUtether(double z, double beta, double lam, double z_rep_inf,
                           double z_rep_on, double z_att_on, double z_att_inf) {
  if (z <= z_rep_inf) {
    // throw std::runtime_error("Fatal error: z = " + std::to_string(z) +
    //                          " reached repulsive singularity");
    return std::numeric_limits<double>::infinity();
  }
  if (z >= z_att_inf) {
    // throw std::runtime_error("Fatal error: z = " + std::to_string(z) +
    //                          " reached attractive singularity");
    return std::numeric_limits<double>::infinity();
  }
  if (z < z_rep_on) {
    return normDGtether(z, beta, lam, z_rep_inf, z_rep_on);
  }
  if (z <= z_att_on) {
    return 0;
  }
  return normDGtether(z, beta, lam, z_att_inf, z_att_on);
}

inline double normDUtether_alt(double z, double beta, double lam,
                               double z_rep_inf, double z_rep_on,
                               double z_att_on, double z_att_inf) {
  if (z == 0.0) {
    throw std::runtime_error("Fatal error: z = 0");
  }
  if (z <= z_rep_inf) {
    // printf(
    //     "Relaxing tether constraint: z = %.20f reached repulsive
    //     singularity\n", z);
    return normDUtether_alt(1.01 * z_rep_inf, beta, lam, z_rep_inf, z_rep_on,
                            z_att_on, z_att_inf);
  }
  if (z >= z_att_inf) {
    // printf("Relaxing tether constraint: z = %.20f reached attractive "
    //        "singularity\n",
    //        z);
    return normDUtether_alt(0.99 * z_att_inf, beta, lam, z_rep_inf, z_rep_on,
                            z_att_on, z_att_inf);
  }
  if (z < z_rep_on) {
    return normDGtether(z, beta, lam, z_rep_inf, z_rep_on);
  }
  if (z <= z_att_on) {
    return 0;
  }
  return normDGtether(z, beta, lam, z_att_inf, z_att_on);
}

inline Vec3d tether_force(const Vec3d &x, const Vec3d &xp, double Ltarget,
                          double Ktether, double z_rep_inf, double z_rep_on,
                          double z_att_on, double z_att_inf, double beta,
                          double lam) {

  Vec3d dx = xp - x;
  double norm_dx = dx.norm();
  double z = norm_dx / Ltarget;
  if (z == 1) {
    return Vec3d::Zero();
  }
  double normDU =
      normDUtether(z, beta, lam, z_rep_inf, z_rep_on, z_att_on, z_att_inf);

  if (std::isinf(normDU)) {
    normDU = normDUtether_alt(z, beta, lam, z_rep_inf, z_rep_on, z_att_on,
                              z_att_inf);
  }
  return (Ktether / Ltarget) * normDU * (dx / norm_dx) * (z - 1) /
         std::abs(z - 1);
}

} // namespace meshbrane
