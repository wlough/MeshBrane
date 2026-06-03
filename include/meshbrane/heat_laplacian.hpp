#pragma once

/**
 * @file heat_laplacian.hpp
 * @brief Functions for computing the Laplace-Beltrami operator on meshed
 * triangulated surfaces using heat kernels.
 */

#include "meshbrane/meshbrane_data_types.hpp"
#include <Eigen/Dense>

namespace meshbrane {

// inline Samples3d belkin_laplacian(Samples1d &phi, Samples3d &xyz_coord_V,
//                                   Samples3i &V_cycle_F, Samplesi &area_F,
//                                   double t) {
//   int num_faces = V_cycle_F.rows();
//   int num_vertices = xyz_coord_V.rows();
//   Samples1d lap_phi(num_vertices);
//   lap_phi.setZero();
//   for (int ix = 0; ix < num_vertices; ix++) {
//     Vec3d x = xyz_coord_V.row(ix);
//     double phix = phi[ix];
//     for (int f = 0; f < num_faces; f++) {
//       for (int iy : V_cycle_F.row(f)) {
//         Vec3d y = xyz_coord_V.row(iy);
//         double phiy = phi[iy];
//         double A = area_F[f];
//         lap_phi[ix] +=
//             (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
//       }
//     }
//   }
// }
/**
 * @brief Approximation of the fundamental solution to the heat equation on a 2D
 * surface embedded in Euclidean space.
 *
 * @param x
 * @param y
 * @param t
 * @return double
 */
inline double heat_parametrix2d(Vec3d &x, Vec3d &y, double t) {
  double dist = (y - x).norm();
  return std::exp(-dist * dist / (4 * t)) / (4 * M_PI * t);
}

inline double heat_laplacian_kernel1(Vec3d x, Vec3d y, double t) {
  double dist = (y - x).norm();
  return std::exp(-dist * dist / (4 * t)) / (4 * M_PI * t * t);
}

inline Samples1d belkin_heat_laplacian(const Samples1d &phi,
                                       const Samples3d &xyz_coord_V,
                                       const Samples3i &V_cycle_F,
                                       const Samples1d &area_F,
                                       const double t) {
  int num_faces = V_cycle_F.rows();
  int num_vertices = xyz_coord_V.rows();
  Samples1d lap_phi(num_vertices);
  lap_phi.setZero();
  for (int ix = 0; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V.row(ix);
    double phix = phi[ix];
    for (int f = 0; f < num_faces; f++) {
      for (int iy : V_cycle_F.row(f)) {
        Vec3d y = xyz_coord_V.row(iy);
        double phiy = phi[iy];
        double A = area_F[f];
        lap_phi[ix] +=
            (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
      }
    }
  }
  return lap_phi;
}

inline Samples3d belkin_heat_laplacian(const Samples3d &phi,
                                       const Samples3d &xyz_coord_V,
                                       const Samples3i &V_cycle_F,
                                       const Samples1d &area_F,
                                       const double t) {
  int num_faces = V_cycle_F.rows();
  int num_vertices = xyz_coord_V.rows();
  Samples3d lap_phi(num_vertices, 3);
  lap_phi.setZero();
  for (int ix = 0; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V.row(ix);
    Vec3d phix = phi.row(ix);
    for (int f = 0; f < num_faces; f++) {
      for (int iy : V_cycle_F.row(f)) {
        Vec3d y = xyz_coord_V.row(iy);
        Vec3d phiy = phi.row(iy);
        double A = area_F[f];
        lap_phi.row(ix) +=
            (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
      }
    }
  }
  return lap_phi;
}

inline Samples1d face_restricted_belkin_heat_laplacian(
    const Samples1d &phi, const Samples3d &xyz_coord_V,
    const Samples3i &V_cycle_F, const Samples1d &area_F, const double t,
    const SimplicialSet &F) {
  int num_faces = V_cycle_F.rows();
  int num_vertices = xyz_coord_V.rows();
  Samples1d lap_phi(num_vertices);
  lap_phi.setZero();
  for (int ix = 0; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V.row(ix);
    double phix = phi[ix];
    for (int f : F) {
      for (int iy : V_cycle_F.row(f)) {
        Vec3d y = xyz_coord_V.row(iy);
        double phiy = phi[iy];
        double A = area_F[f];
        lap_phi[ix] +=
            (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
      }
    }
  }
  return lap_phi;
}

inline Samples3d face_restricted_belkin_heat_laplacian(
    const Samples3d &phi, const Samples3d &xyz_coord_V,
    const Samples3i &V_cycle_F, const Samples1d &area_F, const double t,
    const SimplicialSet &F) {
  int num_faces = V_cycle_F.rows();
  int num_vertices = xyz_coord_V.rows();
  Samples3d lap_phi(num_vertices, 3);
  lap_phi.setZero();
  for (int ix = 0; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V.row(ix);
    Vec3d phix = phi.row(ix);
    for (int f : F) {
      for (int iy : V_cycle_F.row(f)) {
        Vec3d y = xyz_coord_V.row(iy);
        Vec3d phiy = phi.row(iy);
        double A = area_F[f];
        lap_phi.row(ix) +=
            (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
      }
    }
  }
  return lap_phi;
}

inline Samples1d guckenberger_heat_laplacian(const Samples1d &phi,
                                             const Samples3d &xyz_coord_V,
                                             const Samples3i &V_cycle_F,
                                             const Samples1d &area_F,
                                             const double t) {
  int num_faces = V_cycle_F.rows();
  int num_vertices = xyz_coord_V.rows();
  Samples1d lap_phi(num_vertices);
  lap_phi.setZero();
  for (int ix = 0; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V.row(ix);
    double phix = phi[ix];
    for (int f = 0; f < num_faces; f++) {
      for (int iy : V_cycle_F.row(f)) {
        Vec3d y = xyz_coord_V.row(iy);
        double phiy = phi[iy];
        double A = area_F[f];
        lap_phi[ix] +=
            (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
      }
    }
  }
  return lap_phi;
}

inline Samples3d guckenberger_heat_laplacian(const Samples3d &phi,
                                             const Samples3d &xyz_coord_V,
                                             const Samples3i &V_cycle_F,
                                             const Samples1d &area_F,
                                             const double t) {
  int num_faces = V_cycle_F.rows();
  int num_vertices = xyz_coord_V.rows();
  Samples3d lap_phi(num_vertices, 3);
  lap_phi.setZero();
  for (int ix = 0; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V.row(ix);
    Vec3d phix = phi.row(ix);
    for (int f = 0; f < num_faces; f++) {
      for (int iy : V_cycle_F.row(f)) {
        Vec3d y = xyz_coord_V.row(iy);
        Vec3d phiy = phi.row(iy);
        double A = area_F[f];
        lap_phi.row(ix) +=
            (A / (3 * t)) * heat_parametrix2d(x, y, t) * (phiy - phix);
      }
    }
  }
  return lap_phi;
}

} // namespace meshbrane
