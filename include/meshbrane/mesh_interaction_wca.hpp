#pragma once

/**
 * @file mesh_interaction_wca.hpp
 * @brief Defines MeshMeshInteractionWCA class
 */

#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/patch.hpp"
#include <cassert>

namespace meshbrane {
class MeshMeshInteractionWCA {
  // constexpr double TWO_POW_MINUS_ONE_THIRD =
  // 0.7937005259840998; // std::pow(2.0, -1.0 / 3.0);
public:
  Patch patch1_;
  Patch patch2_;

  double epsilon_;
  double sigma_;

  double rcutoff_squared_;

  struct ClosestTrianglePairResult {
    Vec3d barycentric_weights1;
    Vec3d barycentric_weights2;
    double squared_distance;
    Vec3d vector_12; // from point in tri1 to point in tri2
  };

  ClosestTrianglePairResult
  compute_closest_triangle_pair(Vec3d &p1_1, Vec3d &p1_2, Vec3d &p1_3,
                                Vec3d &p2_1, Vec3d &p2_2, Vec3d &p2_3) {
    ClosestTrianglePairResult result;
    // placeholder
    return result;
  }

  Vec3d get_wca_force(const Vec3d &x, const double &squared_distance) {
    double r2 = 1 / squared_distance;
    double s2 = sigma_ * sigma_;
    // rcutoff = 2^(1/6) * sigma
    double rcutoff2 = 0.7937005259840998 / s2; // 1/rcutoff^2
    if (rcutoff2 < r2) {
      double sr2 = s2 * r2;
      double sr6 = sr2 * sr2 * sr2;
      double f_r = 48 * epsilon_ * (sr6 - 0.5) * sr6 * r2;
      return f_r * x;
    }
    return Vec3d::Zero();
  }

  MeshMeshInteractionWCA(MatrixMesh &mesh1, MatrixMesh &mesh2, double epsilon,
                         double sigma)
      : patch1_(&mesh1, {}, {}, {}, {}), patch2_(&mesh2, {}, {}, {}, {}),
        epsilon_(epsilon), sigma_(sigma) {}

  void interact() {
    // Compute the face-face WCA interaction between patch1_ and patch2_ and
    // distribute forces to the vertices using barycentric weights.

    int Nf1 = patch1_.num_faces();
    int Nf2 = patch2_.num_faces();

    for (int f1 : patch1_.F_) {
      int h1_1 = patch1_.h_right_f(f1);
      int h1_2 = patch1_.h_next_h(h1_1);
      int h1_3 = patch1_.h_next_h(h1_2);
      int v1_1 = patch1_.v_origin_h(h1_1);
      int v1_2 = patch1_.v_origin_h(h1_2);
      int v1_3 = patch1_.v_origin_h(h1_3);
      Vec3d p1_1 = patch1_.xyz_coord_v(v1_1);
      Vec3d p1_2 = patch1_.xyz_coord_v(v1_2);
      Vec3d p1_3 = patch1_.xyz_coord_v(v1_3);
      for (int f2 : patch2_.F_) {
        int h2_1 = patch2_.h_right_f(f2);
        int h2_2 = patch2_.h_next_h(h2_1);
        int h2_3 = patch2_.h_next_h(h2_2);
        int v2_1 = patch2_.v_origin_h(h2_1);
        int v2_2 = patch2_.v_origin_h(h2_2);
        int v2_3 = patch2_.v_origin_h(h2_3);
        Vec3d p2_1 = patch2_.xyz_coord_v(v2_1);
        Vec3d p2_2 = patch2_.xyz_coord_v(v2_2);
        Vec3d p2_3 = patch2_.xyz_coord_v(v2_3);

        ClosestTrianglePairResult cp_result =
            compute_closest_triangle_pair(p1_1, p1_2, p1_3, p2_1, p2_2, p2_3);
        if (cp_result.squared_distance >=
            rcutoff_squared_) { // no interaction if beyond cutoff
          continue;
        }
        Vec3d force =
            get_wca_force(cp_result.vector_12, cp_result.squared_distance);
        // distribute force to vertices of f1 and f2 using barycentric weights
        patch1_.supermesh_->force_V_.row(v1_1) +=
            force * cp_result.barycentric_weights1[0];
        patch1_.supermesh_->force_V_.row(v1_2) +=
            force * cp_result.barycentric_weights1[1];
        patch1_.supermesh_->force_V_.row(v1_3) +=
            force * cp_result.barycentric_weights1[2];
        patch2_.supermesh_->force_V_.row(v2_1) -=
            force * cp_result.barycentric_weights2[0];
        patch2_.supermesh_->force_V_.row(v2_2) -=
            force * cp_result.barycentric_weights2[1];
        patch2_.supermesh_->force_V_.row(v2_3) -=
            force * cp_result.barycentric_weights2[2];
      }
    }
  }
};

} // namespace meshbrane
