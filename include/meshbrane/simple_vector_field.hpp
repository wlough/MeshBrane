#pragma once

/**
 * @file simple_vector_field.hpp
 * @brief Defines SimpleVectorField class used for visualization
 */

#include "meshbrane/math_utils.hpp"
#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane/pretty_pictures.hpp"
#include <Eigen/Dense>
// #include <array>
// #include <cmath>

namespace meshbrane {

/**
 * @brief Simple vector field for visualization with libigl.
 *
 */
struct SimpleVectorField : public MeshBraneObject {
  bool has_been_updated_{false};
  SimpleVectorField() = default;
  SimpleVectorField(Eigen::Matrix<double, Eigen::Dynamic, 3> P0,
                    Eigen::Matrix<double, Eigen::Dynamic, 3> vecs, double scale,
                    Vec3d rgb)
      : vecs_(vecs), scale_(scale), rgb_(rgb) {
    size_t num_vecs = vecs.rows();
    for (int i = 0; i < 3; i++) {
      arrows_[i].resize(num_vecs, 3);
    }
    for (int _ = 0; _ < num_vecs; _++) {
      Vec3d p0 = P0.row(_);
      Vec3d u01 = scale_ * vecs_.row(_);
      Vec3d p1 = p0 + u01;
      Vec3d u01_perp = get_perp_vec(u01);
      Vec3d p2 = p1 - tip_len_ * u01 + tip_len_ * u01_perp;

      arrows_[0].row(_) = p0;
      arrows_[1].row(_) = p1;
      arrows_[2].row(_) = p2;
    }
  }

  Eigen::Matrix<double, Eigen::Dynamic, 3> vecs_;
  std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3> arrows_;
  double scale_{1.0};
  double tip_len_ = 0.1;
  Eigen::RowVector3d rgb_{0.8392, 0.1529, 0.1569};

  std::tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>,
             Eigen::Matrix<double, Eigen::Dynamic, 3>,
             Eigen::Matrix<double, 3, 1>>
  args_add_edges0() {
    return std::make_tuple(arrows_[0], arrows_[1], rgb_);
  }
  std::tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>,
             Eigen::Matrix<double, Eigen::Dynamic, 3>,
             Eigen::Matrix<double, 3, 1>>
  args_add_edges1() {
    return std::make_tuple(arrows_[1], arrows_[2], rgb_);
  }
  Vec3d get_perp_vec(const Vec3d &u) {
    double x = std::abs(u[0]), y = std::abs(u[1]), z = std::abs(u[2]);
    int index_of_smallest = 0;
    if (y < x) {
      index_of_smallest = 1;
    }
    if (z < std::abs(u[index_of_smallest])) {
      index_of_smallest = 2;
    }
    Vec3d n = Vec3d::Zero();
    n[index_of_smallest] = 1;
    Vec3d unit_perp = math::cross(n, u);
    unit_perp /= math::L2norm(unit_perp);
    return math::cross(u, unit_perp);
  }

  void update(Eigen::Matrix<double, Eigen::Dynamic, 3> P0,
              Eigen::Matrix<double, Eigen::Dynamic, 3> vecs, double scale,
              Vec3d rgb) {
    vecs_ = vecs;
    scale_ = scale;
    rgb_ = rgb;
    size_t num_vecs = vecs.rows();
    for (int i = 0; i < 3; i++) {
      arrows_[i].resize(num_vecs, 3);
    }
    for (int _ = 0; _ < num_vecs; _++) {
      Vec3d p0 = P0.row(_);
      Vec3d u01 = scale_ * vecs_.row(_);
      Vec3d p1 = p0 + u01;
      Vec3d u01_perp = get_perp_vec(u01);
      Vec3d p2 = p1 - tip_len_ * u01 + tip_len_ * u01_perp;

      arrows_[0].row(_) = p0;
      arrows_[1].row(_) = p1;
      arrows_[2].row(_) = p2;
    }
    has_been_updated_ = true;
  }

  void update_with_normal(Eigen::Matrix<double, Eigen::Dynamic, 3> P0,
                          Eigen::Matrix<double, Eigen::Dynamic, 3> vecs,
                          double scale, Vec3d rgb) {
    vecs_ = vecs;
    scale_ = scale;
    rgb_ = rgb;
    size_t num_vecs = vecs.rows();
    for (int i = 0; i < 3; i++) {
      arrows_[i].resize(num_vecs, 3);
    }
    for (int _ = 0; _ < num_vecs; _++) {
      Vec3d p0 = P0.row(_);
      Vec3d u01 = scale_ * vecs_.row(_);
      Vec3d p1 = p0 + u01;
      Vec3d u01_perp = get_perp_vec(u01);
      Vec3d p2 = p1 - tip_len_ * u01 + tip_len_ * u01_perp;

      arrows_[0].row(_) = p0;
      arrows_[1].row(_) = p1;
      arrows_[2].row(_) = p2;
    }
  }
};

} // namespace meshbrane
