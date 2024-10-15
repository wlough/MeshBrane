/**
 * @file data_types.hpp
 */

#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <Eigen/Dense> // for Eigen::Vector3d
#include <cstdint>     // For fixed-width integer types
#include <vector>      // for std::vector

using INT_TYPE = std::int32_t;
using FLOAT_TYPE = double;
using Samplesi = std::vector<int32_t>;
using Samples2i = std::vector<std::array<int32_t, 2>>;
using Samples3i = std::vector<std::array<int32_t, 3>>;
using Samples3d = std::vector<Eigen::Vector3d>;

struct VertexFaceSamples {
  Samples3d xyz_coord_V;
  Samples3i V_of_F;
};

struct HalfEdgeSamples {
  Samples3d xyz_coord_V;
  Samplesi h_out_V;
  Samplesi v_origin_H;
  Samplesi h_next_H;
  Samplesi h_twin_H;
  Samplesi f_left_H;
  Samplesi h_bound_F;
  Samplesi h_right_B;
};

#endif /* DATA_TYPES_HPP */