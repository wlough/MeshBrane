/**
 * @file half_edge_cutils.hpp
 */
#ifndef HALF_EDGE_CUTILS_HPP
#define HALF_EDGE_CUTILS_HPP

#include <Eigen/Dense> // Eigen::Vector3d
#include <cstdint>     // std::int32_t

// namespace py = pybind11;
namespace eig = Eigen;

using INT_TYPE = std::int32_t;
using FLOAT_TYPE = double;

using Samplesi = eig::Matrix<INT_TYPE, eig::Dynamic, 1>;
using Samples2i = eig::Matrix<INT_TYPE, eig::Dynamic, 2>;
using Samples3i = eig::Matrix<INT_TYPE, eig::Dynamic, 3>;
using Samples3d = eig::Matrix<FLOAT_TYPE, eig::Dynamic, 3>;

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

INT_TYPE find_halfedge_index_of_twin(const Samples2i &H, const INT_TYPE &h);

std::tuple<Samples3d, Samplesi, Samplesi, Samplesi, Samplesi, Samplesi,
           Samplesi, Samplesi>
vf_samples_to_he_samples(const Samples3d &xyz_coord_V, const Samples3i &V_of_F);

#endif /* HALF_EDGE_CUTILS_HPP */