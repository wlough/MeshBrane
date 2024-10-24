/**
 * @file data_types.hpp
 */
#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <Eigen/Dense> //
#include <cstdint>     // std::int32_t
#include <tuple>       // std::tuple

namespace eig = Eigen;

using INT_TYPE = std::int32_t;
using FLOAT_TYPE = double;
using Coords3d = eig::Vector3d;
using Samplesi = eig::Matrix<INT_TYPE, eig::Dynamic, 1>;
using Samples2i = eig::Matrix<INT_TYPE, eig::Dynamic, 2>;
using Samples3i = eig::Matrix<INT_TYPE, eig::Dynamic, 3>;
using Samples3d = eig::Matrix<FLOAT_TYPE, eig::Dynamic, 3>;
using HalfEdgeSamples = std::tuple<Samples3d, Samplesi, Samplesi, Samplesi,
                                   Samplesi, Samplesi, Samplesi, Samplesi>;
using VertexFaceSamples = std::tuple<Samples3d, Samples3i>;
using VertexEdgeFaceSamples = std::tuple<Samples3d, Samples2i, Samples3i>;

#endif /* DATA_TYPES_HPP */