#pragma once

/**
 * @file meshbrane_data_types.hpp
 * @brief Some data types and simple structs used by meshbrane
 */

#include <Eigen/Dense>  //
#include <Eigen/Sparse> // SparseMatrix
#include <tuple>        // std::tuple
#include <unordered_set>

/**
 * @defgroup MeshBraneDataTypes MeshBrane data types
 * @brief Type aliases and simple structs used in the meshbrane library.
 */

namespace meshbrane {

/** @addtogroup MeshBraneDataTypes
 *  @{
 */

/**
 * @brief Coordinates of a single point in 3D space or affine coordinates in 2D
 */
using Vec3d = Eigen::Vector3d;

/**
 * @brief Column vector of doubles.
 */
using Samples1d = Eigen::Matrix<double, Eigen::Dynamic, 1>;

/**
 * @brief N-by-2 matrix of doubles.
 */
using Samples2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;

/**
 * @brief N-by-3 matrix of doubles. Represents 3D spatial coordinates of a
 * vertices in a vertex set.
 */
using Samples3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;

/**
 * @brief Column vector of integers. Represents a permutation, a combinatorial
 * map, or integer-valued coordinate samples.
 */
using Samplesi = Eigen::VectorXi;

/**
 * @brief N-by-2 matrix of integers
 */
using Samples2i = Eigen::Matrix<int, Eigen::Dynamic, 2>;

/**
 * @brief N-by-3 matrix of integers
 */
using Samples3i = Eigen::Matrix<int, Eigen::Dynamic, 3>;

using Samples4i = Eigen::Matrix<int, Eigen::Dynamic, 4>;

using SamplesXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

using SimplicialSet = std::unordered_set<int>;
using SimplicialTriple = std::array<SimplicialSet, 3>;

////////////////////////////////////////////
// Mesh data types
////////////////////////////////////////////
/**
 * @brief An 8-tuple of matrices containing spatial coordinates of vertices and
 * combinatorial maps between vertices/half-edges/faces used to represent a
 * half-edge mesh.
 * @details Each vector of int labels has a name of the form
 * "a_description_Q", where"a" denotes the type of object associated with the
 * elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for
 * face), "Q" denotes the type of object associated with the indices ("V" for
 * vertex, "H" for half-edge, "F" for face, or "B" for boundary), and
 * "description" is a description of information represented by the data. For
 * example, "v_origin_H_" is a vector of vertices at the origin of each
 * half-edge. The i-th element of vector "a_description_Q" can be accessed using
 * the "a_description_q(i)" method.
 * - xyz_coord_V : Eigen::MatrixXd
 *   - xyz_coord_V[i] = \f$[x, y, z]\f$ coordinates of vertex i
 * - h_out_V : Eigen::VectorXi
 *   - h_out_V[i] = some outgoing half-edge incident on vertex i
 * - v_origin_H : Eigen::VectorXi
 *   - v_origin_H[j] = vertex at the origin of half-edge j
 * - h_next_H : Eigen::VectorXi
 *   - h_next_H[j] = next half-edge after half-edge j in the face cycle
 * - h_twin_H : Eigen::VectorXi
 *   - h_twin_H[j] = half-edge antiparallel to half-edge j
 * - f_left_H : Eigen::VectorXi
 *   - f_left_H[j] = face to the left of half-edge j, if j in interior[M] or a
 * positively oriented boundary of M
 *   - f_left_H[j] = boundary to the left of half-edge j, if j in a negatively
 * oriented boundary
 * - h_right_F : Eigen::VectorXi
 *   - h_right_F[k] = some half-edge on the boundary of face k
 * - h_negative_B_ : Eigen::VectorXi
 *   - h_negative_B_[n] = half-edge to the right of boundary n
 *
 */
using HalfEdgeTuple = std::tuple<Samples3d, Samplesi, Samplesi, Samplesi,
                                 Samplesi, Samplesi, Samplesi, Samplesi>;
/**
 * @brief A 2-tuple \f$(V, F)\f$ of matrices containing
 * spatial coordinates of each vertex and the vertex cycle of each face in a
 * triangle mesh.
 * @details \f$(V, F)\f$ = (xyz_coord_V, V_cycle_F)
 * - xyz_coord_V : Eigen::MatrixXd
 *  - xyz_coord_V[v] = \f$[x, y, z]\f$ coordinates of vertex v
 * - V_cycle_F : Eigen::MatrixXi
 *  - V_cycle_F[f] = \f$[v_0, v_1, v_2]\f$ vertex cycle of face f
 */
using VertexFaceTuple = std::tuple<Samples3d, Samples3i>;
/**
 * @brief 3-tuple \f$(V, E, F)\f$ of matrices containing spatial coordinates of
 * each vertex, the vertex cycle of each edge, and the vertex cycle of each face
 * in a triangle mesh.
 * @details \f$(V, E, F)\f$ = (xyz_coord_V, V_cycle_E, V_cycle_F)
 * - xyz_coord_V : Eigen::MatrixXd
 *  - xyz_coord_V[v] = \f$[x, y, z]\f$ coordinates of vertex v
 * - V_cycle_E : Eigen::MatrixXi
 *  - V_cycle_E[e] = \f$[v_0, v_1]\f$ vertex cycle of face edge
 * - V_cycle_F : Eigen::MatrixXi
 *  - V_cycle_F[f] = \f$[v_0, v_1, v_2]\f$ vertex cycle of face f
 */
using VertexEdgeFaceTuple = std::tuple<Samples3d, Samples2i, Samples3i>;

////////////////////////////////////////////
// Visualization data types
////////////////////////////////////////////

/**
 * @brief A color in RGBA format.
 */
using RGBA = Eigen::Matrix<double, 4, 1>;
/** @}*/ // end of group MeshBraneDataTypes
} // namespace meshbrane

/**
 * @example meshbrane_data_types.hpp
 * Vec3d:
 * @code
 * meshbrane::Vec3d xyz_coord(1, 2, 3);
 * meshbrane::Vec3d xyz_coord = {4, 5, 6};
 * @endcode
 *
 * Samples3d:
 * @code
 * meshbrane::Samples3d xyz_coord_V(4, 3);
 * xyz_coord_V << 1, 2, 3,
 *                4, 5, 6,
 *                7, 8, 9,
 *                10, 11, 12;
 *
 * # from a double array
 * double data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
 * meshbrane::Samples3d xyz_coord_V = Eigen::Map<Eigen::Matrix<double,
 * Eigen::Dynamic, 3>>(data, 4, 3);
 * @endcode
 *
 * Samplesi:
 * @code
 * meshbrane::Samplesi v_origin_H(5);
 * v_origin_H << 0, 1, 2, 3, 4;
 *
 * meshbrane::Samplesi v_origin_H;
 * v_origin_H.resize(5);
 * v_origin_H << 0, 1, 2, 3, 4;
 * @endcode
 *
 * Samples3i:
 * @code
 * meshbrane::Samples3i V_cycle_F(4, 3);
 * V_cycle_F << 0, 1, 2,
 *             3, 4, 5,
 *            6, 7, 8,
 *           9, 10, 11;
 */
