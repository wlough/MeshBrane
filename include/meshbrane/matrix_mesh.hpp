#pragma once

/**
 * @file matrix_mesh.hpp
 * @brief Defines the MatrixMesh class.
 */

#include "meshbrane/mesh_builder.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane/patch.hpp"
#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simple_vector_field.hpp"
#include <Eigen/Sparse>
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional> // std::optional
#include <stdexcept>
#include <string> // std::string
#include <tuple>  // std::tuple
#include <utility>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

struct MinimalHalfEdgeMeshData {
  //////////////////////////
  Samples3d xyz_coord_V_; //
                          //
  Samplesi h_out_V_;      //
  Samplesi h_right_F_;    //
  Samplesi h_negative_B_; //
                          //
  Samplesi v_origin_H_;   //
  Samplesi f_left_H_;     //
                          //
  Samplesi h_next_H_;     //
  Samplesi h_twin_H_;     //
  //////////////////////////
};

/**
 * @brief Check that faces only contain vertex indices in range
 * `0 <= v < Nv` where `Nv=xyz_coord_V.rows()`
 *
 * @return bool
 */
bool VF_data_check_all_v_indices_in_range(Samples3d &xyz_coord_V,
                                          Samples3i &V_cycle_F);

/**
 * @brief Check that every vertex appears in at least one face.
 *
 * @return bool
 */
bool VF_data_check_no_isolated_vertices(Samples3d &xyz_coord_V,
                                        Samples3i &V_cycle_F);

bool VF_data_check_face_v_indices_unique(Samples3d &xyz_coord_V,
                                         Samples3i &V_cycle_F);

bool VF_data_check(Samples3d &xyz_coord_V, Samples3i &V_cycle_F);

bool min_HE_data_check_faces_are_triangles(
    Samples3d &xyz_coord_V, Samplesi &h_out_V, Samplesi &h_right_F,
    Samplesi &v_origin_H, Samplesi &f_left_H, Samplesi &h_next_H,
    Samplesi &h_twin_H, Samplesi &h_negative_B);

bool min_HE_data_check_twin_cycles_are_length_2(
    Samples3d &xyz_coord_V, Samplesi &h_out_V, Samplesi &h_right_F,
    Samplesi &v_origin_H, Samplesi &f_left_H, Samplesi &h_next_H,
    Samplesi &h_twin_H, Samplesi &h_negative_B);

bool min_HE_data_check(Samples3d &xyz_coord_V, Samplesi &h_out_V,
                       Samplesi &h_right_F, Samplesi &v_origin_H,
                       Samplesi &f_left_H, Samplesi &h_next_H,
                       Samplesi &h_twin_H, Samplesi &h_negative_B);

bool combinatorial_map_check(Samplesi &h_out_V, Samplesi &h_directed_E,
                             Samplesi &h_right_F, Samplesi &v_origin_H,
                             Samplesi &e_undirected_H, Samplesi &f_left_H,
                             Samplesi &h_next_H, Samplesi &h_twin_H,
                             Samplesi &h_negative_B);

/**
 * @defgroup CoreStructures Core MeshBrane structures
 * @brief Core classes for MeshBrane.
 * @{
 */

/**
 * @brief A half-edge mesh class that uses matrices of integers to represent
 combinatorial maps
 * @details
 * # Data members
 * ## Naming conventions
 * The core data structure used by the `MatrixMesh` class consists of matrices
 that describe geometric and topological properties of the mesh. Each matrix has
 a name of the form `a_description_Q`, where `a` denotes the type of object
 associated with the elements (`xyz` for position,
 * `v` for vertex, `e` for edge, `f` for face, `h` for half-edge, or `b` for
 boundary.), `Q` denotes the type of
 * object associated with the indices (`V` for vertex, `E` for edge, `F`
 * for face, `H` for half-edge, or `B` for boundary), and `description` is a
 description of
 * information represented by the data. For example, `v_origin_H_` is a vector
 * of vertices at the origin of each half-edge. The i-th element of vector
 * `a_description_Q` can be accessed using the `a_description_q(i)` method.
 * ## Simplices
 *
 * - `Vertex`
 *   - `xyz_coord_V_` : `meshbrane::Samples3d`
 *     - `xyz_coord_V_(i)` = \f$(x, y, z)\f$ coordinates of vertex i
 *   - `h_out_V_` : `meshbrane::Samplesi`
 *     - `h_out_V_(i)` = some outgoing half-edge incident on vertex i
 * - `Edge`
 *   - `V_cycle_E_` : `meshbrane::Samples2i`
 *     - `V_cycle_E_(i)` = vertex cycle of edge e
 *   - `h_directed_E_` : `meshbrane::Samplesi`
 *     - `h_directed_E_(j)` = some half-edge along edge j
 * - `Face`
 *   - `V_cycle_F_` : `meshbrane::Samples3i`
 *     - `V_cycle_F_(i)` = vertex cycle of face f
 *   - `h_right_F_` : `meshbrane::Samplesi`
 *     - `h_right_F_(k)` = some half-edge on the right-handed boundary of face k
 *
 * ## Half-edges
 * - `HalfEdge`
 *   - `v_origin_H_` : `meshbrane::Samplesi`
 *     - `v_origin_H_(j)` = vertex at the origin of half-edge j
 *   - `e_undirected_H_` : `meshbrane::Samplesi`
 *     - `e_undirected_H_(j)` = edge associated with half-edge j
 *   - `f_left_H_` : `meshbrane::Samplesi`
 *     - `f_left_H_(j)` = face to the left of half-edge j, if j in interior(M)
 or a positively oriented boundary of M
 *     - `f_left_H_(j)` = boundary containing half-edge j, if j in a
 negatively
 * oriented boundary
 *   - `h_next_H_` : `meshbrane::Samplesi`
 *     - `h_next_H_(j)` = next half-edge after half-edge j in the face cycle
 *   - `h_twin_H_` : `meshbrane::Samplesi`
 *     - `h_twin_H_(j)` = half-edge antiparallel to half-edge j
 *
 * ## Boundaries
 * - `h_negative_B_` : `meshbrane::Samplesi`
 *   - `h_negative_B_(n)` = half-edge in the negatively oriented boundary n
 *
 *
 * # Initialization
 * The MatrixMesh class can be initialized in several ways:
 * - From a ply file (binary/ascii) containing half-edge mesh samples:
 *   @code
 *   MatrixMesh(ply_path);
 *   @endcode
 * - From half-edge mesh samples:
 *   @code
 *   MatrixMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
 * f_left_H, h_right_F, h_negative_B);
 *   @endcode
 * - From vertex positions and and face samples:
 *   @code
 *   MatrixMesh::from_vf_samples(xyz_coord_V, V_cycle_F);
 *   @endcode
 * - From a ply file (binary/ascii) containing vertex/face data:
 *   @code
 *   MatrixMesh::from_vf_ply(ply_path);
 *   @endcode
 *
 * # Methods
 *
 * ## Combinatorial maps
 * The following methods provide access to the vertex positions and
 combinatorial maps:
 * - xyz_coord_v(int v) : meshbrane::Vec3d
 * - xyz_coord_v(const Eigen::VectorXi &indices) : meshbrane::Samples3d
 * - h_out_v(int v) : int
 * - h_out_v(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 * - v_origin_h(int h) : int
 * - v_origin_h(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 * - h_next_h(int h) : int
 * - h_next_h(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 * - h_twin_h(int h) : int
 * - h_twin_h(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 * - f_left_h(int h) : int
 * - f_left_h(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 * - h_right_f(int f) : int
 * - h_right_f(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 * - h_negative_b(int b) : int
 * - h_negative_b(const Eigen::VectorXi &indices) : meshbrane::Samplesi
 *
 * ### Derived maps
 * The following methods are derived from compositions of the above
 combinatorial maps:
 * - h_in_v(int v) : int
 *  - half-edge in to vertex v
 * - v_head_h(int h) : int
 *  - vertex at the head of half-edge h
 * - h_prev_h(int h) : int
 *  - previous half-edge in the face cycle of half-edge h
 * - h_rotcw_h(int h) : int
 *  - half-edge rotated clockwise about surface normal from half-edge h
 * - h_rotccw_h(int h) : int
 *  - half-edge rotated counterclockwise about surface normal from half-edge h
 * - h_prev_h_by_rot(int h) : int
 *  - previous half-edge in the face cycle of half-edge h, obtained by rotating
 * h counterclockwise. Faster than h_prev_h for haf-edges on the boundary, or
 * faces with many edges.
 *
 * ## Predicates
 *
 * - some_negative_boundary_contains_h(int h) : bool
 * - some_positive_boundary_contains_h(int h) : bool
 * - some_boundary_contains_h(int h) : bool
 * - some_boundary_contains_v(int v) : bool
 * - h_is_locally_delaunay(int h) : bool
 * - h_is_flippable(int h) : bool
 *
 * ## Generators
 *
 * - generate_V_of_f(int f) : utils::SimpleGenerator<int>
 * - generate_H_out_v_clockwise(int v, int h_start = -1) :
 utils::SimpleGenerator<int>
 * - generate_H_bound_f(int f, int h_start = -1) : utils::SimpleGenerator<int>
 * - generate_H_rotcw_h(int h) : utils::SimpleGenerator<int>
 * - generate_H_next_h(int h) : utils::SimpleGenerator<int>
 * - generate_H_negative_b(int b) : utils::SimpleGenerator<int>
 * - generate_F_incident_v(int v) : utils::SimpleGenerator<int>
 *
 * ## Mutators
 *
 * - update_mat_v(int v,...)
 * - update_mat_h(int h,...)
 * - update_mat_f(int f,...)
 * - flip_hedge(int h) // Updates combinatorial map/halfedge data without
 updating simplex matrices
 * - flip_edge(int e)
 * - flip_non_delaunay()
 *
 * ## Miscellaneous
 * - get_num_vertices() : int
 * - get_num_edges() : int
 * - get_num_half_edges() : int
 * - get_num_faces() : int
 * - get_euler_characteristic() : int
 * - get_num_boundaries() : int
 * - get_genus() : int
 *
 */
class MatrixMesh {
  // class MatrixMesh : public MeshBraneObject {
public:
  std::string name_{"mesh"};
  ////////////////////////////
  // Core data structure /////
  ////////////////////////////
  Samples3d xyz_coord_V_;   //
  Samples2i V_cycle_E_;     //
  Samples3i V_cycle_F_;     //
                            //
  Samplesi h_out_V_;        //
  Samplesi h_directed_E_;   //
  Samplesi h_right_F_;      //
  Samplesi h_negative_B_;   //
                            //
  Samplesi v_origin_H_;     //
  Samplesi e_undirected_H_; //
  Samplesi f_left_H_;       //
                            //
  Samplesi h_next_H_;       //
  Samplesi h_twin_H_;       //
  ////////////////////////////

  bool xyz_coord_V_was_set_{false};
  bool V_cycle_E_was_set_{false};
  bool V_cycle_F_was_set_{false};
  bool h_out_V_was_set_{false};
  bool h_directed_E_was_set_{false};
  bool h_right_F_was_set_{false};
  bool v_origin_H_was_set_{false};
  bool e_undirected_H_was_set_{false};
  bool f_left_H_was_set_{false};
  bool h_next_H_was_set_{false};
  bool h_twin_H_was_set_{false};
  bool h_negative_B_was_set_{false};

  /////////////////////////
  // fixed parameters /////
  /////////////////////////
  YAML::Node parameters_;

  std::filesystem::path ply_path_;

  bool show_half_edges_{false};
  bool show_vertices_{false};
  bool show_edges_{false};
  bool draw_wireframe_{false};
  RGBA rgba_vertex_ = RGBA_DICT.at("purple");
  RGBA rgba_half_edge_ = RGBA_DICT.at("purple");
  RGBA rgba_face_ = RGBA_DICT.at("yellow");
  RGBA rgba_edge_ = RGBA_DICT.at("blue");
  double radius_vertex_{5};

  double belkin_dt_{0.001};
  double heat_dt_multiple_{10.0};
  double belkin_rtol_{1e-8};
  double belkin_atol_{1e-8};
  int belkin_min_ring_{2};
  bool construct_laplacian_matrix_{false};

  enum class LaplacianType { COTAN, BELKIN, GUCKENBERGER };
  LaplacianType laplacian_type_{LaplacianType::COTAN};
  LaplacianType
  laplacian_type_from_string(const std::string &laplacian_type) const {
    if (laplacian_type == "cotan") {
      return LaplacianType::COTAN;
    } else if (laplacian_type == "belkin") {
      return LaplacianType::BELKIN;
    } else if (laplacian_type == "guckenberger") {
      return LaplacianType::GUCKENBERGER;
    } else {
      throw std::runtime_error(
          "laplacian_type_from_string: Unknown laplacian type");
    }
  }

  enum class GaussianCurvatureType { ANGLE_DEFECT, LAPLACIAN };
  GaussianCurvatureType gaussian_curvature_type_{
      GaussianCurvatureType::ANGLE_DEFECT};
  GaussianCurvatureType gaussian_curvature_type_from_string(
      const std::string &gaussian_curvature_type) const {
    if (gaussian_curvature_type == "angle_defect") {
      return GaussianCurvatureType::ANGLE_DEFECT;
    } else if (gaussian_curvature_type == "laplacian") {
      return GaussianCurvatureType::LAPLACIAN;
    } else {
      throw std::runtime_error("gaussian_curvature_type_from_string: Unknown "
                               "gaussian curvature type");
    }
  }

  Samples3d bary_coord_Q_;
  Samples1d dimensionless_quad_weight_Q_;

  ////////////////////
  // Cached data /////
  ////////////////////

  std::vector<Samplesi> V_cycle_B_;
  Samples3d vec_H_;
  Samples1d length_E_;
  Samples1d area_F_;
  Samples3d normal_F_;
  Samples3d normal_V_;
  Samples1d area_V_;
  double average_edge_length_{-1.0};
  double average_face_area_{-1.0};
  double total_volume_{-1.0};
  // Curvature
  Samples3d mcvec_V_;
  Samples1d mean_curvature_V_;
  Samples1d gaussian_curvature_V_;
  Samples1d lap_mean_curvature_V_;

  Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian_matrix_V_;

  Eigen::Matrix<double, Eigen::Dynamic, 1> radius_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_H;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_F;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_E_;

  std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3>
      shifted_half_edge_arrows_;
  SimpleVectorField vector_field_arrows_;

  //////////////////////
  // Constructors //////
  //////////////////////
  MatrixMesh() = default;

  /**
   * @brief Construct a new MatrixMesh object from a vertex xyz coordinates and
   * face vertex indices.
   *
   * @param xyz_coord_V
   * @param V_cycle_F
   */
  MatrixMesh(const Samples3d &xyz_coord_V, const Samples3i &V_cycle_F);

  /**
   * @brief Construct a new MatrixMesh object from minimal half-edge data
   *
   * @param xyz_coord_V
   * @param h_out_V
   * @param v_origin_H
   * @param h_next_H
   * @param h_twin_H
   * @param f_left_H
   * @param h_right_F
   * @param h_negative_B
   * @return MatrixMesh
   */
  MatrixMesh(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
             const Samplesi &h_right_F, const Samplesi &v_origin_H,
             const Samplesi &f_left_H, const Samplesi &h_next_H,
             const Samplesi &h_twin_H, const Samplesi &h_negative_B);

  // MatrixMesh(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
  //            const Samplesi &v_origin_H, const Samplesi &h_next_H,
  //            const Samplesi &h_twin_H, const Samplesi &f_left_H,
  //            const Samplesi &h_right_F, const Samplesi &h_negative_B);

  MatrixMesh(const YAML::Node &parameters);
  ///////////////////
  // Operators //////
  ///////////////////

  MatrixMesh &operator+=(const MatrixMesh &m);
  MatrixMesh operator+(const MatrixMesh &m) const;

  /////////////////////////////////////
  // Initialization and mesh I/O //////
  /////////////////////////////////////

  void write_he_ply(const std::filesystem::path &ply_path) const;

  /**
   * @brief Compute edges and combinatorial map from triangular faces. Requires
   * xyz_coord_V_ and V_cycle_F_ data.
   *
   * @details Half-edges are identified with oriented boundary components of
   * faces and are assigned integer labels h_{fk}=3*f+k.
   * ```
   *         v2
   *        /  \
   *       /    \
   *      //   /\\
   *     //h2  h1\\
   *    //_       \\
   *   /     h0     \
   *  /    ______\   \
   * v0---------------v1
   * ```
   */
  void init_core_data_from_V_cycle_F();

  /**
   * @brief Compute simplices and combinatorial map from minimal half-edge data:
   * xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H, h_right_F,
   * and h_negative_B.
   */
  void init_core_data_from_minimal_half_edge_data();

  /**
   * @brief Initialize the cached data from core
   */
  void init_cache_from_core_data();

  /**
   * @brief Initialize core data for an icosohedron
   */
  void init_icosohedron();

  /**
   * @brief Initialize core data for an icosohedron with `num_refinements`
   * subdivisions.
   */
  void init_icososphere(int num_refinements);

  /**
   * @brief Load ply file at `ply_path_` and initialize core data.
   */
  void init_core_data_from_ply();

  /**
   * @brief Set data members using stuff in `parameters_` yaml node.
   */
  void set_attributes_from_parameters();

  void copy_mesh_data(MatrixMesh &m);

  void copy_mesh_cache(MatrixMesh &m);

  // Special surfaces ////

  static MatrixMesh icosohedron();

  void refine_icososphere();

  static MatrixMesh icososphere(int num_refinements);

  static MatrixMesh spherical_shell(int num_refinements, double h);

  /**
   * @brief Construct a tube between closed loops
   * @param xyz_loop0 vertices of boundary loop
   * @param xyz_loop1
   * @param Nz
   * @return MatrixMesh
   */
  static MatrixMesh tube_between_loops(Samples3d xyz_loop0, Samples3d xyz_loop1,
                                       int Nz);

  ////////////////////////////
  // Debugging and checks ////
  ////////////////////////////

  void check_v(int v) const;
  void check_e(int e) const;
  void check_b(int b) const;
  void check_f(int f) const;
  void check_h(int h) const;
  void check_he_matrices() const;
  void print_he_mats();
  void print_simplex_mats();

  ///////////////////////////////////////////////////////////////////
  // Checked access to vertex positions and combinatorial map data //
  ///////////////////////////////////////////////////////////////////

  Vec3d xyz_coord_v(int v) const;
  Samples3d xyz_coord_v(const Samplesi &indices) const;
  int h_out_v(int v) const;
  Samplesi h_out_v(const Samplesi &indices) const;
  int v_origin_h(int h) const;
  Samplesi v_origin_h(const Samplesi &indices) const;
  int h_next_h(int h) const;
  Samplesi h_next_h(const Samplesi &indices) const;
  int h_twin_h(int h) const;
  Samplesi h_twin_h(const Samplesi &indices) const;
  int f_left_h(int h) const;
  Samplesi f_left_h(const Samplesi &indices) const;
  int h_right_f(int f) const;
  Samplesi h_right_f(const Samplesi &indices) const;
  int h_negative_b(int b) const;
  Samplesi h_negative_b(const Samplesi &indices) const;
  int h_directed_e(int e) const;
  Samplesi h_directed_e(const Samplesi &indices) const;
  int e_undirected_h(int h) const;
  Samplesi e_undirected_h(const Samplesi &indices) const;
  // Derived combinatorial maps
  int h_in_v(int v) const;
  int v_head_h(int h) const;
  int h_prev_h(int h) const;
  int h_rotcw_h(int h) const;
  int h_rotccw_h(int h) const;
  int h_prev_h_by_rot(int h) const;

  //////////////////////
  // Cache updaters ////
  //////////////////////

  void update_boundary_cycles();
  void update_mesh_geometric_data_E();
  void update_mesh_geometric_data_F();
  void update_mesh_geometric_data_V();
  void update_mesh_volume();
  void update_mesh_geometric_data();

  //////////////////////
  // Geometry helpers //
  //////////////////////

  double length_e(int e) const {
    return math::L2norm(xyz_coord_v(V_cycle_E_(e, 1)) -
                        xyz_coord_v(V_cycle_E_(e, 0)));
  }

  /**
   * @brief Compute `math::heron_area` of face `f`
   *
   * @param f
   * @return double
   */
  double heron_area_f(int f) const {
    auto x0 = xyz_coord_v(V_cycle_F_(f, 0));
    auto x1 = xyz_coord_v(V_cycle_F_(f, 1));
    auto x2 = xyz_coord_v(V_cycle_F_(f, 2));
    double a = math::L2norm(x0 - x1);
    double b = math::L2norm(x1 - x2);
    double c = math::L2norm(x2 - x0);
    return math::heron_area(a, b, c);
  }

  /**
   * @brief Compute the unit normal vector of face `f`
   *
   * @param f
   * @return Eigen::Vector3d
   */
  Vec3d normal_f(int f) const {
    int h0 = h_right_f(f);
    int h1 = h_next_h(h0);
    int h2 = h_next_h(h1);
    int v0 = v_origin_h(h0);
    int v1 = v_origin_h(h1);
    int v2 = v_origin_h(h2);
    auto x0 = xyz_coord_v(v0);
    auto x1 = xyz_coord_v(v1);
    auto x2 = xyz_coord_v(v2);
    Eigen::Vector3d n =
        math::cross(x0, x1) + math::cross(x1, x2) + math::cross(x2, x0);
    // return n / math::L2norm(n);
    return n.normalized();
  }

  /**
   * @brief Compute the unit normal vector of vertex `v`. Does not use
   * precomputed face areas/normals.
   *
   * @param v
   * @return Eigen::Vector3d
   */
  Vec3d normal_v(int v) const {
    int h_out = h_out_v(v);
    auto x0 = xyz_coord_V_.row(v);
    Eigen::Vector3d n;
    for (int h0 : generate_H_out_v_clockwise(v)) {
      if (some_negative_boundary_contains_h(h0)) {
        continue;
      }
      int h1 = h_next_h(h0);
      int h2 = h_next_h(h1);
      int v1 = v_origin_h(h1);
      int v2 = v_origin_h(h2);
      auto x1 = xyz_coord_V_.row(v1);
      auto x2 = xyz_coord_V_.row(v2);
      n += x0.cross(x1) + x1.cross(x2) + x2.cross(x0);
    }
    return n.normalized();
  }

  double signed_volume_f(int f) const;

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////
  // Visualization /////////////////////////////
  //////////////////////////////////////////////
  /**
   * @brief Get points in half-edges shifted towards triangle centers for
   * visualization
   */
  std::array<Samples3d, 3> compute_shifted_half_edge_arrows();

  void update_shifted_half_edge_arrows();
  void update_mesh_visuals();
  /**
   * @brief Set default rgb values and vertex radii for visualization
   */
  void set_visual_defaults();

  /////////////////////
  // Need to check ////
  /////////////////////

  void reverse_orientation() {
    int Nh = get_num_half_edges();
    int Nf = get_num_faces();
    int Nb = get_num_boundaries();

    for (int f = 0; f < Nf; f++) {
      int h0 = h_right_F_[f];
      int h1 = h_next_H_[h0];
      int h2 = h_next_H_[h1];
      int v0 = v_origin_H_[h0];
      int v1 = v_origin_H_[h1];
      int v2 = v_origin_H_[h2];

      h_out_V_[v0] = h2;
      h_out_V_[v1] = h0;
      h_out_V_[v2] = h1;

      v_origin_H_[h0] = v1;
      v_origin_H_[h1] = v2;
      v_origin_H_[h2] = v0;
      h_next_H_[h0] = h2;
      h_next_H_[h1] = h0;
      h_next_H_[h2] = h1;

      V_cycle_F_.row(f) << v1, v2, v0;
    }

    for (int b = 0; b < Nb; b++) {
      int h0 = h_negative_B_[b];
      std::vector<int> H;
      std::vector<int> H_next;
      std::vector<int> V_next;
      int h = h0;
      do {
        int hn = h_next_H_[h];
        H.push_back(h);
        H_next.push_back(hn);
        V_next.push_back(v_origin_H_[hn]);
        h = hn;
      } while (h != h0);

      for (int i = 0; i < H.size(); i++) {
        int h0 = H[i];
        int h1 = H_next[i];
        int v1 = V_next[i];
        v_origin_H_[h0] = v1;
        h_next_H_[h1] = h0;
      }
    }
  }

  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>
  get_V_cycle_E0() const {
    int Ne = h_directed_E_.size();
    Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> V_cycle_E(Ne, 2);

    const int *H = h_directed_E_.data();
    const int *twin = h_twin_H_.data();
    const int *origin = v_origin_H_.data();
    int *out = V_cycle_E.data();

    for (int e = 0; e < Ne; ++e) {
      int h = H[e];
      out[2 * e] = origin[h];
      out[2 * e + 1] = origin[twin[h]];
    }

    return V_cycle_E;
  }

  Eigen::Matrix<int, Eigen::Dynamic, 2> get_V_cycle_E1() const {
    int Ne = h_directed_E_.size();
    Eigen::Matrix<int, Eigen::Dynamic, 2> V_cycle_E(Ne, 2);

    for (int e = 0; e < Ne; ++e) {
      int h0 = h_directed_E_[e];
      int h1 = h_twin_H_[h0];

      V_cycle_E.row(e) << v_origin_H_[h0], v_origin_H_[h1];
    }

    return V_cycle_E;
  }

  Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>
  get_V_cycle_F0() const {
    int Nf = h_right_F_.size();
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> V_cycle_F(Nf, 3);

    const int *H = h_right_F_.data();
    const int *next = h_next_H_.data();
    const int *origin = v_origin_H_.data();
    int *out = V_cycle_F.data();

    for (int f = 0; f < Nf; ++f) {
      int h0 = H[f];
      int h1 = next[h0];
      int h2 = next[h1];

      out[3 * f] = origin[h0];
      out[3 * f + 1] = origin[h1];
      out[3 * f + 2] = origin[h2];
    }

    return V_cycle_F;
  }

  Eigen::Matrix<int, Eigen::Dynamic, 3> get_V_cycle_F1() const {
    int Nf = h_right_F_.size();
    Eigen::Matrix<int, Eigen::Dynamic, 3> V_cycle_F(Nf, 3);

    for (int f = 0; f < Nf; ++f) {
      int h0 = h_right_F_[f];
      int h1 = h_next_H_[h0];
      int h2 = h_next_H_[h1];

      V_cycle_F.row(f) << v_origin_H_[h0], v_origin_H_[h1], v_origin_H_[h2];
    }

    return V_cycle_F;
  }

  void update_vector_field_arrows(Samples3d X, Samples3d U);

  ///////////////////////////////////////////////////////
  // Fundamental accessors and properties ///////////////
  ///////////////////////////////////////////////////////

  /**
   * @brief Get the number of vertices in the mesh
   *
   * @return int
   */
  int get_num_vertices() const;
  /**
   * @brief Get the number edges in the mesh
   *
   * @return int
   */
  int get_num_edges() const;
  /**
   * @brief Get the numberhalf edges in the mesh
   *
   * @return int
   */
  int get_num_half_edges() const;
  /**
   * @brief Get the number of faces in the mesh
   *
   * @return int
   */
  int get_num_faces() const;
  /**
   * @brief Get the Euler characteristic of the mesh
   *
   * @return int
   */
  int get_euler_characteristic() const;
  /**
   * @brief Get the number of dicconnected boundary components
   *
   * @return int
   */
  int get_num_boundaries() const;
  /**
   * @brief Get the genus of the mesh
   *
   * @return int
   */
  int get_genus() const;

  int get_valence_v(int v) const;

  /**
   * @brief Get the vertex cycles of the faces
   *
   * @return Samples3i
   */
  Samples3i V_cycle_F() const;

  /**
   * @brief Get the vertex positions and vertex cycles of faces in the mesh
   *
   * @return VertexFaceTuple
   */
  VertexFaceTuple vf_samples() const;
  /**
   * @brief Get the vertex positions, vertex cycles of edges, and vertex
   * cycles of faces in the mesh
   *
   *
   * @return HalfEdgeTuple
   */
  VertexEdgeFaceTuple vef_samples() const;
  HalfEdgeTuple he_samples() const;
  /**
   * @brief Get all faces incident on boundary component b
   *
   * @param b
   * @return Samplesi
   */
  Samplesi F_incident_b(int b) const;

  /**
   * @brief Get all edges incident on boundary component b
   *
   * @param b
   * @return Samplesi
   */
  Samplesi E_incident_b(int b) const;

  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  bool some_negative_boundary_contains_h(int h) const;
  bool some_positive_boundary_contains_h(int h) const;
  bool some_boundary_contains_h(int h) const;
  bool some_boundary_contains_v(int v) const;
  bool h_is_locally_delaunay(int h) const;
  bool h_is_flippable(int h) const;
  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  utils::SimpleGenerator<int> generate_V_of_f(int f) const;
  utils::SimpleGenerator<int>
  generate_H_out_v_clockwise(int v, int h_start = -1) const;
  utils::SimpleGenerator<int> generate_H_right_f(int f) const;
  utils::SimpleGenerator<int> generate_H_rotcw_h(int h) const;
  utils::SimpleGenerator<int> generate_H_next_h(int h) const;
  utils::SimpleGenerator<int> generate_H_negative_b(int b) const;
  utils::SimpleGenerator<int> generate_F_incident_v(int v) const;
  ///////////////////////////////////////////////////////
  // Re-indexing helpers ////////////////////////////////
  ///////////////////////////////////////////////////////

  void swap_vertex_indices0(int v0, int v1) {
    check_v(v0);
    check_v(v1);
    if (v0 == v1) {
      return;
    }
    // first check if v0,v1 are adjacent
    bool adjacent = false;
    int h01;
    for (auto h : generate_H_out_v_clockwise(v0)) {
      int ht = h_twin_h(h);
      if (v_origin_h(ht) == v1) {
        adjacent = true;
        h01 = h;
        break;
      }
    }

    if (adjacent) {
      int h10 = h_twin_h(h01);
      // find and update things that output v0/v1 indices
      // EXCEPT edge/half-edges/faces incident on both vertices
      // v_origin_H_, V_cycle_E_, V_cycle_F_
      // v0->v1
      for (auto h : generate_H_out_v_clockwise(v0)) {
        if (h == h01) {
          continue;
        }
        int e = e_undirected_h(h);
        int f = f_left_h(h);
        v_origin_H_[h] = v1;
        if (V_cycle_E_(e, 0) == v0) {
          V_cycle_E_(e, 0) = v1;
        } else {
          V_cycle_E_(e, 1) = v1;
        }
        if (f < 0) {
          continue;
        }
        for (int i_f = 0; i_f < 3; i_f++) {
          if (V_cycle_F_(f, i_f) == v0) {
            V_cycle_F_(f, i_f) = v1;
            break;
          }
        }
      }
      // v1->v0
      for (auto h : generate_H_out_v_clockwise(v1)) {
        if (h == h10) {
          continue;
        }
        int e = e_undirected_h(h);
        int f = f_left_h(h);
        v_origin_H_[h] = v0;
        if (V_cycle_E_(e, 0) == v1) {
          V_cycle_E_(e, 0) = v0;
        } else {
          V_cycle_E_(e, 1) = v0;
        }
        if (f < 0) {
          continue;
        }
        for (int i_f = 0; i_f < 3; i_f++) {
          if (V_cycle_F_(f, i_f) == v1) {
            V_cycle_F_(f, i_f) = v0;
            break;
          }
        }
      }
      // v1<->v0 for stuff incident on both
      int e01 = e_undirected_h(h01);
      int f01 = f_left_h(h01);
      int f10 = f_left_h(h10);
      std::swap(v_origin_H_[h01], v_origin_H_[h10]);
      std::swap(V_cycle_E_(e01, 0), V_cycle_E_(e01, 1));
      int i_v0;
      int i_v1;
      for (int i = 0; i < 3; i++) {
        if (V_cycle_F_(f01, i) == v0) {
          i_v0 = i;
        }
        if (V_cycle_F_(f01, i) == v1) {
          i_v1 = i;
        }
      }
      std::swap(V_cycle_F_(f01, i_v0), V_cycle_F_(f01, i_v1));
      for (int i = 0; i < 3; i++) {
        if (V_cycle_F_(f10, i) == v0) {
          i_v0 = i;
        }
        if (V_cycle_F_(f10, i) == v1) {
          i_v1 = i;
        }
      }
      std::swap(V_cycle_F_(f10, i_v0), V_cycle_F_(f10, i_v1));
      // update things that eat v0/v1 indices
      // xyz_coord_V_, h_out_V_
      xyz_coord_V_.row(v0).swap(xyz_coord_V_.row(v1));
      std::swap(h_out_V_[v0], h_out_V_[v1]);
      return;
    }
    // if not adjacent...
    // find and update things that output v0/v1 indices
    // v_origin_H_, V_cycle_E_, V_cycle_F_
    // v0->v1
    for (auto h : generate_H_out_v_clockwise(v0)) {
      int e = e_undirected_h(h);
      int f = f_left_h(h);
      v_origin_H_[h] = v1;
      if (V_cycle_E_(e, 0) == v0) {
        V_cycle_E_(e, 0) = v1;
      } else {
        V_cycle_E_(e, 1) = v1;
      }
      if (f < 0) {
        continue;
      }
      for (int i_f = 0; i_f < 3; i_f++) {
        if (V_cycle_F_(f, i_f) == v0) {
          V_cycle_F_(f, i_f) = v1;
          break;
        }
      }
    }
    // v1->v0
    for (auto h : generate_H_out_v_clockwise(v1)) {
      int e = e_undirected_h(h);
      int f = f_left_h(h);
      v_origin_H_[h] = v0;
      if (V_cycle_E_(e, 0) == v1) {
        V_cycle_E_(e, 0) = v0;
      } else {
        V_cycle_E_(e, 1) = v0;
      }
      if (f < 0) {
        continue;
      }
      for (int i_f = 0; i_f < 3; i_f++) {
        if (V_cycle_F_(f, i_f) == v1) {
          V_cycle_F_(f, i_f) = v0;
          break;
        }
      }
    }
    // update things that eat v0/v1 indices
    // xyz_coord_V_, h_out_V_
    xyz_coord_V_.row(v0).swap(xyz_coord_V_.row(v1));
    std::swap(h_out_V_[v0], h_out_V_[v1]);
  }

  void swap_vertex_indices1(int v0, int v1) {
    check_v(v0);
    check_v(v1);
    if (v0 == v1) {
      return;
    }
    // first check if v0,v1 are adjacent
    bool adjacent = false;
    int h01;
    for (auto h : generate_H_out_v_clockwise(v0)) {
      int ht = h_twin_h(h);
      if (v_origin_h(ht) == v1) {
        adjacent = true;
        h01 = h;
        break;
      }
    }

    // Exchanges references to v0 and v1.
    auto swap_ref = [v0, v1](int &v) {
      if (v == v0) {
        v = v1;
      } else if (v == v1) {
        v = v0;
      }
    };

    if (adjacent) {
      int h10 = h_twin_h(h01);
      int h12 = h_next_h(h01);
      int h03 = h_next_h(h10);
      int e01 = e_undirected_h(h01);
      int e03 = e_undirected_h(h03);
      int e12 = e_undirected_h(h12);
      int f012 = f_left_h(h01);
      int f103 = f_left_h(h10);

      // find and update things that output v0/v1 indices
      // EXCEPT edge/half-edges/faces incident on both vertices
      // v_origin_H_, V_cycle_E_, V_cycle_F_
      // v0->v1
      for (auto h : generate_H_out_v_clockwise(v0)) {
        if (h == h01) {
          continue;
        }
        if (h == h03) {
          continue;
        }
        int e = e_undirected_h(h);
        int f = f_left_h(h);
        // v_origin_H_[h] = v1;
        swap_ref(v_origin_H_[h]);
        // if (V_cycle_E_(e, 0) == v0) {
        //   V_cycle_E_(e, 0) = v1;
        // } else {
        //   V_cycle_E_(e, 1) = v1;
        // }
        for (int i = 0; i < 2; i++) {
          swap_ref(V_cycle_E_(e, i));
        }
        if (f < 0) {
          continue;
        }
        // for (int i_f = 0; i_f < 3; i_f++) {
        //   if (V_cycle_F_(f, i_f) == v0) {
        //     V_cycle_F_(f, i_f) = v1;
        //     break;
        //   }
        // }
        for (int i = 0; i < 3; i++) {
          swap_ref(V_cycle_F_(f, i));
        }
      }
      // v1->v0
      for (auto h : generate_H_out_v_clockwise(v1)) {
        if (h == h10) {
          continue;
        }
        if (h == h12) {
          continue;
        }
        int e = e_undirected_h(h);
        int f = f_left_h(h);
        // v_origin_H_[h] = v0;
        swap_ref(v_origin_H_[h]);
        // if (V_cycle_E_(e, 0) == v1) {
        //   V_cycle_E_(e, 0) = v0;
        // } else {
        //   V_cycle_E_(e, 1) = v0;
        // }
        for (int i = 0; i < 2; i++) {
          swap_ref(V_cycle_E_(e, i));
        }
        if (f < 0) {
          continue;
        }
        // for (int i_f = 0; i_f < 3; i_f++) {
        //   if (V_cycle_F_(f, i_f) == v1) {
        //     V_cycle_F_(f, i_f) = v0;
        //     break;
        //   }
        // }
        for (int i = 0; i < 3; i++) {
          swap_ref(V_cycle_F_(f, i));
        }
      }
      // v1<->v0 for stuff incident on both
      for (int h : {h01, h10, h12, h03}) {
        swap_ref(v_origin_H_[h]);
      }
      for (int e : {e01, e03, e12}) {
        for (int i = 0; i < 2; i++) {
          swap_ref(V_cycle_E_(e, i));
        }
      }
      for (int f : {f012, f103}) {
        for (int i = 0; i < 3; i++) {
          swap_ref(V_cycle_F_(f, i));
        }
      }

      // int e01 = e_undirected_h(h01);
      // int f01 = f_left_h(h01);
      // int f10 = f_left_h(h10);
      // std::swap(v_origin_H_[h01], v_origin_H_[h10]);
      // std::swap(V_cycle_E_(e01, 0), V_cycle_E_(e01, 1));
      // int i_v0;
      // int i_v1;
      // for (int i = 0; i < 3; i++) {
      //   if (V_cycle_F_(f01, i) == v0) {
      //     i_v0 = i;
      //   }
      //   if (V_cycle_F_(f01, i) == v1) {
      //     i_v1 = i;
      //   }
      // }
      // std::swap(V_cycle_F_(f01, i_v0), V_cycle_F_(f01, i_v1));
      // for (int i = 0; i < 3; i++) {
      //   if (V_cycle_F_(f10, i) == v0) {
      //     i_v0 = i;
      //   }
      //   if (V_cycle_F_(f10, i) == v1) {
      //     i_v1 = i;
      //   }
      // }
      // std::swap(V_cycle_F_(f10, i_v0), V_cycle_F_(f10, i_v1));
      // update things that eat v0/v1 indices
      // xyz_coord_V_, h_out_V_
      xyz_coord_V_.row(v0).swap(xyz_coord_V_.row(v1));
      std::swap(h_out_V_[v0], h_out_V_[v1]);
      return;
    }
    // if not adjacent...
    // find and update things that output v0/v1 indices
    // v_origin_H_, V_cycle_E_, V_cycle_F_
    // v0->v1
    for (auto h : generate_H_out_v_clockwise(v0)) {
      int e = e_undirected_h(h);
      int f = f_left_h(h);
      v_origin_H_[h] = v1;
      if (V_cycle_E_(e, 0) == v0) {
        V_cycle_E_(e, 0) = v1;
      } else {
        V_cycle_E_(e, 1) = v1;
      }
      if (f < 0) {
        continue;
      }
      for (int i_f = 0; i_f < 3; i_f++) {
        if (V_cycle_F_(f, i_f) == v0) {
          V_cycle_F_(f, i_f) = v1;
          break;
        }
      }
    }
    // v1->v0
    for (auto h : generate_H_out_v_clockwise(v1)) {
      int e = e_undirected_h(h);
      int f = f_left_h(h);
      v_origin_H_[h] = v0;
      if (V_cycle_E_(e, 0) == v1) {
        V_cycle_E_(e, 0) = v0;
      } else {
        V_cycle_E_(e, 1) = v0;
      }
      if (f < 0) {
        continue;
      }
      for (int i_f = 0; i_f < 3; i_f++) {
        if (V_cycle_F_(f, i_f) == v1) {
          V_cycle_F_(f, i_f) = v0;
          break;
        }
      }
    }
    // update things that eat v0/v1 indices
    // xyz_coord_V_, h_out_V_
    xyz_coord_V_.row(v0).swap(xyz_coord_V_.row(v1));
    std::swap(h_out_V_[v0], h_out_V_[v1]);
  }

  void swap_vertex_indices(int v0, int v1) {
    check_v(v0);
    check_v(v1);
    if (v0 == v1) {
      return;
    }

    // Collect incident half-edges, edges, and faces.
    std::vector<int> H;
    std::unordered_set<int> E;
    std::unordered_set<int> F;

    for (int v : {v0, v1}) {
      for (int h : generate_H_out_v_clockwise(v)) {
        H.push_back(h);
        E.insert(e_undirected_h(h));
        int f = f_left_h(h);
        if (f >= 0) {
          F.insert(f);
        }
      }
    }
    // Exchange references to v0 and v1.
    auto swap_ref = [v0, v1](int &v) {
      if (v == v0) {
        v = v1;
      } else if (v == v1) {
        v = v0;
      }
    };
    for (int h : H) {
      swap_ref(v_origin_H_[h]);
    }
    for (int e : E) {
      for (int i = 0; i < 2; i++) {
        swap_ref(V_cycle_E_(e, i));
      }
    }
    for (int f : F) {
      for (int i = 0; i < 3; i++) {
        swap_ref(V_cycle_F_(f, i));
      }
    }
    // Exchange vertex-indexed data.
    xyz_coord_V_.row(v0).swap(xyz_coord_V_.row(v1));
    std::swap(h_out_V_[v0], h_out_V_[v1]);
    //
  }

  void swap_edge_indices(int e0, int e1) {
    check_e(e0);
    check_e(e1);
    if (e0 == e1) {
      return;
    }
    int h0 = h_directed_E_[e0];
    int ht0 = h_twin_H_[h0];
    int h1 = h_directed_E_[e1];
    int ht1 = h_twin_H_[h1];
    // update things that output e0/e1 indices
    std::swap(e_undirected_H_[h0], e_undirected_H_[h1]);
    std::swap(e_undirected_H_[ht0], e_undirected_H_[ht1]);
    // update things that eat e0/e1 indices
    std::swap(h_directed_E_[e0], h_directed_E_[e1]);
    V_cycle_E_.row(e0).swap(V_cycle_E_.row(e1));
  }

  void swap_face_indices(int f0, int f1) {
    check_f(f0);
    check_f(f1);
    if (f0 == f1) {
      return;
    }
    // update things that output f0/f1 indices
    for (auto h : generate_H_right_f(f0)) {
      f_left_H_[h] = f1;
    }
    for (auto h : generate_H_right_f(f1)) {
      f_left_H_[h] = f0;
    }
    // update things that eat f0/f1 indices
    std::swap(h_right_F_[f0], h_right_F_[f1]);
    V_cycle_F_.row(f0).swap(V_cycle_F_.row(f1));
  }

  // void swap_boundary_indices(int b0, int b1) {
  //   check_b(b0);
  //   check_b(b1);
  //   if (b0 == b1) {
  //     return;
  //   }
  //   int h0 = h_negative_B_[b0];
  //   int h1 = h_negative_B_[b1];
  //   // update things that output b0/b1 indices
  //   std::swap(f_left_H_[h0], f_left_H_[h1]);
  //   // update things that eat b0/b1 indices
  //   std::swap(h_negative_B_[b0], h_negative_B_[b1]);
  // }
  void swap_boundary_indices(int b0, int b1) {
    check_b(b0);
    check_b(b1);
    if (b0 == b1) {
      return;
    }
    // update things that output b0/b1 indices
    for (int h : generate_H_negative_b(b0)) {
      f_left_H_[h] = -(b1 + 1);
    }
    for (int h : generate_H_negative_b(b1)) {
      f_left_H_[h] = -(b0 + 1);
    }
    // update things that eat b0/b1 indices
    std::swap(h_negative_B_[b0], h_negative_B_[b1]);
  }

  void swap_halfedge_indices1(int h0, int h1) {
    check_h(h0);
    check_h(h1);
    if (h0 == h1) {
      return;
    }

    auto swap_ref = [h0, h1](int &h) {
      if (h == h0) {
        h = h1;
      } else if (h == h1) {
        h = h0;
      }
    };

    int v_origin_h0 = v_origin_H_[h0];
    int e_undirected_h0 = e_undirected_H_[h0];
    int f_left_h0 = f_left_H_[h0];
    int b_left_h0 = -(f_left_h0 + 1);
    int h_twin_h0 = h_twin_H_[h0];
    int h_next_h0 = h_next_H_[h0];
    int h_prev_h0 = h_prev_h(h0);

    int v_origin_h1 = v_origin_H_[h1];
    int e_undirected_h1 = e_undirected_H_[h1];
    int f_left_h1 = f_left_H_[h1];
    int b_left_h1 = -(f_left_h1 + 1);
    int h_twin_h1 = h_twin_H_[h1];
    int h_next_h1 = h_next_H_[h1];
    int h_prev_h1 = h_prev_h(h1);

    // update things that output h0/h1 indices
    // h_out_V_, h_directed_E_, h_right_F_, h_negative_B_
    // h_next_H_, h_twin_H_
    swap_ref(h_out_V_[v_origin_h0]);
    if (v_origin_h1 != v_origin_h0) {
      swap_ref(h_out_V_[v_origin_h1]);
    }

    swap_ref(h_directed_E_[e_undirected_h0]);
    if (e_undirected_h1 != e_undirected_h0) {
      swap_ref(h_directed_E_[e_undirected_h1]);
    }

    auto swap_face_ref = [&](int f) {
      if (f >= 0) {
        swap_ref(h_right_F_[f]);
      } else {
        swap_ref(h_negative_B_[-(f + 1)]);
      }
    };

    swap_face_ref(f_left_h0);
    if (f_left_h1 != f_left_h0) {
      swap_face_ref(f_left_h1);
    }

    std::swap(h_next_H_[h_prev_h0], h_next_H_[h_prev_h1]);
    std::swap(h_twin_H_[h_twin_h0], h_twin_H_[h_twin_h1]);
    // update things that eat h0/h1 indices
    std::swap(v_origin_H_[h0], v_origin_H_[h1]);
    std::swap(e_undirected_H_[h0], e_undirected_H_[h1]);
    std::swap(f_left_H_[h0], f_left_H_[h1]);

    std::swap(h_next_H_[h0], h_next_H_[h1]);
    std::swap(h_twin_H_[h0], h_twin_H_[h1]);
  }

  void swap_halfedge_indices(int h0, int h1) {
    check_h(h0);
    check_h(h1);
    if (h0 == h1) {
      return;
    }

    int v_origin_h0 = v_origin_H_[h0];
    int e_undirected_h0 = e_undirected_H_[h0];
    int f_left_h0 = f_left_H_[h0];
    int b_left_h0 = -(f_left_h0 + 1);
    int h_twin_h0 = h_twin_H_[h0];
    int h_next_h0 = h_next_H_[h0];
    int h_prev_h0 = h_prev_h(h0);

    int v_origin_h1 = v_origin_H_[h1];
    int e_undirected_h1 = e_undirected_H_[h1];
    int f_left_h1 = f_left_H_[h1];
    int b_left_h1 = -(f_left_h1 + 1);
    int h_twin_h1 = h_twin_H_[h1];
    int h_next_h1 = h_next_H_[h1];
    int h_prev_h1 = h_prev_h(h1);

    assert(h_next_H_[h_prev_h0] == h0);
    assert(h_next_H_[h_prev_h1] == h1);

    // update things that output h0/h1 indices
    // h_out_V_, h_directed_E_, h_right_F_, h_negative_B_
    // h_next_H_, h_twin_H_
    if (h_out_V_[v_origin_h0] == h0) {
      h_out_V_[v_origin_h0] = h1;
    }
    if (h_out_V_[v_origin_h1] == h1) {
      h_out_V_[v_origin_h1] = h0;
    }
    if (h_directed_E_[e_undirected_h0] == h0) {
      h_directed_E_[e_undirected_h0] = h1;
    }
    if (h_directed_E_[e_undirected_h1] == h1) {
      h_directed_E_[e_undirected_h1] = h0;
    }
    if (f_left_h0 >= 0) {
      if (h_right_F_[f_left_h0] == h0) {
        h_right_F_[f_left_h0] = h1;
      }
    } else {
      if (h_negative_B_[b_left_h0] == h0) {
        h_negative_B_[b_left_h0] = h1;
      }
    }
    if (f_left_h1 >= 0) {
      if (h_right_F_[f_left_h1] == h1) {
        h_right_F_[f_left_h1] = h0;
      }
    } else {
      if (h_negative_B_[b_left_h1] == h1) {
        h_negative_B_[b_left_h1] = h0;
      }
    }
    std::swap(h_next_H_[h_prev_h0], h_next_H_[h_prev_h1]);
    std::swap(h_twin_H_[h_twin_h0], h_twin_H_[h_twin_h1]);
    // update things that eat h0/h1 indices
    std::swap(v_origin_H_[h0], v_origin_H_[h1]);
    std::swap(e_undirected_H_[h0], e_undirected_H_[h1]);
    std::swap(f_left_H_[h0], f_left_H_[h1]);

    std::swap(h_next_H_[h0], h_next_H_[h1]);
    std::swap(h_twin_H_[h0], h_twin_H_[h1]);
  }

  void move_vertex_indices_to_end(std::vector<int> V) {
    int N = get_num_vertices();
    int n = V.size();
    std::sort(V.begin(), V.end());
    // swap input indices with [N-n,N-n+1,...,N-1]
    // starting with the largest
    for (int i = 0; i < n; i++) {
      swap_vertex_indices(V[n - 1 - i], N - 1 - i);
    }
  }

  void move_edge_indices_to_end(std::vector<int> E) {
    int N = get_num_edges();
    int n = E.size();
    std::sort(E.begin(), E.end());
    // swap input indices with [N-n,N-n+1,...,N-1]
    // starting with the largest
    for (int i = 0; i < n; i++) {
      swap_edge_indices(E[n - 1 - i], N - 1 - i);
    }
  }

  void move_face_indices_to_end(std::vector<int> F) {
    int N = get_num_faces();
    int n = F.size();
    std::sort(F.begin(), F.end());
    // swap input indices with [N-n,N-n+1,...,N-1]
    // starting with the largest
    for (int i = 0; i < n; i++) {
      swap_vertex_indices(F[n - 1 - i], N - 1 - i);
    }
  }

  void move_boundary_indices_to_end(std::vector<int> B) {
    int N = get_num_boundaries();
    int n = B.size();
    std::sort(B.begin(), B.end());
    // swap input indices with [N-n,N-n+1,...,N-1]
    // starting with the largest
    for (int i = 0; i < n; i++) {
      swap_vertex_indices(B[n - 1 - i], N - 1 - i);
    }
  }

  void move_half_edge_indices_to_end(std::vector<int> H) {
    int N = get_num_half_edges();
    int n = H.size();
    std::sort(H.begin(), H.end());
    // swap input indices with [N-n,N-n+1,...,N-1]
    // starting with the largest
    for (int i = 0; i < n; i++) {
      swap_halfedge_indices(H[n - 1 - i], N - 1 - i);
    }
  }

  ///////////////////////////////////////////////////////
  // Mutators ///////////////////////////////////////////
  ///////////////////////////////////////////////////////

  /**
   * @brief Flips h and its twin.
   *
   * @param h
   *
   *```
   *         v1                           v1
   *       /    \                       /  |  \
   *      /      \                     /   |   \
   *     /h3    h2\                   /h3  |  h2\
   *    /    f0    \                 /     |     \
   *   /            \               /  f0  |  f1  \
   *  /      h0      \             /       |       \
   * v2--------------v0  |----->  v2     h0|h1     v0
   *  \      h1      /             \       |       /
   *   \            /               \      |      /
   *    \    f1    /                 \     |     /
   *     \h4    h5/                   \h4  |  h5/
   *      \      /                     \   |   /
   *       \    /                       \  |  /
   *         v3                           v3
   * ```
   */
  void flip_hedge(int h);

  void flip_edge(int e);

  int flip_non_delaunay();

  void rigid_transform(const Vec3d &translation, const Vec3d &rotation);

  ///////////////////////////////////////////////////////
  // Prototyping ////////////////////////////////////////
  ///////////////////////////////////////////////////////
  // mcv with cotan laplacian
  Samples3d mean_curvature_vector_cotan() const;

  Samples3d mean_curvature_vector_graph_laplacian() const;

  Samplesi find_path_point_to_point(int v0, int v1, int max_steps) const;

  Vec3d compute_avg_xyz_coord() const;

  /////////////////////////////////////////////////
  // Simplicial operations ////////////////////////
  /////////////////////////////////////////////////
  SimplicialTriple star_of_vertex(int v) const;
  SimplicialTriple star_of_edge(int e) const;
  SimplicialTriple star(SimplicialSet &V, SimplicialSet &E,
                        SimplicialSet &F) const;
  SimplicialTriple link(SimplicialSet &V, SimplicialSet &E, SimplicialSet &F);
  SimplicialTriple closure(SimplicialTriple &VEF) const;

  ///////////////////////////////////////////////////////
  // Quadrature /////////////////////////////////////////
  ///////////////////////////////////////////////////////

  Vec3d bary_coord_q(int q) const { return bary_coord_Q_.row(q); }
  Vec3d xyz_coord_fq(int f, int q) const;
  double quad_weight_fq(int f, int q) const {
    return dimensionless_quad_weight_Q_(q) * area_F_(f);
  }
  utils::SimpleGenerator<std::tuple<Vec3d, double>>
  generate_quad_points_weights_f(int f) const;
  void init_quad_points_and_weights();

  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  // From Membrane subclass /////////////////////////////
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////

  double get_gaussian_curvature_angle_defect_v(int v) const;
  void update_gaussian_curvature_angle_defect();
  void update_gaussian_curvature_laplacian();
  void update_gaussian_curvature();

  /**
   * @brief Divides face by adding a new vertex at the barycenter of the face
   *
   * @param f
   * @details
   * ```
   *                 v2                                       v2
   *               /   \                                    / | \
   *              /     \                                  /  |  \
   *             /       \                                /   |   \
   *            /         \                              /    |    \
   *           /           \                            /   h3|h6   \
   *          /             \                          /      |      \
   *         /               \                        /       |       \
   *        /h2             h1\                      /h2 f2  v3   f1 h1\
   *       /        f0         \                    /       /   \       \
   *      /                     \                  /      /       \      \
   *     /                       \                /   h7/h4       h8\h5   \
   *    /                         \              /    /      f0       \    \
   *   /                           \            /   /                   \   \
   *  /             h0              \  ---->   /  /          h0           \  \
   * v0 ---------------------------v1         v0 ---------------------------v1
   * ```
   */
  void divide_face_barycentric(int f);
  void divide_faces_barycentric();
  void refine_messy_icososphere();

  /**
   * @brief Divides face by adding a new vertex at the midpoint of each edge
   *
   * @param f
   *
   * ```
   *                  v2                                   v2
   *                /   \                                /   \
   *               /     \                              /     \
   *              /       \                            /       \
   *             /         \                          /         \
   *            /           \                        /           \
   *           /             \                      /     f3      \
   *          /               \                    /               \
   *         /h2             h1\                  v5----------------v4
   *        /        f0         \                /  \              / \
   *       /                     \              /    \     f0     /   \
   *      /                       \            /      \          /     \
   *     /                         \          /        \        /       \
   *    /                           \        /    f1    \      /   f2    \
   *   /                             \      /            \    /           \
   *  /              h0               \    /              \  /             \
   * v0-------------------------------v1  v0---------------v3--------------v1
   * ```
   */
  void divide_faces();

  void remove_f(int f) {
    // TODO: boundary cases
    int Nf0 = get_num_faces();
    int Nf = Nf0 - 1;
    int Nb0 = get_num_boundaries();
    int Nb = Nb0 + 1;
    int b = Nb0;
    int f0 = Nf0 - 1;

    swap_face_indices(f, f0);

    int h0 = h_right_F_[f0];
    int h1 = h_next_H_[h0];
    int h2 = h_next_H_[h1];

    h_negative_B_.conservativeResize(Nb);
    h_negative_B_[b] = h0;

    f_left_H_[h0] = -(b + 1);
    f_left_H_[h1] = -(b + 1);
    f_left_H_[h2] = -(b + 1);

    h_right_F_.conservativeResize(Nf);
    V_cycle_F_.conservativeResize(Nf, 3);
  }

  void glue_boundaries(MatrixMesh &m_other, int b_this, int b_other) {
    int Nb_other = m_other.get_num_boundaries();
    int Nb_this = get_num_boundaries();
    // b_this/b_other will be deleted so swap with final b index
    m_other.swap_boundary_indices(b_other, Nb_other - 1);
    swap_boundary_indices(b_this, Nb_this - 1);
    b_other = Nb_other - 1;
    b_this = Nb_this - 1;

    // Get h cycle along this negatively oriented boundary
    std::vector<int> H_negative_this;
    for (int h : generate_H_negative_b(b_this)) {
      H_negative_this.push_back(h);
    }
    // Find v, e, and h indices along other boundary that will be
    // deleted/overwritten by indices in this boundary after gluing
    std::vector<int> V_other;
    std::vector<int> E_other;
    std::vector<int> H_other;
    int size_b = 0; // counts # v/e in other boundary
    for (int h : m_other.generate_H_negative_b(b_other)) {
      size_b++;
      V_other.push_back(m_other.v_origin_h(h));
      E_other.push_back(m_other.e_undirected_h(h));
      H_other.push_back(h);
      H_other.push_back(m_other.h_twin_h(h));
    }
    if (H_negative_this.size() != size_b) {
      throw std::runtime_error(
          "glue_boundaries requires boundaries to be the same size");
    }

    // Relabel v, e, and h indices along other boundary that will be
    // deleted/overwritten by gluing.
    m_other.move_vertex_indices_to_end(V_other);
    m_other.move_edge_indices_to_end(E_other);
    m_other.move_half_edge_indices_to_end(H_other);

    // Find closest vertex in other boundary to starting vertex in this
    // boundary and reassign h_negative_b_other
    double min_dist = std::numeric_limits<double>::infinity();
    int h_other = -1;
    for (int h : m_other.generate_H_negative_b(b_other)) {
      int ht = m_other.h_twin_h(h);
      int v_other = m_other.v_origin_h(ht);
      int h_this = h_negative_b(b_this);
      int v_this = v_origin_h(h_this);
      double dist = (m_other.xyz_coord_v(v_other) - xyz_coord_v(v_this)).norm();
      if (dist < min_dist) {
        min_dist = dist;
        h_other = h;
      }
    }
    m_other.h_negative_B_[b_other] = h_other;

    //
    int Nv_this = get_num_vertices();
    int Ne_this = get_num_edges();
    int Nf_this = get_num_faces();
    int Nh_this = get_num_half_edges();

    int Nv_other = m_other.get_num_vertices();
    int Ne_other = m_other.get_num_edges();
    int Nf_other = m_other.get_num_faces();
    int Nh_other = m_other.get_num_half_edges();

    int dNv = Nv_other - size_b;
    int dNe = Ne_other - size_b;
    int dNf = Nf_other;
    int dNb = Nb_other - 2;
    int dNh = Nh_other - 2 * size_b;

    int Nv = Nv_this + dNv;
    int Ne = Ne_this + dNe;
    int Nf = Nf_this + dNf;
    int Nb = Nb_this + dNb;
    int Nh = Nh_this + dNh;

    // shift so other indices that will be added to this by gluing.
    Samples3d new_xyz_coord_V = m_other.xyz_coord_V_.topRows(dNv);
    Samplesi new_h_out_V = m_other.h_out_V_.head(dNv).array() + Nh_this;

    Samples2i new_V_cycle_E = m_other.V_cycle_E_.topRows(dNe).array() + Nv_this;
    Samplesi new_h_directed_E =
        m_other.h_directed_E_.head(dNe).array() + Nh_this;

    Samples3i new_V_cycle_F = m_other.V_cycle_F_.array() + Nv_this;
    Samplesi new_h_right_F = m_other.h_right_F_.array() + Nh_this;

    Samplesi new_h_negative_B =
        m_other.h_negative_B_.head(Nb_other - 1).array() + Nh_this;

    Samplesi new_v_origin_H = m_other.v_origin_H_.head(dNh).array() + Nv_this;
    Samplesi new_e_undirected_H =
        m_other.e_undirected_H_.head(dNh).array() + Ne_this;
    Samplesi new_f_left_H = m_other.f_left_H_.head(dNh);
    for (int &f : new_f_left_H) {
      f += (f >= 0) ? Nf_this : -(Nb_this - 1);
      // if (f >= 0) {
      //   f += Nf_this;
      //   continue;
      // }
      // int b = -(f + 1);
      // b += Nb_this - 1;
      // f = -(b + 1);
    }
    Samplesi new_h_next_H = m_other.h_next_H_.head(dNh).array() + Nh_this;
    Samplesi new_h_twin_H = m_other.h_twin_H_.head(dNh).array() + Nh_this;

    // get h indices in other boundary that will be replaced by h's in
    // H_negative_this with order chosen to be consistent with
    // H_negative_this. m_other.generate_H_negative_b(b_other) generates twins
    // of the h's in the order [h_{0}, h_{n-1}, h_{n-2},..., h_{1}], but we
    // want [h_{0}, h_{1}, h_{2},..., h_{n-1}] so we need to re-order
    H_other.resize(size_b);
    int i = 0;
    for (int h : m_other.generate_H_negative_b(b_other)) {
      int ht = m_other.h_twin_h(h);
      H_other[(size_b - i) % size_b] = ht;
      i++;
    }
    if (i != size_b) {
      throw std::runtime_error("something is wrong...");
    }

    // Map to swap references to v indices in other boundary to
    // v indices in this boundary.
    std::unordered_map<int, int> V_map_other_to_this;
    // Collections of e/f's incident on other boundary vertices deleted by
    // gluing that need to update V_cycle refs
    std::unordered_set<int> E_incident_b_other;
    std::unordered_set<int> F_incident_b_other;
    for (int i = 0; i < size_b; i++) {
      int h_other = H_other[i];
      int h_this = H_negative_this[i];
      int v_other = m_other.v_origin_h(h_other);
      int v_this = v_origin_h(h_this);
      V_map_other_to_this[v_other + Nv_this] = v_this;
      // H_map_other_to_this[h_other + Nh_this] = h_this;
      // H_prev_H_other.insert(m_other.h_prev_h(h_other));
      for (int h : m_other.generate_H_out_v_clockwise(v_other)) {
        if (h < dNh) {
          // The h's through other boundary vertices need to update their
          // v_origin refs. We skip h>=dNh since those are getting deleted.
          new_v_origin_H[h] = v_this;
        }
        int e = m_other.e_undirected_h(h);
        if (e < dNe) {
          E_incident_b_other.insert(e);
        }
        int f = m_other.f_left_h(h);
        if (f >= 0) {
          F_incident_b_other.insert(f);
        }
      }
      //
      int h_prev_other = m_other.h_prev_h(h_other);
      int h_next_other = m_other.h_next_h(h_other);
      int f_left_other = m_other.f_left_h(h_other);
      // other interior h's preceding other boundary h's need to update h_next
      // refs
      new_h_next_H[h_prev_other] = h_this;
      // this boundary h's need to update h_next refs
      h_next_H_[h_this] = h_next_other + Nh_this;
      // this boundary h's need to update f_left refs since they will no
      // longer be on a boundary after gluing
      f_left_H_[h_this] = f_left_other + Nf_this;
      if (m_other.h_right_f(f_left_other) == h_other) {
        // other f's may need to update h_right refs if they reference an h
        // deleted by gluing
        new_h_right_F[f_left_other] = h_this;
      }
    }

    auto swap_v_ref = [&V_map_other_to_this](int &v) {
      auto it = V_map_other_to_this.find(v);
      if (it != V_map_other_to_this.end()) {
        v = it->second;
      }
    };
    for (int f : F_incident_b_other) {
      for (int i = 0; i < 3; i++) {
        swap_v_ref(new_V_cycle_F(f, i));
      }
    }
    for (int e : E_incident_b_other) {
      for (int i = 0; i < 2; i++) {
        swap_v_ref(new_V_cycle_E(e, i));
      }
    }

    xyz_coord_V_.conservativeResize(Nv, 3);
    h_out_V_.conservativeResize(Nv);
    V_cycle_E_.conservativeResize(Ne, 2);
    h_directed_E_.conservativeResize(Ne);
    V_cycle_F_.conservativeResize(Nf, 3);
    h_right_F_.conservativeResize(Nf);
    h_negative_B_.conservativeResize(Nb);
    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    f_left_H_.conservativeResize(Nh);
    h_next_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);

    xyz_coord_V_.bottomRows(dNv) = new_xyz_coord_V;
    h_out_V_.tail(dNv) = new_h_out_V;

    V_cycle_E_.bottomRows(dNe) = new_V_cycle_E;
    h_directed_E_.tail(dNe) = new_h_directed_E;

    V_cycle_F_.bottomRows(dNf) = new_V_cycle_F;
    h_right_F_.tail(dNf) = new_h_right_F;

    h_negative_B_.tail(Nb_other - 1) = new_h_negative_B;

    v_origin_H_.tail(dNh) = new_v_origin_H;
    e_undirected_H_.tail(dNh) = new_e_undirected_H;
    f_left_H_.tail(dNh) = new_f_left_H;
    h_next_H_.tail(dNh) = new_h_next_H;
    h_twin_H_.tail(dNh) = new_h_twin_H;

    // check_he_matrices();
  }

  int find_opposite_vertex(int v0, double r_max = 0.1) {
    Vec3d x0 = xyz_coord_v(v0);
    Vec3d n0 = normal_v(v0); // = -ez
    int valance0 = get_valence_v(v0);
    // double r_max = 0.1;

    int v_opposite = -1;
    double dist_opposite = std::numeric_limits<double>::max();
    int Nv = get_num_vertices();
    // closest vertex in cylinder satisfying <n_v, n_v0> < 0
    for (int v = 0; v < Nv; v++) {
      int valance = get_valence_v(v);
      if (valance != valance0) {
        continue;
      }
      Vec3d x = xyz_coord_v(v);
      Vec3d x_x0 = x - x0;
      // double z = -n0.dot(x_x0);
      double r = n0.cross(x_x0).norm();
      if (r > r_max) {
        continue;
      }
      Vec3d n = normal_v(v);
      if (n0.dot(n) >= 0.0) {
        continue;
      }
      double dist = x_x0.norm();
      if (dist < dist_opposite) {
        dist_opposite = dist;
        v_opposite = v;
      }
    }

    if (v_opposite == -1) {
      throw std::runtime_error("Could not find opposite vertex");
    }
    return v_opposite;
  }

  void remove_v_neighborhood(int v) {
    int Nv0 = get_num_vertices();
    int Ne0 = get_num_edges();
    int Nf0 = get_num_faces();
    int Nb0 = get_num_boundaries();
    int Nh0 = get_num_half_edges();

    std::vector<int> E_remove;
    std::vector<int> F_remove;
    std::vector<int> H_remove;
    std::vector<int> H_bdry;

    for (int h : generate_H_out_v_clockwise(v)) {
      int ht = h_twin_h(h);
      int h_bdry = h_next_h(h);
      int e = e_undirected_h(h);
      int f = f_left_h(h);

      E_remove.push_back(e);
      F_remove.push_back(f);
      H_remove.push_back(h);
      H_remove.push_back(ht);
    }

    swap_vertex_indices(v, Nv0 - 1);
    v = Nv0 - 1;
    move_edge_indices_to_end(E_remove);
    move_face_indices_to_end(F_remove);
    move_half_edge_indices_to_end(H_remove);

    int Nv = Nv0 - 1;
    int Ne = Ne0 - E_remove.size();
    int Nf = Nf0 - F_remove.size();
    int Nb = Nb0 + 1;
    int Nh = Nh0 - H_remove.size();

    for (int h : generate_H_out_v_clockwise(v)) {
      int h_bdry = h_next_h(h);
      H_bdry.push_back(h_bdry);
      int ht = h_twin_h(h);
      int vt = v_origin_h(ht);
      if (h_out_V_[vt] == ht) {
        h_out_V_[vt] = h_bdry;
      }
    }
    std::reverse(H_bdry.begin(), H_bdry.end());
    xyz_coord_V_.conservativeResize(Nv, 3);
    h_out_V_.conservativeResize(Nv);
    V_cycle_E_.conservativeResize(Ne, 2);
    h_directed_E_.conservativeResize(Ne);
    V_cycle_F_.conservativeResize(Nf, 3);
    h_right_F_.conservativeResize(Nf);
    h_negative_B_.conservativeResize(Nb);
    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    f_left_H_.conservativeResize(Nh);
    h_next_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);

    h_negative_B_[Nb0] = H_bdry[0];
    int size_bdry = H_bdry.size();
    int f_b = -(Nb0 + 1);
    for (int i_h = 0; i_h < size_bdry; i_h++) {
      int h = H_bdry[i_h];
      int h_next = H_bdry[(i_h + 1) % size_bdry];
      h_next_H_[h] = h_next;
      f_left_H_[h] = f_b;
    }
  }

  void remove_v_and_v_opposite_neighborhood(int v0, double r_max = 0.1) {
    int Nv0 = get_num_vertices();
    int Ne0 = get_num_edges();
    int Nf0 = get_num_faces();
    int Nb0 = get_num_boundaries();
    int Nh0 = get_num_half_edges();

    std::vector<int> E_remove;
    std::vector<int> F_remove;
    std::vector<int> H_remove;
    std::vector<int> H_bdry;

    for (int h : generate_H_out_v_clockwise(v0)) {
      int ht = h_twin_h(h);
      int h_bdry = h_next_h(h);
      int e = e_undirected_h(h);
      int f = f_left_h(h);

      E_remove.push_back(e);
      F_remove.push_back(f);
      H_remove.push_back(h);
      H_remove.push_back(ht);
    }

    swap_vertex_indices(v0, Nv0 - 1);
    v0 = Nv0 - 1;
    move_edge_indices_to_end(E_remove);
    move_face_indices_to_end(F_remove);
    move_half_edge_indices_to_end(H_remove);

    int Nv = Nv0 - 1;
    int Ne = Ne0 - E_remove.size();
    int Nf = Nf0 - F_remove.size();
    int Nb = Nb0 + 1;
    int Nh = Nh0 - H_remove.size();

    for (int h : generate_H_out_v_clockwise(v0)) {
      int h_bdry = h_next_h(h);
      H_bdry.push_back(h_bdry);
      int ht = h_twin_h(h);
      int vt = v_origin_h(ht);
      if (h_out_V_[vt] == ht) {
        h_out_V_[vt] = h_bdry;
      }
    }
    std::reverse(H_bdry.begin(), H_bdry.end());
    xyz_coord_V_.conservativeResize(Nv, 3);
    h_out_V_.conservativeResize(Nv);
    V_cycle_E_.conservativeResize(Ne, 2);
    h_directed_E_.conservativeResize(Ne);
    V_cycle_F_.conservativeResize(Nf, 3);
    h_right_F_.conservativeResize(Nf);
    h_negative_B_.conservativeResize(Nb);
    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    f_left_H_.conservativeResize(Nh);
    h_next_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);

    h_negative_B_[Nb0] = H_bdry[0];
    int size_bdry = H_bdry.size();
    int f_b = -(Nb0 + 1);
    for (int i_h = 0; i_h < size_bdry; i_h++) {
      int h = H_bdry[i_h];
      int h_next = H_bdry[(i_h + 1) % size_bdry];
      h_next_H_[h] = h_next;
      f_left_H_[h] = f_b;
    }
  }

  void update_edges_from_he_mats() {
    int Nh = get_num_half_edges();
    int Ne = Nh / 2;
    h_directed_E_.resize(Ne);
    V_cycle_E_.resize(Ne, 2);
    e_undirected_H_.resize(Nh);

    std::unordered_set<int> H;
    H.reserve(Nh);

    for (int h = 0; h < Nh; ++h) {
      H.insert(h);
    }

    for (int e = 0; e < Ne; ++e) {
      int h = *H.begin();
      int ht = h_twin_h(h);

      int v0 = v_origin_h(h);
      int v1 = v_origin_h(ht);

      H.erase(h);
      H.erase(ht);

      e_undirected_H_[h] = e;
      e_undirected_H_[ht] = e;
      h_directed_E_[e] = h;
      V_cycle_E_.row(e) << v0, v1;
    }
  }

  void update_faces_from_he_mats() {
    //
    //
  }

  /**
   * @brief Collapse edge
   *
   * @param f
   *
   * ```
   *
   *
   * ```
   */

  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  // Laplacian stuff to fix /////////////////////////////
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  Samples1d cotan_laplacian(Samples1d &Q);
  Samples3d cotan_laplacian(Samples3d &Q);
  Samples1d belkin_laplacian(Samples1d &Q);
  Samples3d belkin_laplacian(Samples3d &Q);

  Samples1d adaptive_belkin_laplacian(Samples1d &Q);
  Samples3d adaptive_belkin_laplacian(Samples3d &Q);

  template <typename Samples> Samples guckenberger_laplacian(Samples &phi);

  Samples1d apply_laplacian_matrix(Samples1d &Q);
  Samples3d apply_laplacian_matrix(Samples3d &Q);

  // Eigen::MatrixXd laplacian(Eigen::MatrixXd &Q);
  Samples1d laplacian(Samples1d &Q);
  Samples3d laplacian(Samples3d &Q);

  void update_laplacian_matrix_cotan();
  void update_laplacian_matrix_belkin();
  void update_laplacian_matrix();
  void update_mean_curvature();

  ////////////////////////////////////////
  // Stuff that uses integration_patch_ //
  ////////////////////////////////////////

  // Patch integration_patch_;
  // void reset_integration_patch();
  //
  // template <typename Samples>
  // Samples higher_order_quad_heat_laplacian(Samples &phi);
  // template <typename Samples> Samples heat_laplacian(Samples &phi);
  //
  // void update_laplacian_matrix_adaptive_belkin();
  // void update_laplacian_matrix_heat();
  //
  // template <typename Samples>
  // Samples adaptive_guckenberger_laplacian(Samples &phi);

  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  // To be deprecated ///////////////////////////////////
  ///////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////
  void update_vef_from_he();

  int interactive_plot(int argc, char *argv[]);
  int spinning_plot(int argc, char *argv[]);
  int flipping_plot(int argc, char *argv[]);
  int uniform_flip_sweep();

  ///////////////////////////
  // Stuff to get rid of? /////
  ////////////////////////////
  // double heat_dt_{0.001};
  // adaptive_belkin_laplacian
  /**
   * @brief Initialize mesh data from half-edge matrices, set visual defaults,
   *
   */
  // void init_mesh();

  // /**
  //  * @brief Load minimal half-edge data from ply file at `ply_path_`.
  //  */
  // void load_ply();

  /**
   * @brief Recompute `V_cycle_E_` `V_cycle_F_` `e_undirected_H_`
   * `h_directed_E_` from half-edge matrices
   *
   */
  // void update_simplices_from_he();

  // /**
  //  * @brief Initialize the mesh from current half-edge matrices
  //  */
  // void init_from_he_mats();

  // /**
  //  * @brief Initialize the mesh from a ply file at current `ply_path_`
  //  */
  // void init_from_ply();

  // void update_mat_v(int v, const std::optional<Vec3d> &xyz_coord =
  // std::nullopt,
  //                   const std::optional<int> &h_out = std::nullopt);
  // void update_mat_h(int h, const std::optional<int> &v_origin = std::nullopt,
  //                   const std::optional<int> &h_next = std::nullopt,
  //                   const std::optional<int> &h_twin = std::nullopt,
  //                   const std::optional<int> &f_left = std::nullopt);
  // void update_mat_f(int f, const std::optional<int> &h_left = std::nullopt);

  // void update_he_from_vf();
};

/**
 * }
 */

} // namespace meshbrane
