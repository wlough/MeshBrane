#pragma once

/**
 * @file matrix_mesh.hpp
 * @brief Defines the MatrixMesh class.
 */

// #include "meshbrane/mesh_builder.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
// #include "meshbrane/meshbrane_object.hpp"
// #include "meshbrane/mesh_helpers.hpp"
#include "meshbrane/patch.hpp"
#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simple_vector_field.hpp"
#include <Eigen/Sparse>
// #include <algorithm>
#include <cassert>
#include <filesystem>
// #include <iostream>
// #include <limits>
// #include <numeric>
#include <stdexcept>
#include <string> // std::string
// #include <tuple>  // std::tuple
// #include <utility>
#include <yaml-cpp/yaml.h>

namespace meshbrane {

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

  ///////////////////////////////////////////////////
  // Constructors, initialization and mesh I/O //////
  ///////////////////////////////////////////////////

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

  /**
   * @brief Save mesh data as a .ply file.
   *
   * TODO: update me
   *
   * @param xyz_coord_V
   * ...
   * @param ply_path
   * @param use_binary
   */
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

  ///////////////////
  // Operators //////
  ///////////////////

  MatrixMesh &operator+=(const MatrixMesh &m);
  MatrixMesh operator+(const MatrixMesh &m) const;

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

  ////////////////////////////////////////////
  // Misc accessors/properties ///////////////
  ////////////////////////////////////////////

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

  /**
   * @brief Get the vertex cycles of the faces. dep???
   *
   * @return Samples3i
   */
  Samples3i V_cycle_F() const;

  /**
   * @brief Get the vertex positions and vertex cycles of faces in the mesh.
   * dep???
   *
   * @return VertexFaceTuple
   */
  VertexFaceTuple vf_samples() const;

  /**
   * @brief Get the vertex positions, vertex cycles of edges, and vertex
   * cycles of faces in the mesh. dep???
   *
   *
   * @return HalfEdgeTuple
   */
  VertexEdgeFaceTuple vef_samples() const;

  // dep???
  HalfEdgeTuple he_samples() const;

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

  //////////////////////
  // Cache updaters ////
  //////////////////////

  void update_boundary_cycles();
  void update_mesh_geometric_data_E();
  void update_mesh_geometric_data_F();
  void update_mesh_geometric_data_V();
  void update_mesh_volume();
  void update_mesh_geometric_data();

  /////////////////////////////
  // Simple geometry helpers //
  /////////////////////////////

  double length_e(int e) const;

  /**
   * @brief Compute `math::heron_area` of face `f`
   *
   * @param f
   * @return double
   */
  double heron_area_f(int f) const;

  /**
   * @brief Compute the unit normal vector of face `f`
   *
   * @param f
   * @return Eigen::Vector3d
   */
  Vec3d normal_f(int f) const;

  /**
   * @brief Compute the unit normal vector of vertex `v`. Does not use
   * precomputed face areas/normals.
   *
   * @param v
   * @return Eigen::Vector3d
   */
  Vec3d normal_v(int v) const;

  double signed_volume_f(int f) const;

  Vec3d compute_avg_xyz_coord() const;

  //////////////////////////////
  // Gaussian curvature stuff //
  //////////////////////////////

  double get_gaussian_curvature_angle_defect_v(int v) const;
  void update_gaussian_curvature_angle_defect();
  void update_gaussian_curvature_laplacian();
  void update_gaussian_curvature();

  //////////////////////////////////////////////
  // Visualization /////////////////////////////
  //////////////////////////////////////////////
  /**
   * @brief Get points in half-edges shifted towards triangle centers for
   * visualization
   */
  std::array<Samples3d, 3> compute_shifted_half_edge_arrows();

  void update_shifted_half_edge_arrows();

  void update_vector_field_arrows(Samples3d X, Samples3d U);

  void update_mesh_visuals();
  /**
   * @brief Set default rgb values and vertex radii for visualization
   */
  void set_visual_defaults();

  ///////////////////////////////////////////////////////
  // Quadrature /////////////////////////////////////////
  ///////////////////////////////////////////////////////

  Vec3d bary_coord_q(int q) const { return bary_coord_Q_.row(q); }
  Vec3d xyz_coord_fq(int f, int q) const;
  double quad_weight_fq(int f, int q) const {
    return dimensionless_quad_weight_Q_(q) * area_F_(f);
  }
  void init_quad_points_and_weights();

  ///////////////////////////////
  // Re-indexing ////////////////
  ///////////////////////////////

  void swap_vertex_indices(int v0, int v1);

  void swap_edge_indices(int e0, int e1);

  void swap_face_indices(int f0, int f1);

  void swap_boundary_indices(int b0, int b1);

  void swap_halfedge_indices1(int h0, int h1);

  void swap_halfedge_indices(int h0, int h1);

  void move_vertex_indices_to_end(std::vector<int> V);

  void move_edge_indices_to_end(std::vector<int> E);

  void move_face_indices_to_end(std::vector<int> F);

  void move_boundary_indices_to_end(std::vector<int> B);

  void move_half_edge_indices_to_end(std::vector<int> H);

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

  /////////////////////////
  // Convergence testing //
  /////////////////////////

  void refine_messy_icososphere();

  /////////////////////////////////////////////////
  // Simplicial operations ////////////////////////
  /////////////////////////////////////////////////
  SimplicialTriple star_of_vertex(int v) const;
  SimplicialTriple star_of_edge(int e) const;
  SimplicialTriple star(SimplicialSet &V, SimplicialSet &E,
                        SimplicialSet &F) const;
  SimplicialTriple link(SimplicialSet &V, SimplicialSet &E, SimplicialSet &F);
  SimplicialTriple closure(SimplicialTriple &VEF) const;

  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // Prototyping ///////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  Samples3d mean_curvature_vector_graph_laplacian() const;

  void remove_f(int f);

  void glue_boundaries(MatrixMesh &m_other, int b_this, int b_other);

  void glue_boundaries(int b0, int b1);

  int find_opposite_vertex(int v0, double r_max = 0.1);

  void remove_v_neighborhood(int v);

  void remove_v_and_v_opposite_neighborhood(int v0, double r_max = 0.1);

  void reverse_orientation();

  Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> get_V_cycle_E0() const;

  Eigen::Matrix<int, Eigen::Dynamic, 2> get_V_cycle_E1() const;

  Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> get_V_cycle_F0() const;

  Eigen::Matrix<int, Eigen::Dynamic, 3> get_V_cycle_F1() const;

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
  // void update_vef_from_he();

  ///////////////////////////
  // Stuff to get rid of? ///
  ///////////////////////////

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

  // void update_edges_from_he_mats() {
  //   int Nh = get_num_half_edges();
  //   int Ne = Nh / 2;
  //   h_directed_E_.resize(Ne);
  //   V_cycle_E_.resize(Ne, 2);
  //   e_undirected_H_.resize(Nh);
  //
  //   std::unordered_set<int> H;
  //   H.reserve(Nh);
  //
  //   for (int h = 0; h < Nh; ++h) {
  //     H.insert(h);
  //   }
  //
  //   for (int e = 0; e < Ne; ++e) {
  //     int h = *H.begin();
  //     int ht = h_twin_h(h);
  //
  //     int v0 = v_origin_h(h);
  //     int v1 = v_origin_h(ht);
  //
  //     H.erase(h);
  //     H.erase(ht);
  //
  //     e_undirected_H_[h] = e;
  //     e_undirected_H_[ht] = e;
  //     h_directed_E_[e] = h;
  //     V_cycle_E_.row(e) << v0, v1;
  //   }
  // }
};

/**
 * }
 */

} // namespace meshbrane
