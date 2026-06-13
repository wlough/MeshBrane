#pragma once

/**
 * @file matrix_mesh.hpp
 * @brief Defines the MatrixMesh class.
 */

#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane/patch.hpp"
#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simple_vector_field.hpp"
#include <Eigen/Sparse>
#include <filesystem>
#include <iostream>
#include <optional> // std::optional
#include <string>   // std::string
#include <tuple>    // std::tuple
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
 * - generate_H_right_b(int b) : utils::SimpleGenerator<int>
 * - generate_F_incident_v(int v) : utils::SimpleGenerator<int>
 *
 * ## Mutators
 *
 * - update_mat_v(int v,...)
 * - update_mat_h(int h,...)
 * - update_mat_f(int f,...)
 * - flip_hedge(int h) // does not update simplices
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
class MatrixMesh : public MeshBraneObject {
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

  void set_attributes_from_yaml_node(const YAML::Node &node) override;
  void init(const YAML::Node &node) override;

  void init_matrixmesh_from_attributes();

  void check_he_matrices() const {
    printf("MatrixMesh::check_he_matrices\n");
    int Nv = get_num_vertices();
    int Ne = get_num_edges();
    int Nf = get_num_faces();
    int Nh = get_num_half_edges();
    int Nb = get_num_boundaries();

    for (int v = 0; v < Nv; v++) {
      int h_out = h_out_V_(v);
      if (h_out < 0 || h_out >= Nh) {
        std::cout << "v=" << v << '\n';
        std::cout << "h_out=" << h_out << '\n';
        throw std::out_of_range("h_out_V_[v]: Half-edge index out of range");
      }
    }
    for (int e = 0; e < Ne; e++) {
      int h_directed = h_directed_E_[e];
      if (h_directed < 0 || h_directed >= Nh) {
        std::cout << "e=" << e << '\n';
        std::cout << "h_directed=" << h_directed << '\n';
        throw std::out_of_range(
            "h_directed_E_[e]: Half-edge index out of range");
      }
    }
    for (int f = 0; f < Nf; f++) {
      int h_right = h_right_F_[f];
      if (h_right < 0 || h_right >= Nh) {
        std::cout << "f=" << f << '\n';
        std::cout << "h_right=" << h_right << '\n';
        throw std::out_of_range("h_right_F_[f]: Half-edge index out of range");
      }
    }
    for (int b = 0; b < Nb; b++) {
      int h_negative = h_negative_B_[b];
      if (h_negative < 0 || h_negative >= Nh) {
        std::cout << "b=" << b << '\n';
        std::cout << "h_negative=" << h_negative << '\n';
        throw std::out_of_range(
            "h_negative_B_[b]: Half-edge index out of range");
      }
    }
    for (int h = 0; h < Nh; h++) {
      int v_origin = v_origin_H_[h];
      if (v_origin < 0 || v_origin >= Nv) {
        std::cout << "h=" << h << '\n';
        std::cout << "v_origin=" << v_origin << '\n';
        throw std::out_of_range("v_origin_H_[h]: Vertex index out of range");
      }
      int e_undirected = e_undirected_H_[h];
      if (e_undirected < 0 || e_undirected >= Ne) {
        std::cout << "h=" << h << '\n';
        std::cout << "e_undirected=" << e_undirected << '\n';
        throw std::out_of_range("e_undirected_H_[h]: Edge index out of range");
      }
      int f_left = f_left_H_[h];
      if (f_left < -Nb || f_left >= Nf) {
        std::cout << "h=" << h << '\n';
        std::cout << "f_left=" << f_left << '\n';
        throw std::out_of_range("f_left_H_[h]: Face index out of range");
      }
      int h_next = h_next_H_[h];
      if (h_next < 0 || h_next >= Nh) {
        std::cout << "h=" << h << '\n';
        std::cout << "h_next=" << h_next << '\n';
        throw std::out_of_range("h_next_H_[h]: Half-edge index out of range");
      }
      int h_twin = h_twin_H_[h];
      if (h_twin < 0 || h_twin >= Nh) {
        std::cout << "h=" << h << '\n';
        std::cout << "h_twin=" << h_twin << '\n';
        throw std::out_of_range("h_twin_H_[h]: Half-edge index out of range");
      }
    }
  }

  enum class LaplacianType { COTAN, BELKIN, GUCKENBERGER, HEAT };
  enum class GaussianCurvatureType { ANGLE_DEFECT, LAPLACIAN };

  LaplacianType
  laplacian_type_from_string(const std::string &laplacian_type) const {
    if (laplacian_type == "cotan") {
      return LaplacianType::COTAN;
    } else if (laplacian_type == "belkin") {
      return LaplacianType::BELKIN;
    } else if (laplacian_type == "guckenberger") {
      return LaplacianType::GUCKENBERGER;
    } else if (laplacian_type == "heat") {
      return LaplacianType::HEAT;
    } else {
      throw std::runtime_error(
          "laplacian_type_from_string: Unknown laplacian type");
    }
  }

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

  LaplacianType laplacian_type_{LaplacianType::COTAN};

  GaussianCurvatureType gaussian_curvature_type_{
      GaussianCurvatureType::ANGLE_DEFECT};

  ////////////////////////////////
  // Stuff from parameters file //
  ////////////////////////////////
  YAML::Node parameters_;

  std::filesystem::path ply_path_;

  bool show_half_edges_{false};
  bool show_vertices_{false};
  bool show_edges_{true};
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

  ////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////
  // to be categorized start /////////////////////////////////////
  ////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////
  double heat_dt_{0.001};
  ////////////////////////////
  // Cached geometric data ///
  ////////////////////////////
  std::vector<Samplesi> V_cycle_B_;
  Samples3d vec_H_;
  Samples1d length_E_;
  Samples1d area_F_;
  Samples3d normal_F_;
  Samples3d normal_V_;
  Samples1d area_V_;
  double average_edge_length_{0.0};
  double average_face_area_{0.0};
  double total_volume_{0.0};
  // Curvature
  Samples3d mcvec_V_;
  Samples1d mean_curvature_V_;
  Samples1d gaussian_curvature_V_;
  Samples1d lap_mean_curvature_V_;

  ////////////////////////////
  // Laplacian data //////////
  ////////////////////////////
  // Patch integration_patch_test;
  Patch integration_patch_;

  Eigen::SparseMatrix<double, Eigen::RowMajor>
      laplacian_matrix_V_; // row major for insertion efficiency
  ////////////////////////////
  // Other data //////////////
  ////////////////////////////
  Samples3d force_V_;

  Eigen::Matrix<double, Eigen::Dynamic, 1> radius_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_H;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_F;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_E_;

  std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3>
      shifted_half_edge_arrows_;
  SimpleVectorField vector_field_arrows_;

  ////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////
  // to be categorized end ///////////////////////////////////////
  ////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////

  void set_attributes_from_parameters() {
    printf("MatrixMesh::set_attributes_from_parameters\n");
    if (parameters_["ply_path"]) {
      ply_path_ =
          std::filesystem::path(parameters_["ply_path"].as<std::string>());
    }
    if (parameters_["draw_wireframe"]) {
      draw_wireframe_ = parameters_["draw_wireframe"].as<bool>(); // true
    }
    if (parameters_["show_half_edges"]) {
      show_half_edges_ = parameters_["show_half_edges"].as<bool>(); // true
    }
    if (parameters_["show_vertices"]) {
      show_vertices_ = parameters_["show_vertices"].as<bool>(); // true
    }
    if (parameters_["show_edges"]) {
      show_edges_ = parameters_["show_edges"].as<bool>(); // true
    }
    if (parameters_["rgba_face"]) {
      rgba_face_ = Eigen::Map<Eigen::Vector4d>(
          parameters_["rgba_face"].as<std::vector<double>>().data());
    }
    if (parameters_["rgba_edge"]) {
      rgba_edge_ = Eigen::Map<Eigen::Vector4d>(
          parameters_["rgba_edge"].as<std::vector<double>>().data());
    }
    if (parameters_["rgba_vertex"]) {
      rgba_vertex_ = Eigen::Map<Eigen::Vector4d>(
          parameters_["rgba_vertex"].as<std::vector<double>>().data());
    }
    if (parameters_["rgba_half_edge"]) {
      rgba_half_edge_ = Eigen::Map<Eigen::Vector4d>(
          parameters_["rgba_half_edge"].as<std::vector<double>>().data());
    }
    if (parameters_["radius_vertex"]) {
      radius_vertex_ = parameters_["radius_vertex"].as<double>();
    }
    if (parameters_["laplacian_type"]) {
      laplacian_type_ = laplacian_type_from_string(
          parameters_["laplacian_type"].as<std::string>());
    }
    if (parameters_["atol"]) {
      belkin_atol_ = parameters_["atol"].as<double>();
    }
    if (parameters_["rtol"]) {
      belkin_rtol_ = parameters_["rtol"].as<double>();
    }
    if (parameters_["belkin_dt"]) {
      belkin_dt_ = parameters_["belkin_dt"].as<double>();
    }
    if (parameters_["belkin_min_ring"]) {
      belkin_min_ring_ = parameters_["belkin_min_ring"].as<int>();
    }
    if (parameters_["heat_dt_multiple"]) {
      heat_dt_multiple_ = parameters_["heat_dt_multiple"].as<double>();
    }
    if (parameters_["construct_laplacian_matrix"]) {
      construct_laplacian_matrix_ =
          parameters_["construct_laplacian_matrix"].as<bool>();
    }
    if (parameters_["gaussian_curvature_type"]) {
      gaussian_curvature_type_ = gaussian_curvature_type_from_string(
          parameters_["gaussian_curvature_type"].as<std::string>());
    }
  }

  /**
   * @brief Initialize the mesh from current half-edge matrices
   */
  void init_from_he_mats() {
    update_simplices_from_he();
    update_boundary_cycles();
    update_mesh_geometric_data_E();
    update_mesh_geometric_data_F();
    update_mesh_geometric_data_V();
    update_mesh_volume();
    set_visual_defaults();
    update_mesh_visuals();
    init_quad_points_and_weights();
  }

  /**
   * @brief Initialize the mesh from a ply file at current `ply_path_`
   */
  void init_from_ply() {
    load_ply();
    init_from_he_mats();
    integration_patch_ = Patch(this);
  }

  void update_vector_field_arrows(Samples3d X, Samples3d U);
  ////////////////////////////////////
  // Cache updaters //////////////////
  ////////////////////////////////////

  void load_ply();

  /**
   * @brief Recompute `V_cycle_E_` `V_cycle_F_` `e_undirected_H_`
   * `h_directed_E_` from half-edge matrices
   *
   */
  void update_simplices_from_he();
  void update_boundary_cycles();

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

  double signed_volume_f(int f) const;
  void update_mesh_geometric_data_E();
  void update_mesh_geometric_data_F();
  void update_mesh_geometric_data_V();
  void update_mesh_volume();
  void update_mesh_geometric_data();

  /**
   * @brief Initialize mesh data from half-edge matrices, set visual defaults,
   *
   */
  void init_mesh();

  ///////////////////////////////////////////////////////
  // Constructors and Mesh I/O //////////////////////////
  ///////////////////////////////////////////////////////

  MatrixMesh(const YAML::Node &parameters) {
    parameters_ = parameters;
    set_attributes_from_parameters();
    if (ply_path_.empty()) {
      throw std::runtime_error(
          "MatrixMesh constructor: ply_path is required in parameters");
    }
    init_from_ply();
  }

  MatrixMesh() = default;
  ~MatrixMesh() = default;

  /**
   * @brief Construct a new MatrixMesh object from the vectors in a
   * `meshbrane::MatrixMeshSamples` tuple.
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
             const Samplesi &v_origin_H, const Samplesi &h_next_H,
             const Samplesi &h_twin_H, const Samplesi &f_left_H,
             const Samplesi &h_right_F, const Samplesi &h_negative_B);

  MatrixMesh(const std::filesystem::path &ply_path);
  /**
   * @brief Construct a new MatrixMesh object from ply file of half-edge
   * samples.
   * @param ply_path
   * @return MatrixMesh
   */
  static MatrixMesh from_he_ply(const std::filesystem::path &ply_path);
  static MatrixMesh from_icosohedron();
  void init_icosohedron();
  static MatrixMesh from_icososphere(int n);
  void write_he_ply(const std::filesystem::path &ply_path) const;

  /**
   * @brief Construct a new MatrixMesh object from the vectors in a
   * `meshbrane::VertexFaceSamples` tuple.
   * @param xyz_coord_V
   * @param V_cycle_F
   * @return MatrixMesh<VertexType>
   */
  static MatrixMesh from_vf_samples(const Samples3d &xyz_coord_V,
                                    const Samples3i &V_cycle_F);

  /**
   * @brief Construct a new MatrixMesh object from ply file of vertex-face
   * samples.
   *
   * @param ply_path
   * @return MatrixMesh
   */
  static MatrixMesh from_vf_ply(const std::filesystem::path &ply_path);
  void write_vf_ply(const std::filesystem::path &ply_path) const;

  ///////////////////////////////////////////////////////
  // Fundamental accessors and properties ///////////////
  ///////////////////////////////////////////////////////
  const Samples3d &get_xyz_coord_V() const;
  void set_xyz_coord_V(const Samples3d &value);
  const Samplesi &get_h_out_V() const;
  void set_h_out_V(const Samplesi &value);
  const Samplesi &get_v_origin_H() const;
  void set_v_origin_H(const Samplesi &value);
  const Samplesi &get_h_next_H() const;
  void set_h_next_H(const Samplesi &value);
  const Samplesi &get_h_twin_H() const;
  void set_h_twin_H(const Samplesi &value);
  const Samplesi &get_f_left_H() const;
  void set_f_left_H(const Samplesi &value);
  const Samplesi &get_h_right_F() const;
  void set_h_right_F(const Samplesi &value);
  const Samplesi &get_h_negative_B() const;
  void set_h_negative_B(const Samplesi &value);

  const Samplesi &get_h_directed_E() const { return h_directed_E_; };
  const Samplesi &get_e_undirected_H() const { return e_undirected_H_; };
  const Samples3d &get_vec_H() const { return vec_H_; };
  const Samples1d &get_length_E() const { return length_E_; };
  const Samples1d &get_area_F() const { return area_F_; };
  const Samples3d &get_normal_F() const { return normal_F_; };
  const Samples3d &get_normal_V() const { return normal_V_; };
  const Samples1d &get_area_V() const { return area_V_; };

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

  Samples3i V_cycle_E() const;
  /**
   * @brief Get the vertex cycles of the faces
   *
   * @return Samples3i
   */
  Samples3i V_cycle_F() const;
  // Samples2i V_of_H() const;
  // Samples2i V_cycle_E() const;
  /**
   * @brief Get the vertex positions and vertex cycles of faces in the mesh
   *
   * @return VertexFaceTuple
   */
  VertexFaceTuple vf_samples() const;
  /**
   * @brief Get the vertex positions, vertex cycles of edges, and vertex cycles
   * of faces in the mesh
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

  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////

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
  utils::SimpleGenerator<int> generate_H_right_b(int b) const;
  utils::SimpleGenerator<int> generate_F_incident_v(int v) const;
  ///////////////////////////////////////////////////////
  // Mutators ///////////////////////////////////////////
  ///////////////////////////////////////////////////////
  void update_mat_v(int v, const std::optional<Vec3d> &xyz_coord = std::nullopt,
                    const std::optional<int> &h_out = std::nullopt);
  void update_mat_h(int h, const std::optional<int> &v_origin = std::nullopt,
                    const std::optional<int> &h_next = std::nullopt,
                    const std::optional<int> &h_twin = std::nullopt,
                    const std::optional<int> &f_left = std::nullopt);
  void update_mat_f(int f, const std::optional<int> &h_left = std::nullopt);
  /**
   * @brief Flips edge h.
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
  Samples3d bary_coord_Q_;
  Samples1d dimensionless_quad_weight_Q_;
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
  void update_he_from_vf();
  void reset_integration_patch();
  double get_gaussian_curvature_angle_defect_v(int v) const;
  void update_gaussian_curvature_angle_defect();
  void update_gaussian_curvature_laplacian();
  void update_gaussian_curvature();
  //////////////////////
  // Laplacians ////////
  //////////////////////
  Samples1d cotan_laplacian(Samples1d &Q);
  Samples3d cotan_laplacian(Samples3d &Q);
  Samples1d belkin_laplacian(Samples1d &Q);
  Samples3d belkin_laplacian(Samples3d &Q);

  Samples1d adaptive_belkin_laplacian(Samples1d &Q);
  Samples3d adaptive_belkin_laplacian(Samples3d &Q);

  template <typename Samples> Samples guckenberger_laplacian(Samples &phi);
  template <typename Samples>
  Samples adaptive_guckenberger_laplacian(Samples &phi);

  template <typename Samples>
  Samples higher_order_quad_heat_laplacian(Samples &phi);
  template <typename Samples> Samples heat_laplacian(Samples &phi);

  Eigen::MatrixXd laplacian(Eigen::MatrixXd &Q);
  Samples1d laplacian(Samples1d &Q);
  Samples3d laplacian(Samples3d &Q);

  void update_laplacian_matrix_cotan();
  void update_laplacian_matrix_adaptive_belkin();
  void update_laplacian_matrix_heat();
  void update_laplacian_matrix_belkin();
  void update_laplacian_matrix();
  Samples1d apply_laplacian_matrix(Samples1d &Q);
  Samples3d apply_laplacian_matrix(Samples3d &Q);

  void update_mean_curvature();

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
  void refine_icososphere();
};

/**
 * }
 */

} // namespace meshbrane
