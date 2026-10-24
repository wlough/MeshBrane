#pragma once

/**
 * @file simplicial_complex_1d.hpp
 * @brief Defines the SimplicialComplex1d class.
 */

// #include "meshbrane/meshbrane_object.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simple_vector_field.hpp"
#include <filesystem>
#include <iostream>
#include <string> // std::string
#include <tuple>  // std::tuple
#include <yaml-cpp/yaml.h>

namespace meshbrane {

class SimplicialComplex1d {
public:
  std::string name_{"line"};
  ////////////////////////////
  // Core data structure /////
  ////////////////////////////
  Samples3d xyz_coord_V_;   //
  Samples2i V_cycle_E_;     //
                            //
  Samplesi h_out_V_;        //
  Samplesi h_directed_E_;   //
  Samplesi h_ghost_B_;      //
                            //
  Samplesi v_origin_H_;     //
  Samplesi e_undirected_H_; //
                            //
  Samplesi h_twin_H_;       //
  Samplesi h_rot_H_;        //
  ////////////////////////////

  /////////////////////////
  // fixed parameters /////
  /////////////////////////
  YAML::Node parameters_;

  std::filesystem::path ply_path_;

  bool show_half_edges_{false};
  bool show_vertices_{true};
  bool show_edges_{true};
  RGBA rgba_vertex_ = RGBA_DICT.at("purple");
  RGBA rgba_half_edge_ = RGBA_DICT.at("purple");
  RGBA rgba_edge_ = RGBA_DICT.at("green");
  double radius_vertex_{20};

  /////////////////////////////
  // Interaction accumulator //
  /////////////////////////////
  Samples3d force_V_;

  ////////////////////
  // Cached data /////
  ////////////////////

  Samples3d vec_H_;
  Samples1d length_E_;
  double average_edge_length_{0.0};
  // // Curvature
  Samples3d mcvec_V_;
  Samples1d curvature_V_;

  Eigen::Matrix<double, Eigen::Dynamic, 1> radius_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_H;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_E_;
  //
  std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3>
      shifted_half_edge_arrows_;
  SimpleVectorField vector_field_arrows_;

  ////////////////////////
  // Initialization //////
  ////////////////////////
  SimplicialComplex1d() = default;

  //
  // SimplicialComplex1d(const YAML::Node &parameters);
  //

  //

  void set_attributes_from_parameters() {
    printf("SimplicialComplex1d::set_attributes_from_parameters\n");
    if (parameters_["ply_path"]) {
      ply_path_ =
          std::filesystem::path(parameters_["ply_path"].as<std::string>());
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
  }

  // void update_vector_field_arrows(Samples3d X, Samples3d U);
  void init_straight_chain(size_t Nv, Vec3d axis, double edge_length) {
    // Samples3d xyz_coord_V_;   //
    // Samples2i V_cycle_E_;     //
    //                           //
    // Samplesi h_out_V_;        //
    // Samplesi h_directed_E_;   //
    // Samplesi h_ghost_B_;      //
    //                           //
    // Samplesi v_origin_H_;     //
    // Samplesi e_undirected_H_; //
    //                           //
    // Samplesi h_twin_H_;       //
    // Samplesi h_rot_H_;        //
    size_t Ne = Nv - 1;
    size_t Nh = 2 * Ne;
    size_t Nb = 2;

    xyz_coord_V_.resize(Nv, 3);
    h_out_V_.resize(Nv);

    V_cycle_E_.resize(Ne, 2);
    h_directed_E_.resize(Ne);

    h_ghost_B_.resize(Nb);

    v_origin_H_.resize(Nh);
    e_undirected_H_.resize(Nh);
    h_twin_H_.resize(Nh);
    h_rot_H_.resize(Nh);

    for (int v = 0; v < Nv; v++) {
      xyz_coord_V_.row(v) = v * edge_length * axis;
      h_out_V_[v] = v;
    }

    for (int e = 0; e < Ne; e++) {
      int v0 = e;
      int v1 = v0 + 1;
      int h = e;
      int ht = h + (Ne - h) + (Ne - h - 1);
      V_cycle_E_.row(e) << v0, v1;
      h_directed_E_[e] = h;
    }

    for (int h = 0; h < Nh; h++) {

      int ht = Nh - 1 - h;
      // int ht = -(h+1) % Nh;
      // ht = (ht+Nh) % Nh;
      int hr = (Nh - h) % Nh;
      // int hr = -h % Nh;
      // hr = (hr+Nh) % Nh;
      h_twin_H_[h] = ht;
      h_rot_H_[h] = hr;
    }
    // Ne =4, Nh=8
    //
    // .<-h7--.<-h6--.<-h5--.<-h4--.
    // .--h0->.--h1->.--h2->.--h3->.--h4->.--h5->.--h6->.--h7->
  }

  void init_single_loop(size_t Nv, Vec3d loop_center, Vec3d loop_normal,
                        double loop_radius) {
    printf("SimplicialComplex1d::init_single_loop\n");
    size_t Ne = Nv;
    size_t Nh = 2 * Ne;
    size_t Nb = 0;

    printf("  Nv= %lu\n", Nv);
    printf("  Ne= %lu\n", Ne);
    printf("  Nb= %lu\n", Nb);
    printf("  Nh= %lu\n", Nh);

    xyz_coord_V_.resize(Nv, 3);
    h_out_V_.resize(Nv);

    V_cycle_E_.resize(Ne, 2);
    h_directed_E_.resize(Ne);

    h_ghost_B_.resize(Nb);

    v_origin_H_.resize(Nh);
    e_undirected_H_.resize(Nh);
    h_twin_H_.resize(Nh);
    h_rot_H_.resize(Nh);

    Samplesi h_next_H(Nh);

    printf("SimplicialComplex1d::init_single_loop --- 0\n");
    printf("  define loop frame\n");

    Vec3d Ez{0, 0, 1};
    Vec3d Ey{0, 1, 0};
    Vec3d Ex{1, 0, 0};

    Vec3d ez = loop_normal / loop_normal.norm();
    Vec3d ey = ez.cross(Ex);
    double norm_ey = ey.norm();
    if (norm_ey < 1e-6) {
      ey = ez.cross(Ey);
      norm_ey = ey.norm();
    }
    ey /= norm_ey;
    Vec3d ex = ey.cross(ez);

    printf("SimplicialComplex1d::init_single_loop --- 2\n");
    printf("  assign data to vertices\n");
    // assign data to vertices
    for (int v = 0; v < Nv; v++) {
      int h = v;
      double phi_v = 2 * M_PI * v / Nv;
      xyz_coord_V_.row(v) = loop_center + loop_radius * (std::cos(phi_v) * ex +
                                                         std::sin(phi_v) * ey);
      h_out_V_[v] = h;
    }
    printf("SimplicialComplex1d::init_single_loop --- 3\n");
    printf("  assign data to edges\n");
    // assign data to edges
    for (int e = 0; e < Ne; e++) {
      int v0 = e;
      int v1 = (v0 + 1) % Nv;
      int h = e;
      printf("    e= %d\n", e);
      printf("    v0, v1= %d, %d\n", v0, v1);
      V_cycle_E_.row(e) << v0, v1;
      h_directed_E_[e] = h;
    }
    printf("SimplicialComplex1d::init_single_loop --- 4\n");
    printf("  assign twin/edge data to half-edges\n");
    printf("  assign vertex/next to inner/cw half-edges\n");
    // assign twin/edge data to half-edges
    // assign vertex/next to inner/cw half-edges
    for (int h = 0; h < Ne; h++) {
      int v = h;
      int e = h;
      v_origin_H_[h] = v;
      e_undirected_H_[h] = e;

      int ht = Nh - (h + 1);
      int hn = (h + 1) % Ne;
      h_twin_H_[h] = ht;
      h_twin_H_[ht] = h;
      h_next_H[h] = hn;
      e_undirected_H_[ht] = e;

      printf("    h = %d\n", h);
      printf("    ht = %d\n", ht);
      printf("    hn = %d\n", hn);
      printf("    v = %d\n", v);
      printf("    e = %d\n", e);
    }
    printf("SimplicialComplex1d::init_single_loop --- 5\n");
    printf("  assign next to outer/ccw half-edges\n");
    // assign next to outer/ccw half-edges
    for (int h = Ne; h < Nh; h++) {
      int h_reduced = h - Ne;
      int hn_reduced = (h_reduced + 1) % Ne;
      int hn = hn_reduced + Ne;
      h_next_H[h] = hn;

      printf("    h = %d\n", h);
      printf("    hn = %d\n", hn);
    }

    printf("  h_twin_H_ data\n");
    for (int h = 0; h < Nh; h++) {
      int ht = h_twin_H_[h];
      printf("    ht = %d\n", ht);
    }
    printf("  h_rot_H_ data\n");
    for (int h = 0; h < Nh; h++) {
      int hr = h_rot_H_[h];
      printf("    hr = %d\n", hr);
    }
    printf("  h_next_H_ data\n");
    for (int h = 0; h < Nh; h++) {
      int hn = h_next_H[h];
      printf("    hn = %d\n", hn);
    }

    printf("SimplicialComplex1d::init_single_loop --- 6\n");
    printf("  assign rot to half-edges\n");
    // assign rot to half-edges
    for (int h = 0; h < Nh; h++) {
      printf("    h= %d\n", h);
      int ht = h_twin_H_[h];
      printf("    ht= %d\n", ht);
      int hr = h_next_H[ht];
      printf("    hr= %d\n", ht);
      h_rot_H_[h] = hr;
      // int hr = h_next_H[h_twin_H_[h]];
      // h_rot_H_[h] = hr;
    }
    printf("SimplicialComplex1d::init_single_loop --- 7\n");
    printf("  assign vertex to outer/ccw half-edges\n");
    // assign vertex to outer/ccw half-edges
    for (int h = Ne; h < Nh; h++) {
      int hr = h_rot_H_[h];
      int v = v_origin_H_[hr];
      printf("    h= %d\n", h);
      printf("    hr= %d\n", hr);
      printf("    v= %d\n", v);
      v_origin_H_[h] = v;
    }
    printf("SimplicialComplex1d::init_single_loop - done\n");
    check_he_matrices();
  }

  void check_he_matrices() const {
    printf("SimplicialComplex1d::check_he_matrices\n");
    int Nv = get_num_vertices();
    int Ne = get_num_edges();
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

    for (int b = 0; b < Nb; b++) {
      int h_ghost = h_ghost_B_[b];
      if (h_ghost < 0 || h_ghost >= Nh) {
        std::cout << "b=" << b << '\n';
        std::cout << "h_ghost=" << h_ghost << '\n';
        throw std::out_of_range("h_ghost_B_[b]: Half-edge index out of range");
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
      int h_rot = h_rot_H_[h];
      if (h_rot < 0 || h_rot >= Nh) {
        std::cout << "h=" << h << '\n';
        std::cout << "hrott=" << h_rot << '\n';
        throw std::out_of_range("h_rot_H_[h]: Half-edge index out of range");
      }
      int h_twin = h_twin_H_[h];
      if (h_twin < 0 || h_twin >= Nh) {
        std::cout << "h=" << h << '\n';
        std::cout << "h_twin=" << h_twin << '\n';
        throw std::out_of_range("h_twin_H_[h]: Half-edge index out of range");
      }
    }
  }

  ////////////////////////////////////
  // Cache updaters //////////////////
  ////////////////////////////////////
  void update_complex_geometric_data_V() {}
  void update_complex_geometric_data_E() {
    size_t num_edges = get_num_edges();
    int num_half_edges = get_num_half_edges();
    vec_H_.resize(num_half_edges, 3);
    // resize the length vector
    // length_E_.setZero();
    length_E_.resize(num_edges);
    average_edge_length_ = 0.0;
    for (int e = 0; e < num_edges; e++) {
      int h = h_directed_E_(e);
      int ht = h_twin_h(h);
      int v = v_origin_h(h);
      int vt = v_origin_h(ht);
      Vec3d x = xyz_coord_v(v);
      Vec3d xt = xyz_coord_v(vt);
      vec_H_.row(h) = xt - x;
      vec_H_.row(ht) = -vec_H_.row(h);
      // length_E_(e) = math::L2norm(vec_H_.row(h));
      length_E_(e) = vec_H_.row(h).norm();
      average_edge_length_ += length_E_(e);
    }
    average_edge_length_ /= num_edges;
  }
  void update_complex_geometric_data() {
    update_complex_geometric_data_V();
    update_complex_geometric_data_E();
  }

  // void load_ply();

  /**
   * @brief Recompute `V_cycle_E_` `V_cycle_F_` `e_undirected_H_`
   * `h_directed_E_` from half-edge matrices
   *
   */
  // void update_simplices_from_he();
  // void update_boundary_cycles();
  //
  // double length_e(int e) const {
  //   return math::L2norm(xyz_coord_v(V_cycle_E_(e, 1)) -
  //                       xyz_coord_v(V_cycle_E_(e, 0)));
  // }

  /**
   * @brief Initialize mesh data from half-edge matrices, set visual defaults,
   *
   */
  // void init_mesh();

  ///////////////////////////////////////////////////////
  // Constructors and Mesh I/O //////////////////////////
  ///////////////////////////////////////////////////////

  /**
   * @brief Construct a new MatrixMesh object from ply file of half-edge
   * samples.
   * @param ply_path
   * @return MatrixMesh
   */
  // static MatrixMesh from_he_ply(const std::filesystem::path &ply_path);
  // static MatrixMesh from_icosohedron();
  // void init_icosohedron();
  // static MatrixMesh from_icososphere(int n);
  // void write_he_ply(const std::filesystem::path &ply_path) const;
  //
  // /**
  //  * @brief Construct a new MatrixMesh object from the vectors in a
  //  * `meshbrane::VertexFaceSamples` tuple.
  //  * @param xyz_coord_V
  //  * @param V_cycle_F
  //  * @return MatrixMesh<VertexType>
  //  */
  // static SimplicialComplex1d from_ve_samples(const Samples3d &xyz_coord_V,
  //                                 const Samples3i &E_cycle_F);

  ///////////////////////////////////////////////////////
  // Fundamental accessors and properties ///////////////
  ///////////////////////////////////////////////////////
  const Samples3d &get_xyz_coord_V() const { return xyz_coord_V_; }
  const Samplesi &get_h_out_V() const { return h_out_V_; }

  const Samples2i &get_V_cycle_E() const { return V_cycle_E_; }
  const Samplesi &get_h_directed_E() const { return h_directed_E_; }

  const Samplesi &get_h_ghost_B() const { return h_ghost_B_; }

  const Samplesi &get_v_origin_H() const { return v_origin_H_; }
  const Samplesi &get_e_undirected_H() const { return e_undirected_H_; }
  const Samplesi &get_h_twin_H() const { return h_twin_H_; }
  const Samplesi &get_h_rot_H() const { return h_rot_H_; }

  /**
   * @brief Get the number of vertices in the mesh
   *
   * @return int
   */
  int get_num_vertices() const { return h_out_V_.size(); }
  /**
   * @brief Get the number edges in the mesh
   *
   * @return int
   */
  int get_num_edges() const { return h_directed_E_.size(); }
  /**
   * @brief Get the numberhalf edges in the mesh
   *
   * @return int
   */
  int get_num_half_edges() const { return h_twin_H_.size(); }

  /**
   * @brief Get the Euler characteristic of the mesh
   *
   * @return int
   */
  int get_euler_characteristic() const {
    return get_num_vertices() - get_num_edges();
  }
  /**
   * @brief Get the number of dicconnected boundary components
   *
   * @return int
   */
  int get_num_boundaries() const { return h_ghost_B_.size(); }

  int get_valence_v(int v) const;

  /**
   * @brief Get the vertex positions and vertex cycles of edges
   *
   *
   * @return std::tuple<Samples3d, Samples2i>
   */
  std::tuple<Samples3d, Samples2i> ve_samples() const {
    return {xyz_coord_V_, V_cycle_E_};
  }

  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////
  Vec3d xyz_coord_v(int v) const { return xyz_coord_V_.row(v); }
  int h_out_v(int v) const { return h_out_V_[v]; }
  int v_origin_h(int h) const { return v_origin_H_[h]; }
  int h_twin_h(int h) const { return h_twin_H_[h]; }
  int h_rot_h(int h) const { return h_rot_H_[h]; }
  int h_ghost_b(int b) const { return h_ghost_B_[b]; }
  int h_directed_e(int e) const { return h_directed_E_[e]; }
  int e_undirected_h(int h) const { return e_undirected_H_[h]; }

  int h_next_h(int h) const { return h_rot_H_[h_twin_H_[h]]; }

  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  // bool some_negative_boundary_contains_h(int h) const;
  // bool some_positive_boundary_contains_h(int h) const;
  // bool some_boundary_contains_h(int h) const;
  bool some_boundary_contains_v(int v) const {
    int h = h_out_V_[v];
    int hr = h_rot_H_[h];
    return (h == hr);
  }

  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////

  //////////////////////////////////////////////
  // Visualization /////////////////////////////
  //////////////////////////////////////////////
  void set_visual_defaults() {

    int num_vertices = get_num_vertices();
    int num_half_edges = get_num_half_edges();
    int num_edges = get_num_edges();

    radius_V.resize(num_vertices, 1);
    rgba_V.resize(num_vertices, 4);
    for (int i = 0; i < num_vertices; i++) {
      radius_V(i) = radius_vertex_;
      rgba_V.row(i) = rgba_vertex_;
    }
    rgba_H.resize(num_half_edges, 4);
    for (int i = 0; i < num_half_edges; i++) {
      rgba_H.row(i) = rgba_half_edge_;
    }
    rgba_E_.resize(num_edges, 4);
    for (int i = 0; i < num_edges; i++) {
      rgba_E_.row(i) = rgba_edge_;
    }
  }

  /**
   * @brief Divides edge by adding a new vertex at its midpoint
   *
   * @param f
   *
   * ```
   *
   *
   *
   * \        e0            /     \     e0       e1      /
   *  \      -h0->         /       \   -h0->    -h3->   /
   *   v0----------------v1   -->   v0-------v2-------v1
   *  /      <-h1-         \       /   <-h2-    <-h1-   \
   * /                      \     /                      \
   *
   *
   *
   *
   * ```
   */
  void split_edge(int e) {

    int Nv0 = get_num_vertices();
    int Ne0 = get_num_edges();
    int Nh0 = get_num_half_edges();
    int Nv = Nv0 + 1;
    int Ne = Ne0 + 1;
    int Nh = Nh0 + 2;

    xyz_coord_V_.conservativeResize(Nv, 3);
    h_out_V_.conservativeResize(Nv);

    V_cycle_E_.conservativeResize(Ne, 2);
    h_directed_E_.conservativeResize(Ne);

    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);
    h_rot_H_.conservativeResize(Nh);

    int e0 = e;
    int h0 = h_directed_E_[e0];
    int h1 = h_twin_H_[h0];
    int v0 = v_origin_H_[h0];
    int v1 = v_origin_H_[h1];

    // new simplices
    int v2 = Nv0;
    int e1 = Ne0;
    // new darts
    int h2 = Nh0;
    int h3 = Nh0 + 1;

    xyz_coord_V_.row(v2) = (xyz_coord_v(v0) + xyz_coord_v(v1)) / 2.0;
    h_out_V_[v2] = h3;

    V_cycle_E_.row(e0) << v0, v2;

    V_cycle_E_.row(e1) << v2, v1;
    h_directed_E_[e1] = h3;

    h_twin_H_[h0] = h2;

    e_undirected_H_[h1] = e1;
    h_twin_H_[h1] = h3;

    v_origin_H_[h2] = v2;
    e_undirected_H_[h2] = e0;
    h_twin_H_[h2] = h0;
    h_rot_H_[h2] = h3;

    v_origin_H_[h3] = v2;
    e_undirected_H_[h3] = e1;
    h_twin_H_[h3] = h1;
    h_rot_H_[h3] = h2;

    // radius_V
    // rgba_V
    // rgba_H
    // rgba_E_
    // rgba_vertex_
    // rgba_half_edge_
    // rgba_edge_

    radius_V.conservativeResize(Nv);
    radius_V[v2] = radius_vertex_;
    rgba_V.conservativeResize(Nv, 4);
    rgba_V.row(v2) = rgba_vertex_;

    rgba_E_.conservativeResize(Ne, 4);
    rgba_E_.row(e1) = rgba_edge_;

    rgba_H.conservativeResize(Nh, 4);
    rgba_H.row(h2) = rgba_half_edge_;
    rgba_H.row(h3) = rgba_half_edge_;
  }
};

} // namespace meshbrane
