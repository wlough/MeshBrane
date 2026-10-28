
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
  std::string name_{"mesh"};
  ////////////////////////////
  // Core data structure /////
  ////////////////////////////
  Samples3d xyz_coord_V_; //
  Samplesi h_out_V_;      //
  //
  Samples2i V_cycle_E_;   //
  Samplesi h_directed_E_; //
  //
  Samplesi v_origin_H_;     //
  Samplesi e_undirected_H_; //
  Samplesi h_twin_H_;       //
  Samplesi h_rot_H_;        //
  ////////////////////////////
  ////////////////////////////
  ////////////////////////////
  Samplesi h_ghost_B_; //

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

  ////////////////////
  // Cached data /////
  ////////////////////

  Samples3d vec_H_;
  Samples1d length_E_;
  double average_edge_length_{0.0};
  // // Curvature
  // Samples3d mcvec_V_;
  // Samples1d curvature_V_;

  Eigen::Matrix<double, Eigen::Dynamic, 1> radius_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_V;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_H;
  Eigen::Matrix<double, Eigen::Dynamic, 4> rgba_E_;
  //
  // std::array<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3>
  //     shifted_half_edge_arrows_;
  // SimpleVectorField vector_field_arrows_;

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

  void init_single_loop_topo(Samples3d xyz_coord_V) {
    printf("SimplicialComplex1d::init_single_loop_topo\n");

    size_t Nv = xyz_coord_V.rows();
    size_t Ne = Nv;
    size_t Nh = 2 * Ne;
    size_t Nb = 0;

    xyz_coord_V_ = xyz_coord_V;
    h_out_V_.resize(Nv);

    V_cycle_E_.resize(Ne, 2);
    h_directed_E_.resize(Ne);

    h_ghost_B_.resize(Nb);

    v_origin_H_.resize(Nh);
    e_undirected_H_.resize(Nh);
    h_twin_H_.resize(Nh);
    h_rot_H_.resize(Nh);

    Samplesi h_next_H(Nh);
    printf("SimplicialComplex1d::init_single_loop_topo -- 0\n");
    // assign data to vertices
    for (int v = 0; v < Nv; v++) {
      int h = v;
      h_out_V_[v] = h;
    }
    printf("SimplicialComplex1d::init_single_loop_topo -- 1\n");
    // assign data to edges
    for (int e = 0; e < Ne; e++) {
      int v0 = e;
      int v1 = (v0 + 1) % Nv;
      int h = e;
      V_cycle_E_.row(e) << v0, v1;
      h_directed_E_[e] = h;
    }
    printf("SimplicialComplex1d::init_single_loop_topo -- 2\n");
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
    }
    printf("SimplicialComplex1d::init_single_loop_topo -- 3\n");
    // assign next to outer/ccw half-edges
    for (int h = Ne; h < Nh; h++) {
      int h_reduced = h - Ne;
      int hn_reduced = (h_reduced + 1) % Ne;
      int hn = hn_reduced + Ne;
      h_next_H[h] = hn;
    }
    printf("SimplicialComplex1d::init_single_loop_topo -- 4\n");
    // assign rot to half-edges
    for (int h = 0; h < Nh; h++) {
      int ht = h_twin_H_[h];
      int hr = h_next_H[ht];
      h_rot_H_[h] = hr;
      // int hr = h_next_H[h_twin_H_[h]];
      // h_rot_H_[h] = hr;
    }
    printf("SimplicialComplex1d::init_single_loop_topo -- 5\n");
    // assign vertex to outer/ccw half-edges
    for (int h = Ne; h < Nh; h++) {
      int hr = h_rot_H_[h];
      int v = v_origin_H_[hr];
      v_origin_H_[h] = v;
    }
    printf("SimplicialComplex1d::init_single_loop_topo -done\n");
  }

  void init_single_loop(size_t Nv, Vec3d loop_center, Vec3d loop_normal,
                        double loop_radius) {
    printf("SimplicialComplex1d::init_single_loop\n");
    // size_t Ne = Nv;
    // size_t Nh = 2 * Ne;
    // size_t Nb = 0;

    // printf("  Nv= %lu\n", Nv);
    // printf("  Ne= %lu\n", Ne);
    // printf("  Nb= %lu\n", Nb);
    // printf("  Nh= %lu\n", Nh);

    // xyz_coord_V_.resize(Nv, 3);
    // h_out_V_.resize(Nv);
    //
    // V_cycle_E_.resize(Ne, 2);
    // h_directed_E_.resize(Ne);
    //
    // h_ghost_B_.resize(Nb);
    //
    // v_origin_H_.resize(Nh);
    // e_undirected_H_.resize(Nh);
    // h_twin_H_.resize(Nh);
    // h_rot_H_.resize(Nh);
    //
    // Samplesi h_next_H(Nh);

    Samples3d xyz_coord_V(Nv, 3);

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
      xyz_coord_V.row(v) = loop_center + loop_radius * (std::cos(phi_v) * ex +
                                                        std::sin(phi_v) * ey);
      // h_out_V_[v] = h;
    }
    init_single_loop_topo(xyz_coord_V);
    // printf("SimplicialComplex1d::init_single_loop --- 3\n");
    // printf("  assign data to edges\n");
    // // assign data to edges
    // for (int e = 0; e < Ne; e++) {
    //   int v0 = e;
    //   int v1 = (v0 + 1) % Nv;
    //   int h = e;
    //   printf("    e= %d\n", e);
    //   printf("    v0, v1= %d, %d\n", v0, v1);
    //   V_cycle_E_.row(e) << v0, v1;
    //   h_directed_E_[e] = h;
    // }
    // printf("SimplicialComplex1d::init_single_loop --- 4\n");
    // printf("  assign twin/edge data to half-edges\n");
    // printf("  assign vertex/next to inner/cw half-edges\n");
    // // assign twin/edge data to half-edges
    // // assign vertex/next to inner/cw half-edges
    // for (int h = 0; h < Ne; h++) {
    //   int v = h;
    //   int e = h;
    //   v_origin_H_[h] = v;
    //   e_undirected_H_[h] = e;
    //
    //   int ht = Nh - (h + 1);
    //   int hn = (h + 1) % Ne;
    //   h_twin_H_[h] = ht;
    //   h_twin_H_[ht] = h;
    //   h_next_H[h] = hn;
    //   e_undirected_H_[ht] = e;
    //
    //   printf("    h = %d\n", h);
    //   printf("    ht = %d\n", ht);
    //   printf("    hn = %d\n", hn);
    //   printf("    v = %d\n", v);
    //   printf("    e = %d\n", e);
    // }
    // printf("SimplicialComplex1d::init_single_loop --- 5\n");
    // printf("  assign next to outer/ccw half-edges\n");
    // // assign next to outer/ccw half-edges
    // for (int h = Ne; h < Nh; h++) {
    //   int h_reduced = h - Ne;
    //   int hn_reduced = (h_reduced + 1) % Ne;
    //   int hn = hn_reduced + Ne;
    //   h_next_H[h] = hn;
    //
    //   printf("    h = %d\n", h);
    //   printf("    hn = %d\n", hn);
    // }
    //
    // printf("  h_twin_H_ data\n");
    // for (int h = 0; h < Nh; h++) {
    //   int ht = h_twin_H_[h];
    //   printf("    ht = %d\n", ht);
    // }
    // printf("  h_rot_H_ data\n");
    // for (int h = 0; h < Nh; h++) {
    //   int hr = h_rot_H_[h];
    //   printf("    hr = %d\n", hr);
    // }
    // printf("  h_next_H_ data\n");
    // for (int h = 0; h < Nh; h++) {
    //   int hn = h_next_H[h];
    //   printf("    hn = %d\n", hn);
    // }
    //
    // printf("SimplicialComplex1d::init_single_loop --- 6\n");
    // printf("  assign rot to half-edges\n");
    // // assign rot to half-edges
    // for (int h = 0; h < Nh; h++) {
    //   printf("    h= %d\n", h);
    //   int ht = h_twin_H_[h];
    //   printf("    ht= %d\n", ht);
    //   int hr = h_next_H[ht];
    //   printf("    hr= %d\n", ht);
    //   h_rot_H_[h] = hr;
    //   // int hr = h_next_H[h_twin_H_[h]];
    //   // h_rot_H_[h] = hr;
    // }
    // printf("SimplicialComplex1d::init_single_loop --- 7\n");
    // printf("  assign vertex to outer/ccw half-edges\n");
    // // assign vertex to outer/ccw half-edges
    // for (int h = Ne; h < Nh; h++) {
    //   int hr = h_rot_H_[h];
    //   int v = v_origin_H_[hr];
    //   printf("    h= %d\n", h);
    //   printf("    hr= %d\n", hr);
    //   printf("    v= %d\n", v);
    //   v_origin_H_[h] = v;
    // }
    // printf("SimplicialComplex1d::init_single_loop - done\n");
    // check_he_matrices();
  }

  void init_point_topo(Vec3d xyz) {
    int Nv = 1;
    int Ne = 0;
    int Nb = 1;
    int Nh = 1;

    xyz_coord_V_.resize(Nv, 3);
    h_out_V_.resize(Nv);

    V_cycle_E_.resize(Ne, 2);
    h_directed_E_.resize(Ne);

    h_ghost_B_.resize(Nb);

    v_origin_H_.resize(Nh);
    e_undirected_H_.resize(Nh);
    h_twin_H_.resize(Nh);
    h_rot_H_.resize(Nh);

    int v = 0, h = 0, b = 0;
    xyz_coord_V_.row(v) = xyz;
    h_out_V_[v] = h;

    h_ghost_B_[b] = h;

    v_origin_H_[h] = v;
    e_undirected_H_[h] = -(b + 1);
    h_twin_H_[h] = h;
    h_rot_H_[h] = h;
  }

  void init_edge_topo(Samples3d xyz_coord_V) {
    int Nv = xyz_coord_V.rows();
    if (Nv != 2) {
      throw std::out_of_range("xyz_coord_V must contain exactly 2 points");
    }
    int Ne = 1;
    int Nb = 2;
    int Nh = 2 * Ne + Nb;

    xyz_coord_V_.resize(Nv, 3);
    h_out_V_.resize(Nv);

    V_cycle_E_.resize(Ne, 2);
    h_directed_E_.resize(Ne);

    h_ghost_B_.resize(Nb);

    v_origin_H_.resize(Nh);
    e_undirected_H_.resize(Nh);
    h_twin_H_.resize(Nh);
    h_rot_H_.resize(Nh);

    int v0 = 0, v1 = 1;
    int e0 = 0;
    int b0 = 0, b1 = 1;
    int h0 = 0, h1 = 1, h2 = 2, h3 = 3;

    xyz_coord_V_ = xyz_coord_V;
    h_out_V_[v0] = h0;
    h_out_V_[v1] = h1;

    V_cycle_E_.row(e0) << v0, v1;
    h_directed_E_[e0] = h0;

    h_ghost_B_[b0] = h2;
    h_ghost_B_[b1] = h3;

    v_origin_H_[h0] = v0;
    e_undirected_H_[h0] = e0;
    h_twin_H_[h0] = h1;
    h_rot_H_[h0] = h2;

    v_origin_H_[h1] = v1;
    e_undirected_H_[h1] = e0;
    h_twin_H_[h1] = h0;
    h_rot_H_[h1] = h3;

    v_origin_H_[h2] = v0;
    e_undirected_H_[h2] = -(b0 + 1);
    h_twin_H_[h2] = h2;
    h_rot_H_[h2] = h0;

    v_origin_H_[h3] = v1;
    e_undirected_H_[h3] = -(b1 + 1);
    h_twin_H_[h3] = h3;
    h_rot_H_[h3] = h1;
  }

  void init_fib_sphere(size_t Nv, Vec3d center, double radius) {

    Samples3d xyz_coord_V(Nv, 3);

    double ga = M_PI * (3.0 - std::sqrt(5.0)); // golden angle

    for (int v = 0; v < Nv; v++) {
      double z = 1.0 - 2.0 * v / (Nv - 1); // -1<=z<=1
      double r = std::sqrt(1.0 - z * z);   // cylindrical radius at z
      double theta = ga * v;               // angle
      double x = r * std::cos(theta);
      double y = r * std::sin(theta);
      Vec3d xyz{x, y, z};
      xyz_coord_V.row(v) = center + radius * xyz;
    }
    init_single_loop_topo(xyz_coord_V);
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
      int b = -(e_undirected + 1);
      if ((e_undirected < 0 && b >= Nb) || e_undirected >= Ne) {
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
  /**
   * @brief The unique dart with a different vertex and the same edge
   * h=(v, e) --> h'=(v', e). Orbit is the set of all (2) darts through a fixed
   * edge.
   *
   * @param h
   *
   * ```
   * \                     /     \                     /
   *  \      -h->         /       \        e          /
   *   v-----------------v'   -->  v-----------------v'
   *  /        e           \      /      <-h'-         \
   * /                      \    /                      \
   *
   * ```
   */
  int h_twin_h(int h) const { return h_twin_H_[h]; }
  /**
   * @brief h=(v, e) --> h'=(v, e') with e'=e at boundaries. Orbit is the set of
   * all darts through a fixed vertex.
   *
   * @param h
   *
   * ```
   * \                      \
   *  \                      \
   *   \                      \
   *    \  -h->  e             \
   *     v---------   -->       v---------
   *    /    e'                / /
   *   /                   e' / h'
   *  /                      / /_
   * /                      /
   *
   *
   *   -h->  e           -h->  e
   * v---------  -->   v---------
   *
   *
   * ```
   */
  int h_rot_h(int h) const { return h_rot_H_[h]; }
  int h_ghost_b(int b) const { return h_ghost_B_[b]; }
  int h_directed_e(int e) const { return h_directed_E_[e]; }
  int e_undirected_h(int h) const { return e_undirected_H_[h]; }

  int h_next_h(int h) const { return h_rot_H_[h_twin_H_[h]]; }
  int h_unrot_h(int h) const {
    int h0 = h;
    int h1 = h_rot_H_[h0];
    while (h1 != h) {
      h0 = h1;
      h1 = h_rot_H_[h1];
    }
    return h0;
  }

  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  // bool some_negative_boundary_contains_h(int h) const;
  // bool some_positive_boundary_contains_h(int h) const;
  // bool some_boundary_contains_h(int h) const;
  bool some_boundary_contains_v(int v) const {
    int h = h_out_V_[v];
    int hr = h_rot_H_[h];
    return h == hr;
  }

  bool is_ghost_h(int h) {
    // return e_undirected_H_[h] < 0;
    return h_twin_H_[h] == h;
  }

  bool is_isolated_v(int v) {
    int h = h_out_V_[v];
    return h == h_rot_H_[h];
  }

  bool is_boundary_v(int v) {
    int h = h_out_V_[v];
    int hr = h_rot_H_[h];
    return (is_ghost_h(h) || is_ghost_h(hr));
  }

  bool is_nonisolated_boundary_v(int v) {
    int h = h_out_V_[v];
    int hr = h_rot_H_[h];
    return (hr != h && (is_ghost_h(h) || is_ghost_h(hr)));
  }

  bool adjacent_vv(int v0, int v1) {
    int h_start = h_out_V_[v0];
    int h = h_start;

    do {
      int ht = h_twin_H_[h];
      int v = v_origin_H_[ht];
      if (v == v1) {
        return true;
      }
      h = h_rot_H_[h];
    } while (h != h_start);
    return false;
  }

  bool incident_ve(int v, int e) {
    int h0 = h_directed_E_[e];
    int h1 = h_twin_H_[h0];
    return (v_origin_H_[h0] == v || v_origin_H_[h1] == v);
  }

  bool invalid_v(int v) { return (v < 0 || v >= get_num_vertices()); }
  bool invalid_e(int e) { return (e < 0 || e >= get_num_edges()); }
  bool invalid_b(int b) { return (b < 0 || b >= get_num_boundaries()); }
  bool invalid_h(int h) { return (h < 0 || h >= get_num_half_edges()); }

  void check_v(int v) {
    if (v < 0 || v >= get_num_vertices()) {
      throw std::out_of_range("v index " + std::to_string(v) + " out of range");
    }
  }
  void check_e(int e) {
    if (e < 0 || e >= get_num_edges()) {
      throw std::out_of_range("e index " + std::to_string(e) + " out of range");
    }
  }
  void check_b(int b) {
    if (b < 0 || b >= get_num_boundaries()) {
      throw std::out_of_range("b index " + std::to_string(b) + " out of range");
    }
  }
  void check_h(int h) {
    if (h < 0 || h >= get_num_half_edges()) {
      throw std::out_of_range("h index " + std::to_string(h) + " out of range");
    }
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

  int valance_v(int v) {
    check_v(v);
    // int valance = 0;
    // int h_start = h_out_V_[v];
    // int h0 = h_start;
    // int h1 = h_rot_H_[h0];
    // while (h1 != h_start) {
    //   valance++;
    //   h0 = h1;
    //   h1 = h_rot_H_[h1];
    // }
    // int valance = 0;
    // int h_start = h_out_V_[v];
    // int h = h_rot_H_[h_start];
    // while (h != h_start) {
    //   valance++;
    //   h = h_rot_H_[h];
    // }
    // return valance;
    int valance = 0;
    int h_start = h_out_V_[v];
    int h = h_start;
    do {
      if (h_twin_H_[h] != h) {
        valance++;
      }
      h = h_rot_H_[h];
    } while (h != h_start);
    return valance;
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
  int split_edge(int e) {
    check_e(e);
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
    return h3;
  }

  // *       \   _               \   _
  // *     \  \ \              \  \ \
  // *     h1  \ h0            h1  \ h0   e1
  // *      _\  \ \ -h2->       _\  \ \  -h2->      -h4->
  // *          v0          -->     v0-----------v1
  // *                                   <-h3-

  /**
   * @brief grow new edge incident on a vertex v
   *
   * @param h
   *
   * ```
   * With v0=v
   *
   * valance(v0) > 1
   *
   *  \   _         \   _
   *   \ \           \ \
   *    \ h0          \ h0   e2
   *     \ \           \ \  -h2->
   *     v0    -->     v0-----------v1 -h4>
   *     / /           / /  <-h3-
   *    / h1          / h1
   *   / /_          / /_
   *  /             /
   *
   * valance(v0) = 1
   *
   * \        e0                      \     e0       e1
   *  \      -h1->         -h2->       \   -h1->    -h3->   -h2->
   *   v1----------------v0       -->   v1-------v2-------v0
   *  /      <-h0-                     /   <-h4-    <-h0-
   * /                                /
   *
   *
   *
   * valance(v0) = 0
   *                          e0
   *   b0             b0     -h1->       b1
   * <-h0-v0   -->  <-h0-v0-----------v1-h3->
   *                         <-h2-
   * ```
   */
  int add_edge_incident_v(int v, Vec3d direction) {
    // printf("SimplicialComplex1d::seed_edge\n");
    check_v(v);
    int Nv0 = get_num_vertices();
    int Ne0 = get_num_edges();
    int Nb0 = get_num_boundaries();
    int Nh0 = get_num_half_edges();

    if (is_isolated_v(v)) {
      // valance(v0) = 0 case
      int Nv = Nv0 + 1;
      int Ne = Ne0 + 1;
      int Nb = Nb0 + 1;
      int Nh = Nh0 + 3;

      xyz_coord_V_.conservativeResize(Nv, 3);
      h_out_V_.conservativeResize(Nv);

      V_cycle_E_.conservativeResize(Ne, 2);
      h_directed_E_.conservativeResize(Ne);

      h_ghost_B_.conservativeResize(Nb);

      v_origin_H_.conservativeResize(Nh);
      e_undirected_H_.conservativeResize(Nh);
      h_twin_H_.conservativeResize(Nh);
      h_rot_H_.conservativeResize(Nh);

      // existing elements
      int v0 = v;
      int h0 = h_out_V_[v0];
      int b0 = -(e_undirected_H_[h0]);
      // new elements
      int v1 = Nv0;
      int e0 = Ne0;
      int h1 = Nh0, h2 = Nh0 + 1, h3 = Nh0 + 2;
      int b1 = Nb0;

      // update existing elements
      h_out_V_[v0] = h1;
      h_rot_H_[h0] = h1;
      // assign data to new elements
      xyz_coord_V_.row(v1) = xyz_coord_v(v0) + direction;
      h_out_V_[v1] = h2;

      V_cycle_E_.row(e0) << v0, v1;
      h_directed_E_[e0] = h1;

      h_ghost_B_[b1] = h3;

      v_origin_H_[h1] = v0;
      e_undirected_H_[h1] = e0;
      h_twin_H_[h1] = h2;
      h_rot_H_[h1] = h0;

      v_origin_H_[h2] = v1;
      e_undirected_H_[h2] = e0;
      h_twin_H_[h2] = h1;
      h_rot_H_[h2] = h3;

      v_origin_H_[h3] = v1;
      e_undirected_H_[h3] = -(b1 + 1);
      h_twin_H_[h3] = h3;
      h_rot_H_[h3] = h2;

      radius_V.conservativeResize(Nv);
      radius_V[v1] = radius_vertex_;
      rgba_V.conservativeResize(Nv, 4);
      rgba_V.row(v1) = rgba_vertex_;

      rgba_E_.conservativeResize(Ne, 4);
      rgba_E_.row(e0) = rgba_edge_;

      rgba_H.conservativeResize(Nh, 4);
      rgba_H.row(h1) = rgba_half_edge_;
      rgba_H.row(h2) = rgba_half_edge_;
      rgba_H.row(h3) = rgba_half_edge_;
      // printf("SimplicialComplex1d::seed_edge - done\n");
      return h2;
    }
    if (is_boundary_v(v)) {
      // valance(v0) = 1 case
      int v0 = v;
      int v2 = Nv0;

      int h0 = h_out_v(v0);
      if (is_ghost_h(h0)) {
        h0 = h_rot_h(h0);
      }
      int e0 = e_undirected_h(h0);

      int h3 = split_edge(e0); // add new vertex at midpoint

      // set coords of new midpoint vertex to old endpoint
      // and update endpoint coords
      xyz_coord_V_.row(v2) = xyz_coord_v(v0);
      xyz_coord_V_.row(v0) = xyz_coord_v(v0) + direction;
      return h3;
    }

    int Nv = Nv0 + 1;
    int Ne = Ne0 + 1;
    int Nb = Nb0 + 1;
    int Nh = Nh0 + 3;

    xyz_coord_V_.conservativeResize(Nv, 3);
    h_out_V_.conservativeResize(Nv);

    V_cycle_E_.conservativeResize(Ne, 2);
    h_directed_E_.conservativeResize(Ne);

    h_ghost_B_.conservativeResize(Nb);

    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);
    h_rot_H_.conservativeResize(Nh);

    // existing elements
    int v0 = v;
    int h0 = h_out_V_[v0];
    int h1 = h_rot_H_[h0];
    // new elements
    int v1 = Nv0;
    int e0 = Ne0;
    int b0 = Nb0;
    int h2 = Nh0, h3 = Nh0 + 1, h4 = Nh0 + 2;
    // update existing elements
    h_rot_H_[h0] = h2;
    // assign data to new elements
    xyz_coord_V_.row(v1) = xyz_coord_v(v0) + direction;
    h_out_V_[v1] = h3;

    V_cycle_E_.row(e0) << v0, v1;
    h_directed_E_[e0] = h2;

    h_ghost_B_[b0] = h4;

    v_origin_H_[h2] = v0;
    e_undirected_H_[h2] = e0;
    h_twin_H_[h2] = h3;
    h_rot_H_[h2] = h1;

    v_origin_H_[h3] = v1;
    e_undirected_H_[h3] = e0;
    h_twin_H_[h3] = h2;
    h_rot_H_[h3] = h4;

    v_origin_H_[h4] = v1;
    e_undirected_H_[h4] = -(b0 + 1);
    h_twin_H_[h4] = h4;
    h_rot_H_[h4] = h3;

    radius_V.conservativeResize(Nv);
    radius_V[v1] = radius_vertex_;
    rgba_V.conservativeResize(Nv, 4);
    rgba_V.row(v1) = rgba_vertex_;

    rgba_E_.conservativeResize(Ne, 4);
    rgba_E_.row(e0) = rgba_edge_;

    rgba_H.conservativeResize(Nh, 4);
    rgba_H.row(h2) = rgba_half_edge_;
    rgba_H.row(h3) = rgba_half_edge_;
    rgba_H.row(h4) = rgba_half_edge_;
    return h3;
  }

  int add_isolated_vertex(Vec3d xyz) {
    // printf("***************** 0");
    int Nv0 = get_num_vertices();
    int Nb0 = get_num_boundaries();
    int Nh0 = get_num_half_edges();

    int Nv = Nv0 + 1;
    int Nb = Nb0 + 1;
    int Nh = Nh0 + 1;

    xyz_coord_V_.conservativeResize(Nv, 3);
    h_out_V_.conservativeResize(Nv);

    h_ghost_B_.conservativeResize(Nb);

    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);
    h_rot_H_.conservativeResize(Nh);

    int v = Nv0;
    int b = Nb0;
    int h = Nh0;
    // printf("***************** 1");
    xyz_coord_V_.row(v) = xyz;
    h_out_V_[v] = h;

    h_ghost_B_[b] = h;

    v_origin_H_[h] = v;
    e_undirected_H_[h] = -(b + 1);
    h_twin_H_[h] = h;
    h_rot_H_[h] = h;

    // printf("***************** 2");

    radius_V.conservativeResize(Nv);
    radius_V[v] = radius_vertex_;
    rgba_V.conservativeResize(Nv, 4);
    rgba_V.row(v) = rgba_vertex_;
    rgba_H.conservativeResize(Nh, 4);
    rgba_H.row(h) = rgba_half_edge_;
    rgba_H.row(h) = rgba_half_edge_;
    rgba_H.row(h) = rgba_half_edge_;
    // printf("***************** 3");
    return h;
  }

  /**
   * @brief Add new edge between non-adjacent vertices
   *
   * @param f
   *
   * ```
   * 2 interior vertices
   *
   * h0_              _h3     h0_      e0      _h3
   *   \              /         \     -h4->    /
   *   v0            v1   -->   v0------------v1
   *   /_            _\         /_    <-h5-   _\
   * h2                h1     h2                h1
   *
   *
   * ```
   */
  int add_edge_between_vertices(int v0, int v1) {
    check_v(v0);
    check_v(v1);
    if (adjacent_vv(v0, v1)) {
      return -1;
    }

    bool v0_is_non_isolated_boundary = is_nonisolated_boundary_v(v0);
    bool v1_is_non_isolated_boundary = is_nonisolated_boundary_v(v1);

    // t/t covers valance 1/1
    if (v0_is_non_isolated_boundary && v1_is_non_isolated_boundary) {
      int Ne0 = get_num_edges();
      int Nb0 = get_num_boundaries();
      int Ne = Ne0 + 1;
      int Nb = Nb0 - 2;

      int h0 = h_out_V_[v0];
      if (!is_ghost_h(h0)) {
        h0 = h_rot_H_[h0];
      }
      int b_v0 = e_undirected_H_[h0];
      b_v0 = -(b_v0 + 1);
      int b0 = Nb0 - 2;
      swap_boundary_indices(b0, b_v0);

      int h1 = h_out_V_[v1];
      if (!is_ghost_h(h1)) {
        h1 = h_rot_H_[h1];
      }
      int b_v1 = e_undirected_H_[h1];
      b_v1 = -(b_v1 + 1);
      int b1 = Nb0 - 1;
      swap_boundary_indices(b1, b_v1);

      // new elements
      int e0 = Ne0;
      h_ghost_B_.conservativeResize(Nb);
      h_directed_E_.conservativeResize(Ne);
      V_cycle_E_.conservativeResize(Ne, 2);

      // update existing
      e_undirected_H_[h0] = e0;
      h_twin_H_[h0] = h1;
      e_undirected_H_[h1] = e0;
      h_twin_H_[h1] = h0;
      // assign new
      h_directed_E_[e0] = h0;
      V_cycle_E_.row(e0) << v0, v1;
      // visual data
      rgba_E_.conservativeResize(Ne, 4);
      rgba_E_.row(e0) = rgba_edge_;
      return h0;
    }

    if (v0_is_non_isolated_boundary) {
      std::swap(v0, v1);
      std::swap(v0_is_non_isolated_boundary, v1_is_non_isolated_boundary);
    }
    // f/t covers valance 0/1,2+/1
    if (!v0_is_non_isolated_boundary && v1_is_non_isolated_boundary) {
      int Ne0 = get_num_edges();
      int Nb0 = get_num_boundaries();
      int Nh0 = get_num_half_edges();
      int Ne = Ne0 + 1;
      int Nb = Nb0 - 1;
      int Nh = Nh0 + 1;

      int h0 = h_out_V_[v0];
      int h2 = h_rot_h(h0); // = h0 for isolated v0

      int h1 = h_out_V_[v1];
      if (!is_ghost_h(h1)) {
        h1 = h_rot_H_[h1];
      }
      int b1 = e_undirected_H_[h1];
      b1 = -(b1 + 1);
      swap_boundary_indices(b1, Nb0 - 1);

      // new elements
      int e0 = Ne0;
      int h3 = Nh0;
      h_directed_E_.conservativeResize(Ne);
      V_cycle_E_.conservativeResize(Ne, 2);
      h_ghost_B_.conservativeResize(Nb);
      v_origin_H_.conservativeResize(Nh);
      e_undirected_H_.conservativeResize(Nh);
      h_twin_H_.conservativeResize(Nh);
      h_rot_H_.conservativeResize(Nh);
      // update existing
      h_rot_H_[h0] = h3;
      e_undirected_H_[h1] = e0;
      h_twin_H_[h1] = h3;
      // assign new
      V_cycle_E_.row(e0) << v0, v1;
      h_directed_E_[e0] = h1;
      v_origin_H_[h3] = v0;
      e_undirected_H_[h3] = e0;
      h_twin_H_[h3] = h1;
      h_rot_H_[h3] = h2;
      // visual data
      rgba_E_.conservativeResize(Ne, 4);
      rgba_E_.row(e0) = rgba_edge_;
      rgba_H.conservativeResize(Nh, 4);
      rgba_H.row(h3) = rgba_half_edge_;
      return h3;
    }

    // f/f covers valance 0/0,0/2+,2+/2+
    int Ne0 = get_num_edges();
    int Nh0 = get_num_half_edges();
    int Ne = Ne0 + 1;
    int Nh = Nh0 + 2;

    int h0 = h_out_V_[v0];
    int h1 = h_out_V_[v1];
    int h2 = h_rot_h(h0); // = h0 for isolated v0
    int h3 = h_rot_h(h1); // = h1 for isolated v1

    // new elements
    int h4 = Nh0, h5 = Nh0 + 1;
    int e0 = Ne0;
    h_directed_E_.conservativeResize(Ne);
    V_cycle_E_.conservativeResize(Ne, 2);
    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);
    h_rot_H_.conservativeResize(Nh);

    // update existing
    h_rot_H_[h0] = h4;
    h_rot_H_[h1] = h5;
    // assign new
    V_cycle_E_.row(e0) << v0, v1;
    h_directed_E_[e0] = h4;

    v_origin_H_[h4] = v0;
    e_undirected_H_[h4] = e0;
    h_twin_H_[h4] = h5;
    h_rot_H_[h4] = h2;

    v_origin_H_[h5] = v1;
    e_undirected_H_[h5] = e0;
    h_twin_H_[h5] = h4;
    h_rot_H_[h5] = h3;
    // visual data
    rgba_E_.conservativeResize(Ne, 4);
    rgba_E_.row(e0) = rgba_edge_;
    rgba_H.conservativeResize(Nh, 4);
    rgba_H.row(h4) = rgba_half_edge_;
    rgba_H.row(h5) = rgba_half_edge_;
    return h4;
  }

  /**
   * @brief Remove edge from mesh
   *
   * @param f
   *
   * ```
   * 2 interior vertices
   *
   * h0_      e       _h3     h0_              _h3
   *   \     -h4->    /         \              /
   *   v0------------v1   -->   v0            v1
   *   /_    <-h5-   _\         /_            _\
   * h2                h1     h2                h1
   *
   *
   *
   * h2_      e             h0_
   *   \     -h0->            \
   *   v0------------v1 -->   v0
   *   /_                     /_
   * h1                     h2
   *
   *
   * ```
   */
  void remove_e(int e) {
    // *** check if things need to be deleted and swap with final elements FIRST
    // ***
    printf("SimplicialComplex1d::remove_e\n");
    check_e(e);
    int Ne0 = get_num_edges();
    int Nb0 = get_num_boundaries();
    int Nh0 = get_num_half_edges();

    int Ne = Ne0 - 1;
    int Nb = Nb0;
    int Nh = Nh0;
    printf("SimplicialComplex1d::remove_e - 0\n");
    int e0 = Ne0 - 1;
    swap_edge_indices(e, e0);
    printf("SimplicialComplex1d::remove_e - 1\n");

    int h0 = h_directed_E_[e0];
    int h1 = h_rot_H_[h0];
    int h2 = h_unrot_h(h0);

    int ht0 = h_twin_H_[h0];
    int ht1 = h_rot_H_[ht0];
    int ht2 = h_unrot_h(ht0);
    printf("SimplicialComplex1d::remove_e - 2\n");
    bool v0_valance_is_2 = (h1 == h2 && h_twin_H_[h1] != h1);
    bool vt0_valance_is_2 = (ht1 == ht2 && h_twin_H_[ht1] != ht1);

    printf("SimplicialComplex1d::remove_e - 3\n");

    if (v0_valance_is_2 && vt0_valance_is_2) {
      // dNe=-1,dNb=+2,dNh=0
      // create b0, bt0
      Nb = Nb0 + 2;

      int b0 = Nb0, bt0 = Nb0 + 1;
      int e0_ghost = -(b0 + 1), et0_ghost = -(bt0 + 1);

      V_cycle_E_.conservativeResize(Ne, 2);
      h_directed_E_.conservativeResize(Ne);
      rgba_E_.conservativeResize(Ne, 4);
      h_ghost_B_.conservativeResize(Nb);

      // update existing
      e_undirected_H_[h0] = e0_ghost;
      h_twin_H_[h0] = h0;
      e_undirected_H_[ht0] = et0_ghost;
      h_twin_H_[ht0] = ht0;
      // assign new
      h_ghost_B_[b0] = h0;
      h_ghost_B_[bt0] = ht0;
      return;
    }
    if (v0_valance_is_2 && !vt0_valance_is_2) {
      // dNe=-1,dNb=+1,dNh=-1
      // create b0, delete ht0
      Nb = Nb0 + 1;
      Nh = Nh0 - 1;

      int vt0 = v_origin_H_[ht0];
      int b0 = Nb0;
      int e0_ghost = -(b0 + 1);

      swap_halfedge_indices(ht0, Nh0 - 1);
      ht0 = Nh0 - 1;

      V_cycle_E_.conservativeResize(Ne, 2);
      h_directed_E_.conservativeResize(Ne);
      rgba_E_.conservativeResize(Ne, 4);

      v_origin_H_.conservativeResize(Nh);
      e_undirected_H_.conservativeResize(Nh);
      h_twin_H_.conservativeResize(Nh);
      h_rot_H_.conservativeResize(Nh);
      rgba_H.conservativeResize(Nh, 4);

      h_ghost_B_.conservativeResize(Nb);

      // update existing
      e_undirected_H_[h0] = e0_ghost;
      h_twin_H_[h0] = h0;
      if (h_out_V_[vt0] == ht0) {
        h_out_V_[vt0] = ht1;
      }
      h_rot_H_[ht2] = ht1;
      // assign new
      h_ghost_B_[b0] = h0;
      return;
    }
    if (!v0_valance_is_2 && vt0_valance_is_2) {
      // dNe=-1,dNb=+1,dNh=-1
      // create bt0, delete h0
      Nb = Nb0 + 1;
      Nh = Nh0 - 1;

      int v0 = v_origin_H_[h0];
      int bt0 = Nb0;
      int et0_ghost = -(bt0 + 1);

      swap_halfedge_indices(h0, Nh0 - 1);
      h0 = Nh0 - 1;

      V_cycle_E_.conservativeResize(Ne, 2);
      h_directed_E_.conservativeResize(Ne);
      rgba_E_.conservativeResize(Ne, 4);
      v_origin_H_.conservativeResize(Nh);
      e_undirected_H_.conservativeResize(Nh);
      h_twin_H_.conservativeResize(Nh);
      h_rot_H_.conservativeResize(Nh);
      rgba_H.conservativeResize(Nh, 4);
      h_ghost_B_.conservativeResize(Nb);
      // update existing
      if (h_out_V_[v0] == h0) {
        h_out_V_[v0] = h1;
      }
      h_rot_H_[h2] = h1;
      e_undirected_H_[ht0] = et0_ghost;
      h_twin_H_[ht0] = ht0;
      // assign new
      h_ghost_B_[bt0] = ht0;
      return;
    }
    // if (!v0_valance_is_2 && !vt0_valance_is_2)
    // dNe=-1,dNb=0,dNh=-2
    // delete h0, ht0
    int v0 = v_origin_H_[h0];
    int vt0 = v_origin_H_[ht0];
    printf("SimplicialComplex1d::remove_e - 3.5\n");
    Nh = Nh0 - 2;
    swap_halfedge_indices(h0, Nh0 - 2);
    printf("SimplicialComplex1d::remove_e - 4\n");
    swap_halfedge_indices(ht0, Nh0 - 1);
    printf("SimplicialComplex1d::remove_e - 5\n");
    h0 = Nh0 - 2, ht0 = Nh0 - 1;

    //
    //
    // *** check if things need to be deleted and swap with final elements FIRST
    // ***
    h1 = h_rot_H_[h0];
    h2 = h_unrot_h(h0);
    ht1 = h_rot_H_[ht0];
    ht2 = h_unrot_h(ht0);
    //
    //
    //

    V_cycle_E_.conservativeResize(Ne, 2);
    h_directed_E_.conservativeResize(Ne);
    rgba_E_.conservativeResize(Ne, 4);

    v_origin_H_.conservativeResize(Nh);
    e_undirected_H_.conservativeResize(Nh);
    h_twin_H_.conservativeResize(Nh);
    h_rot_H_.conservativeResize(Nh);
    rgba_H.conservativeResize(Nh, 4);
    printf("SimplicialComplex1d::remove_e - 6\n");
    // update existing
    if (h_out_V_[v0] == h0) {
      h_out_V_[v0] = h1;
    }
    printf("SimplicialComplex1d::remove_e - 7\n");
    h_rot_H_[h2] = h1;
    if (h_out_V_[vt0] == ht0) {
      h_out_V_[vt0] = ht1;
    }
    printf("SimplicialComplex1d::remove_e - 8\n");
    h_rot_H_[ht2] = ht1;
    // assign new
    printf("SimplicialComplex1d::remove_e - 9\n");
    return;

    // int h0 = h_directed_E_[e];
    // int h1 = h_twin_H_[h0];
    // int v0 = v_origin_H_[h0];
    // int v1 = v_origin_H_[h1];
    //
    // int valance0 = valance_v(v0); // = 1, 2, or 2+
    // int valance1 = valance_v(v1);
    //
    // int delta_Ne = -1;
    // int delta_Nb = 0;
    // bool del_h0 = true;
    // bool del_h1 = true;
    //
    // if (valance0 == 2) {
    //   delta_Nb += 1;
    //   del_h0 = false;
    // }
    // if (valance1 == 2) {
    //   delta_Nb += 1;
    //   del_h1 = false;
    // }
    // int delta_Nh = 2 * delta_Ne + delta_Nb;
    // swap_edge_indices(e, Ne0 - 1);
    // int h_swap = Nh0 - 1;
    // if (del_h0) {
    //   swap_halfedge_indices(h0, h_swap);
    //   h_swap = Nh0 - 2;
    // }
    // if (del_h1) {
    //   swap_halfedge_indices(h1, h_swap);
    // }
  }

  void swap_vertex_indices(int v0, int v1) {
    check_v(v0);
    check_v(v1);
    if (v0 == v1) {
      return;
    }
    bool adjacent = false;
    int h01;

    // find and update things that output v0/v1 indices
    // EXCEPT edge and half-edges incident on both vertices
    // v0->v1
    int h_start = h_out_V_[v0];
    int h = h_start;
    do {
      // check other vertex on this edge
      int ht = h_twin_H_[h];
      if (v_origin_H_[ht] == v1) {
        adjacent = true;
        h01 = h;
        h = h_rot_H_[h];
        continue;
      }
      v_origin_H_[h] = v1;
      int e = e_undirected_H_[h];
      if (e < 0) {
        h = h_rot_H_[h];
        continue;
      }
      if (V_cycle_E_(e, 0) == v0) {
        V_cycle_E_(e, 0) = v1;
      } else {
        V_cycle_E_(e, 1) = v1;
      }
      h = h_rot_H_[h];
    } while (h != h_start);
    // v1->v0
    h_start = h_out_V_[v1];
    h = h_start;
    do {
      int ht = h_twin_H_[h];
      if (ht == h01) {
        h = h_rot_H_[h];
        continue;
      }
      v_origin_H_[h] = v0;
      int e = e_undirected_H_[h];
      if (e < 0) {
        h = h_rot_H_[h];
        continue;
      }
      int i = V_cycle_E_(e, 0);
      if (V_cycle_E_(e, 0) == v1) {
        V_cycle_E_(e, 0) = v0;
      } else {
        V_cycle_E_(e, 1) = v0;
      }
      h = h_rot_H_[h];
    } while (h != h_start);

    if (adjacent) {
      int h10 = h_twin_H_[h01];
      v_origin_H_[h01] = v1;
      v_origin_H_[h10] = v0;
      int e = e_undirected_H_[h01];
      if (V_cycle_E_(e, 0) == v0) {
        V_cycle_E_.row(e) << v1, v0;
      } else {
        V_cycle_E_.row(e) << v0, v1;
      }
    }
    // update things that eat v0/v1 indices
    auto xyz_coord_v0 = xyz_coord_v(v0);
    auto h_out_v0 = h_out_V_[v0];
    xyz_coord_V_.row(v0) = xyz_coord_v(v1);
    h_out_V_[v0] = h_out_V_[v1];
    xyz_coord_V_.row(v1) = xyz_coord_v0;
    h_out_V_[v1] = h_out_v0;
    // visual data
    auto radius_v0 = radius_V[v0];
    auto rgba_v0 = rgba_V.row(v0).eval();
    radius_V[v0] = radius_V[v1];
    rgba_V.row(v0) = rgba_V.row(v1).eval();
    radius_V[v1] = radius_v0;
    rgba_V.row(v1) = rgba_v0;
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
    auto V_cycle_e0 = V_cycle_E_.row(e0).eval();

    // update things that output e0/e1 indices
    e_undirected_H_[h0] = e1;
    e_undirected_H_[ht0] = e1;
    e_undirected_H_[h1] = e0;
    e_undirected_H_[ht1] = e0;

    // update things that eat e0/e1 indices
    h_directed_E_[e0] = h1;
    V_cycle_E_.row(e0) = V_cycle_E_.row(e1);
    h_directed_E_[e1] = h0;
    V_cycle_E_.row(e1) = V_cycle_e0;
    // visual data
    rgba_E_.row(e0).swap(rgba_E_.row(e1));
  }

  void swap_boundary_indices(int b0, int b1) {
    check_b(b0);
    check_b(b1);
    if (b0 == b1) {
      return;
    }
    int h0 = h_ghost_B_[b0];
    int h1 = h_ghost_B_[b1];
    // update things that output b0/b1 indices
    e_undirected_H_[h0] = -(b1 + 1);
    e_undirected_H_[h1] = -(b0 + 1);
    // update things that eat b0/b1 indices
    h_ghost_B_[b0] = h1;
    h_ghost_B_[b1] = h0;
  }

  void swap_halfedge_indices(int h0, int h1) {
    printf("SimplicialComplex1d::swap_halfedge_indices\n");
    printf("h0=%d, h1=%d\n", h0, h1);
    // check_h(h0);
    // check_h(h1);
    // int v0 = v_origin_H_[h0];
    // int e0 = e_undirected_H_[h0];
    // int b0 = -(e0 + 1);
    // int h_twin0 = h_twin_H_[h0];
    // int h_rot0 = h_rot_H_[h0];
    // int h_unrot0 = h_unrot_h(h0);
    //
    // int v1 = v_origin_H_[h1];
    // int e1 = e_undirected_H_[h1];
    // int b1 = -(e1 + 1);
    // int h_twin1 = h_twin_H_[h1];
    // int h_rot1 = h_rot_H_[h1];
    // int h_unrot1 = h_unrot_h(h1);
    //
    // // update things that output h0/h1 indices
    // if (h_out_V_[v0] == h0) {
    //   h_out_V_[v0] = h1;
    // }
    // if (e0 < 0) {
    //   h_ghost_B_[b0] = h1;
    // } else if (h_directed_E_[e0] == h0) {
    //   h_directed_E_[e0] = h1;
    // }
    // h_twin_H_[h_twin0] = h1; // **
    // h_rot_H_[h_unrot0] = h1; // **
    //
    // if (h_out_V_[v1] == h1) {
    //   h_out_V_[v1] = h0;
    // }
    // if (e1 < 0) {
    //   h_ghost_B_[b1] = h0;
    // } else if (h_directed_E_[e1] == h1) {
    //   h_directed_E_[e1] = h0;
    // }
    // h_twin_H_[h_twin1] = h0; //**
    // h_rot_H_[h_unrot1] = h0; // **
    //
    // // update things that eat h0/h1 indices
    // v_origin_H_[h0] = v1;
    // e_undirected_H_[h0] = e1;
    // h_twin_H_[h0] = h_twin1; // ***
    // h_rot_H_[h0] = h_rot1;
    //
    // v_origin_H_[h1] = v0;
    // e_undirected_H_[h1] = e0;
    // h_twin_H_[h1] = h_twin0;
    // h_rot_H_[h1] = h_rot0;
    // // visual data
    // auto rgba_h0 = rgba_H.row(h0).eval();
    // rgba_H.row(h0) = rgba_H.row(h1);
    // rgba_H.row(h1) = rgba_h0;
    check_h(h0);
    check_h(h1);
    if (h0 == h1) {
      return;
    }
    printf("SimplicialComplex1d::swap_halfedge_indices - 0\n");
    int v0 = v_origin_H_[h0];
    int e0 = e_undirected_H_[h0];
    int b0 = -(e0 + 1);
    int h_twin0 = h_twin_H_[h0];
    int h_rot0 = h_rot_H_[h0];
    int h_unrot0 = h_unrot_h(h0);
    printf("v0=%d, e0=%d, b0=%d, h_twin0=%d, h_rot0=%d, h_unrot0=%d\n", v0, e0,
           b0, h_twin0, h_rot0, h_unrot0);
    printf("SimplicialComplex1d::swap_halfedge_indices - 1\n");
    int v1 = v_origin_H_[h1];
    int e1 = e_undirected_H_[h1];
    int b1 = -(e1 + 1);
    int h_twin1 = h_twin_H_[h1];
    int h_rot1 = h_rot_H_[h1];
    int h_unrot1 = h_unrot_h(h1);
    printf("v1=%d, e1=%d, b1=%d, h_twin1=%d, h_rot1=%d, h_unrot1=%d\n", v1, e1,
           b1, h_twin1, h_rot1, h_unrot1);
    printf("SimplicialComplex1d::swap_halfedge_indices - 2\n");
    //
    // int new_h0 = h1;
    // int new_h_twin0 = (h_twin0 == h0) ? new_h0 : h_twin0;
    // int new_h_rot0 = (h_rot0 == h0) ? new_h0 : h_rot0; // ***
    // int new_h_unrot0 = (h_unrot0 == h0) ? new_h0 : h_unrot0;
    // int new_h1 = h0;
    // int new_h_twin1 = (h_twin1 == h1) ? new_h1 : h_twin1;
    // int new_h_rot1 = (h_rot1 == h1) ? new_h1 : h_rot1;
    // int new_h_unrot1 = (h_unrot1 == h1) ? new_h1 : h_unrot1;

    int new_h0 = h1;
    int new_h1 = h0;

    int new_h_twin0 = (h_twin0 == h0)   ? new_h0
                      : (h_twin0 == h1) ? new_h1
                                        : h_twin0;
    int new_h_rot0 = (h_rot0 == h0) ? new_h0 : (h_rot0 == h1) ? new_h1 : h_rot0;
    int new_h_unrot0 = (h_unrot0 == h0)   ? new_h0
                       : (h_unrot0 == h1) ? new_h1
                                          : h_unrot0;

    int new_h_twin1 = (h_twin1 == h1)   ? new_h1
                      : (h_twin1 == h0) ? new_h0
                                        : h_twin1;
    int new_h_rot1 = (h_rot1 == h1) ? new_h1 : (h_rot1 == h0) ? new_h0 : h_rot1;
    int new_h_unrot1 = (h_unrot1 == h1)   ? new_h1
                       : (h_unrot1 == h0) ? new_h0
                                          : h_unrot1;
    printf("SimplicialComplex1d::swap_halfedge_indices - 3\n");
    printf("v0=%d, e0=%d, b0=%d, new_h_twin0=%d\n", v0, e0, b0, new_h_twin0);
    // update things that output h0/h1 indices
    if (h_out_V_[v0] == h0) {
      h_out_V_[v0] = new_h0;
    }
    printf("SimplicialComplex1d::swap_halfedge_indices - 3.1\n");
    if (e0 < 0) {
      h_ghost_B_[b0] = new_h0;
    } else if (h_directed_E_[e0] == h0) {
      printf("SimplicialComplex1d::swap_halfedge_indices - 3.2\n");
      h_directed_E_[e0] = new_h0;
    }
    printf("SimplicialComplex1d::swap_halfedge_indices - 3.3\n");
    h_twin_H_[new_h_twin0] = new_h0; // **
    printf("SimplicialComplex1d::swap_halfedge_indices - 3.4\n");
    h_rot_H_[new_h_unrot0] = new_h0; // **
    printf("SimplicialComplex1d::swap_halfedge_indices - 4\n");
    if (h_out_V_[v1] == h1) {
      h_out_V_[v1] = new_h1;
    }
    if (e1 < 0) {
      h_ghost_B_[b1] = new_h1;
    } else if (h_directed_E_[e1] == h1) {
      h_directed_E_[e1] = new_h1;
    }
    h_twin_H_[new_h_twin1] = new_h1; //***
    h_rot_H_[new_h_unrot1] = new_h1; // **
    printf("SimplicialComplex1d::swap_halfedge_indices - 5\n");
    // update things that eat h0/h1 indices
    v_origin_H_[new_h0] = v0;
    e_undirected_H_[new_h0] = e0;
    h_twin_H_[new_h0] = new_h_twin0;
    h_rot_H_[new_h0] = new_h_rot0;
    printf("SimplicialComplex1d::swap_halfedge_indices - 6\n");
    v_origin_H_[new_h1] = v1;
    e_undirected_H_[new_h1] = e1;
    h_twin_H_[new_h1] = new_h_twin1; // ***
    h_rot_H_[new_h1] = new_h_rot1;
    printf("SimplicialComplex1d::swap_halfedge_indices - 7\n");
    // visual data
    auto rgba_h0 = rgba_H.row(h0).eval();
    rgba_H.row(new_h1) = rgba_H.row(h1);
    rgba_H.row(new_h0) = rgba_h0;
    printf("SimplicialComplex1d::swap_halfedge_indices - 8\n");
  }

  void print_he_mats() {
    std::cout << "h_out_V = " << h_out_V_.transpose() << '\n';

    std::cout << "h_directed_E = " << h_directed_E_.transpose() << '\n';

    std::cout << "h_ghost_B = " << h_ghost_B_.transpose() << '\n';

    std::cout << "v_origin_H = " << v_origin_H_.transpose() << '\n';
    std::cout << "e_undirected_H = " << e_undirected_H_.transpose() << '\n';
    std::cout << "h_twin_H = " << h_twin_H_.transpose() << '\n';
    std::cout << "h_rot_H = " << h_rot_H_.transpose() << '\n';
  }
};

} // namespace meshbrane
