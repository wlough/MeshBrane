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
  RGBA rgba_edge_ = RGBA_DICT.at("blue");
  double radius_vertex_{5};

  /////////////////////////////
  // Interaction accumulator //
  /////////////////////////////
  Samples3d force_V_;

  ////////////////////
  // Cached data /////
  ////////////////////

  Samples3d vec_H_;
  Samples1d length_E_;
  // double average_edge_length_{0.0};
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
  ////////////////////////////////////
  // Cache updaters //////////////////
  ////////////////////////////////////

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
  // bool some_boundary_contains_v(int v) const;

  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////

  //////////////////////////////////////////////
  // Visualization /////////////////////////////
  //////////////////////////////////////////////
};

} // namespace meshbrane
