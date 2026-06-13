#pragma once

/**
 * @file patch.hpp
 * @brief Defines the Patch class. Represents a nice subset of a meshed surface
 */

#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/meshbrane_object.hpp"
#include "meshbrane/pretty_pictures.hpp"
#include "meshbrane/simple_generator.hpp"

namespace meshbrane {
class MatrixMesh; // forward declaration

// class SubComplex : public MeshBraneObject {
// public:
//   MatrixMesh *supermesh_{nullptr};
//   SimplicialSet F_;
//   SubComplex() = default;
//   ~SubComplex() = default;
//   SubComplex(MatrixMesh *supermesh, SimplicialSet F)
//       : supermesh_(supermesh), F_(F) {};
// };

/**
 * @brief  A submanifold of a HalfEdgeMesh.
 *
 *  @param supermesh (HalfEdgeMesh): mesh containing the patch
 *  @param V (set): set of vertices in the patch
 *  @param E (set): set of edges in the patch
 *  @param F (set): set of faces in the patch
 *  @param H (set): set of half-edges in the patch
 *
 */
class Patch {
  // class Patch : public MeshBraneObject {
public:
  MatrixMesh *supermesh_{nullptr};
  SimplicialSet V_;
  SimplicialSet E_;
  SimplicialSet F_;
  SimplicialSet H_;
  SimplicialSet bdryV_;
  SimplicialSet frontierV_;
  SimplicialSet frontierE_;
  SimplicialSet frontierF_;
  SimplicialSet newF_;
  SimplicialSet newV_;
  SimplicialSet coloredF_;
  SimplicialSet coloredE_;
  Samplesi h_negative_B_;
  RGBA rgba_face_ = RGBA_DICT.at("meshbrane_red");
  RGBA rgba_seed_vertex_ = RGBA_DICT.at("black");
  RGBA rgba_frontier_face_ = RGBA_DICT.at("yellow");
  RGBA rgba_boundary_vertex_ = RGBA_DICT.at("meshbrane_orange");
  RGBA rgba_edge_ = RGBA_DICT.at("meshbrane_red");
  int seed_vertex_{0};
  //////////////////////////////////////////
  // Initialization ////////////////////////
  //////////////////////////////////////////
  void set_supermesh(MatrixMesh &mesh) { supermesh_ = &mesh; }

  Patch() = default;
  ~Patch() = default;
  Patch(MatrixMesh *supermesh) : supermesh_(supermesh) {};
  Patch(MatrixMesh *supermesh, SimplicialSet V, SimplicialSet E,
        SimplicialSet F, SimplicialSet H)
      : supermesh_(supermesh), V_(V), E_(E), F_(F), H_(H) {};
  static Patch from_seed_vertex(MatrixMesh *supermesh, int i);

  static Patch from_ball(MatrixMesh *supermesh, Vec3d p0, double r_max);
  static Patch from_cylinder(MatrixMesh *supermesh, Vec3d p0, Vec3d ez,
                             double r_max, double delta_z = 1.0);
  void update_patch();
  //////////////////////////////////////////
  // Getters ///////////////////////////////
  //////////////////////////////////////////
  Vec3d xyz_coord_v(int v);
  int h_out_v(int v); //
  int h_directed_e(int e);
  int h_right_f(int f);
  int v_origin_h(int h);
  int e_undirected_h(int h);
  int f_left_h(int h); //
  int h_next_h(int h); //
  int h_twin_h(int h);

  size_t num_vertices() { return V_.size(); }
  size_t num_edges() { return E_.size(); }
  size_t num_faces() { return F_.size(); }
  size_t num_half_edges() { return H_.size(); }

  ///////////////////////////////////////////////////////
  // Generators /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  utils::SimpleGenerator<int> generate_H_next_h(int h);
  utils::SimpleGenerator<int> generate_H_negative_bdry();
  utils::SimpleGenerator<int> generate_V_negative_bdry();

  ///////////////////////////////////////////////////////
  // Predicates /////////////////////////////////////////
  ///////////////////////////////////////////////////////
  bool V_contains(int v) { return V_.find(v) != V_.end(); }
  bool E_contains(int e) { return E_.find(e) != E_.end(); }
  bool F_contains(int f) { return F_.find(f) != F_.end(); }
  bool H_contains(int h) { return E_.find(e_undirected_h(h)) != E_.end(); }
  bool some_negative_boundary_contains_h(int h);
  bool some_positive_boundary_contains_h(int h);
  ///////////////////////////////////////////////////////
  // Visualization //////////////////////////////////////
  ///////////////////////////////////////////////////////
  void color_edges();
  void uncolor_edges();
  void color_faces();
  void uncolor_faces();
  //////////////////////////////////////////
  // Embiggeners and unembiggeners /////////
  //////////////////////////////////////////
  void add_vertex(int v);
  void add_neighborhood_v(int v);

  int move_towards_cylinder(Vec3d p0, Vec3d ez, double r_max,
                            double z_min = 0.0,
                            double z_max = std::numeric_limits<double>::max());
  int move_towards_sphere(Vec3d p0, double r_max);
  void expand_by_one_ring();
  //////////////////////////////////////////
  // Utility functions /////////////////////
  //////////////////////////////////////////
  void patch_check_v(int v) {
    if (V_.find(v) == V_.end()) {
      throw std::invalid_argument("Vertex not in patch");
    }
  }
  void patch_check_e(int e) {
    if (E_.find(e) == E_.end()) {
      throw std::invalid_argument("Edge not in patch");
    }
  }
  void patch_check_f(int f) {
    if (F_.find(f) == F_.end()) {
      throw std::invalid_argument("Face not in patch");
    }
  }
  void patch_check_h(int h) {
    if (E_.find(e_undirected_h(h)) == E_.end()) {
      throw std::invalid_argument("Half-edge not in patch");
    }
  }
  void clear();
  /**
   * @brief Find vertices in the boundary of the patch and store them in bdryV_.
   *
   */
  void find_bdryV();
  /**
   * @brief Find faces in supermesh which are incident on the patch boundary and
   * store them in frontierF_.
   *
   */
  void find_frontierF();
  //////////////////////////////////////////
  // Unused ////////////////////////////////
  //////////////////////////////////////////
  static Patch from_seed_to_cylinder(MatrixMesh *supermesh, int seed_vertex,
                                     Vec3d p0, Vec3d ez, double r_max);
  VertexFaceTuple get_vf_tuple();
};

} // namespace meshbrane
