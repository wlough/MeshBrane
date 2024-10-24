/**
 * @file chalf_edge.hpp
 */
#ifndef CHALF_EDGE_HPP
#define CHALF_EDGE_HPP
#include <cgenerators.hpp>
#include <data_types.hpp>
#include <optional> // std::optional
#include <string>   // std::string

class HalfEdgeMesh {
public:
  ///////////////////////////////////////////////////////
  // Constructors ///////////////////////////////////////
  ///////////////////////////////////////////////////////
  HalfEdgeMesh(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
               const Samplesi &v_origin_H, const Samplesi &h_next_H,
               const Samplesi &h_twin_H, const Samplesi &f_left_H,
               const Samplesi &h_bound_F, const Samplesi &h_right_B);
  static HalfEdgeMesh from_he_ply(const std::string &ply_path);

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
  const Samplesi &get_h_bound_F() const;
  void set_h_bound_F(const Samplesi &value);
  const Samplesi &get_h_right_B() const;
  void set_h_right_B(const Samplesi &value);

  int get_num_vertices() const;
  int get_num_edges() const;
  int get_num_half_edges() const;
  int get_num_faces() const;
  int get_euler_characteristic() const;
  int get_num_boundaries() const;
  int get_genus() const;

  Samples3i V_of_F() const;
  // Samples2i V_of_H() const;
  // Samples2i V_of_E() const;
  VertexFaceSamples vf_samples() const;
  HalfEdgeSamples he_samples() const;
  Samplesi F_incident_b(int b) const;
  ///////////////////////////////////////////////////////
  // Combinatorial maps /////////////////////////////////
  ///////////////////////////////////////////////////////
  Coords3d xyz_coord_v(int v) const;
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
  int h_bound_f(int f) const;
  Samplesi h_bound_f(const Samplesi &indices) const;
  int h_right_b(int b) const;
  Samplesi h_right_b(const Samplesi &indices) const;
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
  SimpleGenerator<int> generate_V_of_f(int f) const;
  SimpleGenerator<int> generate_H_out_v_clockwise(int v,
                                                  int h_start = -1) const;
  SimpleGenerator<int> generate_H_bound_f(int f, int h_start = -1) const;
  SimpleGenerator<int> generate_H_rotcw_h(int h) const;
  SimpleGenerator<int> generate_H_next_h(int h) const;
  SimpleGenerator<int> generate_H_right_b(int b) const;
  ///////////////////////////////////////////////////////
  // Mutators ///////////////////////////////////////////
  ///////////////////////////////////////////////////////
  void update_vertex(int v,
                     const std::optional<Coords3d> &xyz_coord = std::nullopt,
                     const std::optional<int> &h_out = std::nullopt);
  void update_half_edge(int h,
                        const std::optional<int> &v_origin = std::nullopt,
                        const std::optional<int> &h_next = std::nullopt,
                        const std::optional<int> &h_twin = std::nullopt,
                        const std::optional<int> &f_left = std::nullopt);
  void update_face(int f, const std::optional<int> &h_left = std::nullopt);
  /**
   * @brief Flips edge h.
   *
   * @param h
   * h cannot be on boundary!
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
   */
  void flip_edge(int h);
  int flip_non_delaunay();

private:
  ///////////////
  // Attributes /
  ///////////////
  Samples3d _xyz_coord_V; //
  Samplesi _h_out_V;      //
  Samplesi _v_origin_H;   //
  Samplesi _h_next_H;     //
  Samplesi _h_twin_H;     //
  Samplesi _f_left_H;     //
  Samplesi _h_bound_F;    //
  Samplesi _h_right_B;    //
};

#endif /* CHALF_EDGE_HPP */