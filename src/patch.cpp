/**
 * @file patch.cpp
 */

#include "meshbrane/patch.hpp"
#include "meshbrane/geometric_predicates.hpp"
#include "meshbrane/matrix_mesh.hpp"
#include <iostream>

namespace meshbrane {

//////////////////////////////////////////
// Initialization ////////////////////////
//////////////////////////////////////////

Patch Patch::from_seed_vertex(MatrixMesh *supermesh, int i) {
  // printf("Patch::from_seed_vertex\n");
  Patch p;
  p.seed_vertex_ = i;
  p.supermesh_ = supermesh;
  p.V_.insert(i);
  int h = supermesh->h_out_v(i);
  for (int h : supermesh->generate_H_out_v_clockwise(i)) {
    int h_next = supermesh->h_next_h(h);
    int v = supermesh->v_origin_h(h_next);
    int e = supermesh->e_undirected_h(h);
    p.V_.insert(v);
    p.E_.insert(e);
    if (supermesh->some_negative_boundary_contains_h(h)) {
      continue;
    }
    int f = supermesh->f_left_h(h);
    int e_next = supermesh->e_undirected_h(h_next);
    p.F_.insert(f);
    p.frontierF_.insert(f);
    p.E_.insert(e_next);
  }
  p.find_bdryV();

  // printf("  #V: %d\n", p.V_.size());
  // printf("  #E: %d\n", p.E_.size());
  // printf("  #F: %d\n", p.F_.size());
  return p;
}

Patch Patch::from_ball(MatrixMesh *supermesh, Vec3d p0, double r_max) {
  // printf("Patch::from_ball\n");
  // printf("  p0 = (%f, %f, %f)\n", p0[0], p0[1], p0[2]);
  // printf("  r_max = %f\n", r_max);
  //////////////////////////////////////////
  double z_seed = std::numeric_limits<double>::min();
  int seed_vertex = -1;
  for (int v = 0; v < supermesh->get_num_vertices(); v++) {
    Vec3d p = supermesh->xyz_coord_V_.row(v);
    if (point_is_in_ball(p, p0, r_max)) {
      seed_vertex = v;
      break;
    }
  }
  if (seed_vertex == -1) {

    return Patch();
  }

  ////////////////////////////////////////////
  Patch s = from_seed_vertex(supermesh, seed_vertex);
  int dNf = 1;
  do {
    dNf = s.move_towards_sphere(p0, r_max);
    // printf("  dNf = %d\n", dNf);
  } while (dNf > 0);

  return s;
}

Patch Patch::from_cylinder(MatrixMesh *supermesh, Vec3d p0, Vec3d ez,
                           double r_max, double delta_z) {
  printf("Patch::from_cylinder\n");
  printf("  p0 = (%f, %f, %f)\n", p0[0], p0[1], p0[2]);
  printf("  ez = (%f, %f, %f\n)", ez[0], ez[1], ez[2]);
  printf("  r_max = %f\n", r_max);
  printf("  delta_z = %f\n", delta_z);
  //////////////////////////////////////////
  double z_seed = std::numeric_limits<double>::min();
  int seed_vertex = 0;
  for (int v = 0; v < supermesh->get_num_vertices(); v++) {
    Vec3d p = supermesh->xyz_coord_V_.row(v);
    Vec3d p_p0 = p - p0;
    double z = math::dot(ez, p_p0);
    if (z > z_seed) {
      z_seed = z;
      seed_vertex = v;
    }
  }
  ////////////////////////////////////////////
  double z_min = z_seed - delta_z;
  double z_max = z_seed;
  Vec3d p = supermesh->xyz_coord_v(seed_vertex);
  printf("  seed_vertex: %d\n", seed_vertex);
  printf("p: %f, %f, %f\n", p[0], p[1], p[2]);
  if (!point_is_in_cylinder(p, p0, ez, r_max, z_min, z_max)) {
    throw std::invalid_argument("Seed vertex not in cylinder");
  }
  z_min = -math::min(delta_z, z_seed);
  z_max = 0.0;
  Patch s = from_seed_vertex(supermesh, seed_vertex);
  int dNf = 1;
  do {
    dNf = s.move_towards_cylinder(p, ez, r_max, z_min, z_max);
    printf("  dNf = %d\n", dNf);
  } while (dNf > 0);

  return s;
}

void Patch::update_patch() {
  printf("Patch::update_patch\n");
  find_bdryV();
  find_frontierF();
}
//////////////////////////////////////////
// Getters ///////////////////////////////
//////////////////////////////////////////
Vec3d Patch::xyz_coord_v(int v) {
  patch_check_v(v);
  return supermesh_->xyz_coord_v(v);
}
int Patch::h_out_v(int v) {
  patch_check_v(v);
  for (int h : supermesh_->generate_H_out_v_clockwise(v)) {
    if (!H_contains(h)) {
      continue;
    } else if (F_contains(supermesh_->f_left_h(h))) {
      return h;
    }
  }
  throw std::invalid_argument("No outgoing half-edge in patch");
}
int Patch::h_directed_e(int e) {
  patch_check_e(e);
  return supermesh_->h_directed_e(e);
}
int Patch::h_right_f(int f) {
  patch_check_f(f);
  return supermesh_->h_right_f(f);
}
int Patch::v_origin_h(int h) {
  patch_check_h(h);
  return supermesh_->v_origin_h(h);
}
int Patch::e_undirected_h(int h) {
  patch_check_h(h);
  return supermesh_->e_undirected_h(h);
}
int Patch::f_left_h(int h) {
  patch_check_h(h);
  if (F_contains(supermesh_->f_left_h(h))) {
    return supermesh_->f_left_h(h);
  } else {
    printf("***NOT IMPLEMENTED***");
    return -1;
  }
}
int Patch::h_next_h(int h) {
  patch_check_h(h);
  if (F_contains(supermesh_->f_left_h(h))) {
    return supermesh_->h_next_h(h);
  }
  int n = supermesh_->h_next_h(h);
  while (!H_contains(n)) {
    n = supermesh_->h_rotcw_h(n);
  }
  return n;
}
int Patch::h_twin_h(int h) {
  patch_check_h(h);
  return supermesh_->h_twin_h(h);
}
///////////////////////////////////////////
// Generators /////////////////////////////
///////////////////////////////////////////

utils::SimpleGenerator<int> Patch::generate_H_next_h(int h) {
  int h_start = h;
  do {
    co_yield h;
    h = h_next_h(h);
  } while (h != h_start);
}

utils::SimpleGenerator<int> Patch::generate_H_negative_bdry() {
  for (int h : h_negative_B_) {
    for (int hh : generate_H_next_h(h)) {
      co_yield hh;
    }
  }
}

utils::SimpleGenerator<int> Patch::generate_V_negative_bdry() {
  for (int h : generate_H_negative_bdry()) {
    co_yield v_origin_h(h);
  }
}

///////////////////////////////////////////////////////
// Predicates /////////////////////////////////////////
///////////////////////////////////////////////////////
bool Patch::some_negative_boundary_contains_h(int h) {
  int f = supermesh_->f_left_h(h);
  int ht = supermesh_->h_twin_h(h);
  int ft = supermesh_->f_left_h(ht);
  return (F_contains(ft) && !F_contains(f));
}
bool Patch::some_positive_boundary_contains_h(int h) {
  int f = supermesh_->f_left_h(h);
  int ht = supermesh_->h_twin_h(h);
  int ft = supermesh_->f_left_h(ht);
  return (!F_contains(ft) && F_contains(f));
}
///////////////////////////////////////////
// Visualization //////////////////////////
///////////////////////////////////////////

void Patch::color_edges() {
  uncolor_edges();
  for (int e : E_) {
    supermesh_->rgba_E_.row(e) = rgba_edge_;
    coloredE_.insert(e);
  }
}
void Patch::uncolor_edges() {
  for (int e : coloredE_) {
    supermesh_->rgba_E_.row(e) = supermesh_->rgba_edge_;
  }
  coloredE_.clear();
}

void Patch::color_faces() {
  uncolor_faces();
  for (int f : F_) {
    supermesh_->rgba_F.row(f) = rgba_face_;
    coloredF_.insert(f);
  }
}

void Patch::uncolor_faces() {
  for (int f : coloredF_) {
    supermesh_->rgba_F.row(f) = supermesh_->rgba_face_;
  }
  coloredF_.clear();
}

//////////////////////////////////////////
// Embiggeners and unembiggeners /////////
//////////////////////////////////////////

void Patch::add_vertex(int v0) {
  // printf("Patch::add_vertex\n");
  // std::cout << "supermesh_.name_=" << supermesh_->name_ << '\n';
  // std::cout << "inside Patch::add_vertex\n";
  // std::cout << "  supermesh_         = " << supermesh_ << '\n';
  // std::cout << "  &h_out_V_          = " << &supermesh_->h_out_V_ << '\n';
  // std::cout << "  h_out_V_.data()    = " << supermesh_->h_out_V_.data() <<
  // '\n'; std::cout << "  h_out_V_ rows/cols = " << supermesh_->h_out_V_.rows()
  // << " "
  //           << supermesh_->h_out_V_.cols() << '\n';
  // std::cout << "  h_out_V_(0,0)      = " << supermesh_->h_out_V_(0, 0) <<
  // '\n'; supermesh_->check_he_matrices(); // fails here... but is fine when
  // called just
  //                                  // outside this function
  V_.insert(v0);
  bdryV_.insert(v0);
  // supermesh_->check_he_matrices();
  for (int h : supermesh_->generate_H_out_v_clockwise(v0)) {
    // printf("Patch::add_vertex 2 --- h=%d\n", h);
    // std::cout << "Patch::add_vertex 2 h=" << h << '\n';
    // printf("Patch::add_vertex 2 --- h_max=%zu\n",
    //        supermesh_->get_num_half_edges());
    if (supermesh_->some_negative_boundary_contains_h(h)) {
      continue;
    }
    int f = supermesh_->f_left_h(h);
    frontierF_.insert(f);
  }
}

void Patch::add_neighborhood_v(int v0) {
  // printf("Patch::add_neighborhood_v\n");
  if (!V_contains(v0)) {
    newV_.insert(v0);
    V_.insert(v0);
  }

  // printf("  V_.size() = %d\n", V_.size());
  for (int h : supermesh_->generate_H_out_v_clockwise(v0)) {
    // printf("    h = %d\n", h);
    int h_next = supermesh_->h_next_h(h);
    int v = supermesh_->v_origin_h(h_next);
    int e = supermesh_->e_undirected_h(h);
    // V_.insert(v);
    if (!V_contains(v)) {
      newV_.insert(v);
      V_.insert(v);
    }
    E_.insert(e);
    if (supermesh_->some_negative_boundary_contains_h(h)) {
      continue;
    }
    int f = supermesh_->f_left_h(h);
    int e_next = supermesh_->e_undirected_h(h_next);

    if (!F_contains(f)) {
      newF_.insert(f);
      F_.insert(f);
    }

    frontierF_.insert(f);
    E_.insert(e_next);
  }
  find_bdryV();
  find_frontierF();
}

int Patch::move_towards_sphere(Vec3d p0, double r_max) {
  int dNf = 0;
  int num_faces = F_.size();
  find_bdryV();
  find_frontierF();

  for (int f : frontierF_) {
    int h0 = supermesh_->h_right_f(f);
    int h1 = supermesh_->h_next_h(h0);
    int h2 = supermesh_->h_next_h(h1);
    int v0 = supermesh_->v_origin_h(h0);
    int v1 = supermesh_->v_origin_h(h1);
    int v2 = supermesh_->v_origin_h(h2);
    int e0 = supermesh_->e_undirected_h(h0);
    int e1 = supermesh_->e_undirected_h(h1);
    int e2 = supermesh_->e_undirected_h(h2);
    Vec3d x0 = supermesh_->xyz_coord_v(v0);
    Vec3d x1 = supermesh_->xyz_coord_v(v1);
    Vec3d x2 = supermesh_->xyz_coord_v(v2);
    Vec3d x = (x0 + x1 + x2) / 3.0;
    if (point_is_in_ball(x, p0, r_max)) {
      F_.insert(f);
      E_.insert(e0);
      E_.insert(e1);
      E_.insert(e2);
      V_.insert(v0);
      V_.insert(v1);
      V_.insert(v2);
      continue;
    }
    if (!F_contains(f)) {
      continue;
    }
    // point is outside sphere and face is in patch
    F_.erase(f);
    E_.erase(e0);
    E_.erase(e1);
    E_.erase(e2);
    V_.erase(v0);
    V_.erase(v1);
    V_.erase(v2);
    for (int h : {h0, h1, h2}) {
      int ht = supermesh_->h_twin_h(h);
      int ft = supermesh_->f_left_h(ht);
      if (F_contains(ft)) {
        int v = supermesh_->v_origin_h(h);
        int vt = supermesh_->v_origin_h(ht);
        int e = supermesh_->e_undirected_h(h);
        V_.insert(v);
        V_.insert(vt);
        E_.insert(e);
        frontierF_.insert(ft);
      }
    }
  }
  dNf = F_.size() - num_faces;
  return dNf;
}

int Patch::move_towards_cylinder(Vec3d p0, Vec3d ez, double r_max, double z_min,
                                 double z_max) {
  // printf("Patch::move_towards_cylinder\n");
  int dNf = 0;
  int num_faces = F_.size();
  find_bdryV();
  find_frontierF();

  for (int f : frontierF_) {
    int h0 = supermesh_->h_right_f(f);
    int h1 = supermesh_->h_next_h(h0);
    int h2 = supermesh_->h_next_h(h1);
    int v0 = supermesh_->v_origin_h(h0);
    int v1 = supermesh_->v_origin_h(h1);
    int v2 = supermesh_->v_origin_h(h2);
    int e0 = supermesh_->e_undirected_h(h0);
    int e1 = supermesh_->e_undirected_h(h1);
    int e2 = supermesh_->e_undirected_h(h2);
    Vec3d x0 = supermesh_->xyz_coord_v(v0);
    Vec3d x1 = supermesh_->xyz_coord_v(v1);
    Vec3d x2 = supermesh_->xyz_coord_v(v2);
    Vec3d x = (x0 + x1 + x2) / 3.0;
    if (point_is_in_cylinder(x, p0, ez, r_max, z_min, z_max)) {
      F_.insert(f);
      E_.insert(e0);
      E_.insert(e1);
      E_.insert(e2);
      V_.insert(v0);
      V_.insert(v1);
      V_.insert(v2);
      continue;
    }
    if (!F_contains(f)) {
      continue;
    }
    // point is outside cylinder and face is in patch
    F_.erase(f);
    E_.erase(e0);
    E_.erase(e1);
    E_.erase(e2);
    V_.erase(v0);
    V_.erase(v1);
    V_.erase(v2);
    for (int h : {h0, h1, h2}) {
      int ht = supermesh_->h_twin_h(h);
      int ft = supermesh_->f_left_h(ht);
      if (F_contains(ft)) {
        int v = supermesh_->v_origin_h(h);
        int vt = supermesh_->v_origin_h(ht);
        int e = supermesh_->e_undirected_h(h);
        V_.insert(v);
        V_.insert(vt);
        E_.insert(e);
        frontierF_.insert(ft);
      }
    }
  }
  dNf = F_.size() - num_faces;
  return dNf;
}

void Patch::expand_by_one_ring() {
  // printf("Patch::expand_by_one_ring\n");
  // find_bdryV();
  // find_frontierF();
  newF_.clear();
  newV_.clear();

  for (int f : frontierF_) {
    if (!F_contains(f)) {
      newF_.insert(f);
    }

    int h0 = supermesh_->h_right_f(f);
    int h1 = supermesh_->h_next_h(h0);
    int h2 = supermesh_->h_next_h(h1);
    int v0 = supermesh_->v_origin_h(h0);
    int v1 = supermesh_->v_origin_h(h1);
    int v2 = supermesh_->v_origin_h(h2);
    int e0 = supermesh_->e_undirected_h(h0);
    int e1 = supermesh_->e_undirected_h(h1);
    int e2 = supermesh_->e_undirected_h(h2);

    F_.insert(f);
    E_.insert(e0);
    E_.insert(e1);
    E_.insert(e2);
    // V_.insert(v0);
    // V_.insert(v1);
    // V_.insert(v2);
    if (!V_contains(v0)) {
      newV_.insert(v0);
      V_.insert(v0);
    }
    if (!V_contains(v1)) {
      newV_.insert(v1);
      V_.insert(v1);
    }
    if (!V_contains(v2)) {
      newV_.insert(v2);
      V_.insert(v2);
    }
    H_.insert(h0);
    H_.insert(h1);
    H_.insert(h2);
  }
  find_bdryV();
  find_frontierF();
}

//////////////////////////////////////////
// Utility functions /////////////////////
//////////////////////////////////////////

void Patch::clear() {
  // printf("Patch::clear\n");
  // uncolor_faces();
  V_.clear();
  E_.clear();
  F_.clear();
  H_.clear();
  bdryV_.clear();
  frontierF_.clear();
  frontierE_.clear();
  frontierV_.clear();
  newF_.clear();
  newV_.clear();
}

void Patch::find_bdryV() {
  // printf("Patch::find_bdryV\n");
  bdryV_.clear();

  for (int f : frontierF_) {
    for (int h : supermesh_->generate_H_right_f(f)) {
      if (some_positive_boundary_contains_h(h)) {
        bdryV_.insert(supermesh_->v_origin_h(h));
      }
    }
  }
  // printf("  #bdryV: %d\n", bdryV_.size());
}

void Patch::find_frontierF() {
  // printf("Patch::find_frontierF\n");
  frontierF_.clear();

  for (int v : bdryV_) {
    for (int f : supermesh_->generate_F_incident_v(v)) {
      frontierF_.insert(f);
    }
  }
  // printf("  #frontierF: %d\n", frontierF_.size());
}

//////////////////////////////////////////
// Unused ////////////////////////////////
//////////////////////////////////////////

Patch Patch::from_seed_to_cylinder(MatrixMesh *supermesh, int seed_vertex,
                                   Vec3d p0, Vec3d ez, double r_max) {
  Vec3d p = supermesh->xyz_coord_v(seed_vertex);
  if (!point_is_in_cylinder(p, p0, ez, r_max, 0,
                            std::numeric_limits<double>::max())) {
    throw std::invalid_argument("Seed vertex not in cylinder");
  }
  Patch s = from_seed_vertex(supermesh, seed_vertex);
  int dNf = 0, dNe = 0, dNv = 0;
  do {
    dNf = s.move_towards_cylinder(p0, ez, r_max);
    printf("dNf = %d\n", dNf);
  } while (dNf > 0);

  return s;
}

VertexFaceTuple Patch::get_vf_tuple() {
  int num_vertices = V_.size();
  int num_faces = F_.size();
  Samples3d xyz_coord_V(num_vertices, 3);
  Samples3i V_cycle_F(num_faces, 3);
  std::vector<int> V_indices;
  V_indices.reserve(num_vertices);
  std::unordered_map<int, int> V_map;
  int v = 0;
  for (int v_ : V_) {
    // int v = V_indices.size();
    // V_indices.push_back(v_);
    printf("adding to V_map: v_ = %d\n", v_);
    V_map[v_] = v;
    xyz_coord_V.row(v) = supermesh_->xyz_coord_v(v_);
    v++;
  }
  // std::vector<int> F_indices;
  // F_indices.reserve(num_faces);
  int f = 0;
  for (int f_ : F_) {
    // int f = F_indices.size();
    // F_indices.push_back(f_);
    int h0 = supermesh_->h_right_f(f);
    int h1 = supermesh_->h_next_h(h0);
    int h2 = supermesh_->h_next_h(h1);
    int v0_ = supermesh_->v_origin_h(h0);
    int v1_ = supermesh_->v_origin_h(h1);
    int v2_ = supermesh_->v_origin_h(h2);
    printf("v0_ = %d, v1_ = %d, v2_ = %d\n", v0_, v1_, v2_);
    if (!V_.contains(v0_) || !V_.contains(v1_) || !V_.contains(v2_)) {
      throw std::invalid_argument("Vertex not in patch");
    }
    int v0 = V_map.at(v0_);
    int v1 = V_map.at(v1_);
    int v2 = V_map.at(v2_);
    printf("v0 = %d, v1 = %d, v2 = %d\n", v0, v1, v2);

    V_cycle_F.row(f) << v0, v1, v2;
    f++;
  }
  return {xyz_coord_V, V_cycle_F};
}

} // namespace meshbrane
