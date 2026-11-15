
/**
 * @file matrix_mesh.cpp
 */
#include "meshbrane/matrix_mesh.hpp"
#include "meshbrane/combinatorics.hpp"
#include "meshbrane/heat_laplacian.hpp"
#include "meshbrane/mesh_builder.hpp"
#include "meshbrane/meshbrane_data_types.hpp"
#include "meshbrane/simple_generator.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath> // For M_PI and std::acos
#include <coroutine>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unordered_set> // std::unordered_set

namespace fs = std::filesystem;

namespace meshbrane {

// Some helper functions

bool VF_data_check_all_v_indices_in_range(Samples3d &xyz_coord_V,
                                          Samples3i &V_cycle_F) {
  auto [min_it, max_it] = std::minmax_element(
      V_cycle_F.data(), V_cycle_F.data() + V_cycle_F.size());

  int v_min = *min_it;
  int v_max = *max_it;

  return (v_min >= 0 || v_max < xyz_coord_V.size());
}

bool VF_data_check_no_isolated_vertices(Samples3d &xyz_coord_V,
                                        Samples3i &V_cycle_F) {
  size_t Nv = xyz_coord_V.rows();
  Samplesi seen_V = Samplesi::Zero(Nv);

  for (int f = 0; f < V_cycle_F.rows(); f++) {
    for (auto v : V_cycle_F.row(f)) {
      seen_V[v] = 1;
    }
  }
  for (auto seen : seen_V) {
    if (seen == 0) {
      return false;
    }
  }

  return true;
}

bool VF_data_check_face_v_indices_unique(Samples3d &xyz_coord_V,
                                         Samples3i &V_cycle_F) {
  for (int f = 0; f < V_cycle_F.rows(); f++) {
    int v0 = V_cycle_F(f, 0);
    int v1 = V_cycle_F(f, 1);
    int v2 = V_cycle_F(f, 2);
    if (v0 == v1 || v1 == v2 || v2 == v0) {
      return false;
    }
  }
  return true;
}

bool VF_data_check(Samples3d &xyz_coord_V, Samples3i &V_cycle_F) {
  int Nv = xyz_coord_V.rows();
  int Nf = V_cycle_F.rows();
  auto [min_it, max_it] = std::minmax_element(
      V_cycle_F.data(), V_cycle_F.data() + V_cycle_F.size());
  int v_min = *min_it;
  int v_max = *max_it;
  if (v_min < 0 || v_max >= Nv) {
    // v index in face out of range
    return false;
  }
  Samplesi seen_V = Samplesi::Zero(Nv);
  for (int f = 0; f < Nf; f++) {
    int v0 = V_cycle_F(f, 0);
    int v1 = V_cycle_F(f, 1);
    int v2 = V_cycle_F(f, 2);
    if (v0 == v1 || v1 == v2 || v2 == v0) {
      // v indices in face not unique
      return false;
    }
    seen_V[v0] = 1;
    seen_V[v1] = 1;
    seen_V[v2] = 1;
  }
  for (auto seen : seen_V) {
    if (seen == 0) {
      // isolated vertex
      return false;
    }
  }
  return true;
}

bool min_HE_data_check_faces_are_triangles(
    Samples3d &xyz_coord_V, Samplesi &h_out_V, Samplesi &h_right_F,
    Samplesi &v_origin_H, Samplesi &f_left_H, Samplesi &h_next_H,
    Samplesi &h_twin_H, Samplesi &h_negative_B) {

  for (int f = 0; f < h_right_F.size(); f++) {
    int h0 = h_right_F[f];
    int h1 = h_next_H[h0];
    int h2 = h_next_H[h2];
    if (h_next_H[h2] != h0) {
      return false;
    }
  }
  return true;
}

bool min_HE_data_check_twin_cycles_are_length_2(
    Samples3d &xyz_coord_V, Samplesi &h_out_V, Samplesi &h_right_F,
    Samplesi &v_origin_H, Samplesi &f_left_H, Samplesi &h_next_H,
    Samplesi &h_twin_H, Samplesi &h_negative_B) {

  for (int h = 0; h < h_twin_H.size(); h++) {
    int ht = h_twin_H[h];
    if (ht == h || h_twin_H[ht] != h) {
      return false;
    }
  }
  return true;
}

// bool min_HE_data_check(Samples3d &xyz_coord_V, Samplesi &h_out_V,
//                        Samplesi &h_right_F, Samplesi &v_origin_H,
//                        Samplesi &f_left_H, Samplesi &h_next_H,
//                        Samplesi &h_twin_H, Samplesi &h_negative_B) {
//
//   int Nv = xyz_coord_V.rows();
//   if (h_out_V.size() != Nv) {
//     return false;
//   }
//   int Nf = h_right_F.size();
//   int Nh = v_origin_H.size();
//   if (Nh % 2 != 0) {
//     return false;
//   }
//   if (f_left_H.size() != Nh || h_next_H.size() != Nh || h_twin_H.size() !=
//   Nh) {
//     return false;
//   }
//   int Ne = Nh / 2;
//   int Nb = h_negative_B.size();
//
//   // Check v index ranges
//   for (int v : v_origin_H) {
//     if (v < 0 || v >= Nv) {
//       return false;
//     }
//   }
//   // Check f/b index ranges
//   for (int f : f_left_H) {
//     if (f < -Nb || f >= Nf) {
//       return false;
//     }
//   }
//   // Check h indices in range
//   for (const auto *samples :
//        {&h_out_V, &h_right_F, &h_next_H, &h_twin_H, &h_negative_B}) {
//     for (int h : *samples) {
//       if (h < 0 || h >= Nh) {
//         return false;
//       }
//     }
//   }
//
//   // check h_out_V, v_origin_H compatibility
//   for (int v = 0; v < Nv; v++) {
//     int h = h_out_V[v];
//     if (v_origin_H[h] != v) {
//       return false;
//     }
//   }
//   // check h_right_F, f_left_H compatibility
//   for (int f = 0; f < Nf; f++) {
//     int h = h_right_F[f];
//     if (f_left_H[h] != f) {
//       return false;
//     }
//   }
//   // check h_negative_B, f_left_H compatibility
//   for (int b = 0; b < Nb; b++) {
//     int h = h_negative_B[b];
//     if (f_left_H[h] != -(b + 1)) {
//       return false;
//     }
//   }
//
//   for (int h0 = 0; h0 < Nh; h0++) {
//     int ht = h_twin_H[h0];
//     if (ht == h0 || h_twin_H[ht] != h0) {
//       // twin cycle not length 2
//       return false;
//     }
//     int h1 = h_next_H[h0];
//     int h2 = h_next_H[h1];
//     if (h_next_H[h2] != h0) {
//       // next cycle not length 2
//       return false;
//     }
//   }
//
//   return true;
// }

bool combinatorial_map_check(Samplesi &h_out_V, Samplesi &h_directed_E,
                             Samplesi &h_right_F, Samplesi &v_origin_H,
                             Samplesi &e_undirected_H, Samplesi &f_left_H,
                             Samplesi &h_next_H, Samplesi &h_twin_H,
                             Samplesi &h_negative_B) {

  int Nv = h_out_V.size();
  int Ne = h_directed_E.size();
  int Nf = h_right_F.size();
  int Nh = v_origin_H.size();
  int Nb = h_negative_B.size();
  if (Nh % 2 != 0) {
    return false;
  }
  if (e_undirected_H.size() != Nh || f_left_H.size() != Nh ||
      h_next_H.size() != Nh || h_twin_H.size() != Nh) {
    return false;
  }
  if (Ne != Nh / 2) {
    return false;
  }

  // Check v index ranges
  for (int v : v_origin_H) {
    if (v < 0 || v >= Nv) {
      return false;
    }
  }
  // Check e index ranges
  for (int e : e_undirected_H) {
    if (e < 0 || e >= Nv) {
      return false;
    }
  }
  // Check f/b index ranges
  for (int f : f_left_H) {
    if (f < -Nb || f >= Nf) {
      return false;
    }
  }
  // Check h indices in range
  for (const auto *samples : {&h_out_V, &h_directed_E, &h_right_F, &h_next_H,
                              &h_twin_H, &h_negative_B}) {
    for (int h : *samples) {
      if (h < 0 || h >= Nh) {
        return false;
      }
    }
  }

  // check h_out_V, v_origin_H compatibility
  for (int v = 0; v < Nv; v++) {
    if (v_origin_H[h_out_V[v]] != v) {
      return false;
    }
  }
  // check h_right_F, f_left_H compatibility
  for (int f = 0; f < Nf; f++) {
    if (f_left_H[h_right_F[f]] != f) {
      return false;
    }
  }
  // check h_negative_B, f_left_H compatibility
  for (int b = 0; b < Nb; b++) {
    if (f_left_H[h_negative_B[b]] != -(b + 1)) {
      return false;
    }
  }

  for (int h = 0; h < Nh; h++) {
    int ht = h_twin_H[h];
    if (v_origin_H[h] == v_origin_H[ht] ||
        e_undirected_H[h] != e_undirected_H[ht] ||
        f_left_H[h] == f_left_H[ht]) {
      // twin ~ (v,e,f)->(v',e,f')
      return false;
    }
    int hn = h_next_H[h];
    if (v_origin_H[h] == v_origin_H[hn] ||
        e_undirected_H[h] == e_undirected_H[hn] ||
        f_left_H[h] != f_left_H[hn]) {
      // next ~ (v,e,f)->(v',e',f)
      return false;
    }
  }

  return true;
}

//////////////////////
// Constructors //////
//////////////////////

MatrixMesh::MatrixMesh(const Samples3d &xyz_coord_V, const Samples3i &V_cycle_F)
    : xyz_coord_V_{xyz_coord_V}, V_cycle_F_{V_cycle_F},
      xyz_coord_V_was_set_{true}, V_cycle_F_was_set_{true} {}

// MatrixMesh::MatrixMesh(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
//                        const Samplesi &v_origin_H, const Samplesi &h_next_H,
//                        const Samplesi &h_twin_H, const Samplesi &f_left_H,
//                        const Samplesi &h_right_F, const Samplesi
//                        &h_negative_B)
//     : xyz_coord_V_(xyz_coord_V), h_out_V_(h_out_V), v_origin_H_(v_origin_H),
//       h_next_H_(h_next_H), h_twin_H_(h_twin_H), f_left_H_(f_left_H),
//       h_right_F_(h_right_F), h_negative_B_(h_negative_B) {
//   throw std::runtime_error(
//       "MatrixMesh::MatrixMesh(...stuff from he tuple...) ");
// }
MatrixMesh::MatrixMesh(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                       const Samplesi &h_right_F, const Samplesi &v_origin_H,
                       const Samplesi &f_left_H, const Samplesi &h_next_H,
                       const Samplesi &h_twin_H, const Samplesi &h_negative_B)
    : xyz_coord_V_(xyz_coord_V), h_out_V_(h_out_V), h_right_F_(h_right_F),
      f_left_H_(f_left_H), v_origin_H_(v_origin_H), h_next_H_(h_next_H),
      h_twin_H_(h_twin_H), h_negative_B_(h_negative_B),
      xyz_coord_V_was_set_{true}, h_out_V_was_set_{true},
      h_right_F_was_set_{true}, v_origin_H_was_set_{true},
      f_left_H_was_set_{true}, h_next_H_was_set_{true}, h_twin_H_was_set_{true},
      h_negative_B_was_set_{true} {}

MatrixMesh::MatrixMesh(const YAML::Node &parameters) {
  parameters_ = parameters;
  set_attributes_from_parameters();
  init_core_data_from_ply();
  init_cache_from_core_data();
}
///////////////////
// Operators //////
///////////////////

MatrixMesh &MatrixMesh::operator+=(const MatrixMesh &m) {

  if (this == &m) {
    MatrixMesh copy(m);
    return *this += copy;
  }

  int Nv_other = m.get_num_vertices();
  int Ne_other = m.get_num_edges();
  int Nf_other = m.get_num_faces();
  int Nb_other = m.get_num_boundaries();
  int Nh_other = m.get_num_half_edges();

  int Nv_this = get_num_vertices();
  int Ne_this = get_num_edges();
  int Nf_this = get_num_faces();
  int Nb_this = get_num_boundaries();
  int Nh_this = get_num_half_edges();

  int Nv = Nv_this + Nv_other;
  int Ne = Ne_this + Ne_other;
  int Nf = Nf_this + Nf_other;
  int Nb = Nb_this + Nb_other;
  int Nh = Nh_this + Nh_other;

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

  xyz_coord_V_.bottomRows(Nv_other) = m.xyz_coord_V_;
  h_out_V_.tail(Nv_other) = m.h_out_V_.array() + Nh_this;

  V_cycle_E_.bottomRows(Ne_other) = m.V_cycle_E_.array() + Nv_this;
  h_directed_E_.tail(Ne_other) = m.h_directed_E_.array() + Nh_this;

  V_cycle_F_.bottomRows(Nf_other) = m.V_cycle_F_.array() + Nv_this;
  h_right_F_.tail(Nf_other) = m.h_right_F_.array() + Nh_this;

  h_negative_B_.tail(Nb_other) = m.h_negative_B_.array() + Nh_this;

  v_origin_H_.tail(Nh_other) = m.v_origin_H_.array() + Nv_this;
  e_undirected_H_.tail(Nh_other) = m.e_undirected_H_.array() + Ne_this;
  Samplesi new_f_left_H = m.f_left_H_;
  for (int &f : new_f_left_H) {
    f += (f >= 0) ? Nf_this : -Nb_this;
  }
  f_left_H_.tail(Nh_other) = new_f_left_H;
  h_next_H_.tail(Nh_other) = m.h_next_H_.array() + Nh_this;
  h_twin_H_.tail(Nh_other) = m.h_twin_H_.array() + Nh_this;

  return *this;
}

MatrixMesh MatrixMesh::operator+(const MatrixMesh &m) const {
  MatrixMesh out = *this; // copy
  out += m;               // reuse checked in-place op
  return out;
}

//////////////////////
// Initialization ////
//////////////////////

void MatrixMesh::init_core_data_from_V_cycle_F() {
  if (!xyz_coord_V_was_set_ || !V_cycle_F_was_set_) {
    throw std::runtime_error(
        "MatrixMesh::init_core_data_from_V_cycle_F "
        "requires xyz_coord_V_was_set_ and V_cycle_F_was_set_");
  }
  if (!VF_data_check_all_v_indices_in_range(xyz_coord_V_, V_cycle_F_)) {
    throw std::out_of_range(
        "MatrixMesh::init_core_data_from_V_cycle_F found v out of range");
  }
  if (!VF_data_check_all_v_indices_in_range(xyz_coord_V_, V_cycle_F_)) {
    throw std::out_of_range(
        "MatrixMesh::init_core_data_from_V_cycle_F found isolated vertex");
  }
  // Complete initialization requires:
  //
  // Vertices V
  // - xyz_coord_V_[v]
  //   - for v in V (0<=v<Nv)
  // - h_out_V_[v]
  //   - for v in V (0<=v<Nv)
  // Edges E=E_interior+E_bdry
  // - V_cycle_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|)
  //   - for e in E_bdry (|E_interior|<=e<|E|)
  // - h_directed_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|)
  //   - for e in E_bdry (|E_interior|<=e<|E|)
  // Faces F
  // - V_cycle_F_[f]
  //   - for f in F (0<=f<Nf)
  // - h_right_F_[f]
  //   - for f in F (0<=f<Nf)
  // Half-edges H=H_interior+H_bdry_plus+H_bdry_minus
  // - v_origin_H_[h]
  //   - for h in H_interior+H_bdry_plus (0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - e_undirected_H_[h]
  //   - for h in H_interior+H_bdry_plus (0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - f_left_H_[h]
  //   - for h in H_interior+H_bdry_plus (0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - h_next_H_[h]
  //   - for h in H_interior+H_bdry_plus (0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - h_twin_H_[h]
  //   - for h in H_interior+H_bdry_plus (0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)

  size_t Nv = xyz_coord_V_.rows();
  size_t Nf = V_cycle_F_.rows();
  size_t Nh0 = 3 * Nf; // = 2*|E_interior|+|E_bdry|
  // First, we build a combinatorial map for the Nh0=3*|F| half-edges
  // whose orientations are compatible with triangular face boundaries using the
  // convention that boundary half-edges are fixed points of the twin map.
  // $\beta_1(h)=h_twin_H_[h]$  //(v,e,f)->(v',e,f')
  // $\beta_2(h)=h_next_H_[h]$ // (v,e,f)->(v',e',f)
  // $\sigma_0(v)=h_out_V_[v]$ //
  // $\sigma_1(e)=h_directed_E_[e]$
  // $\sigma_2(f)=h_right_F_[f]$ //

  h_out_V_.resize(Nv);
  V_cycle_E_.resize(Nh0, 2);
  h_directed_E_.resize(Nh0);
  // h_right_F_[f] = 3*f for 0<=f<Nf
  h_right_F_ = 3 * Samplesi::LinSpaced(Nf, 0, Nf - 1);

  v_origin_H_.resize(Nh0);
  e_undirected_H_.resize(Nh0);
  h_next_H_.resize(Nh0);
  // h_twin_H_[h] = h for 0<=h<Nh0
  h_twin_H_.setLinSpaced(Nh0, 0, Nh0 - 1);
  f_left_H_.resize(Nh0);

  std::unordered_map<uint64_t, int> h_directed_EdgeKey;
  h_directed_EdgeKey.reserve(Nh0 / 2);
  auto edge_key_vv = [](int v0, int v1) -> uint64_t {
    uint32_t a = std::min(v0, v1);
    uint32_t b = std::max(v0, v1);
    return (uint64_t(a) << 32) | b;
  };

  // This loop assigns ***
  //
  // Vertices V
  // - xyz_coord_V_[v]
  //   - for v in V (0<=v<Nv) -Done
  // - h_out_V_[v]
  //   - for v in V (0<=v<Nv) ***
  // Edges E=E_interior+E_bdry
  // - V_cycle_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|) ***
  //   - for e in E_bdry (|E_interior|<=e<|E|)
  // - h_directed_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|) ***
  //   - for e in E_bdry (|E_interior|<=e<|E|)
  // Faces F
  // - V_cycle_F_[f]
  //   - for f in F (0<=f<Nf) -Done
  // - h_right_F_[f]
  //   - for f in F (0<=f<Nf) -Done
  // Half-edges H=H_interior+H_bdry_plus+H_bdry_minus
  // - v_origin_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - e_undirected_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - f_left_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - h_next_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - h_twin_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0)
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // Boundaries B
  // - h_negative_B_[b]
  //   - for b in B (0<=b<Nb)
  //
  // This loop incements:
  // e=0,...,|E_interior|
  //
  // This loop collects:
  // h_directed_EdgeKey[key(v0,v1)] for all edges
  int e = 0;
  for (int h = 0; h < Nh0; h++) {
    // h = 3*f + i
    int f = h / 3;
    int i = h - 3 * f;
    // int i_next = (i + 1) % 3;
    int i_next = i + 1 - 3 * (i == 2);
    // h_next = h + 1 - 3 * (i == 2);
    int h_next = 3 * f + i_next;
    int v = V_cycle_F_(f, i);

    h_out_V_[v] = h; // will end up being the largest h with v at its origin

    v_origin_H_[h] = v;
    f_left_H_[h] = f;
    h_next_H_[h] = h_next;

    int v_next = V_cycle_F_(f, i_next);
    uint64_t edge_key = edge_key_vv(v, v_next);
    // collect first h through edge
    auto [it, inserted] = h_directed_EdgeKey.try_emplace(edge_key, h);

    if (!inserted) {
      // If not inserted that means we've already seen its twin
      // For a manifold, h_twin should still be a fixed point of h_twin_H_
      int h_twin = it->second;
      if (h_twin_H_[h_twin] != h_twin) {
        // This would mean h_twin was previously assigned a twin
        throw std::runtime_error("Non-manifold edge");
      }
      int f_twin = h_twin / 3;
      int i_twin = h_twin - 3 * f_twin;
      // int i_twin_next = (i_twin + 1) % 3;
      // int i_twin_next = i_twin + 1 - 3 * (i_twin == 2);
      // int v_twin = V_cycle_F_(f_twin, i_twin);
      // int v_twin_next = V_cycle_F_(f_twin, i_twin_next);
      if (V_cycle_F_(f_twin, i_twin) != v_next ||
          V_cycle_F_(f_twin, i_twin + 1 - 3 * (i_twin == 2)) != v) {
        throw std::runtime_error("Inconsistent face orientation");
      }
      h_twin_H_[h] = h_twin;
      h_twin_H_[h_twin] = h;

      e_undirected_H_[h] = e;
      e_undirected_H_[h_twin] = e;
      V_cycle_E_.row(e) << v_next, v; // orientation agrees with first h
      h_directed_E_[e] = h_twin;
      e++; // count interior edge
    }
  }
  // should have e=|E_interior|=|E|-|E_bdry|
  size_t Ne = h_directed_EdgeKey.size();
  size_t Nh = 2 * Ne;
  size_t Nh_bdry = Nh - Nh0; //

  V_cycle_E_.conservativeResize(Ne, 2); // need for E_bdry
  h_directed_E_.conservativeResize(Ne); // need for E_bdry

  v_origin_H_.conservativeResize(Nh);     // need for H_bdry_minus
  e_undirected_H_.conservativeResize(Nh); // need for H_bdry_plus+H_bdry_minus
  f_left_H_.conservativeResize(Nh);       // need for H_bdry_minus

  h_next_H_.conservativeResize(Nh); // need for H_bdry_minus
  h_twin_H_.conservativeResize(Nh); // need for H_bdry_plus+H_bdry_minus

  // This loop assigns ***
  //
  // Vertices V
  // - xyz_coord_V_[v]
  //   - for v in V (0<=v<Nv) -Done
  // - h_out_V_[v]
  //   - for v in V (0<=v<Nv) -Done
  // Edges E=E_interior+E_bdry
  // - V_cycle_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|) -Done
  //   - for e in E_bdry (|E_interior|<=e<|E|) ***
  // - h_directed_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|) -Done
  //   - for e in E_bdry (|E_interior|<=e<|E|) ***
  // Faces F
  // - V_cycle_F_[f]
  //   - for f in F (0<=f<Nf) -Done
  // - h_right_F_[f]
  //   - for f in F (0<=f<Nf) -Done
  // Half-edges H=H_interior+H_bdry_plus+H_bdry_minus
  // - v_origin_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) ***
  // - e_undirected_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) ***
  // - f_left_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - h_next_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh)
  // - h_twin_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) ***
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) ***
  // Boundaries B
  // - h_negative_B_[b]
  //   - for b in B (0<=b<Nb)
  int h_twin = Nh0;
  for (int h = 0; h < Nh0; h++) {
    if (h_twin_H_[h] != h) {
      // if h in H_interior
      continue;
    }
    // otherwise h is in H_bdry_plus
    // h_twin is in H_bdry_minus
    // e is in E_bdry
    // int v = v_origin_H_[h];
    int v_twin = v_origin_H_[h_next_H_[h]];
    V_cycle_E_.row(e) << v_origin_H_[h], v_twin;
    h_directed_E_[e] = h;
    e_undirected_H_[h] = e;
    e_undirected_H_[h_twin] = e;
    v_origin_H_[h_twin] = v_twin;
    h_twin_H_[h] = h_twin;
    h_twin_H_[h_twin] = h;
    e++;      // count e in E_bdry
    h_twin++; // count h_twin in H_bdry_minus
  }

  // This loop assigns ***
  //
  // Vertices V
  // - xyz_coord_V_[v]
  //   - for v in V (0<=v<Nv) -Done
  // - h_out_V_[v]
  //   - for v in V (0<=v<Nv) -Done
  // Edges E=E_interior+E_bdry
  // - V_cycle_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|) -Done
  //   - for e in E_bdry (|E_interior|<=e<|E|) -Done
  // - h_directed_E_[e]
  //   - for e in E_interior (0<=e<|E_interior|) -Done
  //   - for e in E_bdry (|E_interior|<=e<|E|) -Done
  // Faces F
  // - V_cycle_F_[f]
  //   - for f in F (0<=f<Nf) -Done
  // - h_right_F_[f]
  //   - for f in F (0<=f<Nf) -Done
  // Half-edges H=H_interior+H_bdry_plus+H_bdry_minus
  // - v_origin_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) -Done
  // - e_undirected_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) -Done
  // - f_left_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) ***
  // - h_next_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) ***
  // - h_twin_H_[h]
  //   - for h in H_interior (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_plus (subset of 0<=h<=Nh0) -Done
  //   - for h in H_bdry_minus (Nh0<=h<=Nh) -Done
  // Boundaries B
  // - h_negative_B_[b]
  //   - for b in B (0<=b<Nb) ***
  int b = 0;
  Eigen::VectorXi H_bdry_minus_seen = Eigen::VectorXi::Zero(Nh_bdry);
  for (int i_h = 0; i_h < Nh_bdry; i_h++) {
    if (H_bdry_minus_seen[i_h]) {
      continue;
    }
    int h_negative_b = i_h + Nh0;
    h_negative_B_.conservativeResize(b + 1);
    h_negative_B_[b] = h_negative_b;
    int h = h_negative_b;
    // follow prev cycle along boundary b until we get back to h=h_negative_b
    do {
      H_bdry_minus_seen[h - Nh0] = 1;
      f_left_H_[h] = -(b + 1);
      int h_prev = h_twin_H_[h];
      // rot cw through h's pointing in to v_origin_h
      // until we find h_prev in boundary b
      do {
        h_prev = h_twin_H_[h_next_H_[h_prev]];
      } while (h_prev < Nh0);
      h_next_H_[h_prev] = h;
      h = h_prev;
    } while (h != h_negative_b);
    b++;
  }
  h_out_V_was_set_ = true;
  V_cycle_E_was_set_ = true;
  h_directed_E_was_set_ = true;
  h_right_F_was_set_ = true;
  h_negative_B_was_set_ = true;
  v_origin_H_was_set_ = true;
  e_undirected_H_was_set_ = true;
  f_left_H_was_set_ = true;
  h_next_H_was_set_ = true;
  h_twin_H_was_set_ = true;
}

void MatrixMesh::init_core_data_from_minimal_half_edge_data() {
  if (!xyz_coord_V_was_set_ || !h_out_V_was_set_ || !h_right_F_was_set_ ||
      !v_origin_H_was_set_ || !f_left_H_was_set_ || !h_next_H_was_set_ ||
      !h_twin_H_was_set_ || !h_negative_B_was_set_) {
    throw std::runtime_error(
        "MatrixMesh::init_core_data_from_minimal_half_edge_data "
        "requires xyz_coord_V_was_set_, h_out_V_was_set_, h_right_F_was_set_, "
        "v_origin_H_was_set_, f_left_H_was_set_, h_next_H_was_set_, "
        "h_twin_H_was_set_, and h_negative_B_was_set_");
  }
  int Nh = v_origin_H_.size();
  int Ne = Nh / 2;
  int Nf = h_right_F_.size();

  V_cycle_E_.resize(Ne, 2);
  h_directed_E_.resize(Ne);
  V_cycle_F_.resize(Nf, 3);
  e_undirected_H_.resize(Nh);

  for (int f = 0; f < Nf; f++) {
    int h0 = h_right_F_[f];
    int h1 = h_next_H_[h0];
    V_cycle_F_.row(f) << v_origin_H_[h0], v_origin_H_[h1],
        v_origin_H_[h_next_H_[h1]];
  }

  int e = 0;
  for (int h = 0; h < Nh; h++) {
    int ht = h_twin_H_[h];
    if (h > ht) {
      continue;
    }
    V_cycle_E_.row(e) << v_origin_H_[h], v_origin_H_[ht];
    h_directed_E_[e] = h;
    e_undirected_H_[h] = e;
    e_undirected_H_[ht] = e;
    e++;
  }
  assert(e == Ne);
  V_cycle_E_was_set_ = true;
  h_directed_E_was_set_ = true;
  V_cycle_F_was_set_ = true;
  e_undirected_H_was_set_ = true;
}

void MatrixMesh::init_cache_from_core_data() {
  update_boundary_cycles();
  update_mesh_geometric_data_E();
  update_mesh_geometric_data_F();
  update_mesh_geometric_data_V();
  update_mesh_volume();
  set_visual_defaults();
  update_mesh_visuals();
  init_quad_points_and_weights();
}

void MatrixMesh::init_icosohedron() {
  printf("MatrixMesh::init_icosohedron\n");
  double phi = (1.0 + sqrt(5.0)) * 0.5; // golden ratio
  double a = 1.0;
  double b = 1.0 / phi;

  int num_vertices = 12;
  int num_faces = 20;
  // Samples3d xyz_coord_V(num_vertices, 3);
  // Samples3i V_cycle_F(num_faces, 3);
  xyz_coord_V_.resize(num_vertices, 3);
  V_cycle_F_.resize(num_faces, 3);
  xyz_coord_V_.row(0) << 0.0, b, -a;
  xyz_coord_V_.row(1) << b, a, 0.0;
  xyz_coord_V_.row(2) << -b, a, 0.0;
  xyz_coord_V_.row(3) << 0.0, b, a;
  xyz_coord_V_.row(4) << 0.0, -b, a;
  xyz_coord_V_.row(5) << -a, 0.0, b;
  xyz_coord_V_.row(6) << 0.0, -b, -a;
  xyz_coord_V_.row(7) << a, 0.0, -b;
  xyz_coord_V_.row(8) << a, 0.0, b;
  xyz_coord_V_.row(9) << -a, 0.0, -b;
  xyz_coord_V_.row(10) << b, -a, 0.0;
  xyz_coord_V_.row(11) << -b, -a, 0.0;

  double rad = std::sqrt(a * a + b * b);
  xyz_coord_V_ /= rad;

  V_cycle_F_.row(0) << 2, 1, 0;
  V_cycle_F_.row(1) << 1, 2, 3;
  V_cycle_F_.row(2) << 5, 4, 3;
  V_cycle_F_.row(3) << 4, 8, 3;
  V_cycle_F_.row(4) << 7, 6, 0;
  V_cycle_F_.row(5) << 6, 9, 0;
  V_cycle_F_.row(6) << 11, 10, 4;
  V_cycle_F_.row(7) << 10, 11, 6;
  V_cycle_F_.row(8) << 9, 5, 2;
  V_cycle_F_.row(9) << 5, 9, 11;
  V_cycle_F_.row(10) << 8, 7, 1;
  V_cycle_F_.row(11) << 7, 8, 10;
  V_cycle_F_.row(12) << 2, 5, 3;
  V_cycle_F_.row(13) << 8, 1, 3;
  V_cycle_F_.row(14) << 9, 2, 0;
  V_cycle_F_.row(15) << 1, 7, 0;
  V_cycle_F_.row(16) << 11, 9, 6;
  V_cycle_F_.row(17) << 7, 10, 6;
  V_cycle_F_.row(18) << 5, 11, 4;
  V_cycle_F_.row(19) << 10, 8, 4;

  xyz_coord_V_was_set_ = true;
  V_cycle_F_was_set_ = true;

  init_core_data_from_V_cycle_F();
}

void MatrixMesh::init_icososphere(int num_refinements) {
  init_icosohedron();
  for (int i = 0; i < num_refinements; i++) {
    refine_icososphere();
  }
}

void MatrixMesh::init_core_data_from_ply() {

  if (ply_path_.empty()) {
    printf("MatrixMesh::init_core_data_from_ply: ply_path_ is empty");
    return;
  }
  HalfEdgeTuple het = mesh_io::load_he_samples_from_ply(ply_path_);
  xyz_coord_V_ = std::get<0>(het);
  h_out_V_ = std::get<1>(het);
  v_origin_H_ = std::get<2>(het);
  h_next_H_ = std::get<3>(het);
  h_twin_H_ = std::get<4>(het);
  f_left_H_ = std::get<5>(het);
  h_right_F_ = std::get<6>(het);
  h_negative_B_ = std::get<7>(het);

  xyz_coord_V_was_set_ = true;
  h_out_V_was_set_ = true;
  v_origin_H_was_set_ = true;
  h_next_H_was_set_ = true;
  h_twin_H_was_set_ = true;
  f_left_H_was_set_ = true;
  h_right_F_was_set_ = true;
  h_negative_B_was_set_ = true;
  init_core_data_from_minimal_half_edge_data();
}

void MatrixMesh::set_attributes_from_parameters() {
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

void MatrixMesh::copy_mesh_data(MatrixMesh &m) {
  xyz_coord_V_ = m.xyz_coord_V_;
  h_out_V_ = m.h_out_V_;
  V_cycle_E_ = m.V_cycle_E_;
  h_directed_E_ = m.h_directed_E_;
  V_cycle_F_ = m.V_cycle_F_;
  h_right_F_ = m.h_right_F_;
  h_negative_B_ = m.h_negative_B_;
  v_origin_H_ = m.v_origin_H_;
  e_undirected_H_ = m.e_undirected_H_;
  f_left_H_ = m.f_left_H_;
  h_next_H_ = m.h_next_H_;
  h_twin_H_ = m.h_twin_H_;

  xyz_coord_V_was_set_ = true;
  h_out_V_was_set_ = true;
  V_cycle_E_was_set_ = true;
  h_directed_E_was_set_ = true;
  V_cycle_F_was_set_ = true;
  h_right_F_was_set_ = true;
  h_negative_B_was_set_ = true;
  v_origin_H_was_set_ = true;
  e_undirected_H_was_set_ = true;
  f_left_H_was_set_ = true;
  h_next_H_was_set_ = true;
  h_twin_H_was_set_ = true;
}

void MatrixMesh::copy_mesh_cache(MatrixMesh &m) {
  V_cycle_B_ = m.V_cycle_B_;
  vec_H_ = m.vec_H_;
  length_E_ = m.length_E_;
  area_F_ = m.area_F_;
  normal_F_ = m.normal_F_;
  normal_V_ = m.normal_V_;
  area_V_ = m.area_V_;
  average_edge_length_ = m.average_edge_length_;
  average_face_area_ = m.average_face_area_;
  total_volume_ = m.total_volume_;
  mcvec_V_ = m.mcvec_V_;
  mean_curvature_V_ = m.mean_curvature_V_;
  gaussian_curvature_V_ = m.gaussian_curvature_V_;
  lap_mean_curvature_V_ = m.lap_mean_curvature_V_;
  laplacian_matrix_V_ = m.laplacian_matrix_V_;
  radius_V = m.radius_V;
  rgba_V = m.rgba_V;
  rgba_H = m.rgba_H;
  rgba_F = m.rgba_F;
  rgba_E_ = m.rgba_E_;
  shifted_half_edge_arrows_ = m.shifted_half_edge_arrows_;
  vector_field_arrows_ = m.vector_field_arrows_;
}

///////////////////////////////////////////
// Static methods for special surfaces ////
///////////////////////////////////////////

// xyz_coord_V_was_set_ = true;
// h_out_V_was_set_ = true;
// v_origin_H_was_set_ = true;
// h_next_H_was_set_ = true;
// h_twin_H_was_set_ = true;
// f_left_H_was_set_ = true;
// h_right_F_was_set_ = true;
// h_negative_B_was_set_ = true;
//
// xyz_coord_V_was_set_ = true;
// V_cycle_F_was_set_ = true;

MatrixMesh MatrixMesh::icosohedron() {
  MatrixMesh m;
  m.init_icosohedron();
  return m;
}

void MatrixMesh::refine_icososphere() {
  // printf("MatrixMesh::refine_icososphere\n");

  auto vertex_pair_key = [](int v0, int v1) -> int64_t {
    return std::min(v0, v1) * 1000000 + std::max(v0, v1);
  };

  int num_faces_pre = get_num_faces();
  // printf("  num_faces_pre=%d\n", num_faces_pre);
  int num_faces = 4 * num_faces_pre;
  int num_vertices_pre = get_num_vertices();

  // Samples3d xyz_coord_V = xyz_coord_V_;
  Samples3i V_cycle_F(num_faces, 3);

  std::vector<Vec3d> vecV;
  for (int v = 0; v < num_vertices_pre; v++) {
    vecV.push_back(xyz_coord_v(v));
  }
  // std::vector<Eigen::Vector3i> vecF;

  std::unordered_map<int64_t, int> v_midpt_vv;
  int v_count = num_vertices_pre;
  int f_count = 0;
  for (int f = 0; f < num_faces_pre; f++) {
    // printf("  f=%d\n", f);
    int h0 = h_right_f(f);
    int h1 = h_next_h(h0);
    int h2 = h_next_h(h1);
    int v0 = v_origin_h(h0);
    int v1 = v_origin_h(h1);
    int v2 = v_origin_h(h2);
    // printf("  v0,v1,v2=%d,%d,%d\n", v0, v1, v2);
    int64_t key01 = vertex_pair_key(v0, v1);
    int64_t key12 = vertex_pair_key(v1, v2);
    int64_t key20 = vertex_pair_key(v2, v0);
    // std::cout << "  key01, key12, key20=" << key01 << "," << key12 << ","
    //           << key20 << std::endl;

    int v01 = (v_midpt_vv.count(key01) == 1) ? v_midpt_vv[key01] : -1;
    int v12 = (v_midpt_vv.count(key12) == 1) ? v_midpt_vv[key12] : -1;
    int v20 = (v_midpt_vv.count(key20) == 1) ? v_midpt_vv[key20] : -1;
    // printf("  v01,v12,v20=%d,%d,%d\n", v01, v12, v20);
    if (v01 == -1) {
      v01 = v_count++;

      Vec3d xyz01 = (vecV[v0] + vecV[v1]) / 2.0;
      xyz01 *= 1 / xyz01.norm();
      // vecV[v01] = xyz01;
      vecV.push_back(xyz01);
      v_midpt_vv[key01] = v01;
    }
    if (v12 == -1) {
      v12 = v_count++;
      Vec3d xyz12 = (vecV[v1] + vecV[v2]) / 2.0;
      xyz12 *= 1 / xyz12.norm();
      // vecV[v12] = xyz12;
      vecV.push_back(xyz12);
      v_midpt_vv[key12] = v12;
    }
    if (v20 == -1) {
      v20 = v_count++;
      Vec3d xyz20 = (vecV[v2] + vecV[v0]) / 2.0;
      xyz20 *= 1 / xyz20.norm();
      // vecV[v20] = xyz20;
      vecV.push_back(xyz20);
      v_midpt_vv[key20] = v20;
    }

    // printf("  v01,v12,v20=%d,%d,%d\n", v01, v12, v20);

    V_cycle_F.row(f_count++) << v0, v01, v20;
    V_cycle_F.row(f_count++) << v01, v1, v12;
    V_cycle_F.row(f_count++) << v20, v12, v2;
    V_cycle_F.row(f_count++) << v01, v12, v20;
  }

  xyz_coord_V_.conservativeResize(vecV.size(), 3);
  for (int v = 0; v < vecV.size(); v++) {
    vecV[v] /= vecV[v].norm();
    xyz_coord_V_.row(v) = vecV[v];
  }

  V_cycle_F_ = V_cycle_F;
  init_core_data_from_V_cycle_F();
}

MatrixMesh MatrixMesh::icososphere(int num_refinements) {
  MatrixMesh m;
  m.init_icososphere(num_refinements);
  return m;
}

MatrixMesh MatrixMesh::tube_between_loops(Samples3d xyz_loop0,
                                          Samples3d xyz_loop1, int Nz) {
  int Nphi = xyz_loop0.rows();
  if (Nphi != xyz_loop1.rows()) {
    throw std::runtime_error("MatrixMesh::tube_between_loops requires loops "
                             "with the same number of vertices");
  }
  if (Nphi < 3) {
    throw std::runtime_error(
        "MatrixMesh::tube_between_loops requires loops with >= 3 vertices");
  }
  if (Nz < 2) {
    throw std::runtime_error("MatrixMesh::tube_between_loops requires Nz >= 2");
  }

  int Nv = Nphi * Nz;
  int Nf = 2 * Nphi * (Nz - 1);
  Samples3d xyz_coord_V(Nv, 3);
  Samples3i V_cycle_F(Nf, 3);
  Samples3d Delta_xyz = (xyz_loop1 - xyz_loop0) / (Nz - 1);
  for (int nz = 0; nz < Nz; nz++) {
    for (int nphi = 0; nphi < Nphi; nphi++) {
      int v = nphi + nz * Nphi;
      xyz_coord_V.row(v) = xyz_loop0.row(nphi) + nz * Delta_xyz.row(nphi);
    }
  }

  for (int nz = 0; nz < Nz - 1; nz++) {
    int nz0 = nz;
    int nz1 = nz;
    int nz2 = nz + 1;
    int nz3 = nz + 1;
    for (int nphi = 0; nphi < Nphi; nphi++) {
      int nphi0 = nphi;
      int nphi1 = (nphi + 1) % Nphi;
      int nphi2 = (nphi + 1) % Nphi;
      int nphi3 = nphi;
      int v0 = nphi0 + nz0 * Nphi;
      int v1 = nphi1 + nz1 * Nphi;
      int v2 = nphi2 + nz2 * Nphi;
      int v3 = nphi3 + nz3 * Nphi;
      int f0 = 2 * v0;
      int f1 = 2 * v0 + 1;
      V_cycle_F.row(f0) << v0, v1, v2;
      V_cycle_F.row(f1) << v2, v3, v0;
    }
  }

  MatrixMesh m(xyz_coord_V, V_cycle_F);
  return m;
}

MatrixMesh MatrixMesh::spherical_shell(int num_refinements, double h) {
  MatrixMesh m = MatrixMesh::icosohedron();

  for (int i = 0; i < num_refinements; i++) {
    m.refine_icososphere();
  }

  MatrixMesh m_inner = m;
  m_inner.xyz_coord_V_ *= (1.0 - h);
  m_inner.reverse_orientation();
  m += m_inner;
  return m;
}

////////////////////////////
// Debugging and checks ////
////////////////////////////

void MatrixMesh::check_v(int v) const {
  if (v < 0 || v >= get_num_vertices()) {
    throw std::out_of_range("v index " + std::to_string(v) + " out of range");
  }
}
void MatrixMesh::check_e(int e) const {
  if (e < 0 || e >= get_num_edges()) {
    throw std::out_of_range("e index " + std::to_string(e) + " out of range");
  }
}
void MatrixMesh::check_b(int b) const {
  if (b < 0 || b >= get_num_boundaries()) {
    throw std::out_of_range("b index " + std::to_string(b) + " out of range");
  }
}
void MatrixMesh::check_f(int f) const {
  if (f < 0 || f >= get_num_faces()) {
    throw std::out_of_range("f index " + std::to_string(f) + " out of range");
  }
}
void MatrixMesh::check_h(int h) const {
  if (h < 0 || h >= get_num_half_edges()) {
    throw std::out_of_range("h index " + std::to_string(h) + " out of range");
  }
}

void MatrixMesh::check_he_matrices() const {
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
    if (v_origin_H_[h_out] != v) {
      std::cout << "v=" << v << '\n';
      std::cout << "h_out=" << h_out << '\n';
      throw std::runtime_error("v_origin_H_[h_out] != v");
    }
  }
  for (int e = 0; e < Ne; e++) {
    int h_directed = h_directed_E_[e];
    if (h_directed < 0 || h_directed >= Nh) {
      std::cout << "e=" << e << '\n';
      std::cout << "h_directed=" << h_directed << '\n';
      throw std::out_of_range("h_directed_E_[e]: Half-edge index out of range");
    }
    if (e_undirected_H_[h_directed] != e) {
      std::cout << "e=" << e << '\n';
      std::cout << "h_directed=" << h_directed << '\n';
      throw std::runtime_error("e_undirected_H_[h_directed] != e");
    }
  }
  for (int f = 0; f < Nf; f++) {
    int h_right = h_right_F_[f];
    if (h_right < 0 || h_right >= Nh) {
      std::cout << "f=" << f << '\n';
      std::cout << "h_right=" << h_right << '\n';
      throw std::out_of_range("h_right_F_[f]: Half-edge index out of range");
    }
    if (f_left_H_[h_right] != f) {
      std::cout << "f=" << f << '\n';
      std::cout << "h_right=" << h_right << '\n';
      throw std::runtime_error("f_left_H_[h_right] != f");
    }
  }
  for (int b = 0; b < Nb; b++) {
    int h_negative = h_negative_B_[b];
    if (h_negative < 0 || h_negative >= Nh) {
      std::cout << "b=" << b << '\n';
      std::cout << "h_negative=" << h_negative << '\n';
      throw std::out_of_range("h_negative_B_[b]: Half-edge index out of range");
    }
    if (f_left_H_[h_negative] != -(b + 1)) {
      std::cout << "b=" << b << '\n';
      std::cout << "h_negative=" << h_negative << '\n';
      throw std::runtime_error("f_left_H_[h_negative] != -(b+1)");
    }
  }
  for (int h = 0; h < Nh; h++) {
    // check index ranges
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
    // check compatibility
    int h_prev = h_prev_h(h);
    if (h_next_H_[h_prev] != h) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_prev=" << h_prev << '\n';
      throw std::runtime_error("h_next_H_[h_prev] != h");
    }
    if (h_prev_h(h_next) != h) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_next=" << h_next << '\n';
      std::cout << "h_prev_h(h_next)=" << h_prev_h(h_next) << '\n';
      throw std::runtime_error("h_prev_h(h_next) != h");
    }
    if (h_twin_H_[h_twin] != h) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_twin=" << h_twin << '\n';
      std::cout << "h_twin_H_[h_twin]=" << h_twin_H_[h_twin] << '\n';
      throw std::runtime_error("h_twin_H_[h_twin] != h");
    }

    if (f_left_H_[h_next] != f_left_H_[h]) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_next=" << h_next << '\n';
      std::cout << "f_left_H_[h]=" << f_left_H_[h] << '\n';
      std::cout << "f_left_H_[h_next]=" << f_left_H_[h_next] << '\n';
      throw std::runtime_error("f_left_H_[h_next] != f_left_H_[h]");
    }
    if (f_left_H_[h_prev] != f_left_H_[h]) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_prev=" << h_prev << '\n';
      throw std::runtime_error("f_left_H_[h_prev] != f_left_H_[h]");
    }

    if (e_undirected_H_[h_twin] != e_undirected_H_[h]) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_twin=" << h_twin << '\n';
      throw std::runtime_error("e_undirected_H_[h_twin] != e_undirected_H_[h]");
    }

    if (h_twin == h) {
      std::cout << "h=h_twin=" << h << '\n';
      throw std::runtime_error("h_twin == h");
    }
    if (h_next == h) {
      std::cout << "h=h_next=" << h << '\n';
      throw std::runtime_error("h_next == h");
    }
    if (v_origin_H_[h_next] == v_origin_H_[h]) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_next=" << h_next << '\n';
      std::cout << "v_origin_H_[h]=v_origin_H_[h_next]=" << v_origin_H_[h]
                << '\n';
      throw std::runtime_error("v_origin_H_[h_next] == v_origin_H_[h]");
    }
    if (v_origin_H_[h_twin] == v_origin_H_[h]) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_twin=" << h_twin << '\n';
      std::cout << "v_origin_H_[h]=v_origin_H_[h_twin]=" << v_origin_H_[h]
                << '\n';
      throw std::runtime_error("v_origin_H_[h_twin] == v_origin_H_[h]");
    }
    if (f_left_H_[h_twin] == f_left_H_[h]) {
      std::cout << "h=" << h << '\n';
      std::cout << "h_twin=" << h_twin << '\n';
      std::cout << "f_left_H_[h]=f_left_H_[h_twin]=" << f_left_H_[h] << '\n';
      throw std::runtime_error("f_left_H_[h_twin] == f_left_H_[h]");
    }
  }
}

void MatrixMesh::print_he_mats() {
  std::cout << "h_out_V = " << h_out_V_.transpose() << '\n';

  std::cout << "h_directed_E = " << h_directed_E_.transpose() << '\n';

  std::cout << "h_right_F = " << h_right_F_.transpose() << '\n';

  std::cout << "h_negative_B = " << h_negative_B_.transpose() << '\n';

  std::cout << "v_origin_H = " << v_origin_H_.transpose() << '\n';
  std::cout << "e_undirected_H = " << e_undirected_H_.transpose() << '\n';
  std::cout << "f_left_H = " << f_left_H_.transpose() << '\n';
  std::cout << "h_next_H = " << h_next_H_.transpose() << '\n';
  std::cout << "h_twin_H = " << h_twin_H_.transpose() << '\n';
}

void MatrixMesh::print_simplex_mats() {
  // std::cout << "V_cycle_E = " << V_cycle_E_ << '\n';

  std::cout << "V_cycle_F = " << V_cycle_F_ << '\n';
}

//////////////////////
// Cache updaters ////
//////////////////////

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

//////////////////////
// Cache updaters ////
//////////////////////

void MatrixMesh::update_boundary_cycles() {
  int num_boundaries = get_num_boundaries();
  V_cycle_B_.resize(num_boundaries);
  for (int b = 0; b < num_boundaries; b++) {
    std::vector<int> Vnegative;
    for (auto h : generate_H_next_h(h_negative_b(b))) {
      Vnegative.push_back(v_origin_h(h));
    }
    int Nv = Vnegative.size();
    V_cycle_B_[b].resize(Nv);
    for (int _i = 0; _i < Nv; _i++) {
      int i = Nv - 1 - _i;
      V_cycle_B_[b][i] = Vnegative[_i];
    }
  }
}

void MatrixMesh::update_mesh_geometric_data_E() {
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

void MatrixMesh::update_mesh_geometric_data_F() {
  size_t num_faces = get_num_faces();
  // area_F_.setZero();
  // normal_F_.setZero();
  area_F_.resize(num_faces);
  normal_F_.resize(num_faces, 3);
  average_face_area_ = 0.0;
  for (int f = 0; f < num_faces; f++) {
    area_F_(f) = heron_area_f(f);
    normal_F_.row(f) = normal_f(f);
    average_face_area_ += area_F_(f);
  }
  average_face_area_ /= num_faces;
}

void MatrixMesh::update_mesh_geometric_data_V() {
  int num_vertices = get_num_vertices();
  // area_V_.setZero();
  // normal_V_.setZero();
  area_V_.resize(num_vertices);
  normal_V_.resize(num_vertices, 3);
  for (int v = 0; v < num_vertices; v++) {
    area_V_(v) = 0;
    normal_V_.row(v) = Eigen::Vector3d::Zero();
    int norm_n = 0;
    for (auto f : generate_F_incident_v(v)) {
      // printf("v,f=%d,%d\n", v, f);
      area_V_(v) += area_F_[f] / 3;
      normal_V_.row(v) += area_F_[f] * normal_F_.row(f);
    }
    normal_V_.row(v) /= math::L2norm(normal_V_.row(v));
  }
}

void MatrixMesh::update_mesh_volume() {
  total_volume_ = 0.0;
  for (int f = 0; f < get_num_faces(); f++) {
    total_volume_ += signed_volume_f(f);
  }
  total_volume_ = std::abs(total_volume_);
}

void MatrixMesh::update_mesh_geometric_data() {
  update_mesh_geometric_data_E();
  update_mesh_geometric_data_F();
  update_mesh_geometric_data_V();
  update_mesh_volume();
}

/////////////////////
// Need to check ////
/////////////////////

/////////////////////////
// Convergence testing //
/////////////////////////

void MatrixMesh::refine_messy_icososphere() {
  divide_faces_barycentric();
  int num_vertices = xyz_coord_V_.rows();
  for (int v{0}; v < num_vertices; v++) {
    Vec3d xyz = xyz_coord_V_.row(v);
    double normxyz = xyz.norm();
    // xyz = xyz / normxyz;
    xyz_coord_V_.row(v) = xyz / normxyz;
  }
  flip_non_delaunay();
}

void MatrixMesh::update_vector_field_arrows(Samples3d X, Samples3d U) {
  Eigen::Vector3d rgb = vector_field_arrows_.rgb_;
  double scale = 1.0;
  vector_field_arrows_.update(X, U, scale, rgb);
}
/////////////////////////////////////
// Misc precomputed geometric data //
/////////////////////////////////////

double MatrixMesh::signed_volume_f(int f) const {
  // auto x0 = xyz_coord_v(V_cycle_F_(f, 0));
  // auto x1 = xyz_coord_v(V_cycle_F_(f, 1));
  // auto x2 = xyz_coord_v(V_cycle_F_(f, 2));
  // Eigen::Vector3d n01, n12, n20;
  // math::cross_inplace(x0, x1, n01);
  // math::cross_inplace(x1, x2, n12);
  // math::cross_inplace(x2, x0, n20);
  // Eigen::Vector3d n = n01 + n12 + n20;
  // double val;
  // math::dot_inplace(n, x0, val);
  // return val / 6.0;
  int h1 = h_right_f(f);
  int h2 = h_next_h(h1);
  int h3 = h_next_h(h2);
  Vec3d u = xyz_coord_v(v_origin_h(h1));
  Vec3d v = xyz_coord_v(v_origin_h(h2));
  Vec3d w = xyz_coord_v(v_origin_h(h3));
  return (u[1] * v[2] * w[0] - u[2] * v[1] * w[0] + u[2] * v[0] * w[1] -
          u[0] * v[2] * w[1] + u[0] * v[1] * w[2] - u[1] * v[0] * w[2]) /
         6.0;
}

///////////////////////////////////////////////////////
// Constructors and Mesh I/O //////////////////////////
///////////////////////////////////////////////////////

// /**
//  * @brief Construct a new MatrixMesh object from a half-edge ply.
//  *
//  * @param ply_path
//  * @return MatrixMesh
//  */
// MatrixMesh MatrixMesh::from_he_ply(const fs::path &ply_path) {
//   mesh_io::MeshBuilder mc = mesh_io::MeshBuilder::from_he_ply(ply_path,
//   false); auto [xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
//   f_left_H,
//         h_right_F, h_negative_B] = mc.he_samples;
//   return MatrixMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
//                     f_left_H, h_right_F, h_negative_B);
// }

/**
 * @brief Save the mesh to a half-edge ply file.
 *
 * @param ply_path
 */
void MatrixMesh::write_he_ply(const fs::path &ply_path) const {
  // printf("Saving ply file %s\n", ply_path.c_str());
  meshbrane::mesh_io::write_he_samples_to_ply(
      xyz_coord_V_, h_out_V_, v_origin_H_, h_next_H_, h_twin_H_, f_left_H_,
      h_right_F_, h_negative_B_, ply_path);
}

// /**
//  * @brief Construct a new MatrixMesh object from a vertex-face samples. See
//  * `meshbrane::VertexFaceTuple`.
//  *
//  * @param xyz_coord_V
//  * @param V_cycle_F
//  * @return MatrixMesh
//  */
// MatrixMesh MatrixMesh::from_vf_samples(const Samples3d &xyz_coord_V,
//                                        const Samples3i &V_cycle_F) {
//   mesh_io::MeshBuilder mc =
//       mesh_io::MeshBuilder::from_vf_samples(xyz_coord_V, V_cycle_F, true);
//   auto [xyz_coord_V0, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H,
//         h_right_F, h_negative_B] = mc.he_samples;
//   return MatrixMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
//                     f_left_H, h_right_F, h_negative_B);
// }

// MatrixMesh MatrixMesh::from_vf_ply(const fs::path &ply_path) {
//   mesh_io::MeshBuilder mc = mesh_io::MeshBuilder::from_vf_ply(ply_path,
//   true); auto [xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
//   f_left_H,
//         h_right_F, h_negative_B] = mc.he_samples;
//   return MatrixMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
//                     f_left_H, h_right_F, h_negative_B);
// }

void MatrixMesh::update_vef_from_he() {

  // auto [xyz_coord_V, h_out_V, v_origin_H_, h_next_H_, h_twin_H, f_left_H_,
  //       h_right_F_, h_negative_B] = he_samples();

  int Nf = h_right_F_.rows();
  int Nh = get_num_half_edges();
  int Ne = Nh / 2;

  // Samples3i V_cycle_F_ = Samples3i(Nf, 3);
  // Samples2i V_cycle_E_ = Samples2i(Ne, 2);
  V_cycle_E_.resize(Ne, 2);
  V_cycle_F_.resize(Nf, 3);
  e_undirected_H_.resize(Nh);
  h_directed_E_.resize(Ne);
  std::set<std::vector<int>> setV_of_E;
  for (int f = 0; f < Nf; ++f) {
    int h0 = h_right_F_[f];
    int h1 = h_next_H_[h0];
    int h2 = h_next_H_[h1];
    int v0 = v_origin_H_[h0];
    int v1 = v_origin_H_[h1];
    int v2 = v_origin_H_[h2];
    V_cycle_F_.row(f) << v0, v1, v2;

    std::vector<int> edge0 = {std::min(v0, v1), std::max(v0, v1)};
    std::vector<int> edge1 = {std::min(v1, v2), std::max(v1, v2)};
    std::vector<int> edge2 = {std::min(v2, v0), std::max(v2, v0)};
    // Insert edges into the set and update V_cycle_E
    if (setV_of_E.find(edge0) == setV_of_E.end()) {
      V_cycle_E_.row(setV_of_E.size()) << edge0[0], edge0[1];
      setV_of_E.insert(edge0);
    }
    if (setV_of_E.find(edge1) == setV_of_E.end()) {
      V_cycle_E_.row(setV_of_E.size()) << edge1[0], edge1[1];
      setV_of_E.insert(edge1);
    }
    if (setV_of_E.find(edge2) == setV_of_E.end()) {
      V_cycle_E_.row(setV_of_E.size()) << edge2[0], edge2[1];
      setV_of_E.insert(edge2);
    }
  }
  // return {xyz_coord_V, V_cycle_E, V_cycle_F};
}

///////////////////////////////////////////////////////
// Fundamental accessors and properties ///////////////
///////////////////////////////////////////////////////

// Misc getters
int MatrixMesh::get_num_vertices() const { return h_out_V_.size(); }

int MatrixMesh::get_num_edges() const { return v_origin_H_.size() / 2; }

int MatrixMesh::get_num_half_edges() const { return v_origin_H_.size(); }

int MatrixMesh::get_num_faces() const { return h_right_F_.size(); }

int MatrixMesh::get_euler_characteristic() const {
  return get_num_vertices() - get_num_edges() + get_num_faces();
}

int MatrixMesh::get_num_boundaries() const { return h_negative_B_.size(); }

int MatrixMesh::get_genus() const {
  return (2 - get_euler_characteristic() - get_num_boundaries()) / 2;
}

int MatrixMesh::get_valence_v(int v) const {
  int valence = 0;
  for (auto h : generate_H_out_v_clockwise(v)) {
    valence++;
  }
  return valence;
}

Samples3i MatrixMesh::V_cycle_F() const {
  Samples3i V_cycle_F(get_num_faces(), 3);
  for (int f = 0; f < get_num_faces(); f++) {
    V_cycle_F.row(f) << v_origin_H_(h_right_F_(f)),
        v_origin_H_(h_next_H_(h_right_F_(f))),
        v_origin_H_(h_next_H_(h_next_H_(h_right_F_(f))));
  }
  return V_cycle_F;
}

VertexFaceTuple MatrixMesh::vf_samples() const {
  return {xyz_coord_V_, V_cycle_F()};
}

VertexEdgeFaceTuple MatrixMesh::vef_samples() const {

  auto [xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H,
        h_right_F, h_negative_B] = he_samples();
  int Nf = h_right_F.rows();
  int Ne = v_origin_H.rows() / 2;
  Samples3i V_cycle_F = Samples3i(Nf, 3);
  Samples2i V_cycle_E = Samples2i(Ne, 2);
  std::set<std::vector<int>> setV_of_E;
  for (int f = 0; f < Nf; ++f) {
    int h0 = h_right_F[f];
    int h1 = h_next_H[h0];
    int h2 = h_next_H[h1];
    int v0 = v_origin_H[h0];
    int v1 = v_origin_H[h1];
    int v2 = v_origin_H[h2];
    V_cycle_F.row(f) << v0, v1, v2;
    std::vector<int> edge0 = {std::min(v0, v1), std::max(v0, v1)};
    std::vector<int> edge1 = {std::min(v1, v2), std::max(v1, v2)};
    std::vector<int> edge2 = {std::min(v2, v0), std::max(v2, v0)};
    // Insert edges into the set and update V_cycle_E
    if (setV_of_E.find(edge0) == setV_of_E.end()) {
      V_cycle_E.row(setV_of_E.size()) << edge0[0], edge0[1];
      setV_of_E.insert(edge0);
    }
    if (setV_of_E.find(edge1) == setV_of_E.end()) {
      V_cycle_E.row(setV_of_E.size()) << edge1[0], edge1[1];
      setV_of_E.insert(edge1);
    }
    if (setV_of_E.find(edge2) == setV_of_E.end()) {
      V_cycle_E.row(setV_of_E.size()) << edge2[0], edge2[1];
      setV_of_E.insert(edge2);
    }
  }
  return {xyz_coord_V, V_cycle_E, V_cycle_F};
}

HalfEdgeTuple MatrixMesh::he_samples() const {
  return {xyz_coord_V_, h_out_V_,  v_origin_H_, h_next_H_,
          h_twin_H_,    f_left_H_, h_right_F_,  h_negative_B_};
}

Samplesi MatrixMesh::F_incident_b(int b) const {
  std::unordered_set<int> setF_incident_b;
  for (auto h : generate_H_negative_b(b)) {
    int v = v_origin_h(h);
    for (auto h_out : generate_H_out_v_clockwise(v, h)) {
      if (some_negative_boundary_contains_h(h_out)) {
        continue;
      }
      setF_incident_b.insert(f_left_h(h_out));
    }
  }
  Samplesi F_incident_b(setF_incident_b.size());
  std::copy(setF_incident_b.begin(), setF_incident_b.end(),
            F_incident_b.begin());
  return F_incident_b;
}

Samplesi MatrixMesh::E_incident_b(int b) const {
  std::unordered_set<int> setE_incident_b;
  for (auto h : generate_H_negative_b(b)) {
    int v = v_origin_h(h);
    for (auto h_out : generate_H_out_v_clockwise(v, h)) {
      setE_incident_b.insert(e_undirected_h(h_out));
    }
  }
  Samplesi E_incident_b(setE_incident_b.size());
  std::copy(setE_incident_b.begin(), setE_incident_b.end(),
            E_incident_b.begin());
  return E_incident_b;
}
///////////////////////////////////////////////////////
// Combinatorial maps /////////////////////////////////
///////////////////////////////////////////////////////

Vec3d MatrixMesh::xyz_coord_v(int v) const {
  if (v < 0 || v >= xyz_coord_V_.rows()) {
    throw std::out_of_range("xyz_coord_v: Vertex index out of range");
  }
  return xyz_coord_V_.row(v);
}

Samples3d MatrixMesh::xyz_coord_v(const Samplesi &indices) const {
  Samples3d result(indices.size(), 3);
  for (int i = 0; i < indices.size(); ++i) {
    result.row(i) = xyz_coord_V_.row(indices(i));
  }
  return result;
}

int MatrixMesh::h_out_v(int v) const {
  if (v < 0 || v >= h_out_V_.size()) {
    throw std::out_of_range("h_out_v: Vertex index out of range");
  }
  return h_out_V_(v);
}

Samplesi MatrixMesh::h_out_v(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = h_out_V_(indices(i));
  }
  return result;
}

int MatrixMesh::v_origin_h(int h) const {
  if (h < 0 || h >= v_origin_H_.size()) {
    throw std::out_of_range("v_origin_h: Half-edge index out of range");
  }
  return v_origin_H_(h);
}

Samplesi MatrixMesh::v_origin_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = v_origin_H_(indices(i));
  }
  return result;
}

int MatrixMesh::h_next_h(int h) const {
  if (h < 0 || h >= h_next_H_.size()) {
    throw std::out_of_range("h_next_h: Half-edge index out of range");
  }
  return h_next_H_(h);
}

Samplesi MatrixMesh::h_next_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = h_next_H_(indices(i));
  }
  return result;
}

int MatrixMesh::h_twin_h(int h) const {
  if (h < 0 || h >= h_twin_H_.size()) {
    throw std::out_of_range("h_twin_h: Half-edge index out of range");
  }
  return h_twin_H_(h);
}

Samplesi MatrixMesh::h_twin_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = h_twin_H_(indices(i));
  }
  return result;
}

int MatrixMesh::f_left_h(int h) const {
  if (h < 0 || h >= f_left_H_.size()) {
    throw std::out_of_range("f_left_h: Half-edge index out of range");
  }
  return f_left_H_(h);
}

Samplesi MatrixMesh::f_left_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = f_left_H_(indices(i));
  }
  return result;
}

int MatrixMesh::h_right_f(int f) const {
  if (f < 0 || f >= h_right_F_.size()) {
    throw std::out_of_range("h_right_f: Face index out of range");
  }
  return h_right_F_(f);
}

Samplesi MatrixMesh::h_right_f(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = h_right_F_(indices(i));
  }
  return result;
}

int MatrixMesh::h_negative_b(int b) const {
  if (b < 0 || b >= h_negative_B_.size()) {
    throw std::out_of_range("h_negative_b: Boundary index out of range");
  }
  return h_negative_B_(b);
}

Samplesi MatrixMesh::h_negative_b(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = h_negative_B_(indices(i));
  }
  return result;
}

int MatrixMesh::h_directed_e(int e) const {
  if (e < 0 || e >= h_directed_E_.size()) {
    throw std::out_of_range("h_directed_e: Edge index out of range");
  }
  return h_directed_E_(e);
}

Samplesi MatrixMesh::h_directed_e(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = h_directed_E_(indices(i));
  }
  return result;
}

int MatrixMesh::e_undirected_h(int h) const {
  if (h < 0 || h >= e_undirected_H_.size()) {
    throw std::out_of_range("e_undirected_h: Half-edge index out of range");
  }
  return e_undirected_H_(h);
}

Samplesi MatrixMesh::e_undirected_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = e_undirected_H_(indices(i));
  }
  return result;
}

// Derived combinatorial maps
int MatrixMesh::h_in_v(int v) const { return h_twin_h(h_out_v(v)); }

int MatrixMesh::v_head_h(int h) const { return v_origin_h(h_twin_h(h)); }

// int MatrixMesh::h_prev_h(int h) const {
//   int h_prev;
//   int h_next = h_next_h(h);
//   while (h_next != h) {
//     h_prev = h_next;
//     h_next = h_next_h(h_prev);
//   }
//   return h_prev;
// }

int MatrixMesh::h_prev_h(int h) const {
  int h_prev = h;
  while (h_next_h(h_prev) != h) {
    h_prev = h_next_h(h_prev);
  }
  return h_prev;
}

int MatrixMesh::h_rotcw_h(int h) const { return h_next_h(h_twin_h(h)); }

int MatrixMesh::h_rotccw_h(int h) const { return h_twin_h(h_prev_h(h)); }

int MatrixMesh::h_prev_h_by_rot(int h) const {
  int p_h = h_twin_h(h);
  int n_h = h_next_h(p_h);
  while (n_h != h) {
    p_h = h_twin_h(n_h);
    n_h = h_next_h(p_h);
  }
  return p_h;
}
///////////////////////////////////////////////////////
// Predicates /////////////////////////////////////////
///////////////////////////////////////////////////////

bool MatrixMesh::some_negative_boundary_contains_h(int h) const {
  return f_left_h(h) < 0;
}

bool MatrixMesh::some_positive_boundary_contains_h(int h) const {
  return f_left_h(h_twin_h(h)) < 0;
}

bool MatrixMesh::some_boundary_contains_h(int h) const {
  return some_negative_boundary_contains_h(h) ||
         some_positive_boundary_contains_h(h);
}

bool MatrixMesh::some_boundary_contains_v(int v) const {
  int h_start = h_out_v(v);
  //   do {
  //     if (some_boundary_contains_h(h)) {
  //       return true;
  //     }
  //     h = h_twin_h(h_next_h(h));
  //   } while (h != h_out_v(v));
  //   return false;
  for (auto h : generate_H_out_v_clockwise(v, h_start)) {
    if (some_boundary_contains_h(h)) {
      return true;
    }
  }
  return false;
}

bool MatrixMesh::h_is_locally_delaunay(int h) const {
  int vi = v_head_h(h_next_h(h_twin_h(h)));
  int vj = v_head_h(h);
  int vk = v_head_h(h_next_h(h));
  int vl = v_origin_h(h);

  Vec3d rij = xyz_coord_v(vj) - xyz_coord_v(vi);
  Vec3d ril = xyz_coord_v(vl) - xyz_coord_v(vi);

  Vec3d rkj = xyz_coord_v(vj) - xyz_coord_v(vk);
  Vec3d rkl = xyz_coord_v(vl) - xyz_coord_v(vk);

  double alphai = std::acos(rij.dot(ril) / (rij.norm() * ril.norm()));
  double alphak = std::acos(rkl.dot(rkj) / (rkl.norm() * rkj.norm()));

  return alphai + alphak <= M_PI;
}

bool MatrixMesh::h_is_flippable(int h) const {
  if (some_boundary_contains_h(h)) {
    return false;
  }
  int hlj = h;
  int hjk = h_next_h(hlj);
  int hli = h_next_h(h_twin_h(hlj));
  int vi = v_head_h(hli);
  int vk = v_head_h(hjk);
  for (auto him : generate_H_out_v_clockwise(vi)) {
    if (v_head_h(him) == vk) {
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////
// Generators /////////////////////////////////////////
///////////////////////////////////////////////////////

utils::SimpleGenerator<int>
MatrixMesh::generate_H_out_v_clockwise(int v, int h_start) const {
  if (h_start < 0) {
    h_start = h_out_v(v);
  } else if (v_origin_h(h_start) != v) {
    throw std::invalid_argument(
        "Starting half-edge does not originate at vertex v");
  }
  int h = h_start;
  do {
    co_yield h;
    h = h_rotcw_h(h);
    //     h = h_next_h(h_twin_h(h));
  } while (h != h_start);
}

utils::SimpleGenerator<int> MatrixMesh::generate_H_rotcw_h(int h) const {
  int h_start = h;
  do {
    co_yield h;
    h = h_rotcw_h(h);
    //     h = h_next_h(h_twin_h(h));
  } while (h != h_start);
}

utils::SimpleGenerator<int> MatrixMesh::generate_H_next_h(int h) const {
  int h_start = h;
  do {
    co_yield h;
    h = h_next_h(h);
  } while (h != h_start);
}

utils::SimpleGenerator<int> MatrixMesh::generate_H_right_f(int f) const {
  int h_start = h_right_f(f);
  int h = h_start;
  do {
    co_yield h;
    h = h_next_h(h);
  } while (h != h_start);
}

utils::SimpleGenerator<int> MatrixMesh::generate_H_negative_b(int b) const {
  int h_start = h_negative_b(b);
  int h = h_start;
  do {
    co_yield h;
    h = h_next_h(h);
  } while (h != h_start);
}

utils::SimpleGenerator<int> MatrixMesh::generate_F_incident_v(int v) const {
  int h_start = h_out_v(v);
  int h = h_start;
  do {
    if (!some_negative_boundary_contains_h(h)) {
      co_yield f_left_h(h);
    }
    h = h_rotcw_h(h);
  } while (h != h_start);
}

///////////////////////////////////////////////////////
// Mutators ///////////////////////////////////////////
///////////////////////////////////////////////////////

void MatrixMesh::flip_hedge(int h) {
  if (!h_is_flippable(h)) {
    throw std::invalid_argument("Edge is not flippable");
  }
  int h0 = h;
  int h1 = h_twin_h(h0);
  int h2 = h_next_h(h0);
  int h3 = h_next_h(h2);
  int h4 = h_next_h(h1);
  int h5 = h_next_h(h4);
  int v0 = v_origin_h(h1);
  int v1 = v_origin_h(h3);
  int v2 = v_origin_h(h0);
  int v3 = v_origin_h(h5);
  int f0 = f_left_h(h0);
  int f1 = f_left_h(h1);
  //   update vertices
  if (h_out_v(v0) == h1) {
    h_out_V_(v0) = h2;
  }
  if (h_out_v(v2) == h0) {
    h_out_V_(v2) = h4;
  }
  // update half-edges
  // h
  // v_origin_H_[h] = v;
  // h_next_H_[h] = h;
  // h_twin_H_[h] = h;
  // f_left_H_[h] = f;
  // update_mat_h(h0, v3, h3, std::nullopt, std::nullopt);
  v_origin_H_[h0] = v3;
  h_next_H_[h0] = h3;
  // update_mat_h(h1, v1, h5, std::nullopt, std::nullopt);
  v_origin_H_[h1] = v1;
  h_next_H_[h1] = h5;
  // update_mat_h(h2, std::nullopt, h1, std::nullopt, f1);
  h_next_H_[h2] = h1;
  f_left_H_[h2] = f1;
  // update_mat_h(h3, std::nullopt, h4, std::nullopt, std::nullopt);
  h_next_H_[h3] = h4;
  // update_mat_h(h4, std::nullopt, h0, std::nullopt, f0);
  h_next_H_[h4] = h0;
  f_left_H_[h4] = f0;
  // update_mat_h(h5, std::nullopt, h2, std::nullopt, std::nullopt);
  h_next_H_[h5] = h2;
  // update faces
  if (h_right_f(f0) == h2) {
    h_right_F_(f0) = h3;
  }
  if (h_right_f(f1) == h4) {
    h_right_F_(f1) = h5;
  }
}

void MatrixMesh::flip_edge(int e) {
  int h = h_directed_E_(e);
  if (!h_is_flippable(h)) {
    throw std::invalid_argument("Edge is not flippable");
  }
  int h0 = h;
  int h1 = h_twin_h(h0);
  int h2 = h_next_h(h0);
  int h3 = h_next_h(h2);
  int h4 = h_next_h(h1);
  int h5 = h_next_h(h4);
  int v0 = v_origin_h(h1);
  int v1 = v_origin_h(h3);
  int v2 = v_origin_h(h0);
  int v3 = v_origin_h(h5);
  int f0 = f_left_h(h0);
  int f1 = f_left_h(h1);
  //   update vertices
  if (h_out_v(v0) == h1) {
    h_out_V_(v0) = h2;
  }
  if (h_out_v(v2) == h0) {
    h_out_V_(v2) = h4;
  }
  // update half-edges
  // update_mat_h(h0, v3, h3, std::nullopt, std::nullopt);
  v_origin_H_[h0] = v3;
  h_next_H_[h0] = h3;
  // update_mat_h(h1, v1, h5, std::nullopt, std::nullopt);
  v_origin_H_[h1] = v1;
  h_next_H_[h1] = h5;
  // update_mat_h(h2, std::nullopt, h1, std::nullopt, f1);
  h_next_H_[h2] = h1;
  f_left_H_[h2] = f1;
  // update_mat_h(h3, std::nullopt, h4, std::nullopt, std::nullopt);
  h_next_H_[h3] = h4;
  // update_mat_h(h4, std::nullopt, h0, std::nullopt, f0);
  h_next_H_[h4] = h0;
  f_left_H_[h4] = f0;
  // update_mat_h(h5, std::nullopt, h2, std::nullopt, std::nullopt);
  h_next_H_[h5] = h2;
  // update faces
  if (h_right_f(f0) == h2) {
    h_right_F_(f0) = h3;
  }
  if (h_right_f(f1) == h4) {
    h_right_F_(f1) = h5;
  }

  // V_cycle_E_.row(e) << v_origin_h(h0), v_origin_h(h_twin_h(h0));
  // V_cycle_F_.row(f0) << v_origin_h(h_right_f(f0)),
  //     v_origin_h(h_next_h(h_right_f(f0))),
  //     v_origin_h(h_next_h(h_next_h(h_right_f(f0))));
  // V_cycle_F_.row(f1) << v_origin_h(h_right_f(f1)),
  //     v_origin_h(h_next_h(h_right_f(f1))),
  //     v_origin_h(h_next_h(h_next_h(h_right_f(f1))));
  V_cycle_E_.row(e) << v_origin_h(h0), v_origin_h(h_twin_h(h0));
  V_cycle_F_.row(f0) << v_origin_h(h_right_f(f0)),
      v_origin_h(h_next_h(h_right_f(f0))),
      v_origin_h(h_next_h(h_next_h(h_right_f(f0))));
  V_cycle_F_.row(f1) << v_origin_h(h_right_f(f1)),
      v_origin_h(h_next_h(h_right_f(f1))),
      v_origin_h(h_next_h(h_next_h(h_right_f(f1))));

  // double L0 = length_E_[e];
  // vec_H_.row(h0) = xyz_coord_v(v1) - xyz_coord_v(v3);
  // vec_H_.row(h1) = -vec_H_.row(h0);
  // length_E_[e] = math::L2norm(vec_H_.row(h0));
  // double L1 = length_E_[e];
  // double L2 = length_E_[e_undirected_h(h2)];
  // double L3 = length_E_[e_undirected_h(h3)];
  // double L4 = length_E_[e_undirected_h(h4)];
  // double L5 = length_E_[e_undirected_h(h5)];
  // area_F_[f0] = math::heron_area(L1, L3, L4);
  // area_F_[f1] = math::heron_area(L1, L5, L2);

  // int num_edges = get_num_edges();
  // average_edge_length_ =
  //     (num_edges * average_edge_length_ - L0 + L1) / num_edges;
}

int MatrixMesh::flip_non_delaunay() {
  int flip_count = 0;
  for (int h = 0; h < get_num_half_edges(); h++) {
    if (h_is_flippable(h) && !h_is_locally_delaunay(h)) {
      flip_hedge(h);
      flip_count++;
    }
  }
  return flip_count;
}

void MatrixMesh::rigid_transform(const Eigen::Vector3d &translation,
                                 const Eigen::Vector3d &angle_vec) {
  for (int _v = 0; _v < get_num_vertices(); ++_v) {
    xyz_coord_V_.row(_v) =
        lie::rigid_transform(translation, angle_vec, xyz_coord_V_.row(_v));
  }
}

void MatrixMesh::divide_face_barycentric(int f) {

  int h0 = h_right_f(f);
  int h1 = h_next_h(h0);
  int h2 = h_next_h(h1);
  int v0 = v_origin_h(h0);
  int v1 = v_origin_h(h1);
  int v2 = v_origin_h(h2);
  int e0 = e_undirected_h(h0);
  int e1 = e_undirected_h(h1);
  int e2 = e_undirected_h(h2);
  int f0 = f;

  int num_faces_pre = get_num_faces();
  int num_edges_pre = get_num_edges();
  int num_vertices_pre = get_num_vertices();
  int num_half_edges_pre = get_num_half_edges();

  int num_vertices = num_vertices_pre + 1;
  int num_edges = num_edges_pre + 3;
  int num_faces = num_faces_pre + 2;
  int num_half_edges = num_half_edges_pre + 6;

  int v3 = num_vertices_pre;
  int e3 = num_edges_pre;
  int e4 = num_edges_pre + 1;
  int e5 = num_edges_pre + 2;
  int f1 = num_faces_pre;
  int f2 = num_faces_pre + 1;

  int h3 = num_half_edges_pre;
  int h4 = num_half_edges_pre + 1;
  int h5 = num_half_edges_pre + 2;
  int h6 = num_half_edges_pre + 3;
  int h7 = num_half_edges_pre + 4;
  int h8 = num_half_edges_pre + 5;

  xyz_coord_V_.conservativeResize(num_vertices, 3);
  h_out_V_.conservativeResize(num_vertices);
  Vec3d x012 = (xyz_coord_v(v0) + xyz_coord_v(v1) + xyz_coord_v(v2)) / 3;
  xyz_coord_V_.row(v3) = x012;
  h_out_V_[v3] = h3;

  h_directed_E_.conservativeResize(num_edges);
  V_cycle_E_.conservativeResize(num_edges, 2);
  h_directed_E_[e3] = h3;
  h_directed_E_[e4] = h4;
  h_directed_E_[e5] = h5;
  V_cycle_E_.row(e3) << v3, v2;
  V_cycle_E_.row(e4) << v3, v0;
  V_cycle_E_.row(e5) << v3, v1;

  h_right_F_.conservativeResize(num_faces);
  V_cycle_F_.conservativeResize(num_faces, 3);
  h_right_F_[f1] = h1;
  h_right_F_[f2] = h2;
  V_cycle_F_.row(f1) << v1, v2, v3;
  V_cycle_F_.row(f2) << v2, v0, v3;

  v_origin_H_.conservativeResize(num_half_edges);
  h_next_H_.conservativeResize(num_half_edges);
  h_twin_H_.conservativeResize(num_half_edges);
  f_left_H_.conservativeResize(num_half_edges);
  e_undirected_H_.conservativeResize(num_half_edges);

  h_next_H_[h0] = h8;
  // f_left_H_[h0] = f0;

  h_next_H_[h1] = h6;
  f_left_H_[h1] = f1;

  h_next_H_[h2] = h7;
  f_left_H_[h2] = f2;

  v_origin_H_[h3] = v3;
  h_next_H_[h3] = h2;
  h_twin_H_[h3] = h6;
  f_left_H_[h3] = f2;
  e_undirected_H_[h3] = e3;

  v_origin_H_[h4] = v3;
  h_next_H_[h4] = h0;
  h_twin_H_[h4] = h7;
  f_left_H_[h4] = f0;
  e_undirected_H_[h4] = e4;

  v_origin_H_[h5] = v3;
  h_next_H_[h5] = h1;
  h_twin_H_[h5] = h8;
  f_left_H_[h5] = f1;
  e_undirected_H_[h5] = e5;

  v_origin_H_[h6] = v2;
  h_next_H_[h6] = h5;
  h_twin_H_[h6] = h3;
  f_left_H_[h6] = f1;
  e_undirected_H_[h6] = e3;

  v_origin_H_[h7] = v0;
  h_next_H_[h7] = h3;
  h_twin_H_[h7] = h4;
  f_left_H_[h7] = f2;
  e_undirected_H_[h7] = e4;

  v_origin_H_[h8] = v1;
  h_next_H_[h8] = h4;
  h_twin_H_[h8] = h5;
  f_left_H_[h8] = f0;
  e_undirected_H_[h8] = e5;
}

void MatrixMesh::divide_faces_barycentric() {
  int num_faces_pre = get_num_faces();
  for (int f = 0; f < num_faces_pre; f++) {
    divide_face_barycentric(f);
  }
}

void MatrixMesh::divide_faces() {
  printf("MatrixMesh::divide_faces\n");

  auto vertex_pair_key = [](int v0, int v1) -> int64_t {
    return std::min(v0, v1) * 1000000 + std::max(v0, v1);
  };

  int num_faces_pre = get_num_faces();
  printf("  num_faces_pre=%d\n", num_faces_pre);
  int num_faces = 4 * num_faces_pre;
  int num_vertices_pre = get_num_vertices();

  // Samples3d xyz_coord_V = xyz_coord_V_;
  Samples3i V_cycle_F(num_faces, 3);

  std::vector<Vec3d> vecV;
  for (int v = 0; v < num_vertices_pre; v++) {
    vecV.push_back(xyz_coord_v(v));
  }
  // std::vector<Eigen::Vector3i> vecF;

  std::unordered_map<int64_t, int> v_midpt_vv;
  int v_count = num_vertices_pre;
  int f_count = 0;
  for (int f = 0; f < num_faces_pre; f++) {
    printf("  f=%d\n", f);
    int h0 = h_right_f(f);
    int h1 = h_next_h(h0);
    int h2 = h_next_h(h1);
    int v0 = v_origin_h(h0);
    int v1 = v_origin_h(h1);
    int v2 = v_origin_h(h2);
    // int v0 = V_cycle_F_(f, 0);
    // int v1 = V_cycle_F_(f, 1);
    // int v2 = V_cycle_F_(f, 2);
    printf("  v0,v1,v2=%d,%d,%d\n", v0, v1, v2);
    int64_t key01 = vertex_pair_key(v0, v1);
    int64_t key12 = vertex_pair_key(v1, v2);
    int64_t key20 = vertex_pair_key(v2, v0);
    std::cout << "  key01, key12, key20=" << key01 << "," << key12 << ","
              << key20 << std::endl;

    // int count01 = v_midpt_vv.count(key01);
    // int count12 = v_midpt_vv.count(key12);
    // int count20 = v_midpt_vv.count(key20);
    // int v01;
    // int v12;
    // int v20;
    // if (count01 == 0) {
    //   v01 = v_count;
    //   v_midpt_vv[key01] = v01;
    //   Vec3d xyz01 = (vecV[v0] + vecV[v1]) / 2.0;
    //   xyz01 *= 1 / xyz01.norm();
    //   // vecV[v01] = xyz01;
    //   vecV.push_back(xyz01);
    //   v_count++;
    // } else if (count01 == 1) {
    //   v01 = v_midpt_vv[key01];
    // } else {
    //   throw std::runtime_error("count01 == 1");
    // }
    // if (count12 == 0) {
    //   v12 = v_count;
    //   v_midpt_vv[key12] = v12;
    //   Vec3d xyz12 = (vecV[v1] + vecV[v2]) / 2.0;
    //   xyz12 *= 1 / xyz12.norm();
    //   // vecV[v12] = xyz12;
    //   vecV.push_back(xyz12);
    //   v_count++;
    // } else if (count12 == 1) {
    //   v12 = v_midpt_vv[key12];
    // } else {
    //   throw std::runtime_error("count12 == 1");
    // }
    // if (count20 == 0) {
    //   v20 = v_count;
    //   v_midpt_vv[key20] = v20;
    //   Vec3d xyz20 = (vecV[v2] + vecV[v0]) / 2.0;
    //   xyz20 *= 1 / xyz20.norm();
    //   // vecV[v20] = xyz20;
    //   vecV.push_back(xyz20);
    //   v_count++;
    // } else if (count20 == 1) {
    //   v20 = v_midpt_vv[key20];
    // } else {
    //   throw std::runtime_error("count20 == 1");
    // }

    int v01 = (v_midpt_vv.count(key01) == 1) ? v_midpt_vv[key01] : -1;
    int v12 = (v_midpt_vv.count(key12) == 1) ? v_midpt_vv[key12] : -1;
    int v20 = (v_midpt_vv.count(key20) == 1) ? v_midpt_vv[key20] : -1;
    printf("  v01,v12,v20=%d,%d,%d\n", v01, v12, v20);
    if (v01 == -1) {
      v01 = v_count++;

      Vec3d xyz01 = (vecV[v0] + vecV[v1]) / 2.0;
      xyz01 *= 1 / xyz01.norm();
      // vecV[v01] = xyz01;
      vecV.push_back(xyz01);
      v_midpt_vv[key01] = v01;
    }
    if (v12 == -1) {
      v12 = v_count++;
      Vec3d xyz12 = (vecV[v1] + vecV[v2]) / 2.0;
      xyz12 *= 1 / xyz12.norm();
      // vecV[v12] = xyz12;
      vecV.push_back(xyz12);
      v_midpt_vv[key12] = v12;
    }
    if (v20 == -1) {
      v20 = v_count++;
      Vec3d xyz20 = (vecV[v2] + vecV[v0]) / 2.0;
      xyz20 *= 1 / xyz20.norm();
      // vecV[v20] = xyz20;
      vecV.push_back(xyz20);
      v_midpt_vv[key20] = v20;
    }

    printf("  v01,v12,v20=%d,%d,%d\n", v01, v12, v20);

    V_cycle_F.row(f_count++) << v0, v01, v20;
    V_cycle_F.row(f_count++) << v01, v1, v12;
    V_cycle_F.row(f_count++) << v20, v12, v2;
    V_cycle_F.row(f_count++) << v01, v12, v20;
  }

  xyz_coord_V_.conservativeResize(vecV.size(), 3);
  for (int v = 0; v < vecV.size(); v++) {
    // vecV[v] /= vecV[v].norm();
    xyz_coord_V_.row(v) = vecV[v];
  }

  // xyz_coord_V_ = xyz_coord_V;
  V_cycle_F_ = V_cycle_F;

  init_core_data_from_V_cycle_F();
}

///////////////////////////////////////////////////////
// Prototyping ////////////////////////////////////////
///////////////////////////////////////////////////////
Samples3d MatrixMesh::mean_curvature_vector_cotan() const {
  // int numH = get_num_half_edges();
  Samples1d H;
  Eigen::MatrixXd HN;
  Eigen::SparseMatrix<double> L, M, Minv;
  auto [V, F] = vf_samples();
  // igl::cotmatrix(V, F, L);
  // igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
  // igl::invert_diag(M, Minv);
  // HN = -Minv * (L * V) / 2;
  H = HN.rowwise().norm(); // up to sign
  return HN;
}

Samples3d MatrixMesh::mean_curvature_vector_graph_laplacian() const {
  int num_vertices = get_num_vertices();
  int valance;
  int h_start;
  int vj;
  Samples1d H(num_vertices);
  Samples3d Q = xyz_coord_V_;
  Samples3d lapQ(num_vertices, 3);

  for (int v = 0; v < num_vertices; v++) {
    h_start = h_out_v(v);
    valance = 0;
    lapQ.row(v) = Vec3d::Zero();
    for (auto h : generate_H_out_v_clockwise(v, h_start)) {
      valance++;
      vj = v_head_h(h);
      lapQ.row(v) += Q.row(vj) - Q.row(v);
    }
    lapQ.row(v) /= valance;
  }
  return lapQ;
}

Vec3d MatrixMesh::compute_avg_xyz_coord() const {
  Vec3d avg = Vec3d::Zero();
  for (int v = 0; v < get_num_vertices(); v++) {
    avg += xyz_coord_v(v);
  }
  return avg / get_num_vertices();
}

int MatrixMesh::uniform_flip_sweep() {
  int flip_count = 0;
  int num_edges = get_num_edges();
  for (int e = 0; e < num_edges; e++) {
    double _r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    if (_r < 0.02) {
      int h = h_directed_E_(e);
      if (h_is_flippable(h)) {
        printf("Flipping edge %d\n", e);
        flip_edge(e);
        flip_count++;
      } else {
        printf("Edge %d is not flippable\n", e);
      }
    }
  }
  return flip_count;
}

//////////////////////////////////////////////
// Visualization /////////////////////////////
//////////////////////////////////////////////

void MatrixMesh::set_visual_defaults() {

  // radius_vertex_ = 5;
  // rgba_vertex_ = RGBA_DICT.at("purple");
  // rgba_half_edge_ = RGBA_DICT.at("purple");
  // rgba_face_ = RGBA_DICT.at("meshbrane_blue80");
  // rgba_face_ = RGBA_DICT.at("purple50");

  int num_vertices = get_num_vertices();
  int num_half_edges = get_num_half_edges();
  int num_faces = get_num_faces();
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
  rgba_F.resize(num_faces, 4);
  for (int i = 0; i < num_faces; i++) {
    rgba_F.row(i) = rgba_face_;
  }
  rgba_E_.resize(num_edges, 4);
  for (int i = 0; i < num_edges; i++) {
    rgba_E_.row(i) = rgba_edge_;
  }
}

std::array<Samples3d, 3> MatrixMesh::compute_shifted_half_edge_arrows() {

  Samples3d xyz_o_H = xyz_coord_v(v_origin_H_);
  Samples3d xyz_on_H = xyz_coord_v(v_origin_h(h_next_H_));
  Samples3d xyz_onn_H = xyz_coord_v(v_origin_h(h_next_h(h_next_H_)));

  Samples3d P1(get_num_half_edges(), 3);
  Samples3d P2(get_num_half_edges(), 3);
  Samples3d P3(get_num_half_edges(), 3);
  double shift_to_center = 0.15;
  double shaft_len = 0.75;
  double tip_len = 0.25;
  for (int h = 0; h < get_num_half_edges(); h++) {
    Vec3d p1 = xyz_coord_v(v_origin_h(h));
    Vec3d p2 = xyz_coord_v(v_origin_h(h_next_h(h)));
    Vec3d p3 = xyz_coord_v(v_origin_h(h_next_h(h_next_h(h))));
    Vec3d p12 = (p1 + p2) / 2;
    if (some_negative_boundary_contains_h(h)) {
      int ht = h_twin_h(h);

      Vec3d p3 = xyz_coord_v(v_origin_h(h_next_h(h_next_h(ht))));
      p3 = p12 - (p3 - p12);
    }

    Vec3d c = (p1 + p2 + p3) / 3;
    p1 = p1 + shift_to_center * (c - p1);
    p2 = p2 + shift_to_center * (c - p2);

    p1 = p12 + shaft_len * (p1 - p12);
    p2 = p12 + shaft_len * (p2 - p12);
    p3 = p2 + tip_len * (c - p2);

    P1.row(h) = p1;
    P2.row(h) = p2;
    P3.row(h) = p3;
  }
  std::array<Samples3d, 3> result = {P1, P2, P3};
  return result;
}

void MatrixMesh::update_shifted_half_edge_arrows() {
  if (!show_half_edges_) {
    return;
  }
  // shifted_half_edge_arrows_ = compute_shifted_half_edge_arrows();
  size_t num_half_edges = get_num_half_edges();
  for (int i = 0; i < 3; i++) {
    shifted_half_edge_arrows_[i].resize(num_half_edges, 3);
  }
  Samples3d xyz_o_H = xyz_coord_v(v_origin_H_);
  Samples3d xyz_on_H = xyz_coord_v(v_origin_h(h_next_H_));
  Samples3d xyz_onn_H = xyz_coord_v(v_origin_h(h_next_h(h_next_H_)));

  // Samples3d P1(num_half_edges, 3);
  // Samples3d P2(num_half_edges, 3);
  // Samples3d P3(num_half_edges, 3);
  double shift_to_center = 0.15;
  double shaft_len = 0.75;
  double tip_len = 0.25;
  for (int h = 0; h < num_half_edges; h++) {
    Vec3d p1 = xyz_coord_v(v_origin_h(h));
    Vec3d p2 = xyz_coord_v(v_origin_h(h_next_h(h)));
    Vec3d p3 = xyz_coord_v(v_origin_h(h_next_h(h_next_h(h))));
    Vec3d p12 = (p1 + p2) / 2;
    if (some_negative_boundary_contains_h(h)) {
      int ht = h_twin_h(h);

      Vec3d p3 = xyz_coord_v(v_origin_h(h_next_h(h_next_h(ht))));
      p3 = p12 - (p3 - p12);
    }

    Vec3d c = (p1 + p2 + p3) / 3;
    p1 = p1 + shift_to_center * (c - p1);
    p2 = p2 + shift_to_center * (c - p2);

    p1 = p12 + shaft_len * (p1 - p12);
    p2 = p12 + shaft_len * (p2 - p12);
    p3 = p2 + tip_len * (c - p2);

    shifted_half_edge_arrows_[0].row(h) = p1;
    shifted_half_edge_arrows_[1].row(h) = p2;
    shifted_half_edge_arrows_[2].row(h) = p3;
  }
}

void MatrixMesh::update_mesh_visuals() { update_shifted_half_edge_arrows(); }

/////////////////////////////////////////////////
// Simplicial operations ////////////////////////
/////////////////////////////////////////////////

SimplicialTriple MatrixMesh::star_of_vertex(int v) const {
  SimplicialTriple VEF;
  // std::array<SimplicialSet, 3> VEF;
  SimplicialSet *V = &VEF[0];
  SimplicialSet *E = &VEF[1];
  SimplicialSet *F = &VEF[2];
  V->insert(v);
  for (auto h : generate_H_out_v_clockwise(v)) {
    E->insert(e_undirected_h(h));
    if (some_negative_boundary_contains_h(h)) {
      continue;
    }
    F->insert(f_left_h(h));
  }

  return VEF;
}

SimplicialTriple MatrixMesh::star_of_edge(int e) const {
  SimplicialTriple VEF;
  // std::array<SimplicialSet, 3> VEF;
  SimplicialSet *V = &VEF[0];
  SimplicialSet *E = &VEF[1];
  SimplicialSet *F = &VEF[2];
  int h = h_directed_E_(e);
  int ht = h_twin_h(h);
  int f = f_left_h(h);
  int ft = f_left_h(ht);
  E->insert(e);
  if (!some_negative_boundary_contains_h(h)) {
    F->insert(f);
  }
  if (!some_negative_boundary_contains_h(ht)) {
    F->insert(ft);
  }
  return VEF;
}

SimplicialTriple MatrixMesh::closure(SimplicialTriple &VEF) const {
  SimplicialSet *V = &VEF[0];
  SimplicialSet *E = &VEF[1];
  SimplicialSet *F = &VEF[2];

  SimplicialSet clV = VEF[0];
  SimplicialSet clE = VEF[1];
  SimplicialSet clF = VEF[2];

  for (auto e : *E) {
    int h = h_directed_E_(e);
    clV.insert(v_origin_h(h));
    clV.insert(v_head_h(h));
  }

  for (auto f : *F) {
    for (auto h : generate_H_next_h(h_right_f(f))) {
      clE.insert(e_undirected_h(h));
      clV.insert(v_origin_h(h));
    }
  }

  return {clV, clE, clF};
}

///////////////////////////////////////////////////////
// Quadrature /////////////////////////////////////////
///////////////////////////////////////////////////////
void MatrixMesh::init_quad_points_and_weights() {
  printf("MatrixMesh::init_quad_points_and_weights\n");
  // 4th order quadrature
  bary_coord_Q_.resize(7, 3);
  bary_coord_Q_ << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0 / 3,
      1.0 / 3, 1.0 / 3, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.5;
  dimensionless_quad_weight_Q_.resize(7);
  dimensionless_quad_weight_Q_ << 1.0 / 20, 1.0 / 20, 1.0 / 20, 9.0 / 20,
      2.0 / 15, 2.0 / 15, 2.0 / 15;
}
Vec3d MatrixMesh::xyz_coord_fq(int f, int q) const {
  int h0 = h_right_f(f);
  int h1 = h_next_h(h0);
  int h2 = h_next_h(h1);
  int v0 = v_origin_h(h0);
  int v1 = v_origin_h(h1);
  int v2 = v_origin_h(h2);
  Vec3d x0 = xyz_coord_v(v0);
  Vec3d x1 = xyz_coord_v(v1);
  Vec3d x2 = xyz_coord_v(v2);
  Vec3d s = bary_coord_q(q);
  Vec3d x = s[0] * x0 + s[1] * x1 + s[2] * x2;
  return x;
}
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// From Membrane subclass /////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

double MatrixMesh::get_gaussian_curvature_angle_defect_v(int v) const {
  double area = 0.0;
  double defect = 2 * M_PI;
  for (auto h : generate_H_out_v_clockwise(v)) {
    if (some_negative_boundary_contains_h(h)) {
      // # do boundary geodesic curvature stuff;
      continue;
    }
    int e = e_undirected_h(h);
    double L = length_E_[e];
    Vec3d vecL = vec_H_.row(h);
    int hrot = h_rotcw_h(h);
    int erot = e_undirected_h(hrot);
    double Lrot = length_E_[erot];
    Vec3d vecLrot = vec_H_.row(hrot);
    double cos_angle = math::dot(vecL, vecLrot) / (L * Lrot);
    defect -= std::acos(cos_angle);
    area += L * Lrot * std::sqrt(1 - cos_angle * cos_angle) / 6;
  }
  return defect / area;
}

void MatrixMesh::update_gaussian_curvature_angle_defect() {
  int num_vertices = get_num_vertices();
  gaussian_curvature_V_.resize(num_vertices);
  for (int v = 0; v < num_vertices; v++) {
    gaussian_curvature_V_[v] = get_gaussian_curvature_angle_defect_v(v);
  }
}

void MatrixMesh::update_gaussian_curvature_laplacian() {
  // printf("MatrixMesh::update_gaussian_curvature_laplacian\n");
  int num_vertices = get_num_vertices();
  gaussian_curvature_V_.resize(num_vertices);
  Samples3d n_V = mcvec_V_.rowwise().normalized();
  // Samples3d n_V = normal_V_;
  Samples3d lap_n_V = laplacian(n_V);
  for (int v = 0; v < num_vertices; v++) {
    // printf("v = %d\n", v);
    Vec3d lap_r = mcvec_V_.row(v);
    double norm_sqr_lap_r = lap_r.squaredNorm();
    // double norm_sqr_lap_r = 4 * mean_curvature_V_[v] * mean_curvature_V_[v];
    // printf("norm_sqr_lap_r = %.20f\n", norm_sqr_lap_r);
    Vec3d n = n_V.row(v);
    // printf("n = %.20f, %.20f, %.20f\n", n[0], n[1], n[2]);
    Vec3d lap_n = lap_n_V.row(v);
    // printf("lap_n = %.20f, %.20f, %.20f\n", lap_n[0], lap_n[1], lap_n[2]);
    gaussian_curvature_V_[v] = 0.5 * (norm_sqr_lap_r + math::dot(n, lap_n));
    // printf("gaussian_curvature_V_[%d] = %.20f\n", v,
    // gaussian_curvature_V_[v]);
  }
}

void MatrixMesh::update_gaussian_curvature() {
  // printf("MatrixMesh::update_gaussian_curvature\n");
  // if (gaussian_curvature_type_ == "angle_defect") {
  //   update_gaussian_curvature_angle_defect();
  // } else if (gaussian_curvature_type_ == "laplacian") {
  //   update_gaussian_curvature_laplacian();
  // } else {
  //   throw std::runtime_error("Invalid gaussian_curvature_type_");
  // }
  switch (gaussian_curvature_type_) {
  case GaussianCurvatureType::ANGLE_DEFECT:
    update_gaussian_curvature_angle_defect();
    return;
  case GaussianCurvatureType::LAPLACIAN:
    update_gaussian_curvature_laplacian();
    return;
  }
  throw std::runtime_error(
      "update_gaussian_curvature: Unknown gaussian curvature type");
}

//////////////////////
// Laplacians ////////
//////////////////////

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Laplacian stuff to fix /////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

Samples3d MatrixMesh::cotan_laplacian(Samples3d &Q) {
  // printf("MatrixMesh::cotan_laplacian\n");
  int num_vertices = get_num_vertices();
  if (Q.rows() != num_vertices) {
    throw std::runtime_error("Q.size() != num_vertices");
  }
  Samples3d lapQ(num_vertices, 3);
  lapQ.setZero();

  for (int vi = 0; vi < get_num_vertices(); vi++) {
    double Atot = 0.0;
    Vec3d qi = Q.row(vi);

    for (auto hij : generate_H_out_v_clockwise(vi)) {
      int vj = v_head_h(hij);
      Vec3d qj = Q.row(vj);

      int hijm1 = h_next_h(h_twin_h(hij));
      int hijp1 = h_twin_h(h_next_h(h_next_h(hij)));
      int hjjm1 = h_twin_h(h_next_h(h_next_h(h_twin_h(hij))));
      int hjjp1 = h_next_h(hij);
      // printf("hijm1 = %d, hijp1 = %d, hjjm1 = %d, hjjp1 = %d, hij = %d\n",
      //        hijm1, hijp1, hjjm1, hjjp1, hij);

      int eijm1 = e_undirected_h(hijm1);
      int eijp1 = e_undirected_h(hijp1);
      int ejjm1 = e_undirected_h(hjjm1);
      int ejjp1 = e_undirected_h(hjjp1);
      int eij = e_undirected_h(hij);
      // printf("eijm1 = %d, ejjm1 = %d, eijp1 = %d, ejjp1 = %d, eij = %d\n",
      //        eijm1, ejjm1, eijp1, ejjp1, eij);

      double Lijm1 = length_E_[eijm1];
      double Ljjm1 = length_E_[ejjm1];
      double Lijp1 = length_E_[eijp1];
      double Ljjp1 = length_E_[ejjp1];
      double Lij = length_E_[eij];
      // printf("Lijm1 = %.20f, Ljjm1 = %.20f, Lijp1 = %.20f, Ljjp1 = %.20f,
      // Lij
      // "
      //        "= %.20f\n",
      //        Lijm1, Ljjm1, Lijp1, Ljjp1, Lij);

      double thetajm1 = math::heron_angle(Lijm1, Ljjm1, Lij);
      double thetajp1 = math::heron_angle(Lijp1, Ljjp1, Lij);
      // printf("thetajm1 = %.20f, thetajp1 = %.20f\n", thetajm1, thetajp1);

      // auto cos_thetam =
      //     (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1);

      // auto cos_thetap =
      //     (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1);

      // auto cot_thetam = cos_thetam / std::sqrt(1 - math::POW2(cos_thetam));
      // auto cot_thetap = cos_thetap / std::sqrt(1 - math::POW2(cos_thetap));
      double cot_thetam = 1.0 / std::tan(thetajm1);
      double cot_thetap = 1.0 / std::tan(thetajp1);
      if (some_negative_boundary_contains_h(hij)) {
        cot_thetap = 0.0;
      } else if (some_negative_boundary_contains_h(h_twin_h(hij))) {
        cot_thetam = 0.0;
      }

      Atot += math::POW2(Lij) * (cot_thetam + cot_thetap) / 8;
      lapQ.row(vi) += (cot_thetam + cot_thetap) * (qj - qi) / 2;
    }
    lapQ.row(vi) /= Atot;
  }

  return lapQ;
}

Samples1d MatrixMesh::cotan_laplacian(Samples1d &Q) {

  int num_vertices = get_num_vertices();
  if (Q.size() != num_vertices) {
    throw std::runtime_error("Q.size() != num_vertices");
  }
  Samples1d lapQ(num_vertices);
  lapQ.setZero();

  for (int vi = 0; vi < get_num_vertices(); vi++) {
    double Atot = 0.0;
    double qi = Q[vi];

    for (auto hij : generate_H_out_v_clockwise(vi)) {
      int vj = v_head_h(hij);
      double qj = Q[vj];

      int hijm1 = h_next_h(h_twin_h(hij));
      int hijp1 = h_twin_h(h_next_h(h_next_h(hij)));
      int hjjm1 = h_twin_h(h_next_h(h_next_h(h_twin_h(hij))));
      int hjjp1 = h_next_h(hij);

      int eijm1 = e_undirected_h(hijm1);
      int eijp1 = e_undirected_h(hijp1);
      int ejjm1 = e_undirected_h(hjjm1);
      int ejjp1 = e_undirected_h(hjjp1);
      int eij = e_undirected_h(hij);

      double Lijm1 = length_E_[eijm1];
      double Ljjm1 = length_E_[ejjm1];
      double Lijp1 = length_E_[eijp1];
      double Ljjp1 = length_E_[ejjp1];
      double Lij = length_E_[eij];

      double thetajm1 = math::heron_angle(Lijm1, Ljjm1, Lij);
      double thetajp1 = math::heron_angle(Lijp1, Ljjp1, Lij);
      // auto cos_thetam =
      //     (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1);

      // auto cos_thetap =
      //     (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1);

      // auto cot_thetam = cos_thetam / std::sqrt(1 - math::POW2(cos_thetam));
      // auto cot_thetap = cos_thetap / std::sqrt(1 - math::POW2(cos_thetap));
      double cot_thetam = 1.0 / std::tan(thetajm1);
      double cot_thetap = 1.0 / std::tan(thetajp1);

      Atot += math::POW2(Lij) * (cot_thetam + cot_thetap) / 8;
      lapQ[vi] += (cot_thetam + cot_thetap) * (qj - qi) / 2;
    }
    lapQ.row(vi) /= Atot;
  }

  return lapQ;
}

Samples1d MatrixMesh::belkin_laplacian(Samples1d &Q) {
  return belkin_heat_laplacian(Q, xyz_coord_V_, V_cycle_F_, area_F_,
                               belkin_dt_);
}

Samples3d MatrixMesh::belkin_laplacian(Samples3d &Q) {
  return belkin_heat_laplacian(Q, xyz_coord_V_, V_cycle_F_, area_F_,
                               belkin_dt_);
}

template <typename Samples>
Samples MatrixMesh::guckenberger_laplacian(Samples &phi) {
  // printf("MatrixMesh::guckenberger_laplacian\n");
  auto num_vertices = get_num_vertices();
  Eigen::MatrixXd lap_phi(phi);
  lap_phi.setZero();
  for (int ix{0}; ix < num_vertices; ix++) {
    Vec3d x = xyz_coord_V_.row(ix);
    double heat_dt = heat_dt_multiple_ * area_V_[ix];
    for (int f{0}; f < get_num_faces(); f++) {
      for (int iy : V_cycle_F_.row(f)) {
        Vec3d y = xyz_coord_V_.row(iy);
        double A = area_F_[f];
        lap_phi.row(ix) += (A / (3 * heat_dt)) *
                           heat_parametrix2d(x, y, heat_dt) *
                           (phi.row(iy) - phi.row(ix));
      }
    }
  }

  return lap_phi;
}

template Samples1d MatrixMesh::guckenberger_laplacian(Samples1d &phi);
template Samples3d MatrixMesh::guckenberger_laplacian(Samples3d &phi);

Samples1d MatrixMesh::apply_laplacian_matrix(Samples1d &Q) {
  int num_vertices = get_num_vertices();
  if (Q.size() != num_vertices) {
    throw std::runtime_error("Q.size() != num_vertices");
  }
  return laplacian_matrix_V_ * Q;
}

Samples3d MatrixMesh::apply_laplacian_matrix(Samples3d &Q) {
  int num_vertices = get_num_vertices();
  if (Q.rows() != num_vertices) {
    throw std::runtime_error("Q.size() != num_vertices");
  }
  return laplacian_matrix_V_ * Q;
}

Samples1d MatrixMesh::laplacian(Samples1d &Q) {
  if (construct_laplacian_matrix_) {
    return apply_laplacian_matrix(Q);
  }
  switch (laplacian_type_) {
  case LaplacianType::COTAN:
    return cotan_laplacian(Q);
  case LaplacianType::GUCKENBERGER:
    return guckenberger_laplacian(Q);
  case LaplacianType::BELKIN:
    return belkin_laplacian(Q);
  }
  throw std::runtime_error("Unknown laplacian type");
}

Samples3d MatrixMesh::laplacian(Samples3d &Q) {
  if (construct_laplacian_matrix_) {
    return apply_laplacian_matrix(Q);
  }
  switch (laplacian_type_) {
  case LaplacianType::COTAN:
    return cotan_laplacian(Q);
  case LaplacianType::GUCKENBERGER:
    return guckenberger_laplacian(Q);
  case LaplacianType::BELKIN:
    return belkin_laplacian(Q);
  }
  throw std::runtime_error("Unknown laplacian type");
}

void MatrixMesh::update_laplacian_matrix() {
  if (!construct_laplacian_matrix_) {
    return;
  }
  switch (laplacian_type_) {
  case LaplacianType::COTAN:
    update_laplacian_matrix_cotan();
    return;
  case LaplacianType::BELKIN:
    update_laplacian_matrix_belkin();
    return;
  case LaplacianType::GUCKENBERGER:
    std::runtime_error(
        "update_laplacian_matrix: Laplacian type GUCKENBERGER not implemented");
  }
  throw std::runtime_error("update_laplacian_matrix: Unknown laplacian type");
}

void MatrixMesh::update_laplacian_matrix_belkin() {
  int num_vertices = get_num_vertices();
  int num_faces = get_num_faces();
  if (laplacian_matrix_V_.rows() != num_vertices ||
      laplacian_matrix_V_.cols() != num_vertices) {
    laplacian_matrix_V_.resize(num_vertices,
                               num_vertices); // this also initializes to zero
  } else {
    laplacian_matrix_V_.setZero();
  }

  laplacian_matrix_V_.reserve(num_vertices * num_vertices);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      laplacian_matrix_V;
  laplacian_matrix_V.resize(num_vertices, num_vertices);
  laplacian_matrix_V.setZero();

  for (int ix{0}; ix < get_num_vertices(); ix++) {
    Vec3d x = xyz_coord_V_.row(ix);
    double wii = 0.0;
    for (int f{0}; f < get_num_faces(); f++) {
      double Af = area_F_[f];
      for (int iy : V_cycle_F_.row(f)) {
        if (iy == ix) {
          continue;
        }
        Vec3d y = xyz_coord_V_.row(iy);
        double Ay = Af / 3;
        double wij = Ay * heat_parametrix2d(x, y, belkin_dt_) / belkin_dt_;
        // laplacian_matrix_V_.insert(ix, iy) = wij;
        wii -= wij;
        laplacian_matrix_V(ix, iy) += wij;
      }
    }
    // laplacian_matrix_V_.insert(ix, ix) = wii;
    laplacian_matrix_V(ix, ix) = wii;
  }
  laplacian_matrix_V_ = laplacian_matrix_V.sparseView();
}

void MatrixMesh::update_laplacian_matrix_cotan() {
  int num_vertices = get_num_vertices();
  if (laplacian_matrix_V_.rows() != num_vertices ||
      laplacian_matrix_V_.cols() != num_vertices) {
    laplacian_matrix_V_.resize(num_vertices,
                               num_vertices); // this also initializes to zero
  } else {
    laplacian_matrix_V_.setZero();
  }

  laplacian_matrix_V_.reserve(6 * num_vertices);

  for (int vi = 0; vi < get_num_vertices(); vi++) {
    double Atot = 0.0;
    double wii = 0.0;

    for (auto hij : generate_H_out_v_clockwise(vi)) {
      int vj = v_head_h(hij);

      int hijm1 = h_next_h(h_twin_h(hij));
      int hijp1 = h_twin_h(h_next_h(h_next_h(hij)));
      int hjjm1 = h_twin_h(h_next_h(h_next_h(h_twin_h(hij))));
      int hjjp1 = h_next_h(hij);

      int eijm1 = e_undirected_h(hijm1);
      int eijp1 = e_undirected_h(hijp1);
      int ejjm1 = e_undirected_h(hjjm1);
      int ejjp1 = e_undirected_h(hjjp1);
      int eij = e_undirected_h(hij);

      double Lijm1 = length_E_[eijm1];
      double Ljjm1 = length_E_[ejjm1];
      double Lijp1 = length_E_[eijp1];
      double Ljjp1 = length_E_[ejjp1];
      double Lij = length_E_[eij];

      double thetajm1 = math::heron_angle(Lijm1, Ljjm1, Lij);
      double thetajp1 = math::heron_angle(Lijp1, Ljjp1, Lij);

      double cot_thetam = 1.0 / std::tan(thetajm1);
      double cot_thetap = 1.0 / std::tan(thetajp1);

      Atot += math::POW2(Lij) * (cot_thetam + cot_thetap) / 8;
      double wij = (cot_thetam + cot_thetap) / 2;
      laplacian_matrix_V_.insert(vi, vj) = wij;
      wii -= wij;
    }
    laplacian_matrix_V_.insert(vi, vi) = wii;
    laplacian_matrix_V_.row(vi) /= Atot;
  }
}

void MatrixMesh::update_mean_curvature() {
  int num_vertices = get_num_vertices();
  mean_curvature_V_.resize(num_vertices);
  lap_mean_curvature_V_.resize(num_vertices);
  mcvec_V_ = laplacian(xyz_coord_V_);
  for (int v = 0; v < num_vertices; v++) {
    Vec3d mcvec = mcvec_V_.row(v);
    int h = h_out_v(v);
    if (some_negative_boundary_contains_h(h)) {
      h = h_rotcw_h(h);
    }
    int f = f_left_h(h);
    Vec3d n = normal_F_.row(f);
    double mcvec_sign = math::sign(math::dot(mcvec, n));
    mean_curvature_V_[v] = mcvec_sign * math::L2norm(mcvec) / 2;
    //
    // mean_curvature_V_[v] = mcvec_V_.row(v).dot(normal_V_.row(v)) / 2;
  }
  // mean_curvature_V_ =
  //     0.5 * (mcvec_V_.array() * normal_V_.array()).rowwise().sum().matrix();
  lap_mean_curvature_V_ = laplacian(mean_curvature_V_);
}

////////////////////////////////////////
// Stuff that uses integration_patch_ //
////////////////////////////////////////

// void MatrixMesh::reset_integration_patch() {
//   integration_patch_.uncolor_faces();
//   integration_patch_.clear();
// }
//
// Samples1d MatrixMesh::adaptive_belkin_laplacian(Samples1d &phi) {
//   int num_faces = V_cycle_F_.rows();
//   int num_vertices = xyz_coord_V_.rows();
//   Samples1d lap_phi(num_vertices);
//   // Eigen::MatrixXd lap_phi(phi);
//   int n_ring = 0;
//   // lap_phi.setZero();
//   // printf("  num_vertices = %d\n", num_vertices);
//   for (int ix = 0; ix < num_vertices; ix++) {
//     // belkin_dt_ = area_V_[ix];
//     lap_phi[ix] = 0.0;
//     Vec3d x = xyz_coord_v(ix);
//     double phix = phi[ix];
//     reset_integration_patch();
//     integration_patch_.add_neighborhood_v(ix);
//     n_ring += 1;
//     // SimplicialSet F = integration_patch_.F_;
//     lap_phi[ix] = 0.0;
//     double lap_phi0 = 0.0;
//     // printf("   ix = %d\n", ix);
//     for (int f : integration_patch_.F_) {
//       for (int iy : V_cycle_F_.row(f)) {
//         Vec3d y = xyz_coord_v(iy);
//         double phiy = phi[iy];
//         double A = area_F_[f];
//         lap_phi[ix] += (A / (3 * belkin_dt_)) *
//                        heat_parametrix2d(x, y, belkin_dt_) * (phiy - phix);
//       }
//     }
//
//     // printf("    lap_phi[ix] = %.10f\n", lap_phi[ix]);
//     do {
//       lap_phi0 = lap_phi[ix];
//       integration_patch_.expand_by_one_ring();
//       n_ring += 1;
//       // F = integration_patch_.newF_;
//       for (int f : integration_patch_.newF_) {
//         for (int iy : V_cycle_F_.row(f)) {
//           Vec3d y = xyz_coord_V_.row(iy);
//           double phiy = phi[iy];
//           double A = area_F_[f];
//           lap_phi[ix] += (A / (3 * belkin_dt_)) *
//                          heat_parametrix2d(x, y, belkin_dt_) * (phiy - phix);
//         }
//       }
//
//     } while (std::abs(lap_phi[ix] - lap_phi0) >=
//                  belkin_atol_ + belkin_rtol_ * std::abs(lap_phi0) ||
//              n_ring < belkin_min_ring_);
//   }
//
//   return lap_phi;
// }
//
// Samples3d MatrixMesh::adaptive_belkin_laplacian(Samples3d &phi) {
//
//   int num_rows = phi.rows();
//   int num_cols = phi.cols();
//   // if (num_cols == 3) {
//   //   Eigen::RowVector3d lap_phi0;
//   // }
//   int num_vertices = xyz_coord_V_.rows();
//
//   Eigen::MatrixXd lap_phi(phi);
//   lap_phi.setZero();
//   double delta;
//   double mag;
//   int n_ring = 0;
//   for (int ix = 0; ix < num_vertices; ix++) {
//     // belkin_dt_ = area_V_[ix];
//     Vec3d x = xyz_coord_V_.row(ix);
//     Eigen::RowVector3d phix = phi.row(ix);
//     reset_integration_patch();
//     integration_patch_.add_neighborhood_v(ix);
//     n_ring += 1;
//     SimplicialSet F = integration_patch_.F_;
//     for (int f : F) {
//       for (int iy : V_cycle_F_.row(f)) {
//         Vec3d y = xyz_coord_V_.row(iy);
//         Eigen::RowVector3d phiy = phi.row(iy);
//         double A = area_F_[f];
//         lap_phi.row(ix) += (A / (3 * belkin_dt_)) *
//                            heat_parametrix2d(x, y, belkin_dt_) * (phiy -
//                            phix);
//       }
//     }
//
//     do {
//       Eigen::RowVector3d lap_phi0 = lap_phi.row(ix);
//       integration_patch_.expand_by_one_ring();
//       n_ring += 1;
//       F = integration_patch_.newF_;
//       for (int f : F) {
//         for (int iy : V_cycle_F_.row(f)) {
//           Vec3d y = xyz_coord_V_.row(iy);
//           Eigen::RowVector3d phiy = phi.row(iy);
//           double A = area_F_[f];
//           lap_phi.row(ix) += (A / (3 * belkin_dt_)) *
//                              heat_parametrix2d(x, y, belkin_dt_) *
//                              (phiy - phix);
//         }
//       }
//       delta = (lap_phi.row(ix) - lap_phi0).norm();
//       mag = lap_phi0.norm();
//     } while (delta >= belkin_atol_ + belkin_rtol_ * mag ||
//              n_ring < belkin_min_ring_);
//   }
//
//   return lap_phi;
// }
//
// template <typename Samples>
// Samples MatrixMesh::adaptive_guckenberger_laplacian(Samples &phi) {
//
//   auto num_vertices = get_num_vertices();
//   Eigen::MatrixXd lap_phi(phi);
//   lap_phi.setZero();
//
//   for (int ix{0}; ix < num_vertices; ix++) {
//     // printf("  ix = %d\n", ix);
//     Vec3d x = xyz_coord_V_.row(ix);
//     double heat_dt = heat_dt_multiple_ * area_V_[ix];
//
//     reset_integration_patch();
//     // printf("  add_vertex\n");
//     integration_patch_.add_vertex(ix);
//     double delta = std::numeric_limits<double>::infinity();
//     double mag = 0;
//     // printf("  delta = %.10f\n", delta);
//     while (delta >= belkin_atol_ + belkin_rtol_ * mag) {
//       Eigen::RowVectorXd lap_phi0 = lap_phi.row(ix);
//       // printf("  expand_by_one_ring\n");
//       integration_patch_.expand_by_one_ring();
//       for (int f : integration_patch_.newF_) {
//         for (int iy : V_cycle_F_.row(f)) {
//           Vec3d y = xyz_coord_V_.row(iy);
//           double A = area_F_[f];
//           lap_phi.row(ix) += (A / (3 * heat_dt)) *
//                              heat_parametrix2d(x, y, heat_dt) *
//                              (phi.row(iy) - phi.row(ix));
//         }
//       }
//       delta = (lap_phi.row(ix) - lap_phi0).norm();
//       mag = lap_phi0.norm();
//     }
//   }
//
//   return lap_phi;
// }
//
// template Samples1d MatrixMesh::adaptive_guckenberger_laplacian(Samples1d
// &phi); template Samples3d
// MatrixMesh::adaptive_guckenberger_laplacian(Samples3d &phi);
//
// template <typename Samples> Samples MatrixMesh::heat_laplacian(Samples &phi)
// {
//
//   auto num_vertices = get_num_vertices();
//   Eigen::MatrixXd lap_phi(phi);
//   lap_phi.setZero();
//   for (int ix{0}; ix < num_vertices; ix++) {
//     Vec3d x = xyz_coord_V_.row(ix);
//     double heat_dt = heat_dt_multiple_ * area_V_[ix];
//     integration_patch_ = Patch::from_seed_vertex(this, 0);
//     reset_integration_patch();
//     ////////////////////////////////////////////////
//     // MatrixMesh *base = static_cast<MatrixMesh *>(this);
//
//     // std::cout << "before add_vertex\n";
//     // std::cout << "  MatrixMesh*        = " << base << '\n';
//     // std::cout << "  &h_out_V_          = " << &base->h_out_V_ << '\n';
//     // std::cout << "  h_out_V_.data()    = " << base->h_out_V_.data() <<
//     '\n';
//     // std::cout << "  h_out_V_ rows/cols = " << base->h_out_V_.rows() << " "
//     //           << base->h_out_V_.cols() << '\n';
//     // std::cout << "  h_out_V_(0,0)      = " << base->h_out_V_(0, 0) <<
//     '\n';
//
//     /////////////////////////////////////////////////
//     integration_patch_.add_vertex(ix);
//     double delta = std::numeric_limits<double>::infinity();
//     double mag = 0;
//     while (delta >= belkin_atol_ + belkin_rtol_ * mag) {
//       Eigen::RowVectorXd lap_phi0 = lap_phi.row(ix);
//       integration_patch_.expand_by_one_ring();
//       for (int iy : integration_patch_.newV_) {
//         Vec3d y = xyz_coord_v(iy);
//         double A = area_V_[iy];
//         lap_phi.row(ix) += (A / heat_dt) * heat_parametrix2d(x, y, heat_dt) *
//                            (phi.row(iy) - phi.row(ix));
//       }
//       delta = (lap_phi.row(ix) - lap_phi0).norm();
//       mag = lap_phi0.norm();
//     }
//   }
//   return lap_phi;
// }
//
// template Samples1d MatrixMesh::heat_laplacian(Samples1d &phi);
// template Samples3d MatrixMesh::heat_laplacian(Samples3d &phi);
//
// template <typename Samples>
// Samples MatrixMesh::higher_order_quad_heat_laplacian(Samples &phi) {
//
//   auto num_vertices = get_num_vertices();
//   auto num_quad = dimensionless_quad_weight_Q_.size();
//   Eigen::MatrixXd lap_phi(phi);
//   lap_phi.setZero();
//
//   for (int ix{0}; ix < num_vertices; ix++) {
//     // printf("  ix = %d\n", ix);
//     Vec3d x = xyz_coord_V_.row(ix);
//     double heat_dt = heat_dt_multiple_ * area_V_[ix];
//
//     reset_integration_patch();
//     // printf("  add_vertex\n");
//     integration_patch_.add_vertex(ix);
//     double delta = std::numeric_limits<double>::infinity();
//     double mag = 0;
//     // printf("  delta = %.10f\n", delta);
//     while (delta >= belkin_atol_ + belkin_rtol_ * mag) {
//       Eigen::RowVectorXd lap_phi0 = lap_phi.row(ix);
//       // printf("  expand_by_one_ring\n");
//       integration_patch_.expand_by_one_ring();
//       for (int f : integration_patch_.newF_) {
//         int iy0 = V_cycle_F_(f, 0);
//         int iy1 = V_cycle_F_(f, 1);
//         int iy2 = V_cycle_F_(f, 2);
//         Eigen::RowVectorXd phiy0 = phi.row(iy0);
//         Eigen::RowVectorXd phiy1 = phi.row(iy1);
//         Eigen::RowVectorXd phiy2 = phi.row(iy2);
//         for (int q{0}; q < num_quad; q++) {
//           double s0 = bary_coord_Q_(q, 0);
//           double s1 = bary_coord_Q_(q, 1);
//           double s2 = bary_coord_Q_(q, 2);
//           Eigen::RowVectorXd phiy = s0 * phiy0 + s1 * phiy1 + s2 * phiy2;
//           Vec3d y = xyz_coord_fq(f, q);
//           double w = quad_weight_fq(f, q);
//
//           lap_phi.row(ix) += (w / heat_dt) * heat_parametrix2d(x, y, heat_dt)
//           *
//                              (phiy - phi.row(ix));
//         }
//       }
//       delta = (lap_phi.row(ix) - lap_phi0).norm();
//       mag = lap_phi0.norm();
//     }
//   }
//   return lap_phi;
// }
//
// template Samples1d MatrixMesh::higher_order_quad_heat_laplacian(Samples1d
// &phi); template Samples3d
// MatrixMesh::higher_order_quad_heat_laplacian(Samples3d &phi);
//
// void MatrixMesh::update_laplacian_matrix_heat() {
//   // printf("MatrixMesh::update_laplacian_matrix_heat\n");
//   int num_vertices = get_num_vertices();
//   int num_faces = get_num_faces();
//   if (laplacian_matrix_V_.rows() != num_vertices ||
//       laplacian_matrix_V_.cols() != num_vertices) {
//     laplacian_matrix_V_.resize(num_vertices,
//                                num_vertices); // this also initializes to
//                                zero
//   } else {
//     laplacian_matrix_V_.setZero();
//   }
//   int nnz = .25 * num_vertices * num_vertices;
//   laplacian_matrix_V_.reserve(nnz);
//   for (int ix{0}; ix < get_num_vertices(); ix++) {
//     Vec3d x = xyz_coord_V_.row(ix);
//     double Ax = area_V_[ix];
//     double heat_dt = heat_dt_multiple_ * Ax;
//
//     reset_integration_patch();
//     integration_patch_.add_vertex(ix);
//
//     std::vector<double> Li{0.0};
//     std::vector<int> col_indices{ix};
//
//     double norm_dLi = std::numeric_limits<double>::infinity();
//     double norm_Li = 0;
//     int iter = 0;
//     while (norm_dLi >= belkin_atol_ + belkin_rtol_ * norm_Li) {
//       integration_patch_.expand_by_one_ring();
//       if (integration_patch_.newV_.size() == 0) {
//         break;
//       }
//       std::vector<double> dLi{0.0};
//       std::vector<int> dcol_indices{ix};
//       for (int iy : integration_patch_.newV_) {
//         Vec3d y = xyz_coord_V_.row(iy);
//         double Ay = area_V_[iy];
//         dLi.push_back((Ay / heat_dt) * heat_parametrix2d(x, y, heat_dt));
//         dcol_indices.push_back(iy);
//         dLi[0] -= dLi.back();
//       }
//       norm_dLi = math::L2norm(dLi);
//       Li[0] += dLi[0];
//       for (int _ = 1; _ < dLi.size(); _++) {
//         Li.push_back(dLi[_]);
//         col_indices.push_back(dcol_indices[_]);
//       }
//       norm_Li = math::L2norm(Li);
//       iter++;
//     }
//     std::vector<int> argsort_col_indices = math::argsort(col_indices);
//     for (int _ = 0; _ < col_indices.size(); _++) {
//       laplacian_matrix_V_.coeffRef(ix, col_indices[argsort_col_indices[_]]) =
//           Li[argsort_col_indices[_]];
//       // laplacian_matrix_V_.coeffRef(ix, col_indices[_]) = Li[_];
//     }
//   }
// }
//
// void MatrixMesh::update_laplacian_matrix_adaptive_belkin() {
//   int num_vertices = get_num_vertices();
//   int num_faces = get_num_faces();
//   if (laplacian_matrix_V_.rows() != num_vertices ||
//       laplacian_matrix_V_.cols() != num_vertices) {
//     laplacian_matrix_V_.resize(num_vertices,
//                                num_vertices); // this also initializes to
//                                zero
//   } else {
//     laplacian_matrix_V_.setZero();
//   }
//   int nnz = .25 * num_vertices * num_vertices;
//   laplacian_matrix_V_.reserve(nnz);
//   for (int ix{0}; ix < get_num_vertices(); ix++) {
//     Vec3d x = xyz_coord_V_.row(ix);
//     double Ax = area_V_[ix];
//
//     reset_integration_patch();
//     integration_patch_.add_vertex(ix);
//
//     std::vector<double> Li{0.0};
//     std::vector<int> col_indices{ix};
//
//     double norm_dLi = std::numeric_limits<double>::infinity();
//     double norm_Li = 0;
//     int iter = 0;
//     while (norm_dLi >= belkin_atol_ + belkin_rtol_ * norm_Li) {
//       integration_patch_.expand_by_one_ring();
//       if (integration_patch_.newV_.size() == 0) {
//         break;
//       }
//       std::vector<double> dLi{0.0};
//       std::vector<int> dcol_indices{ix};
//       for (int iy : integration_patch_.newV_) {
//         Vec3d y = xyz_coord_V_.row(iy);
//         double Ay = area_V_[iy];
//         dLi.push_back((Ay / belkin_dt_) * heat_parametrix2d(x, y,
//         belkin_dt_)); dcol_indices.push_back(iy); dLi[0] -= dLi.back();
//       }
//       norm_dLi = math::L2norm(dLi);
//       Li[0] += dLi[0];
//       for (int _ = 1; _ < dLi.size(); _++) {
//         Li.push_back(dLi[_]);
//         col_indices.push_back(dcol_indices[_]);
//       }
//       norm_Li = math::L2norm(Li);
//       iter++;
//     }
//     std::vector<int> argsort_col_indices = math::argsort(col_indices);
//     for (int _ = 0; _ < col_indices.size(); _++) {
//       laplacian_matrix_V_.coeffRef(ix, col_indices[argsort_col_indices[_]]) =
//           Li[argsort_col_indices[_]];
//       // laplacian_matrix_V_.coeffRef(ix, col_indices[_]) = Li[_];
//     }
//   }
// }

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// To be deprecated ///////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////

// void MatrixMesh::init_mesh() {
//   load_ply();
//   update_simplices_from_he();
//   update_boundary_cycles();
//   update_mesh_geometric_data_E();
//   update_mesh_geometric_data_F();
//   update_mesh_geometric_data_V();
//   update_mesh_volume();
//   set_visual_defaults();
//   update_mesh_visuals();
//   init_quad_points_and_weights();
// }

// void MatrixMesh::load_ply() {
//   if (ply_path_.empty()) {
//     printf("MatrixMesh::load_ply: ply_path_ is empty");
//     return;
//   }
//   HalfEdgeTuple het = mesh_io::load_he_samples_from_ply(ply_path_);
//   xyz_coord_V_ = std::get<0>(het);
//   h_out_V_ = std::get<1>(het);
//   v_origin_H_ = std::get<2>(het);
//   h_next_H_ = std::get<3>(het);
//   h_twin_H_ = std::get<4>(het);
//   f_left_H_ = std::get<5>(het);
//   h_right_F_ = std::get<6>(het);
//   h_negative_B_ = std::get<7>(het);
//
//   xyz_coord_V_was_set_ = true;
//   h_out_V_was_set_ = true;
//   v_origin_H_was_set_ = true;
//   h_next_H_was_set_ = true;
//   h_twin_H_was_set_ = true;
//   f_left_H_was_set_ = true;
//   h_right_F_was_set_ = true;
//   h_negative_B_was_set_ = true;
// }

// void MatrixMesh::update_simplices_from_he() {
// // replaced by init_core_data_from_minimal_half_edge_data
//   size_t num_vertices = get_num_vertices();
//   size_t num_faces = get_num_faces();
//   size_t num_edges = get_num_edges();
//   size_t num_half_edges = get_num_half_edges();
//   size_t num_boundaries = get_num_boundaries();
//
//   V_cycle_E_.resize(num_edges, 2);
//   V_cycle_F_.resize(num_faces, 3);
//   e_undirected_H_.resize(num_half_edges);
//   h_directed_E_.resize(num_edges);
//   std::unordered_set<size_t> Hmin;
//   for (int f{0}; f < num_faces; ++f) {
//     int h0 = h_right_F_[f];
//     int h1 = h_next_H_[h0];
//     int h2 = h_next_H_[h1];
//     V_cycle_F_.row(f) << v_origin_H_[h0], v_origin_H_[h1], v_origin_H_[h2];
//
//     size_t h = h0;
//     size_t h_start = h;
//     do {
//       size_t ht = h_twin_h(h);
//       size_t hn = h_next_h(h);
//       size_t v = v_origin_h(h);
//       size_t hmin = std::min(h, ht);
//
//       if (Hmin.find(hmin) == Hmin.end()) {
//         size_t e = Hmin.size();
//         size_t vt = v_origin_h(ht);
//         // if (v < vt) {
//         //   V_cycle_E_.row(e) << v, vt;
//         // } else {
//         //   V_cycle_E_.row(e) << vt, v;
//         // }
//         V_cycle_E_.row(e) << v, vt;
//         e_undirected_H_[h] = e;
//         e_undirected_H_[ht] = e;
//         h_directed_E_[e] = h;
//
//         Hmin.insert(hmin);
//       }
//       h = hn;
//     } while (h != h_start);
//   }
// }

// void MatrixMesh::init_from_he_mats() {
//   // update_simplices_from_he();
//   init_core_data_from_minimal_half_edge_data();
//   // update_boundary_cycles();
//   // update_mesh_geometric_data_E();
//   // update_mesh_geometric_data_F();
//   // update_mesh_geometric_data_V();
//   // update_mesh_volume();
//   // set_visual_defaults();
//   // update_mesh_visuals();
//   // init_quad_points_and_weights();
//   init_cache_from_core_data();
// }

// void MatrixMesh::init_from_ply() {
//   init_core_data_from_ply();
//   init_cache_from_core_data();
//   // integration_patch_ = Patch(this);
// }

// void MatrixMesh::update_mat_v(int v, const std::optional<Vec3d> &xyz_coord,
//                               const std::optional<int> &h_out) {
//   if (xyz_coord.has_value()) {
//     xyz_coord_V_.row(v) = xyz_coord.value();
//   }
//   if (h_out.has_value()) {
//     h_out_V_(v) = h_out.value();
//   }
// }
//
// void MatrixMesh::update_mat_h(int h, const std::optional<int> &v_origin,
//                               const std::optional<int> &h_next,
//                               const std::optional<int> &h_twin,
//                               const std::optional<int> &f_left) {
//   if (v_origin.has_value()) {
//     v_origin_H_(h) = v_origin.value();
//   }
//   if (h_next.has_value()) {
//     h_next_H_(h) = h_next.value();
//   }
//   if (h_twin.has_value()) {
//     h_twin_H_(h) = h_twin.value();
//   }
//   if (f_left.has_value()) {
//     f_left_H_(h) = f_left.value();
//   }
// }
//
// void MatrixMesh::update_mat_f(int f, const std::optional<int> &h_left) {
//   if (h_left.has_value()) {
//     f_left_H_(f) = h_left.value();
//   }
// }

// void MatrixMesh::update_he_from_vf() { // replace with
// init_core_data_from_V_cycle_F()
//   // auto [xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H,
//   //       h_right_F, h_negative_B] =
//   //     mesh_io::vf_samples_to_he_samples(xyz_coord_V_, V_cycle_F_);
//   // xyz_coord_V_ = xyz_coord_V;
//   // h_out_V_ = h_out_V;
//   // v_origin_H_ = v_origin_H;
//   // h_next_H_ = h_next_H;
//   // h_twin_H_ = h_twin_H;
//   // f_left_H_ = f_left_H;
//   // h_right_F_ = h_right_F;
//   // h_negative_B_ = h_negative_B;
//   // Samples3d xyz_coord_V = xyz_coord_V_;
//   // Samples3i V_cycle_F = V_cycle_F_;
//   HalfEdgeTuple he_tuple =
//       mesh_io::vf_samples_to_he_samples(xyz_coord_V_, V_cycle_F_);
//   // xyz_coord_V_ = std::get<0>(he_tuple);
//   h_out_V_ = std::get<1>(he_tuple);
//   v_origin_H_ = std::get<2>(he_tuple);
//   h_next_H_ = std::get<3>(he_tuple);
//   h_twin_H_ = std::get<4>(he_tuple);
//   f_left_H_ = std::get<5>(he_tuple);
//   h_right_F_ = std::get<6>(he_tuple);
//   h_negative_B_ = std::get<7>(he_tuple);
//   update_edges_from_he_mats();
// }

} // namespace meshbrane
