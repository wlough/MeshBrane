
/**
 * @file mesh_helpers.cpp
 */
#include "meshbrane/mesh_helpers.hpp"

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

} // namespace meshbrane
