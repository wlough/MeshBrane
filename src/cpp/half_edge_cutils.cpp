/**
 * @file half_edge_cutils.cpp
 */
#include <half_edge_cutils.hpp> // INT_TYPE, Samplesi,..., Samples3d
#include <pybind11/eigen.h>     // pybind11 Eigen<->Numpy conversion
#include <pybind11/pybind11.h>  // PYBIND11_MODULE
#include <tuple>                // std::tuple
#include <unordered_set>        // std::unordered_set
#include <vector>               // std::vector

namespace py = pybind11;
/**
 * @brief Get the index of twin half-edge
 *
 * @param H Nhx2 array of half-edges.
 */
INT_TYPE find_halfedge_index_of_twin(const Samples2i &H, const INT_TYPE &h) {
  auto v0 = H(h, 0);
  auto v1 = H(h, 1);
  for (INT_TYPE h_twin = 0; h_twin < H.rows(); ++h_twin) {
    if ((H(h_twin, 0) == v1) && (H(h_twin, 1) == v0)) {
      return h_twin; // Return the index of the twin edge
    }
  }
  return -1; // Return -1 if no twin edge is found
}

/**
 * @brief Convert vertex-face mesh data to half-edge mesh data.
 *
 * @param xyz_coord_V Nvx3 Eigen matrix of vertex Cartesian coordinates.
 * @param vvv_of_F Nfx3 Eigen matrix of vertex indices of faces.
 * @return A tuple containing:
 * - xyz_coord_V: Nvx3 Eigen matrix of vertex Cartesian coordinates.
 * - h_out_V: Eigen matrix where h_out_V[i] is some outgoing half-edge incident
 * on vertex i.
 * - v_origin_H: Eigen matrix where v_origin_H[j] is the vertex at the origin of
 * half-edge j.
 * - h_next_H: Eigen matrix where h_next_H[j] is the next half-edge after
 * half-edge j in the face cycle.
 * - h_twin_H: Eigen matrix where h_twin_H[j] is the half-edge antiparallel to
 * half-edge j.
 * - f_left_H: Eigen matrix where f_left_H[j] is the face to the left of
 * half-edge j, if j is in the interior or a positively oriented boundary of M,
 * or the boundary to the left of half-edge j, if j is in a negatively oriented
 * boundary.
 * - h_bound_F: Eigen matrix where h_bound_F[k] is some half-edge on the
 * boundary of face k.
 * - h_right_B: Eigen matrix where h_right_B[n] is the half-edge to the right of
 * boundary n.
 */
std::tuple<Samples3d, Samplesi, Samplesi, Samplesi, Samplesi, Samplesi,
           Samplesi, Samplesi>
vf_samples_to_he_samples(const Samples3d &xyz_coord_V,
                         const Samples3i &V_of_F) {

  auto Nv = xyz_coord_V.rows();
  auto Nf = V_of_F.rows();
  // num interior + num positive boundary half-edges
  auto Nh0 = 3 * Nf;
  Samples2i H0 = Samples2i(Nh0, 2);
  // half-edge samples
  // h_out=Nh0 if not assigned
  // h_twin=-1 if not assigned
  Samplesi h_out_V = Samplesi::Constant(Nv, Nh0);
  Samplesi v_origin_H = Samplesi(Nh0);
  Samplesi h_next_H = Samplesi(Nh0);
  Samplesi h_twin_H = Samplesi::Constant(Nh0, -1);
  Samplesi f_left_H = Samplesi(Nh0);
  Samplesi h_bound_F = Samplesi(Nf);
  Samplesi h_right_B;
  // assign h_out for vertices to be minimum of outgoing half-edge indices
  // assign v_origin/f_left/h_next for half-edges in H0
  // assign h_bound for faces
  for (INT_TYPE f = 0; f < Nf; ++f) {
    h_bound_F[f] = 3 * f;
    for (INT_TYPE i = 0; i < 3; ++i) {
      INT_TYPE h = 3 * f + i;
      INT_TYPE h_next = 3 * f + (i + 1) % 3;
      INT_TYPE v0 = V_of_F(f, i);
      INT_TYPE v1 = V_of_F(f, (i + 1) % 3);
      H0.row(h) << v0, v1;
      v_origin_H[h] = v0;
      f_left_H[h] = f;
      h_next_H[h] = h_next;
      // assign h_out for vertices if not already assigned
      // reassign if h is smaller than current h_out_V[v0]
      if (h_out_V[v0] > h) {
        h_out_V[v0] = h;
      }
    }
  }
  // Temporary containers for indices of +/- boundary half-edge
  std::vector<INT_TYPE> H_boundary_plus;
  std::unordered_set<INT_TYPE> H_boundary_minus;
  // find positive boundary half-edges
  // assign h_twin for interior half-edges
  for (INT_TYPE h = 0; h < H0.rows(); ++h) {
    // if h_twin_H[h] is already assigned, skip
    if (h_twin_H[h] != -1) {
      continue;
    }
    INT_TYPE h_twin = find_halfedge_index_of_twin(H0, h);
    if (h_twin == -1) {
      H_boundary_plus.push_back(h);
    } else {
      h_twin_H[h] = h_twin;
      h_twin_H[h_twin] = h;
    }
  }
  INT_TYPE Nh1 = H_boundary_plus.size();
  INT_TYPE Nh = Nh0 + Nh1;
  v_origin_H.conservativeResize(Nh);
  h_next_H.conservativeResize(Nh);
  h_twin_H.conservativeResize(Nh);
  f_left_H.conservativeResize(Nh);
  // define negative boundary half-edges
  // assign v_origin for negative boundary half-edges
  // assign h_twin for boundary half-edges
  for (INT_TYPE i = 0; i < Nh1; ++i) {
    INT_TYPE h = H_boundary_plus[i];
    INT_TYPE h_twin = Nh0 + i;
    // INT_TYPE v0 = H0(h, 0);
    INT_TYPE v1 = H0(h, 1);
    H_boundary_minus.insert(h_twin);
    v_origin_H[h_twin] = v1;
    h_twin_H[h] = h_twin;
    h_twin_H[h_twin] = h;
  }
  // enumerate boundaries b=0,1,...
  // assign h_right for boundaries
  // assign h_next for negative boundary half-edges
  // set f_left=-(b+1) for half-edges in boundary b
  while (!H_boundary_minus.empty()) {
    INT_TYPE b = h_right_B.size();
    INT_TYPE h_right_b = *H_boundary_minus.begin();
    h_right_B.conservativeResize(b + 1);
    h_right_B[b] = h_right_b; // Assign new value
    INT_TYPE h = h_right_b;
    // follow prev cycle along boundary b until we get back to h=h_right_b
    do {
      INT_TYPE h_prev = h_twin_H[h];
      // rotate cw around origin of h until we find h_prev in boundary b
      // erase h from H_boundary_minus
      while (H_boundary_minus.find(h_prev) == H_boundary_minus.end()) {
        h_prev = h_twin_H[h_next_H[h_prev]];
      }
      h_next_H[h_prev] = h;
      h = h_prev;
      H_boundary_minus.erase(h);
      f_left_H[h] = -(b + 1);
    } while (h != h_right_b);
  }
  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_bound_F, h_right_B);
}

////////////////////////////////////////////////////////////////
// Python bindings
////////////////////////////////////////////////////////////////

PYBIND11_MODULE(half_edge_cutils, m) {
  m.doc() = "pybind11 half_edge_cutils plugin"; // module docstring
  m.def("vf_samples_to_he_samples", &vf_samples_to_he_samples,
        "A function to compute half-edge data from vertices of faces");
}

// PYBIND11_MODULE(half_edge_utils, m) {
//   m.doc() = "pybind11 half_edge_utils plugin"; // Optional module docstring
//  m.def("find_halfedge_index_of_twin", &find_halfedge_index_of_twin,
//       "A function to compute vertices of faces from half-edge data");
// m.def("find_halfedge_index_of_twin", &find_halfedge_index_of_twin,
//       "Get the index of the twin half-edge", py::arg("H"), py::arg("h"));
// m.def(
//     "vf_samples_to_he_samples",
//     [](const Samples3d &xyz_coord_V, const Samples3i &V_of_F) {
//       return vf_samples_to_he_samples(xyz_coord_V, V_of_F);
//     },
//     "A function to compute half-edge data from vertices of faces");
// }
