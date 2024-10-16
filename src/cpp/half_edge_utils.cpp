// half_edge_utils.cpp
#include <half_edge_utils.hpp> // INT_TYPE, Samplesi,..., Samples3d
#include <pybind11/eigen.h>    // pybind11 Eigen<->Numpy conversion
#include <pybind11/pybind11.h> // PYBIND11_MODULE
#include <set>                 // std::set
#include <vector>              // std::vector

namespace py = pybind11;
/**
 * @brief Get the index of twin half-edge
 */
INT_TYPE get_halfedge_index_of_twin(const Samples2i &H, const INT_TYPE &h) {
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
 * @brief Compute half-edge data from vertices of faces
 */
std::tuple<Samples3d, Samplesi, Samplesi, Samplesi, Samplesi, Samplesi,
           Samplesi, Samplesi>
vf_samples_to_he_samples(const Samples3d &xyz_coord_V,
                         const Samples3i &V_of_F) {

  auto Nv = xyz_coord_V.rows();
  auto Nf = V_of_F.rows();
  // number of interior half-edges + positive boundary half-edges
  auto Nh0 = 3 * Nf;

  Samples2i H0 = Samples2i::Zero(Nh0, 2);

  Samplesi h_out_V = Samplesi::Constant(Nv, -1);
  Samplesi v_origin_H = Samplesi::Zero(Nh0);
  Samplesi h_next_H = Samplesi::Constant(Nh0, -1);
  Samplesi h_twin_H = Samplesi::Constant(Nh0, -1);
  Samplesi f_left_H = Samplesi::Zero(Nh0);
  Samplesi h_bound_F = Samplesi::Zero(Nf);
  Samplesi h_right_B;
  std::vector<INT_TYPE> H_boundary_plus;
  //
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
      if (h_out_V[v0] == -1) {
        h_out_V[v0] = h;
      }
    }
  }

  // find positive boundary half-edges
  // assign twins for interior half-edges
  for (INT_TYPE h = 0; h < H0.rows(); ++h) {
    if (h_twin_H[h] != -1) {
      continue;
    }
    INT_TYPE h_twin = get_halfedge_index_of_twin(H0, h);
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
  std::set<INT_TYPE> H_boundary_minus;

  // define negative boundary half-edges
  // assign origins for negative boundary half-edges
  // assign twins for boundary half-edges
  // temporarily assign left face for boundary half-edges to -1
  for (INT_TYPE i = 0; i < Nh1; ++i) {
    INT_TYPE h = H_boundary_plus[i];
    INT_TYPE h_twin = Nh0 + i;
    // INT_TYPE v0 = H0(h, 0);
    INT_TYPE v1 = H0(h, 1);
    H_boundary_minus.insert(h_twin);
    v_origin_H[h_twin] = v1;
    h_twin_H[h] = h_twin;
    h_twin_H[h_twin] = h;
    f_left_H[h_twin] = -1;
  }

  // assign next for negative boundary half-edges
  // assign left face for negative boundary half-edges
  while (!H_boundary_minus.empty()) {
    INT_TYPE b = h_right_B.size();
    INT_TYPE h_right_b = *H_boundary_minus.begin();
    // H_boundary_minus.erase(h_right_b);
    f_left_H[h_right_b] = -(b + 1);
    h_right_B.conservativeResize(b + 1);
    h_right_B[b] = h_right_b; // Assign new value

    INT_TYPE h = h_right_b;
    // follow next cycle along boundary b until we get back to h=h_right_b
    do {
      INT_TYPE h_next = h_twin_H[h];
      // rotate ccw around origin of twin until we find next h on boundary b
      // erase h from H_boundary_minus
      while (H_boundary_minus.find(h_next) == H_boundary_minus.end()) {
        h_next = h_twin_H[h_next_H[h_next_H[h_next]]];
      }
      h_next_H[h] = h_next;
      h = h_next;
      H_boundary_minus.erase(h);
    } while (h != h_right_b);
  }

  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_bound_F, h_right_B);
}

////////////////////////////////////////////////////////////////
// Python bindings
////////////////////////////////////////////////////////////////

PYBIND11_MODULE(half_edge_utils, m) {
  m.doc() = "pybind11 half_edge_utils plugin"; // Optional module docstring
  m.def("get_halfedge_index_of_twin", &get_halfedge_index_of_twin,
        "A function to compute vertices of faces from half-edge data");
  m.def("vf_samples_to_he_samples", &vf_samples_to_he_samples,
        "A function to compute half-edge data from vertices of faces");
  // m.def("get_halfedge_index_of_twin", &get_halfedge_index_of_twin,
  //       "Get the index of the twin half-edge", py::arg("H"), py::arg("h"));
  // m.def(
  //     "vf_samples_to_he_samples",
  //     [](const Samples3d &xyz_coord_V, const Samples3i &V_of_F) {
  //       return vf_samples_to_he_samples(xyz_coord_V, V_of_F);
  //     },
  //     "A function to compute half-edge data from vertices of faces");
}
