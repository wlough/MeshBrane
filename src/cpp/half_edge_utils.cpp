// half_edge_utils.cpp
#include <Eigen/Dense>         // for Eigen::Vector3d
#include <algorithm>           // for std::find
#include <array>               // for std::array
#include <pybind11/numpy.h>    // for py::array_t
#include <pybind11/pybind11.h> // for PYBIND11_MODULE
#include <pybind11/stl.h>      // for py::list
#include <set>                 // for std::set
#include <vector>              // for std::vector

namespace py = pybind11;

// V_of_F
// find_h_right_B
// vf_samples_to_he_samples
// he_samples_to_vf_samples
using INT_TYPE = std::int32_t;
using FLOAT_TYPE = double;
using PySamplesi = py::array_t<INT_TYPE>;
using PySamples2i = py::array_t<INT_TYPE>;
using PySamples3i = py::array_t<INT_TYPE>;
using PySamples3d = py::array_t<FLOAT_TYPE>;
using Samplesi = std::vector<INT_TYPE>;
using Samples2i = std::vector<std::array<INT_TYPE, 2>>;
using Samples3i = std::vector<std::array<INT_TYPE, 3>>;
using Samples3d = std::vector<Eigen::Vector3d>;
using VertexFaceSamples = std::pair<Samples3d, Samples3i>;
using HalfEdgeSamples = std::tuple<Samples3d, Samplesi, Samplesi, Samplesi,
                                   Samplesi, Samplesi, Samplesi, Samplesi>;

PySamples3i get_V_of_F(PySamples3d xyz_coord_V, PySamplesi h_out_V,
                       PySamplesi v_origin_H, PySamplesi h_next_H,
                       PySamplesi h_twin_H, PySamplesi f_left_H,
                       PySamplesi h_bound_F, PySamplesi h_right_B) {
  // Request buffer descriptors from the NumPy arrays
  auto buf_h_bound_F = h_bound_F.request();
  auto buf_v_origin_H = v_origin_H.request();
  auto buf_h_next_H = h_next_H.request();

  // Get the number of faces
  int Nf = buf_h_bound_F.shape[0];

  // Create an output array
  PySamples3i V_of_F({Nf, 3});
  auto buf_V_of_F = V_of_F.request();
  int *ptr_V_of_F = static_cast<int *>(buf_V_of_F.ptr);

  // Get pointers to the input arrays
  int *ptr_h_bound_F = static_cast<int *>(buf_h_bound_F.ptr);
  int *ptr_v_origin_H = static_cast<int *>(buf_v_origin_H.ptr);
  int *ptr_h_next_H = static_cast<int *>(buf_h_next_H.ptr);

  // Compute V_of_F
  for (int f = 0; f < Nf; ++f) {
    int h = ptr_h_bound_F[f];
    int h_start = h;
    int _v = 0;
    while (true) {
      ptr_V_of_F[f * 3 + _v] = ptr_v_origin_H[h];
      h = ptr_h_next_H[h];
      _v += 1;
      if (h == h_start) {
        break;
      }
    }
  }

  return V_of_F;
}

PySamplesi find_h_right_B(PySamples3d xyz_coord_V, PySamplesi h_out_V,
                          PySamplesi v_origin_H, PySamplesi h_next_H,
                          PySamplesi h_twin_H, PySamplesi f_left_H,
                          PySamplesi h_bound_F) {
  // Request buffer descriptors from the NumPy arrays
  auto buf_v_origin_H = v_origin_H.request();
  auto buf_h_next_H = h_next_H.request();
  auto buf_f_left_H = f_left_H.request();

  // Get pointers to the data
  //   INT_TYPE *ptr_v_origin_H = static_cast<INT_TYPE *>(buf_v_origin_H.ptr);
  INT_TYPE *ptr_h_next_H = static_cast<INT_TYPE *>(buf_h_next_H.ptr);
  INT_TYPE *ptr_f_left_H = static_cast<INT_TYPE *>(buf_f_left_H.ptr);

  // Get the number of half-edges
  size_t Nhedges = buf_v_origin_H.shape[0];

  // Find and enumerate boundaries -1, -2, ...
  std::set<INT_TYPE> H_need2visit;
  for (size_t h = 0; h < Nhedges; ++h) {
    if (ptr_f_left_H[h] < 0) {
      H_need2visit.insert(static_cast<INT_TYPE>(h));
    }
  }

  std::vector<INT_TYPE> _h_right_B;
  while (!H_need2visit.empty()) {
    size_t b = _h_right_B.size();
    INT_TYPE h_start = *H_need2visit.begin();
    H_need2visit.erase(H_need2visit.begin());
    ptr_f_left_H[h_start] = -(b + 1);
    INT_TYPE h = static_cast<INT_TYPE>(ptr_h_next_H[h_start]);
    _h_right_B.push_back(h);
    while (h != h_start) {
      H_need2visit.erase(h);
      ptr_f_left_H[h] = -(b + 1);
      h = ptr_h_next_H[h];
    }
  }

  // Convert the result to a NumPy array
  PySamplesi h_right_B(_h_right_B.size());
  auto buf_h_right_B = h_right_B.request();
  INT_TYPE *ptr_h_right_B = static_cast<INT_TYPE *>(buf_h_right_B.ptr);
  std::copy(_h_right_B.begin(), _h_right_B.end(), ptr_h_right_B);

  return h_right_B;
}

INT_TYPE get_index_of_twin(const Samples2i &H, const INT_TYPE &h) {
  auto v0 = H[h][0];
  auto v1 = H[h][1];
  for (int32_t h_twin = 0; h_twin < H.size(); ++h_twin) {
    // Check if the edge in E is a twin of e
    if ((H[h_twin][0] == v1) && (H[h_twin][1] == v0)) {
      return h_twin; // Return the index of the twin edge
    }
  }
  return -1; // Return -1 if no twin edge is found
}

HalfEdgeSamples vf_samples_to_he_samples(Samples3d xyz_coord_V,
                                         Samples3i V_of_F) {
  size_t Nfaces = V_of_F.size();
  size_t Nvertices = xyz_coord_V.size();
  size_t _Nhedges = 3 * Nfaces * 2;

  Samples2i _H(_Nhedges);
  Samplesi h_out_V(Nvertices);
  Samplesi _v_origin_H(_Nhedges);
  Samplesi _h_next_H(_Nhedges);
  Samplesi _f_left_H(_Nhedges);
  Samplesi h_bound_F(Nfaces);

  std::fill(h_out_V.begin(), h_out_V.end(), -1);
  std::fill(_h_next_H.begin(), _h_next_H.end(), -1);

  for (size_t f = 0; f < Nfaces; ++f) {
    h_bound_F[f] = 3 * f;
    for (size_t i = 0; i < 3; ++i) {
      size_t h = 3 * f + i;
      size_t h_next = 3 * f + (i + 1) % 3;
      INT_TYPE v0 = V_of_F[f][i];
      INT_TYPE v1 = V_of_F[f][(i + 1) % 3];
      _H[h] = {v0, v1};
      _v_origin_H[h] = v0;
      _f_left_H[h] = f;
      _h_next_H[h] = h_next;
      if (h_out_V[v0] == -1) {
        h_out_V[v0] = h;
      }
    }
  }

  size_t h_count = 3 * Nfaces;
  std::set<INT_TYPE> need_twins;
  for (size_t i = 0; i < h_count; ++i) {
    need_twins.insert(i);
  }

  std::set<INT_TYPE> need_next;
  Samplesi _h_twin_H(_Nhedges);
  std::fill(_h_twin_H.begin(), _h_twin_H.end(), -2);

  while (!need_twins.empty()) {
    INT_TYPE h = *need_twins.begin();
    need_twins.erase(need_twins.begin());
    if (_h_twin_H[h] == -2) {
      INT_TYPE h_twin = get_index_of_twin(_H, h);
      if (h_twin == -1) {
        h_twin = h_count++;
        INT_TYPE v0 = _H[h][0];
        INT_TYPE v1 = _H[h][1];
        _H[h_twin] = {v1, v0};
        _v_origin_H[h_twin] = v1;
        need_next.insert(h_twin);
        _h_twin_H[h] = h_twin;
        _h_twin_H[h_twin] = h;
        _f_left_H[h_twin] = -1;
      } else {
        _h_twin_H[h] = h_twin;
        _h_twin_H[h_twin] = h;
        need_twins.erase(h_twin);
      }
    }
  }

  size_t Nhedges = h_count;
  Samplesi v_origin_H(Nhedges);
  Samplesi h_next_H(Nhedges);
  Samplesi f_left_H(Nhedges);
  Samplesi h_twin_H(Nhedges);

  std::copy(_v_origin_H.begin(), _v_origin_H.end(), v_origin_H.begin());
  std::copy(_h_next_H.begin(), _h_next_H.end(), h_next_H.begin());
  std::copy(_f_left_H.begin(), _f_left_H.end(), f_left_H.begin());
  std::copy(_h_twin_H.begin(), _h_twin_H.end(), h_twin_H.begin());

  while (!need_next.empty()) {
    INT_TYPE h = *need_next.begin();
    need_next.erase(need_next.begin());
    INT_TYPE h_next = h_twin_H[h];
    while (f_left_H[h_next] != -1) {
      h_next = h_twin_H[h_next];
    }
    h_next_H[h] = h_next;
  }

  std::set<INT_TYPE> H_need2visit;
  for (size_t h = 0; h < Nhedges; ++h) {
    if (f_left_H[h] == -1) {
      H_need2visit.insert(h);
    }
  }

  std::vector<INT_TYPE> _h_right_B;
  while (!H_need2visit.empty()) {
    size_t b = _h_right_B.size();
    INT_TYPE h_start = *H_need2visit.begin();
    H_need2visit.erase(H_need2visit.begin());
    f_left_H[h_start] = -(b + 1);
    INT_TYPE h = h_next_H[h_start];
    _h_right_B.push_back(h);
    while (h != h_start) {
      H_need2visit.erase(h);
      f_left_H[h] = -(b + 1);
      h = h_next_H[h];
    }
  }

  Samplesi h_right_B(_h_right_B.size());
  std::copy(_h_right_B.begin(), _h_right_B.end(), h_right_B.begin());

  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_bound_F, h_right_B);
  /*
  Copilot non-verified code
  */
  // Function to find the index of the twin half-edge
  INT_TYPE get_halfedge_index_of_twin_gpt(PySamples2i H, INT_TYPE h) {
    // Request a buffer descriptor from the NumPy array
    auto buf = H.request();
    if (buf.ndim != 2 || buf.shape[1] != 2) {
      throw std::runtime_error(
          "Input should be a 2D NumPy array with shape (n, 2)");
    }

    // Get a pointer to the data
    int *ptr = static_cast<int *>(buf.ptr);

    // Get the number of half-edges
    size_t num_half_edges = buf.shape[0];

    // Flip the half-edge to find its twin
    std::array<int, 2> hedge_twin = {ptr[h * 2 + 1], ptr[h * 2]};

    // Search for the twin half-edge in the list
    for (size_t i = 0; i < num_half_edges; ++i) {
      if (ptr[i * 2] == hedge_twin[0] && ptr[i * 2 + 1] == hedge_twin[1]) {
        return i; // Return the index of the twin half-edge
      }
    }

    // If not found, return -1
    return -1;
  }

  std::tuple<PySamples3d, PySamplesi, PySamplesi, PySamplesi, PySamplesi,
             PySamplesi, PySamplesi, PySamplesi>
  vf_samples_to_he_samples_gpt(PySamples3d xyz_coord_V, PySamples3i V_of_F) {
    auto buf_xyz_coord_V = xyz_coord_V.request();
    auto buf_V_of_F = V_of_F.request();

    py::ssize_t Nfaces = buf_V_of_F.shape[0];
    py::ssize_t Nvertices = buf_xyz_coord_V.shape[0];
    py::ssize_t _Nhedges = 3 * Nfaces * 2;

    PySamples2i _H(py::array::ShapeContainer({_Nhedges, 2}));
    PySamplesi h_out_V(py::array::ShapeContainer({Nvertices}));
    PySamplesi _v_origin_H(py::array::ShapeContainer({_Nhedges}));
    PySamplesi _h_next_H(py::array::ShapeContainer({_Nhedges}));
    PySamplesi _f_left_H(py::array::ShapeContainer({_Nhedges}));
    PySamplesi h_bound_F(py::array::ShapeContainer({Nfaces}));

    auto ptr_H = _H.mutable_data();
    auto ptr_h_out_V = h_out_V.mutable_data();
    auto ptr_v_origin_H = _v_origin_H.mutable_data();
    auto ptr_h_next_H = _h_next_H.mutable_data();
    auto ptr_f_left_H = _f_left_H.mutable_data();
    auto ptr_h_bound_F = h_bound_F.mutable_data();
    auto ptr_V_of_F = static_cast<INT_TYPE *>(buf_V_of_F.ptr);

    std::fill(ptr_h_out_V, ptr_h_out_V + Nvertices, -1);
    std::fill(ptr_h_next_H, ptr_h_next_H + _Nhedges, -1);

    for (py::ssize_t f = 0; f < Nfaces; ++f) {
      ptr_h_bound_F[f] = 3 * f;
      for (py::ssize_t i = 0; i < 3; ++i) {
        py::ssize_t h = 3 * f + i;
        py::ssize_t h_next = 3 * f + (i + 1) % 3;
        INT_TYPE v0 = ptr_V_of_F[f * 3 + i];
        INT_TYPE v1 = ptr_V_of_F[f * 3 + (i + 1) % 3];
        ptr_H[h * 2] = v0;
        ptr_H[h * 2 + 1] = v1;
        ptr_v_origin_H[h] = v0;
        ptr_f_left_H[h] = f;
        ptr_h_next_H[h] = h_next;
        if (ptr_h_out_V[v0] == -1) {
          ptr_h_out_V[v0] = h;
        }
      }
    }

    py::ssize_t h_count = 3 * Nfaces;
    std::set<INT_TYPE> need_twins;
    for (py::ssize_t i = 0; i < h_count; ++i) {
      need_twins.insert(i);
    }

    std::set<INT_TYPE> need_next;
    PySamplesi _h_twin_H(py::array::ShapeContainer({_Nhedges}));
    auto ptr_h_twin_H = _h_twin_H.mutable_data();
    std::fill(ptr_h_twin_H, ptr_h_twin_H + _Nhedges, -2);

    while (!need_twins.empty()) {
      INT_TYPE h = *need_twins.begin();
      need_twins.erase(need_twins.begin());
      if (ptr_h_twin_H[h] == -2) {
        INT_TYPE h_twin = get_halfedge_index_of_twin_gpt(_H, h);
        if (h_twin == -1) {
          h_twin = h_count++;
          INT_TYPE v0 = ptr_H[h * 2];
          INT_TYPE v1 = ptr_H[h * 2 + 1];
          ptr_H[h_twin * 2] = v1;
          ptr_H[h_twin * 2 + 1] = v0;
          ptr_v_origin_H[h_twin] = v1;
          need_next.insert(h_twin);
          ptr_h_twin_H[h] = h_twin;
          ptr_h_twin_H[h_twin] = h;
          ptr_f_left_H[h_twin] = -1;
        } else {
          ptr_h_twin_H[h] = h_twin;
          ptr_h_twin_H[h_twin] = h;
          need_twins.erase(h_twin);
        }
      }
    }

    py::ssize_t Nhedges = h_count;
    PySamplesi v_origin_H(py::array::ShapeContainer({Nhedges}));
    PySamplesi h_next_H(py::array::ShapeContainer({Nhedges}));
    PySamplesi f_left_H(py::array::ShapeContainer({Nhedges}));
    PySamplesi h_twin_H(py::array::ShapeContainer({Nhedges}));

    std::copy(ptr_v_origin_H, ptr_v_origin_H + Nhedges,
              v_origin_H.mutable_data());
    std::copy(ptr_h_next_H, ptr_h_next_H + Nhedges, h_next_H.mutable_data());
    std::copy(ptr_f_left_H, ptr_f_left_H + Nhedges, f_left_H.mutable_data());
    std::copy(ptr_h_twin_H, ptr_h_twin_H + Nhedges, h_twin_H.mutable_data());

    while (!need_next.empty()) {
      INT_TYPE h = *need_next.begin();
      need_next.erase(need_next.begin());
      INT_TYPE h_next = ptr_h_twin_H[h];
      while (ptr_f_left_H[h_next] != -1) {
        h_next = ptr_h_twin_H[ptr_h_next_H[ptr_h_next_H[h_next]]];
      }
      ptr_h_next_H[h] = h_next;
    }

    std::set<INT_TYPE> H_need2visit;
    for (py::ssize_t h = 0; h < Nhedges; ++h) {
      if (ptr_f_left_H[h] == -1) {
        H_need2visit.insert(h);
      }
    }

    std::vector<INT_TYPE> _h_right_B;
    while (!H_need2visit.empty()) {
      py::ssize_t b = _h_right_B.size();
      INT_TYPE h_start = *H_need2visit.begin();
      H_need2visit.erase(H_need2visit.begin());
      ptr_f_left_H[h_start] = -(b + 1);
      INT_TYPE h = ptr_h_next_H[h_start];
      _h_right_B.push_back(h);
      while (h != h_start) {
        H_need2visit.erase(h);
        ptr_f_left_H[h] = -(b + 1);
        h = ptr_h_next_H[h];
      }
    }

    PySamplesi h_right_B(
        py::array::ShapeContainer{static_cast<py::ssize_t>(_h_right_B.size())});
    std::copy(_h_right_B.begin(), _h_right_B.end(), h_right_B.mutable_data());

    return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                           f_left_H, h_bound_F, h_right_B);
  }

  PYBIND11_MODULE(half_edge_utils, m) {
    m.doc() = "pybind11 half_edge_utils plugin"; // Optional module docstring
    m.def("get_V_of_F", &get_V_of_F,
          "A function to compute vertices of faces from half-edge data");
    m.def("get_halfedge_index_of_twin_gpt", &get_halfedge_index_of_twin_gpt,
          "A function to find the index of the twin half-edge");
    m.def("find_h_right_B", &find_h_right_B,
          "A function to find the index of the twin half-edge");
    m.def("vf_samples_to_he_samples_gpt", &vf_samples_to_he_samples_gpt,
          "A function to convert vertex-face samples to half-edge samples");
  }
