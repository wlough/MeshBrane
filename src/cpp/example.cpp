// half_edge_utils.cpp
#include <algorithm>
#include <array>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// Define the function to compute V_of_F
py::array_t<int>
get_V_of_F(py::array_t<double> xyz_coord_V, py::array_t<int> h_out_V,
           py::array_t<int> v_origin_H, py::array_t<int> h_next_H,
           py::array_t<int> h_twin_H, py::array_t<int> f_left_H,
           py::array_t<int> h_bound_F, py::array_t<int> h_right_B) {
  // Request buffer descriptors from the NumPy arrays
  auto buf_h_bound_F = h_bound_F.request();
  auto buf_v_origin_H = v_origin_H.request();
  auto buf_h_next_H = h_next_H.request();

  // Get the number of faces
  int Nf = buf_h_bound_F.shape[0];

  // Create an output array
  py::array_t<int> V_of_F({Nf, 3});
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

// Define the type for half-edges
using HalfEdge = std::array<int, 2>;

// Function to find the index of the twin half-edge
int get_halfedge_index_of_twin(py::array_t<int> H, int h) {
  // Request a buffer descriptor from the NumPy array
  auto buf = H.request();
  if (buf.ndim != 2 || buf.shape[1] != 2) {
    throw std::runtime_error(
        "Input should be a 2D NumPy array with shape (n, 2)");
  }

  // Convert the NumPy array to a vector of HalfEdge
  std::vector<HalfEdge> half_edges(buf.shape[0]);
  for (size_t i = 0; i < buf.shape[0]; ++i) {
    half_edges[i] = {H.at(i, 0), H.at(i, 1)};
  }

  // Flip the half-edge to find its twin
  HalfEdge hedge_twin = {half_edges[h][1], half_edges[h][0]};

  // Search for the twin half-edge in the list
  auto it = std::find(half_edges.begin(), half_edges.end(), hedge_twin);

  // If found, return the index
  if (it != half_edges.end()) {
    return std::distance(half_edges.begin(), it);
  }

  // If not found, return -1
  return -1;
}

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 half_edge_utils plugin"; // Optional module docstring
  m.def("add", &add, "A function which adds two numbers");
  m.def("get_halfedge_index_of_twin", &get_halfedge_index_of_twin,
        "A function to find the index of the twin half-edge");
  m.def("get_V_of_F", &get_V_of_F, "A function to compute vertices of faces");
}