/**
 * @file chalf_edge_pybind.cpp
 */
#include <chalf_edge.hpp>
#include <cply_tools.hpp>
#include <data_types.hpp>
#include <pybind11/eigen.h>    // Eigen<->Numpy conversion
#include <pybind11/pybind11.h> // PYBIND11_MODULE

namespace py = pybind11;

PYBIND11_MODULE(chalf_edge, m) {
  m.doc() = "pybind11 chalf_edge plugin"; // module docstring
  py::class_<HalfEdgeMesh>(m, "HalfEdgeMesh")
      //   Constructors
      .def(py::init<const Samples3d &, const Samplesi &, const Samplesi &,
                    const Samplesi &, const Samplesi &, const Samplesi &,
                    const Samplesi &, const Samplesi &>())
      .def_static("from_he_ply", &HalfEdgeMesh::from_he_ply,
                  py::arg("ply_path"))
      // Attributes
      // Fundamental accessors and properties
      .def("get_xyz_coord_V", &HalfEdgeMesh::get_xyz_coord_V)
      .def("set_xyz_coord_V", &HalfEdgeMesh::set_xyz_coord_V)
      .def("get_h_out_V", &HalfEdgeMesh::get_h_out_V)
      .def("set_h_out_V", &HalfEdgeMesh::set_h_out_V)
      .def("get_v_origin_H", &HalfEdgeMesh::get_v_origin_H)
      .def("set_v_origin_H", &HalfEdgeMesh::set_v_origin_H)
      .def("get_h_next_H", &HalfEdgeMesh::get_h_next_H)
      .def("set_h_next_H", &HalfEdgeMesh::set_h_next_H)
      .def("get_h_twin_H", &HalfEdgeMesh::get_h_twin_H)
      .def("set_h_twin_H", &HalfEdgeMesh::set_h_twin_H)
      .def("get_f_left_H", &HalfEdgeMesh::get_f_left_H)
      .def("set_f_left_H", &HalfEdgeMesh::set_f_left_H)
      .def("get_h_bound_F", &HalfEdgeMesh::get_h_bound_F)
      .def("set_h_bound_F", &HalfEdgeMesh::set_h_bound_F)
      .def("get_h_right_B", &HalfEdgeMesh::get_h_right_B)
      .def("set_h_right_B", &HalfEdgeMesh::set_h_right_B)
      .def("get_num_vertices", &HalfEdgeMesh::get_num_vertices)
      .def("get_num_edges", &HalfEdgeMesh::get_num_edges)
      .def("get_num_half_edges", &HalfEdgeMesh::get_num_half_edges)
      .def("get_num_faces", &HalfEdgeMesh::get_num_faces)
      .def("get_euler_characteristic", &HalfEdgeMesh::get_euler_characteristic)
      .def("get_num_boundaries", &HalfEdgeMesh::get_num_boundaries)
      .def("get_genus", &HalfEdgeMesh::get_genus)
      .def("V_of_F", &HalfEdgeMesh::V_of_F)
      .def("vf_samples", &HalfEdgeMesh::vf_samples)
      .def("he_samples", &HalfEdgeMesh::he_samples)
      .def("F_incident_b", &HalfEdgeMesh::F_incident_b)
      .def("xyz_coord_v",
           py::overload_cast<int>(&HalfEdgeMesh::xyz_coord_v, py::const_))
      .def("xyz_coord_v", py::overload_cast<const Samplesi &>(
                              &HalfEdgeMesh::xyz_coord_v, py::const_))
      .def("h_out_v",
           py::overload_cast<int>(&HalfEdgeMesh::h_out_v, py::const_))
      .def("h_out_v", py::overload_cast<const Samplesi &>(
                          &HalfEdgeMesh::h_out_v, py::const_))
      .def("v_origin_h",
           py::overload_cast<int>(&HalfEdgeMesh::v_origin_h, py::const_))
      .def("v_origin_h", py::overload_cast<const Samplesi &>(
                             &HalfEdgeMesh::v_origin_h, py::const_))
      .def("h_next_h",
           py::overload_cast<int>(&HalfEdgeMesh::h_next_h, py::const_))
      .def("h_next_h", py::overload_cast<const Samplesi &>(
                           &HalfEdgeMesh::h_next_h, py::const_))
      .def("h_twin_h",
           py::overload_cast<int>(&HalfEdgeMesh::h_twin_h, py::const_))
      .def("h_twin_h", py::overload_cast<const Samplesi &>(
                           &HalfEdgeMesh::h_twin_h, py::const_))
      .def("f_left_h",
           py::overload_cast<int>(&HalfEdgeMesh::f_left_h, py::const_))
      .def("f_left_h", py::overload_cast<const Samplesi &>(
                           &HalfEdgeMesh::f_left_h, py::const_))
      .def("h_bound_f",
           py::overload_cast<int>(&HalfEdgeMesh::h_bound_f, py::const_))
      .def("h_bound_f", py::overload_cast<const Samplesi &>(
                            &HalfEdgeMesh::h_bound_f, py::const_))
      .def("h_right_b",
           py::overload_cast<int>(&HalfEdgeMesh::h_right_b, py::const_))
      .def("h_right_b", py::overload_cast<const Samplesi &>(
                            &HalfEdgeMesh::h_right_b, py::const_))
      .def("h_in_v", &HalfEdgeMesh::h_in_v)
      .def("v_head_h", &HalfEdgeMesh::v_head_h)
      .def("h_prev_h", &HalfEdgeMesh::h_prev_h)
      .def("h_rotcw_h", &HalfEdgeMesh::h_rotcw_h)
      .def("h_rotccw_h", &HalfEdgeMesh::h_rotccw_h)
      .def("h_prev_h_by_rot", &HalfEdgeMesh::h_prev_h_by_rot)
      // Predicates
      .def("some_negative_boundary_contains_h",
           &HalfEdgeMesh::some_negative_boundary_contains_h)
      .def("some_positive_boundary_contains_h",
           &HalfEdgeMesh::some_positive_boundary_contains_h)
      .def("some_boundary_contains_h", &HalfEdgeMesh::some_boundary_contains_h)
      .def("some_boundary_contains_v", &HalfEdgeMesh::some_boundary_contains_v)
      .def("h_is_locally_delaunay", &HalfEdgeMesh::h_is_locally_delaunay)
      .def("h_is_flippable", &HalfEdgeMesh::h_is_flippable)
      // Generators
      .def("generate_H_out_v_clockwise",
           &HalfEdgeMesh::generate_H_out_v_clockwise)
      .def("generate_H_rotcw_h", &HalfEdgeMesh::generate_H_rotcw_h)
      // Mesh modification
      .def("update_update_vertex", &HalfEdgeMesh::update_vertex)
      .def("update_half_edge", &HalfEdgeMesh::update_half_edge)
      .def("update_face", &HalfEdgeMesh::update_face)
      .def("flip_edge", &HalfEdgeMesh::flip_edge)
      .def("flip_non_delaunay", &HalfEdgeMesh::flip_non_delaunay);
}

// h_in_v
// v_head_h
// h_prev_h
// h_rotcw_h
// h_rotccw_h
// h_prev_h_by_rot

// .def("xyz_coord_v", &HalfEdgeMesh::xyz_coord_v)
// .def("h_out_v", &HalfEdgeMesh::h_out_v)
// .def("v_origin_h", &HalfEdgeMesh::v_origin_h)
// .def("h_next_h", &HalfEdgeMesh::h_next_h)
// .def("h_twin_h", &HalfEdgeMesh::h_twin_h)
// .def("f_left_h", &HalfEdgeMesh::f_left_h)
// .def("h_bound_f", &HalfEdgeMesh::h_bound_f)
// .def("h_right_b", &HalfEdgeMesh::h_right_b);
