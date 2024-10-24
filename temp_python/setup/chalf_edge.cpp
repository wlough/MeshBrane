/**
 * @file chalf_edge.cpp
 */
#include <cgenerators.hpp>
#include <chalf_edge.hpp>
#include <cmath> // For M_PI and std::acos
#include <cply_tools.hpp>
#include <data_types.hpp>
#include <unordered_set> // std::unordered_set
// #include <exception>
// #include <stdexcept>
// #include <vector>
// #include <coroutine>
// num_vertices
// num_edges
// num_half_edges
// num_faces
// euler_characteristic
// num_boundaries
// genus

// get_num_vertices
// get_num_edges
// get_num_half_edges
// get_num_faces
// get_euler_characteristic
// get_num_boundaries
// get_genus
///////////////////////////////////////////////////////
// Constructors ///////////////////////////////////////
///////////////////////////////////////////////////////
HalfEdgeMesh::HalfEdgeMesh(const Samples3d &xyz_coord_V,
                           const Samplesi &h_out_V, const Samplesi &v_origin_H,
                           const Samplesi &h_next_H, const Samplesi &h_twin_H,
                           const Samplesi &f_left_H, const Samplesi &h_bound_F,
                           const Samplesi &h_right_B)
    : _xyz_coord_V(xyz_coord_V), _h_out_V(h_out_V), _v_origin_H(v_origin_H),
      _h_next_H(h_next_H), _h_twin_H(h_twin_H), _f_left_H(f_left_H),
      _h_bound_F(h_bound_F), _h_right_B(h_right_B) {}

HalfEdgeMesh HalfEdgeMesh::from_he_ply(const std::string &ply_path) {
  MeshConverter mc = MeshConverter::from_he_ply(ply_path);
  auto [xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H,
        h_bound_F, h_right_B] = mc.he_samples;
  return HalfEdgeMesh(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                      f_left_H, h_bound_F, h_right_B);
}
///////////////////////////////////////////////////////
// Fundamental accessors and properties ///////////////
///////////////////////////////////////////////////////
const Samples3d &HalfEdgeMesh::get_xyz_coord_V() const { return _xyz_coord_V; }
void HalfEdgeMesh::set_xyz_coord_V(const Samples3d &value) {
  _xyz_coord_V = value;
}
const Samplesi &HalfEdgeMesh::get_h_out_V() const { return _h_out_V; }
void HalfEdgeMesh::set_h_out_V(const Samplesi &value) { _h_out_V = value; }
const Samplesi &HalfEdgeMesh::get_v_origin_H() const { return _v_origin_H; }
void HalfEdgeMesh::set_v_origin_H(const Samplesi &value) {
  _v_origin_H = value;
}
const Samplesi &HalfEdgeMesh::get_h_next_H() const { return _h_next_H; }
void HalfEdgeMesh::set_h_next_H(const Samplesi &value) { _h_next_H = value; }
const Samplesi &HalfEdgeMesh::get_h_twin_H() const { return _h_twin_H; }
void HalfEdgeMesh::set_h_twin_H(const Samplesi &value) { _h_twin_H = value; }
const Samplesi &HalfEdgeMesh::get_f_left_H() const { return _f_left_H; }
void HalfEdgeMesh::set_f_left_H(const Samplesi &value) { _f_left_H = value; }
const Samplesi &HalfEdgeMesh::get_h_bound_F() const { return _h_bound_F; }
void HalfEdgeMesh::set_h_bound_F(const Samplesi &value) { _h_bound_F = value; }
const Samplesi &HalfEdgeMesh::get_h_right_B() const { return _h_right_B; }
void HalfEdgeMesh::set_h_right_B(const Samplesi &value) { _h_right_B = value; }

int HalfEdgeMesh::get_num_vertices() const { return _h_out_V.size(); }
int HalfEdgeMesh::get_num_edges() const { return _v_origin_H.rows() / 2; }
int HalfEdgeMesh::get_num_half_edges() const { return _v_origin_H.size(); }
int HalfEdgeMesh::get_num_faces() const { return _h_bound_F.size(); }
int HalfEdgeMesh::get_euler_characteristic() const {
  return get_num_vertices() - get_num_edges() + get_num_faces();
}
int HalfEdgeMesh::get_num_boundaries() const { return _h_right_B.size(); }
int HalfEdgeMesh::get_genus() const {
  return (2 - get_euler_characteristic() - get_num_boundaries()) / 2;
}

Samples3i HalfEdgeMesh::V_of_F() const {
  Samples3i V_of_F(get_num_faces(), 3);
  for (int f = 0; f < get_num_faces(); f++) {
    V_of_F.row(f) << _v_origin_H(_h_bound_F(f)),
        _v_origin_H(_h_next_H(_h_bound_F(f))),
        _v_origin_H(_h_next_H(_h_next_H(_h_bound_F(f))));
  }
  return V_of_F;
}
VertexFaceSamples HalfEdgeMesh::vf_samples() const {
  return {_xyz_coord_V, V_of_F()};
}
HalfEdgeSamples HalfEdgeMesh::he_samples() const {
  return {_xyz_coord_V, _h_out_V,  _v_origin_H, _h_next_H,
          _h_twin_H,    _f_left_H, _h_bound_F,  _h_right_B};
}
Samplesi HalfEdgeMesh::F_incident_b(int b) const {
  std::unordered_set<int> setF_incident_b;
  Samplesi F_incident_b;
  for (auto h : generate_H_right_b(b)) {
    int v = v_origin_h(h);
    for (auto h_out : generate_H_out_v_clockwise(v, h)) {
      if (some_negative_boundary_contains_h(h_out)) {
        continue;
      }
      setF_incident_b.insert(f_left_h(h_out));
    }
  }
  F_incident_b.resize(setF_incident_b.size());
  std::copy(setF_incident_b.begin(), setF_incident_b.end(),
            F_incident_b.begin());
  return F_incident_b;
}
// for (auto h : generate_H_out_v_clockwise(v, h_start)) {
//     if (some_boundary_contains_h(h)) {
//       return true;
//     }
//   }
///////////////////////////////////////////////////////
// Combinatorial maps /////////////////////////////////
///////////////////////////////////////////////////////
Coords3d HalfEdgeMesh::xyz_coord_v(int v) const { return _xyz_coord_V.row(v); }
Samples3d HalfEdgeMesh::xyz_coord_v(const Samplesi &indices) const {
  Samples3d result(indices.size(), 3);
  for (int i = 0; i < indices.size(); ++i) {
    result.row(i) = _xyz_coord_V.row(indices(i));
  }
  return result;
}
int HalfEdgeMesh::h_out_v(int v) const { return _h_out_V(v); }
Samplesi HalfEdgeMesh::h_out_v(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _h_out_V(indices(i));
  }
  return result;
}
int HalfEdgeMesh::v_origin_h(int h) const { return _v_origin_H(h); }
Samplesi HalfEdgeMesh::v_origin_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _v_origin_H(indices(i));
  }
  return result;
}
int HalfEdgeMesh::h_next_h(int h) const { return _h_next_H(h); }
Samplesi HalfEdgeMesh::h_next_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _h_next_H(indices(i));
  }
  return result;
}
int HalfEdgeMesh::h_twin_h(int h) const { return _h_twin_H(h); }
Samplesi HalfEdgeMesh::h_twin_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _h_twin_H(indices(i));
  }
  return result;
}
int HalfEdgeMesh::f_left_h(int h) const { return _f_left_H(h); }
Samplesi HalfEdgeMesh::f_left_h(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _f_left_H(indices(i));
  }
  return result;
}
int HalfEdgeMesh::h_bound_f(int f) const { return _h_bound_F(f); }
Samplesi HalfEdgeMesh::h_bound_f(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _h_bound_F(indices(i));
  }
  return result;
}
int HalfEdgeMesh::h_right_b(int b) const { return _h_right_B(b); }
Samplesi HalfEdgeMesh::h_right_b(const Samplesi &indices) const {
  Samplesi result(indices.size());
  for (int i = 0; i < indices.size(); ++i) {
    result(i) = _h_right_B(indices(i));
  }
  return result;
}
// Derived combinatorial maps
int HalfEdgeMesh::h_in_v(int v) const { return h_twin_h(h_out_v(v)); }
int HalfEdgeMesh::v_head_h(int h) const { return v_origin_h(h_twin_h(h)); }
int HalfEdgeMesh::h_prev_h(int h) const {
  int h_prev;
  int h_next = h_next_h(h);
  while (h_next != h) {
    h_prev = h_next;
    h_next = h_next_h(h_prev);
  }
  return h;
}
int HalfEdgeMesh::h_rotcw_h(int h) const { return h_next_h(h_twin_h(h)); }
int HalfEdgeMesh::h_rotccw_h(int h) const { return h_twin_h(h_prev_h(h)); }
int HalfEdgeMesh::h_prev_h_by_rot(int h) const {
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
bool HalfEdgeMesh::some_negative_boundary_contains_h(int h) const {
  return f_left_h(h) < 0;
}
bool HalfEdgeMesh::some_positive_boundary_contains_h(int h) const {
  return f_left_h(h_twin_h(h)) < 0;
}
bool HalfEdgeMesh::some_boundary_contains_h(int h) const {
  return some_negative_boundary_contains_h(h) ||
         some_positive_boundary_contains_h(h);
}
bool HalfEdgeMesh::some_boundary_contains_v(int v) const {
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
bool HalfEdgeMesh::h_is_locally_delaunay(int h) const {
  int vi = v_head_h(h_next_h(h_twin_h(h)));
  int vj = v_head_h(h);
  int vk = v_head_h(h_next_h(h));
  int vl = v_origin_h(h);

  Eigen::Vector3d rij = xyz_coord_v(vj) - xyz_coord_v(vi);
  Eigen::Vector3d ril = xyz_coord_v(vl) - xyz_coord_v(vi);

  Eigen::Vector3d rkj = xyz_coord_v(vj) - xyz_coord_v(vk);
  Eigen::Vector3d rkl = xyz_coord_v(vl) - xyz_coord_v(vk);

  double alphai = std::acos(rij.dot(ril) / (rij.norm() * ril.norm()));
  double alphak = std::acos(rkl.dot(rkj) / (rkl.norm() * rkj.norm()));

  return alphai + alphak <= M_PI;
}
bool HalfEdgeMesh::h_is_flippable(int h) const {
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
SimpleGenerator<int>
HalfEdgeMesh::generate_H_out_v_clockwise(int v, int h_start) const {
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
SimpleGenerator<int> HalfEdgeMesh::generate_H_rotcw_h(int h) const {
  int h_start = h;
  do {
    co_yield h;
    h = h_rotcw_h(h);
    //     h = h_next_h(h_twin_h(h));
  } while (h != h_start);
}
SimpleGenerator<int> HalfEdgeMesh::generate_H_next_h(int h) const {
  int h_start = h;
  do {
    co_yield h;
    h = h_next_h(h);
  } while (h != h_start);
}
SimpleGenerator<int> HalfEdgeMesh::generate_H_right_b(int b) const {
  int h_start = h_right_b(b);
  int h = h_start;
  do {
    co_yield h;
    h = h_next_h(h);
  } while (h != h_start);
}
///////////////////////////////////////////////////////
// Mutators ///////////////////////////////////////////
///////////////////////////////////////////////////////
void HalfEdgeMesh::update_vertex(int v,
                                 const std::optional<Coords3d> &xyz_coord,
                                 const std::optional<int> &h_out) {
  if (xyz_coord.has_value()) {
    _xyz_coord_V.row(v) = xyz_coord.value();
  }
  if (h_out.has_value()) {
    _h_out_V(v) = h_out.value();
  }
}
void HalfEdgeMesh::update_half_edge(int h, const std::optional<int> &v_origin,
                                    const std::optional<int> &h_next,
                                    const std::optional<int> &h_twin,
                                    const std::optional<int> &f_left) {
  if (v_origin.has_value()) {
    _v_origin_H(h) = v_origin.value();
  }
  if (h_next.has_value()) {
    _h_next_H(h) = h_next.value();
  }
  if (h_twin.has_value()) {
    _h_twin_H(h) = h_twin.value();
  }
  if (f_left.has_value()) {
    _f_left_H(h) = f_left.value();
  }
}
void HalfEdgeMesh::update_face(int f, const std::optional<int> &h_left) {
  if (h_left.has_value()) {
    _f_left_H(f) = h_left.value();
  }
}
void HalfEdgeMesh::flip_edge(int h) {
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
    _h_out_V(v0) = h2;
  }
  if (h_out_v(v2) == h0) {
    _h_out_V(v2) = h4;
  }
  // update half-edges
  update_half_edge(h0, v3, h3, std::nullopt, std::nullopt);
  update_half_edge(h1, v1, h5, std::nullopt, std::nullopt);
  update_half_edge(h2, std::nullopt, h1, std::nullopt, f1);
  update_half_edge(h3, std::nullopt, h4, std::nullopt, std::nullopt);
  update_half_edge(h4, std::nullopt, h0, std::nullopt, f0);
  update_half_edge(h5, std::nullopt, h2, std::nullopt, std::nullopt);
  // update faces
  if (h_bound_f(f0) == h2) {
    _h_bound_F(f0) = h3;
  }
  if (h_bound_f(f1) == h4) {
    _h_bound_F(f1) = h5;
  }
}

int HalfEdgeMesh::flip_non_delaunay() {
  int flip_count = 0;
  for (int h = 0; h < get_num_half_edges(); h++) {
    if (h_is_flippable(h) && !h_is_locally_delaunay(h)) {
      flip_edge(h);
      flip_count++;
    }
  }
  return flip_count;
}
////////////////////////////////////////////////////////////////
// Python bindings
////////////////////////////////////////////////////////////////

// PYBIND11_MODULE(cply_tools, m) {
//   m.doc() = "pybind11 chalf_edge plugin"; // module docstring
//   py::class_<HalfEdgeMesh>(m, "HalfEdgeMesh")
//       //   Constructors
//       .def(py::init<const Samples3d &, const Samplesi &, const Samplesi &,
//                     const Samplesi &, const Samplesi &, const Samplesi &,
//                     const Samplesi &, const Samplesi &>())
//       .def_static("from_he_ply", &HalfEdgeMesh::from_he_ply,
//                   py::arg("ply_path"))
//       // Attributes
//       // Fundamental accessors and properties
//       .def("get_xyz_coord_V", &HalfEdgeMesh::get_xyz_coord_V)
//       .def("set_xyz_coord_V", &HalfEdgeMesh::set_xyz_coord_V)
//       .def("get_h_out_V", &HalfEdgeMesh::get_h_out_V)
//       .def("set_h_out_V", &HalfEdgeMesh::set_h_out_V)
//       .def("get_v_origin_H", &HalfEdgeMesh::get_v_origin_H)
//       .def("set_v_origin_H", &HalfEdgeMesh::set_v_origin_H)
//       .def("get_h_next_H", &HalfEdgeMesh::get_h_next_H)
//       .def("set_h_next_H", &HalfEdgeMesh::set_h_next_H)
//       .def("get_h_twin_H", &HalfEdgeMesh::get_h_twin_H)
//       .def("set_h_twin_H", &HalfEdgeMesh::set_h_twin_H)
//       .def("get_f_left_H", &HalfEdgeMesh::get_f_left_H)
//       .def("set_f_left_H", &HalfEdgeMesh::set_f_left_H)
//       .def("get_h_bound_F", &HalfEdgeMesh::get_h_bound_F)
//       .def("set_h_bound_F", &HalfEdgeMesh::set_h_bound_F)
//       .def("get_h_right_B", &HalfEdgeMesh::get_h_right_B)
//       .def("set_h_right_B", &HalfEdgeMesh::set_h_right_B)
//       .def("num_vertices", &HalfEdgeMesh::num_vertices)
//       .def("num_edges", &HalfEdgeMesh::num_edges)
//       .def("num_half_edges", &HalfEdgeMesh::num_half_edges)
//       .def("num_faces", &HalfEdgeMesh::num_faces)
//       .def("euler_characteristic", &HalfEdgeMesh::euler_characteristic)
//       .def("num_boundaries", &HalfEdgeMesh::num_boundaries)
//       .def("genus", &HalfEdgeMesh::genus)
//       .def("V_of_F", &HalfEdgeMesh::V_of_F)
//       .def("vf_samples", &HalfEdgeMesh::vf_samples)
//       .def("he_samples", &HalfEdgeMesh::he_samples)
//       .def("xyz_coord_v", &HalfEdgeMesh::xyz_coord_v)
//       .def("h_out_v", &HalfEdgeMesh::h_out_v)
//       .def("v_origin_h", &HalfEdgeMesh::v_origin_h)
//       .def("h_next_h", &HalfEdgeMesh::h_next_h)
//       .def("h_twin_h", &HalfEdgeMesh::h_twin_h)
//       .def("f_left_h", &HalfEdgeMesh::f_left_h)
//       .def("h_bound_f", &HalfEdgeMesh::h_bound_f)
//       .def("h_right_b", &HalfEdgeMesh::h_right_b);
// }

//   .def_static("from_vf_ply", &MeshConverter::from_vf_ply,
//               py::arg("ply_path"), py::arg("compute_he_stuff") = true)
//   .def_static("from_vf_samples", &MeshConverter::from_vf_samples,
//               py::arg("xyz_coord_V"), py::arg("V_of_F"),
//               py::arg("compute_he_stuff") = true)
//   .def("vf_samples_to_he_samples",
//   &MeshConverter::vf_samples_to_he_samples) .def("vf_ply_data_to_samples",
//   &MeshConverter::vf_ply_data_to_samples) .def("write_vf_ply",
//   &MeshConverter::write_vf_ply, py::arg("ply_path"),
//        py::arg("use_binary") = true)