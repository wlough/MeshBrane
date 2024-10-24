PYBIND11_MODULE(cply_tools, m) {
  m.doc() = "pybind11 cply_tools plugin"; // module docstring
  m.def("vf_samples_to_he_samples", &vf_samples_to_he_samples,
        "A function to compute half-edge data from vertices of faces");
  py::class_<Pet>(m, "Pet")
      .def(py::init<const std::string &>())
      .def_readwrite("name", &Pet::name)
      .def("setName", &Pet::setName)
      .def("getName", &Pet::getName);
  py::class_<MeshConverter>(m, "MeshConverter")
      //   Constructors
      .def(py::init<>())
      .def_static("from_vf_ply", &MeshConverter::from_vf_ply,
                  py::arg("ply_path"), py::arg("compute_he_stuff") = true)
      .def_static("from_vf_samples", &MeshConverter::from_vf_samples,
                  py::arg("xyz_coord_V"), py::arg("V_of_F"),
                  py::arg("compute_he_stuff") = true)
      .def_static("from_he_ply", &MeshConverter::from_he_ply,
                  py::arg("ply_path"), py::arg("compute_vf_stuff") = true)
      .def_static("from_he_samples", &MeshConverter::from_he_samples,
                  py::arg("xyz_coord_V"), py::arg("h_out_V"),
                  py::arg("v_origin_H"), py::arg("h_next_H"),
                  py::arg("h_twin_H"), py::arg("f_left_H"),
                  py::arg("h_bound_F"), py::arg("h_right_B"),
                  py::arg("compute_vf_stuff") = true)
      // Attributes
      .def_readwrite("vf_ply_path", &MeshConverter::vf_ply_path)
      .def_readwrite("vf_samples", &MeshConverter::vf_samples)
      .def_readwrite("he_ply_path", &MeshConverter::he_ply_path)
      .def_readwrite("he_samples", &MeshConverter::he_samples)
      // Methods
      .def("write_vf_ply", &MeshConverter::write_vf_ply, py::arg("ply_path"),
           py::arg("use_binary") = true)
      .def("write_he_ply", &MeshConverter::write_he_ply, py::arg("ply_path"),
           py::arg("use_binary") = true);
};
