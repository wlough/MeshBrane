/**
 * @file pretty_pictures_test.cpp
 * @brief tests for pretty_pictures
 */
#include "pretty_pictures.hpp"
#include <iostream>

void test_load_tri_mesh_data_from_ply() {
  std::string ply_path = "../data/ply_files/dumbbell.ply";
  TriMeshData mesh = load_tri_mesh_data_from_ply(ply_path, false);
  auto &V = mesh.vertices;
  std::uint32_t Nvertices = V.size();
  Eigen::Vector3d com(0.0, 0.0, 0.0);
  for (auto &xyz : V) {
    com += xyz / Nvertices;
  }
  std::cout << "Center of mass: " << com << std::endl;
}

void test_load_write_to_ply() {
  bool preload_into_memory = false;
  bool useBinary = false;

  std::string ply_path0 = "../data/ply_files/dumbbell.ply";
  std::string output_directory = "../data/ply_files";
  std::string filename = "dumbbell_test_copy.ply";
  std::string ply_path = output_directory + "/" + filename;

  TriMeshData mesh0 =
      load_tri_mesh_data_from_ply(ply_path0, preload_into_memory);

  write_tri_mesh_data_to_ply(mesh0, output_directory, filename, useBinary);
  TriMeshData mesh = load_tri_mesh_data_from_ply(ply_path, preload_into_memory);

  auto &V = mesh.vertices;
  std::uint32_t Nvertices = V.size();
  Eigen::Vector3d com(0.0, 0.0, 0.0);
  for (auto &xyz : V) {
    com += xyz / Nvertices;
  }
  std::cout << "Center of mass: " << com << std::endl;
}

// void test_create_pyply_plot_json() {

//   Params_create_pyply_plot_json params;
//   params.ply_path = "./data/ply_files/dumbbell.ply";
//   create_pyply_plot_json(params);
// }

// int main(int argc, char *argv[]) {
// std::string filepath = "./data/ply_files/dumbbell.ply";
// std::pair<std::vector<std::array<double, 3>>,
//           std::vector<std::array<uint32_t, 3>>>
//     vf_mesh = load_vertex_face_list_from_ply(filepath);
// std::vector<std::array<double, 3>> vertices = vf_mesh.first;

// VertexFaceList vf_list = load_face_vertex_list_from_ply(filepath);
// std::vector<std::array<double, 3>> vertices2 = vf_list.vertices;

// for (size_t i = 0; i < vertices.size(); ++i) {
//   std::cout << "Vertex " << i << ": " << vertices[i][0] << " " <<
//   vertices[i][1]
//             << " " << vertices[i][2] << std::endl;
// }

// return EXIT_SUCCESS;
// }

int main() {
  //   test_load_tri_mesh_data_from_ply();
  //   test_create_pyply_plot_json();
  test_load_write_to_ply();
  return EXIT_SUCCESS;
}