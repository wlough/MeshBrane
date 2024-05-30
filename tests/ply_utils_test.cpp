/**
 * @file ply_utils_test.cpp
 * @brief tests for ply_utils
 */
#include "ply_utils.hpp"
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

void test_HalfEdgeMeshData() {
  std::string ply_path = "../data/ply_files/dumbbell_ultracoarse.ply";
  TriMeshData mesh = load_tri_mesh_data_from_ply(ply_path, false);
  HalfEdgeMeshData he_data = buildHalfEdgeMeshDataFromTriMeshData(mesh);
  std::cout << "Number of vertices: " << he_data.V.size() << std::endl;
  std::cout << "Number of edges: " << he_data.E_vertex.size() << std::endl;
  std::cout << "Number of faces: " << he_data.F_edge.size() << std::endl;
}

void test_write_he_mesh_data_to_ply() {
  std::string ply_path = "../data/ply_files/hex_patch.ply";
  TriMeshData mesh = load_tri_mesh_data_from_ply(ply_path, false);
  HalfEdgeMeshData he_data = buildHalfEdgeMeshDataFromTriMeshData(mesh);
  std::string output_directory = "../data/ply_files";
  std::string filename = "hex_patch_he.ply";
  std::string comment = "this is but a test";
  write_he_mesh_data_to_ply(he_data, output_directory, filename, true, comment);
}

void test_load_he_mesh_data_from_ply() {
  std::string ply_path = "../data/ply_files/hex_patch_he.ply";
  HalfEdgeMeshData he_data = load_he_mesh_data_from_ply(ply_path, false, true);
  std::cout << "Number of vertices: " << he_data.V.size() << std::endl;
  std::cout << "Number of edges: " << he_data.E_vertex.size() << std::endl;
  std::cout << "Number of faces: " << he_data.F_edge.size() << std::endl;
}

void test_ply_precision() {
  std::string ply_path_original = "../data/ply_files/dumbbell.ply";
  std::string output_directory = "../data/ply_files";
  TriMeshData mesh_og = load_tri_mesh_data_from_ply(ply_path_original, false);

  std::string filename_yesbinary = "dumbbell_yesbinary.ply";
  std::string ply_path_yesbinary = output_directory + "/" + filename_yesbinary;
  std::string filename_nobinary = "dumbbell_nobinary.ply";
  std::string ply_path_nobinary = output_directory + "/" + filename_nobinary;

  write_tri_mesh_data_to_ply(mesh_og, output_directory, filename_yesbinary,
                             true);
  write_tri_mesh_data_to_ply(mesh_og, output_directory, filename_nobinary,
                             false);

  TriMeshData mesh_yb = load_tri_mesh_data_from_ply(ply_path_yesbinary, false);
  TriMeshData mesh_nb = load_tri_mesh_data_from_ply(ply_path_nobinary, false);

  std::cout.precision(17); // Set precision to 17 decimal places
  std::cout << "og x: " << mesh_og.vertices[0][0] << std::endl;
  std::cout << "yes binary x: " << mesh_yb.vertices[0][0] << std::endl;
  std::cout << "no binary x: " << mesh_nb.vertices[0][0] << std::endl;
}

int main() {
  test_load_tri_mesh_data_from_ply();
  test_load_write_to_ply();
  test_HalfEdgeMeshData();
  test_write_he_mesh_data_to_ply();
  test_ply_precision();
  test_load_he_mesh_data_from_ply();
  return EXIT_SUCCESS;
}