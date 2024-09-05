/**
 * @file ply_utils_test.cpp
 * @brief tests for ply_utils
 */
#include "ply_utils.hpp"
#include <iostream>

std::string output_dir = "../../output";
std::string input_ply_dir = "../../data/half_edge_base/ply";
// std::string surf = "dumbbell";

void test_load_tri_mesh_data_from_ply(const std::string &surf) {
  std::string vf_ply = input_ply_dir + "/" + surf + "_vf.ply";
  TriMeshData mesh = load_tri_mesh_data_from_ply(vf_ply, false);
  auto &V = mesh.vertices;
  std::uint32_t Nvertices = V.size();
  Eigen::Vector3d com(0.0, 0.0, 0.0);
  for (auto &xyz : V) {
    com += xyz / Nvertices;
  }
  std::cout << "--------------------------------" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  std::cout << "test_load_tri_mesh_data_from_ply()" << std::endl;
  std::cout << "Center of mass: (" << com.x() << ", " << com.y() << ", "
            << com.z() << ")" << std::endl;
}

void test_load_write_to_ply(const std::string &surf) {
  std::cout << "--------------------------------" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  std::cout << "test_load_write_to_ply()" << std::endl;
  bool preload_into_memory = false;

  std::string vf_ply = input_ply_dir + "/" + surf + "_vf.ply";
  std::string vf_ply_test = output_dir + "/" + surf + "_vf_test.ply";
  std::string filename_test = surf + "_vf_test.ply";

  {
    std::cout << "useBinary=false" << std::endl;
    bool useBinary = false;
    TriMeshData mesh0 =
        load_tri_mesh_data_from_ply(vf_ply, preload_into_memory);

    write_tri_mesh_data_to_ply(mesh0, output_dir, filename_test, useBinary);
    TriMeshData mesh =
        load_tri_mesh_data_from_ply(vf_ply_test, preload_into_memory);

    auto &V0 = mesh0.vertices;
    std::uint32_t Nvertices0 = V0.size();
    Eigen::Vector3d com0(0.0, 0.0, 0.0);
    for (auto &xyz : V0) {
      com0 += xyz / Nvertices0;
    }
    auto &V = mesh.vertices;
    std::uint32_t Nvertices = V.size();
    Eigen::Vector3d com(0.0, 0.0, 0.0);
    for (auto &xyz : V) {
      com += xyz / Nvertices;
    }
    Eigen::Vector3d com_diff = com - com0;
    std::cout << "com0: (" << com0.x() << ", " << com0.y() << ", " << com0.z()
              << ")" << std::endl;
    std::cout << "com: (" << com.x() << ", " << com.y() << ", " << com.z()
              << ")" << std::endl;
    std::cout << "com_diff: (" << com_diff.x() << ", " << com_diff.y() << ", "
              << com_diff.z() << ")" << std::endl;
  }

  {
    std::cout << "useBinary=true" << std::endl;
    bool useBinary = true;
    TriMeshData mesh0 =
        load_tri_mesh_data_from_ply(vf_ply, preload_into_memory);

    write_tri_mesh_data_to_ply(mesh0, output_dir, filename_test, useBinary);
    TriMeshData mesh =
        load_tri_mesh_data_from_ply(vf_ply_test, preload_into_memory);

    auto &V0 = mesh0.vertices;
    std::uint32_t Nvertices0 = V0.size();
    Eigen::Vector3d com0(0.0, 0.0, 0.0);
    for (auto &xyz : V0) {
      com0 += xyz / Nvertices0;
    }
    auto &V = mesh.vertices;
    std::uint32_t Nvertices = V.size();
    Eigen::Vector3d com(0.0, 0.0, 0.0);
    for (auto &xyz : V) {
      com += xyz / Nvertices;
    }
    Eigen::Vector3d com_diff = com - com0;
    std::cout << "com0: (" << com0.x() << ", " << com0.y() << ", " << com0.z()
              << ")" << std::endl;
    std::cout << "com: (" << com.x() << ", " << com.y() << ", " << com.z()
              << ")" << std::endl;
    std::cout << "com_diff: (" << com_diff.x() << ", " << com_diff.y() << ", "
              << com_diff.z() << ")" << std::endl;
  }
}

void test_HalfEdgeMeshData(const std::string &surf) {
  std::cout << "--------------------------------" << std::endl;
  std::cout << "--------------------------------" << std::endl;
  std::cout << "test_HalfEdgeMeshData()" << std::endl;
  std::string vf_ply = input_ply_dir + "/" + surf + "_vf.ply";
  TriMeshData vf_data = load_tri_mesh_data_from_ply(vf_ply, false);
  HalfEdgeMeshData he_data = buildHalfEdgeMeshDataFromTriMeshData(vf_data);
  std::cout << "Number of vertices: " << he_data.V.size() << std::endl;
  std::cout << "Number of edges: " << he_data.E_vertex.size() << std::endl;
  std::cout << "Number of faces: " << he_data.F_edge.size() << std::endl;
}

// void test_write_he_mesh_data_to_ply() {
//   std::string ply_path = "../data/ply_files/hex_patch.ply";
//   TriMeshData mesh = load_tri_mesh_data_from_ply(ply_path, false);
//   HalfEdgeMeshData he_data = buildHalfEdgeMeshDataFromTriMeshData(mesh);
//   std::string output_directory = "../data/ply_files";
//   std::string filename = "hex_patch_he.ply";
//   std::string comment = "this is but a test";
//   write_he_mesh_data_to_ply(he_data, output_directory, filename, true,
//   comment);
// }

// void test_load_he_mesh_data_from_ply() {
//   std::string ply_path = "../data/ply_files/hex_patch_he.ply";
//   HalfEdgeMeshData he_data = load_he_mesh_data_from_ply(ply_path, false,
//   true); std::cout << "Number of vertices: " << he_data.V.size() <<
//   std::endl; std::cout << "Number of edges: " << he_data.E_vertex.size() <<
//   std::endl; std::cout << "Number of faces: " << he_data.F_edge.size() <<
//   std::endl;
// }

// void test_ply_precision() {
//   std::string ply_path_original = "../data/ply_files/dumbbell.ply";
//   std::string output_directory = "../data/ply_files";
//   TriMeshData mesh_og = load_tri_mesh_data_from_ply(ply_path_original,
//   false);

//   std::string filename_yesbinary = "dumbbell_yesbinary.ply";
//   std::string ply_path_yesbinary = output_directory + "/" +
//   filename_yesbinary; std::string filename_nobinary =
//   "dumbbell_nobinary.ply"; std::string ply_path_nobinary = output_directory +
//   "/" + filename_nobinary;

//   write_tri_mesh_data_to_ply(mesh_og, output_directory, filename_yesbinary,
//                              true);
//   write_tri_mesh_data_to_ply(mesh_og, output_directory, filename_nobinary,
//                              false);

//   TriMeshData mesh_yb = load_tri_mesh_data_from_ply(ply_path_yesbinary,
//   false); TriMeshData mesh_nb =
//   load_tri_mesh_data_from_ply(ply_path_nobinary, false);

//   std::cout.precision(17); // Set precision to 17 decimal places
//   std::cout << "og x: " << mesh_og.vertices[0][0] << std::endl;
//   std::cout << "yes binary x: " << mesh_yb.vertices[0][0] << std::endl;
//   std::cout << "no binary x: " << mesh_nb.vertices[0][0] << std::endl;
// }

// int main() {
//   test_load_tri_mesh_data_from_ply();
//   test_load_write_to_ply();
//   test_HalfEdgeMeshData();
//   // test_write_he_mesh_data_to_ply();
//   // test_ply_precision();
//   // test_load_he_mesh_data_from_ply();
//   return EXIT_SUCCESS;
// }

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <surf>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string surf = argv[1];

  test_load_tri_mesh_data_from_ply(surf);
  test_load_write_to_ply(surf);
  test_HalfEdgeMeshData(surf);

  return EXIT_SUCCESS;
}