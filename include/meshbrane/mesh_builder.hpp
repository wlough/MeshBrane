#pragma once

/**
 * @file mesh_builder.hpp
 * @brief Mesh input/output.
 */

#include "meshbrane/meshbrane_data_types.hpp"
#include <fstream>   // std::ifstream
#include <istream>   // std::istream
#include <stdexcept> // std::runtime_error
#include <streambuf> // std::streambuf
#include <string>    // std::string
#include <vector>    // std::vector

/**
 * @defgroup MeshIO Mesh input/output
 * @brief Tools for reading and writing mesh data.
 * @details The `MeshLoader` class uses the tinyply library for reading and
 * writing .ply files.
 */

namespace meshbrane {
namespace mesh_io {
////////////////////////////////////////////
// misc tinyply helpers ////////////////////
////////////////////////////////////////////
/** @addtogroup utils
 *  @{
 */
/**
 * @brief Read a binary file into a vector of bytes.
 *
 * @param pathToFile
 * @return std::vector<uint8_t>
 */
inline std::vector<uint8_t> read_file_binary(const std::string &pathToFile) {
  std::ifstream file(pathToFile, std::ios::binary);
  std::vector<uint8_t> fileBufferBytes;

  if (file.is_open()) {
    file.seekg(0, std::ios::end);
    size_t sizeBytes = file.tellg();
    file.seekg(0, std::ios::beg);
    fileBufferBytes.resize(sizeBytes);
    if (file.read((char *)fileBufferBytes.data(), sizeBytes))
      return fileBufferBytes;
  } else
    throw std::runtime_error("could not open binary ifstream to path " +
                             pathToFile);
  return fileBufferBytes;
}

/**
 * @brief A streambuf that reads from a memory buffer.
 *
 */
struct memory_buffer : public std::streambuf {
  char *p_start{nullptr};
  char *p_end{nullptr};
  size_t size;

  memory_buffer(char const *first_elem, size_t size)
      : p_start(const_cast<char *>(first_elem)), p_end(p_start + size),
        size(size) {
    setg(p_start, p_start, p_end);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (dir == std::ios_base::cur)
      gbump(static_cast<int>(off));
    else
      setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
    return gptr() - p_start;
  }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(pos, std::ios_base::beg, which);
  }
};

/**
 * @brief A stream that reads from a memory buffer.
 *
 */
struct memory_stream : virtual memory_buffer, public std::istream {
  memory_stream(char const *first_elem, size_t size)
      : memory_buffer(first_elem, size),
        std::istream(static_cast<std::streambuf *>(this)) {}
};

// /**
//  * @brief A timer.
//  */
// class manual_timer {
//   std::chrono::high_resolution_clock::time_point t0;
//   double timestamp{0.0};

// public:
//   void start() { t0 = std::chrono::high_resolution_clock::now(); }
//   void stop() {
//     timestamp = std::chrono::duration<double>(
//                     std::chrono::high_resolution_clock::now() - t0)
//                     .count() *
//                 1000.0;
//   }
//   const double &get() { return timestamp; }
// };
/** @}*/ // end of group utils

////////////////////////////////////////////
// half-edge mesh funs /////////////////////
////////////////////////////////////////////
/**
 * @brief Get the index of twin half-edge
 * @param H Nhx2 array of half-edges.
 * @param h Index of half-edge.
 * @return Index of twin half-edge.
 */
int find_halfedge_index_of_twin(const meshbrane::Samples2i &H, const int &h);

// /**
//  * @brief
//  *
//  * @param h_twin_H
//  *
//  * @return Samples2i
//  */
// Samples2i he_samples_to_edges(const Samplesi &v_origin_H,
//                               const Samplesi &h_twin_H, Samplesi &V_cycle_E)
//                               {
//   std::unordered_set<size_t> Hmin; // keep track of half-edges already
//   processed
//                                    // by saving min(h, h_twin(h))
//   for (int h{0}; h < h_twin_H.size(); h++) {
//     size_t h_min = std::min(h, h_twin_H[h]);
//     if (Hmin.find(h_min) == Hmin.end()) {
//       V_cycle_E.row(Hmin.size()) << v_origin_H[h], v_origin_H[h_twin_H[h]];
//       Hmin.insert(h_min);
//     }
//   }
// }

/** @addtogroup MeshIO
 *  @{
 */
/**
 * @brief Convert vertex-face mesh data to half-edge mesh data. See:
 * `meshbrane::VertexFaceTuple`, `meshbrane::HalfEdgeTuple`.
 * @param xyz_coord_V Nvx3 Eigen matrix of vertex Cartesian coordinates.
 * @param vvv_of_F Nfx3 Eigen matrix of vertex indices of faces.
 * @return A tuple containing:
 */
meshbrane::HalfEdgeTuple
vf_samples_to_he_samples(const meshbrane::Samples3d &xyz_coord_V,
                         const meshbrane::Samples3i &V_cycle_F);

/**
 * @brief loads ply file into meshbrane::VertexFaceTuple tuple.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return meshbrane::VertexFaceTuple
 */
meshbrane::VertexFaceTuple
load_vf_samples_from_ply(const std::string &filepath,
                         const bool preload_into_memory = true,
                         const bool verbose = false);

VertexFaceTuple
he_samples_to_vf_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                         const Samplesi &v_origin_H, const Samplesi &h_next_H,
                         const Samplesi &h_twin_H, const Samplesi &f_left_H,
                         const Samplesi &h_right_F,
                         const Samplesi &h_negative_B);

VertexEdgeFaceTuple
he_samples_to_vef_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                          const Samplesi &v_origin_H, const Samplesi &h_next_H,
                          const Samplesi &h_twin_H, const Samplesi &f_left_H,
                          const Samplesi &h_right_F,
                          const Samplesi &h_negative_B);

/**
 * @brief loads ply file into `meshbrane::HalfEdgeTuple` structure.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return meshbrane::HalfEdgeTuple
 */
meshbrane::HalfEdgeTuple
load_he_samples_from_ply(const std::string &filepath,
                         const bool preload_into_memory = true,
                         const bool verbose = false);

/**
 * @brief writes meshbrane::VertexFaceTuple to a .ply file.
 *
 * @param xyz_coord_V
 * @param V_cycle_F
 * @param ply_path
 * @param use_binary
 */
void write_vf_samples_to_ply(meshbrane::Samples3d &xyz_coord_V,
                             meshbrane::Samples3i &V_cycle_F,
                             const std::string &ply_path,
                             const bool use_binary = true);

/**
 * @brief writes meshbrane::HalfEdgeTuple to a .ply file.
 *
 * @param xyz_coord_V
 * ...
 * @param ply_path
 * @param use_binary
 */
void write_he_samples_to_ply(
    const meshbrane::Samples3d &xyz_coord_V, const meshbrane::Samplesi &h_out_V,
    const meshbrane::Samplesi &v_origin_H, const meshbrane::Samplesi &h_next_H,
    const meshbrane::Samplesi &h_twin_H, const meshbrane::Samplesi &f_left_H,
    const meshbrane::Samplesi &h_right_F,
    const meshbrane::Samplesi &h_negative_B, const std::string &ply_path,
    const bool use_binary = true);

EdgeFaceCellTuple cmap_to_efc_tuple(const CombinatorialMapTuple &cm);

/**
 * @brief writes `SimplicialComplexData` to a .ply file.
 *
 * @param sc_data SimplicialComplexData
 * @param ply_path std::string
 * @param use_binary bool
 */
void write_simplicial_complex_data_to_ply(const SimplicialComplexData &sc_data,
                                          const std::string &ply_path,
                                          const bool use_binary = true);
// void write_simplicial_complex_data_to_ply(const SimplicialComplexData
// &sc_data,
//                                           const std::string &ply_path,
//                                           bool use_binary);
SimplicialComplexData
load_simplicial_complex_data_from_ply(const std::string &filepath,
                                      const bool preload_into_memory = true,
                                      const bool verbose = false);

////////////////////////////////////////////
// mesh converter //////////////////////////
////////////////////////////////////////////
class MeshBuilder {
public:
  /////////////////
  // Constructors /
  /////////////////
  MeshBuilder();

  static MeshBuilder from_vf_ply(const std::string &ply_path,
                                 bool compute_he_stuff = true);
  static MeshBuilder from_vf_samples(const meshbrane::Samples3d &xyz_coord_V,
                                     const meshbrane::Samples3i &V_cycle_F,
                                     bool compute_he_stuff = true);
  static MeshBuilder from_he_ply(const std::string &ply_path,
                                 bool compute_vf_stuff = true);
  static MeshBuilder from_he_samples(
      const meshbrane::Samples3d &xyz_coord_V,
      const meshbrane::Samplesi &h_out_V, const meshbrane::Samplesi &v_origin_H,
      const meshbrane::Samplesi &h_next_H, const meshbrane::Samplesi &h_twin_H,
      const meshbrane::Samplesi &f_left_H, const meshbrane::Samplesi &h_right_F,
      const meshbrane::Samplesi &h_negative_B, bool compute_vf_stuff = true);

  ///////////////
  // Attributes /
  ///////////////
  std::string vf_ply_path;
  meshbrane::VertexFaceTuple vf_samples;
  std::string he_ply_path;
  meshbrane::HalfEdgeTuple he_samples;

  ////////////
  // Methods /
  ////////////
  meshbrane::VertexEdgeFaceTuple get_vef_samples();

  void write_vf_ply(const std::string &ply_path, const bool use_binary = true);
  void write_he_ply(const std::string &ply_path, const bool use_binary = true);

private:
  //   std::string vf_ply_path;
  //   tinyply::PlyData vf_ply_data;
  //   meshbrane::VertexFaceTuple vf_samples;
  //   tinyply::PlyData he_ply_data;
  //   meshbrane::HalfEdgeTuple he_samples;
};

/** @}*/ // end of group MeshIO

} // namespace mesh_io
} // namespace meshbrane
