#pragma once

/**
 * @file mesh_builder.hpp
 * @brief Mesh input/output.
 */

#include "meshbrane/meshbrane_data_types.hpp"
#include <filesystem>
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
 * @param path_to_file
 * @return std::vector<uint8_t>
 */
inline std::vector<std::uint8_t>
read_file_binary(const std::filesystem::path &path_to_file) {
  std::ifstream file(path_to_file,
                     std::ios::binary |
                         std::ios::ate); // open file in binary mode with read
                                         // position at end of file
  if (!file) {
    throw std::runtime_error("could not open binary file: " +
                             path_to_file.string());
  }
  const std::streamsize sizeBytes = file.tellg();

  if (sizeBytes < 0) {
    throw std::runtime_error("could not determine file size: " +
                             path_to_file.string());
  }
  std::vector<std::uint8_t> buffer(static_cast<std::size_t>(sizeBytes));
  file.seekg(0, std::ios::beg); // move read position to begining of file
  if (!file.read(reinterpret_cast<char *>(buffer.data()), sizeBytes)) {
    throw std::runtime_error("could not read binary file: " +
                             path_to_file.string());
  }
  return buffer;
}

// inline std::vector<uint8_t> read_file_binary(const std::string &pathToFile) {
//   std::ifstream file(pathToFile, std::ios::binary);

//   std::vector<uint8_t> fileBufferBytes;

//   if (file.is_open()) {
//     file.seekg(0, std::ios::end);
//     size_t sizeBytes = file.tellg();
//     file.seekg(0, std::ios::beg);
//     fileBufferBytes.resize(sizeBytes);
//     if (file.read((char *)fileBufferBytes.data(), sizeBytes))
//       return fileBufferBytes;
//   } else
//     throw std::runtime_error("could not open binary ifstream to path " +
//                              pathToFile);
//   return fileBufferBytes;
// }

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
 * @brief loads ply file into meshbrane::VertexFaceTuple tuple.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return meshbrane::VertexFaceTuple
 */
meshbrane::VertexFaceTuple
load_vf_samples_from_ply(const std::filesystem::path &filepath,
                         const bool preload_into_memory = true,
                         const bool verbose = false);

/**
 * @brief loads ply file into `meshbrane::HalfEdgeTuple` structure.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return meshbrane::HalfEdgeTuple
 */
meshbrane::HalfEdgeTuple
load_he_samples_from_ply(const std::filesystem::path &filepath,
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
                             const std::filesystem::path &ply_path,
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
    const meshbrane::Samplesi &h_negative_B,
    const std::filesystem::path &ply_path, const bool use_binary = true);

////////////////////////////////////////////
// mesh builder //////////////////////////
////////////////////////////////////////////
class MeshBuilder {
public:
  /////////////////
  // Constructors /
  /////////////////
  MeshBuilder();

  static MeshBuilder from_vf_ply(const std::filesystem::path &ply_path,
                                 bool compute_he_stuff = true);
  static MeshBuilder from_vf_samples(const meshbrane::Samples3d &xyz_coord_V,
                                     const meshbrane::Samples3i &V_cycle_F,
                                     bool compute_he_stuff = true);
  static MeshBuilder from_he_ply(const std::filesystem::path &ply_path,
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
  std::filesystem::path vf_ply_path;
  meshbrane::VertexFaceTuple vf_samples;
  std::filesystem::path he_ply_path;
  meshbrane::HalfEdgeTuple he_samples;

  ////////////
  // Methods /
  ////////////
  meshbrane::VertexEdgeFaceTuple get_vef_samples();

  void write_vf_ply(const std::filesystem::path &ply_path,
                    const bool use_binary = true);
  void write_he_ply(const std::filesystem::path &ply_path,
                    const bool use_binary = true);
};

/** @}*/ // end of group MeshIO

} // namespace mesh_io
} // namespace meshbrane
