/**
 * @file cply_tools.hpp
 */
#ifndef CPLY_TOOLS_HPP
#define CPLY_TOOLS_HPP
#include <chrono> // for std::chrono::high_resolution_clock and std::chrono::duration
#include <data_types.hpp>
#include <fstream>   // for std::ifstream
#include <istream>   // for std::istream
#include <stdexcept> // for std::runtime_error
#include <streambuf> // for std::streambuf
#include <string>    // for std::string
#include <vector>    // for std::vector

////////////////////////////////////////////
// misc tinyply helpers ////////////////////
////////////////////////////////////////////
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

struct memory_stream : virtual memory_buffer, public std::istream {
  memory_stream(char const *first_elem, size_t size)
      : memory_buffer(first_elem, size),
        std::istream(static_cast<std::streambuf *>(this)) {}
};

class manual_timer {
  std::chrono::high_resolution_clock::time_point t0;
  double timestamp{0.0};

public:
  void start() { t0 = std::chrono::high_resolution_clock::now(); }
  void stop() {
    timestamp = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0)
                    .count() *
                1000.0;
  }
  const double &get() { return timestamp; }
};

////////////////////////////////////////////
// half-edge mesh funs /////////////////////
////////////////////////////////////////////
/**
 * @brief Get the index of twin half-edge
 *
 * @param H Nhx2 array of half-edges.
 */
INT_TYPE find_halfedge_index_of_twin(const Samples2i &H, const INT_TYPE &h);

/**
 * @brief Convert vertex-face mesh data to half-edge mesh data.
 *
 * @param xyz_coord_V Nvx3 Eigen matrix of vertex Cartesian coordinates.
 * @param vvv_of_F Nfx3 Eigen matrix of vertex indices of faces.
 * @return A tuple containing:
 * - xyz_coord_V: Nvx3 Eigen matrix of vertex Cartesian coordinates.
 * - h_out_V: Eigen matrix where h_out_V[i] is some outgoing half-edge incident
 * on vertex i.
 * - v_origin_H: Eigen matrix where v_origin_H[j] is the vertex at the origin of
 * half-edge j.
 * - h_next_H: Eigen matrix where h_next_H[j] is the next half-edge after
 * half-edge j in the face cycle.
 * - h_twin_H: Eigen matrix where h_twin_H[j] is the half-edge antiparallel to
 * half-edge j.
 * - f_left_H: Eigen matrix where f_left_H[j] is the face to the left of
 * half-edge j, if j is in the interior or a positively oriented boundary of M,
 * or the boundary to the left of half-edge j, if j is in a negatively oriented
 * boundary.
 * - h_bound_F: Eigen matrix where h_bound_F[k] is some half-edge on the
 * boundary of face k.
 * - h_right_B: Eigen matrix where h_right_B[n] is the half-edge to the right of
 * boundary n.
 */
HalfEdgeSamples vf_samples_to_he_samples(const Samples3d &xyz_coord_V,
                                         const Samples3i &V_of_F);

/**
 * @brief loads ply file into VertexFaceSamples tuple.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return VertexFaceSamples
 */
VertexFaceSamples
load_vf_samples_from_ply(const std::string &filepath,
                         const bool preload_into_memory = true,
                         const bool verbose = false);

/**
 * @brief loads ply file into HalfEdgeMeshData structure.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return HalfEdgeSamples
 */
HalfEdgeSamples load_he_samples_from_ply(const std::string &filepath,
                                         const bool preload_into_memory = true,
                                         const bool verbose = false);

/**
 * @brief writes VertexFaceSamples to a .ply file.
 *
 * @param xyz_coord_V
 * @param V_of_F
 * @param ply_path
 * @param use_binary
 */

void write_vf_samples_to_ply(Samples3d &xyz_coord_V, Samples3i &V_of_F,
                             const std::string &ply_path,
                             const bool use_binary = true);

/**
 * @brief writes HalfEdgeSamples to a .ply file.
 *
 * @param xyz_coord_V
 * ...
 * @param ply_path
 * @param use_binary
 */
void write_he_samples_to_ply(
    const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
    const Samplesi &v_origin_H, const Samplesi &h_next_H,
    const Samplesi &h_twin_H, const Samplesi &f_left_H,
    const Samplesi &h_bound_F, const Samplesi &h_right_B,
    const std::string &ply_path, const bool use_binary = true);
////////////////////////////////////////////
// mesh converter //////////////////////////
////////////////////////////////////////////
class MeshConverter {
public:
  /////////////////
  // Constructors /
  /////////////////
  MeshConverter();

  static MeshConverter from_vf_ply(const std::string &ply_path,
                                   bool compute_he_stuff = true);
  static MeshConverter from_vf_samples(const Samples3d &xyz_coord_V,
                                       const Samples3i &V_of_F,
                                       bool compute_he_stuff = true);
  static MeshConverter from_he_ply(const std::string &ply_path,
                                   bool compute_vf_stuff = true);
  static MeshConverter
  from_he_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                  const Samplesi &v_origin_H, const Samplesi &h_next_H,
                  const Samplesi &h_twin_H, const Samplesi &f_left_H,
                  const Samplesi &h_bound_F, const Samplesi &h_right_B,
                  bool compute_vf_stuff = true);

  ///////////////
  // Attributes /
  ///////////////
  std::string vf_ply_path;
  VertexFaceSamples vf_samples;
  std::string he_ply_path;
  HalfEdgeSamples he_samples;

  ////////////
  // Methods /
  ////////////
  void write_vf_ply(const std::string &ply_path, const bool use_binary = true);
  void write_he_ply(const std::string &ply_path, const bool use_binary = true);

private:
  //   std::string vf_ply_path;
  //   tinyply::PlyData vf_ply_data;
  //   VertexFaceSamples vf_samples;
  //   tinyply::PlyData he_ply_data;
  //   HalfEdgeSamples he_samples;
};

#endif /* CPLY_TOOLS_HPP */