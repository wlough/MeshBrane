/**
 * @file ply_utils.hpp
 */

#ifndef ply_utils_hpp
#define ply_utils_hpp

// #include "data_types.hpp" // for _INT_TYPE_ and _FLOAT_TYPE_
#include <Eigen/Dense> // for Eigen::Vector3d
#include <chrono> // for std::chrono::high_resolution_clock and std::chrono::duration
#include <fstream>   // for std::ifstream
#include <istream>   // for std::istream
#include <stdexcept> // for std::runtime_error
#include <streambuf> // for std::streambuf
#include <string>    // for std::string
#include <vector>    // for std::vector

////////////////////////////////////////////
// ply_utils ///////////////////////////////
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

/**
 * @brief Container for mesh data loaded from a .ply file.
 *
 */
struct TriMeshData {
  std::vector<Eigen::Vector3d> vertices;
  std::vector<std::array<uint32_t, 3>> faces;
};

/**
 * @brief Container for half-edge mesh data.
 *
 */
struct HalfEdgeMeshData {
  HalfEdgeMeshData() {}
  HalfEdgeMeshData(uint32_t Nvertices, uint32_t Nedges, uint32_t Nfaces)
      : V(Nvertices), V_edge(Nvertices), E_vertex(Nedges), E_face(Nedges),
        E_next(Nedges), E_twin(Nedges, -2), F_edge(Nfaces) {}
  std::vector<Eigen::Vector3d> V; // vertex positions
  std::vector<uint32_t> V_edge;   // index of an outgoing half-edge
  std::vector<uint32_t>
      E_vertex; // index of vertex the half-edge is incident upon
  std::vector<uint32_t> E_face; // index of face the half-edge is bounding
  std::vector<uint32_t> E_next; // index of next half-edge
  std::vector<int32_t> E_twin;  // index of twin half-edge, -1 if no twin
  std::vector<uint32_t> F_edge; // index of a half-edge bounding the face
};

/**
 * @brief loads ply file into TriMeshData structure.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return TriMeshData
 */
TriMeshData load_tri_mesh_data_from_ply(const std::string &filepath,
                                        const bool preload_into_memory = true,
                                        const bool verbose = false);

/**
 * @brief writes TriMeshData to a .ply file.
 *
 * @param mesh_data
 * @param output_directory
 * @param filename
 * @param useBinary
 */
void write_tri_mesh_data_to_ply(TriMeshData &mesh_data,
                                const std::string &output_directory,
                                const std::string &filename,
                                const bool useBinary = false);

HalfEdgeMeshData buildHalfEdgeMeshDataFromTriMeshData(const TriMeshData &data);

/**
 * @brief Get the index of twin half-edge
 */
int32_t get_index_of_twin(const std::vector<std::array<uint32_t, 2>> &E,
                          const uint32_t &e);

/**
 * @brief Writes HalfEdgeMeshData to a .ply file.
 *
 * @param mesh_data
 * @param output_directory
 * @param filename
 * @param useBinary
 * @param comment
 */
void write_he_mesh_data_to_ply(HalfEdgeMeshData &mesh_data,
                               const std::string &output_directory,
                               const std::string &filename,
                               const bool useBinary = false,
                               const std::string &comment = "");

/**
 * @brief loads ply file into HalfEdgeMeshData structure.
 *
 * @param filepath
 * @param preload_into_memory
 * @param verbose
 * @return HalfEdgeMeshData
 */
HalfEdgeMeshData
load_he_mesh_data_from_ply(const std::string &filepath,
                           const bool preload_into_memory = true,
                           const bool verbose = false);

/**
 * @brief Container for half-edge mesh data.
 *
 */

struct HalfEdgeData {
  HalfEdgeData() {}
  HalfEdgeData(uint32_t num_vertices, uint32_t num_half_edges,
               uint32_t num_faces, uint32_t num_boundaries)
      : xyz_coord_V(num_vertices), h_out_V(num_vertices),
        v_origin_H(num_half_edges), h_next_H(num_half_edges),
        h_twin_H(num_half_edges), f_left_H(num_half_edges),
        h_bound_F(num_faces), h_right_B(num_boundaries) {}
  std::vector<Eigen::Vector3d> xyz_coord_V; // vertex positions
  std::vector<int32_t> h_out_V;             // index of an outgoing half-edge
  std::vector<int32_t>
      v_origin_H; // index of vertex the half-edge is incident upon
  std::vector<int32_t> h_next_H;  // index of face the half-edge is bounding
  std::vector<int32_t> h_twin_H;  // index of next half-edge
  std::vector<int32_t> f_left_H;  // index of twin half-edge, -1 if no twin
  std::vector<int32_t> h_bound_F; // index of a half-edge bounding the face
  std::vector<int32_t> h_right_B; // index of a half-edge bounding the face
};
#endif /* ply_utils_hpp */
