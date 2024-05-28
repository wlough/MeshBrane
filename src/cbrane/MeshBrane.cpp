// #include <tinyply.h>
// #include <fstream>
// #include <vector>
// #include <iostream>
// #include <cstring>
// #include <brane_utils.hpp>
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <MeshBrane.hpp>
#include <fstream> // std::ifstream

BraneBuilder::BraneBuilder(const std::string &filepath) : filepath_(filepath) {
  set_VHF_from_ply();
};

void BraneBuilder::set_VHF_from_ply() {
  auto &filepath = this->filepath_;
  auto &V = this->V_;
  auto &H = this->H_;
  auto &F = this->F_;
  V.clear();
  H.clear();
  F.clear();
  std::cout << "..............................................................."
               ".........\n";
  std::cout << "Now Reading: " << filepath_ << std::endl;

  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;

  try {
    // For most files < 1gb, pre-loading the entire file upfront and wrapping it
    // into a stream is a net win for parsing speed, about 40% faster.

    file_stream.reset(new std::ifstream(filepath, std::ios::binary));

    if (!file_stream || file_stream->fail())
      throw std::runtime_error("file_stream failed to open " + filepath);

    file_stream->seekg(0, std::ios::end);
    const float size_mb = file_stream->tellg() * float(1e-6);
    file_stream->seekg(0, std::ios::beg);

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    std::cout << "\t[ply_header] Type: "
              << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto &c : file.get_comments())
      std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto &c : file.get_info())
      std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : file.get_elements()) {
      std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
                << std::endl;
      for (const auto &p : e.properties) {
        std::cout << "\t[ply_header] \tproperty: " << p.name
                  << " (type=" << tinyply::PropertyTable[p.propertyType].str
                  << ")";
        if (p.isList)
          std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                    << ")";
        std::cout << std::endl;
      }
    }

    // tinyply treats parsed data
    // as structured/typed byte buffers.
    std::shared_ptr<tinyply::PlyData> vertices, faces;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data.
    try {
      vertices =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    file.read(*file_stream);

    if (vertices)
      std::cout << "\tRead " << vertices->count << " total vertices "
                << std::endl;
    if (faces)
      std::cout << "\tRead " << faces->count << " total faces (triangles) "
                << std::endl;

    // Convert tinyply vertices to std::vector<Eigen::Vector3d> V
    {
      const size_t numVerticesBytes =
          vertices->buffer.size_bytes(); // get the size of the buffer
      V.resize(vertices->count);         // resize the vertices vector
      std::memcpy(V.data(), vertices->buffer.get(), numVerticesBytes);
    }

    // Convert tinyply faces to std::vector<std::array<uint32_t, 3>> F
    {
      const size_t numFacesBytes = faces->buffer.size_bytes();
      F.resize(faces->count);
      std::memcpy(F.data(), faces->buffer.get(), numFacesBytes);
    }
  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  // Set std::vector<std::array<uint32_t, 2>> H from
  // std::vector<std::array<uint32_t, 3>> F
  for (size_t f = 0; f < F.size(); ++f) {
    std::array<uint32_t, 3> face = F[f];
    for (size_t v = 0; v < 3; ++v) {
      std::array<uint32_t, 2> half_edge_vertex_index = {face[v],
                                                        face[(v + 1) % 3]};
      H.push_back(half_edge_vertex_index);
    }
  }

  std::uint32_t Nedges = H.size();
  std::uint32_t Nvertices = V.size();
  std::uint32_t Nfaces = F.size();
  this->V_hedge_.resize(Nvertices);
  this->F_hedge_.resize(Nfaces);
  this->H_next_.resize(Nedges);
  this->H_twin_.resize(Nedges, -1);
  for (size_t i = 0; i < Nedges; ++i) {
    std::array<uint32_t, 2> half_edge = H[i];
    uint32_t v0 = half_edge[0];
    uint32_t v1 = half_edge[1];
    V_hedge_[v0] = i;
    F_hedge_[i / 3] = i;
    H_vertex_.push_back(v0);
    H_next_[i] = i + 1;
  }
}

int32_t BraneBuilder::get_twin_index(uint32_t i) {

  std::array<uint32_t, 2> half_edge = H_[i];
  std::array<uint32_t, 2> twin_half_edge = {half_edge[1], half_edge[0]};

  auto it = std::find(H_.begin(), H_.end(), twin_half_edge);

  if (it != H_.end()) {
    return std::distance(H_.begin(), it);
  } else {
    return -1; // Return -1 if the twin half edge is not found in H
  }
}
// void BraneBuilder::set_half_edges() {
//   auto &H = this->H_;
//   H.clear();
//   for (size_t i = 0; i < F_.size(); ++i) {
//     std::array<uint32_t, 3> face = F_[i];
//     for (size_t j = 0; j < 3; ++j) {
//       std::array<uint32_t, 2> half_edge_vertex_index = {face[j],
//                                                         face[(j + 1) % 3]};
//       H.push_back(half_edge_vertex_index);
//     }
//   }
// };
void BraneBuilder::set_half_edges() { std::vector<HE_edge> edges; }

Brane::Brane(const HalfEdgeMesh &halfEdgeMesh, const BraneParams &braneParams)
    : halfEdgeMesh_(halfEdgeMesh), braneParams_(braneParams){};

// int main() {
//     TriangulatedSurface surface("./data/ply_files/dumbbell.ply"); // Replace
//     with your .ply file path

//     // Now you can access the vertices and faces of the surface
//     std::cout << "Number of vertices: " << surface.vertices.size() / 3 <<
//     "\n"; std::cout << "Number of faces: " << surface.faces.size() / 3;

//     return 0;
// }
