/**
 * @file mesh_builder.cpp
 */

#ifndef TINYPLY_IMPLEMENTATION
#define TINYPLY_IMPLEMENTATION
#endif

#include "meshbrane/mesh_builder.hpp"
#include "meshbrane/manual_timer.hpp"
#include "tinyply.h"
#include <algorithm>     // For std::min and std::max
#include <iostream>      // std::cout
#include <set>           // std::set
#include <tuple>         // std::tuple
#include <unordered_set> // std::unordered_set
#include <vector>        // std::vector

namespace meshbrane {

namespace mesh_io {
////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////

int find_halfedge_index_of_twin(const Samples2i &H, const int &h) {
  auto v0 = H(h, 0);
  auto v1 = H(h, 1);
  for (int h_twin = 0; h_twin < H.rows(); ++h_twin) {
    if ((H(h_twin, 0) == v1) && (H(h_twin, 1) == v0)) {
      return h_twin; // Return the index of the twin edge
    }
  }
  return -1; // Return -1 if no twin edge is found
}

HalfEdgeTuple vf_samples_to_he_samples(const Samples3d &xyz_coord_V,
                                       const Samples3i &V_cycle_F) {

  auto Nv = xyz_coord_V.rows();
  auto Nf = V_cycle_F.rows();
  // num interior + num positive boundary half-edges
  auto Nh0 = 3 * Nf;
  Samples2i H0 = Samples2i(Nh0, 2);
  // half-edge samples
  // h_out=Nh0 if not assigned
  // h_twin=-1 if not assigned
  Samplesi h_out_V = Samplesi::Constant(Nv, Nh0);
  Samplesi v_origin_H = Samplesi(Nh0);
  Samplesi h_next_H = Samplesi(Nh0);
  Samplesi h_twin_H = Samplesi::Constant(Nh0, -1);
  Samplesi f_left_H = Samplesi(Nh0);
  Samplesi h_right_F = Samplesi(Nf);
  Samplesi h_negative_B;
  // assign h_out for vertices to be minimum of outgoing half-edge indices
  // assign v_origin/f_left/h_next for half-edges in H0
  // assign h_bound for faces
  for (int f = 0; f < Nf; ++f) {
    h_right_F[f] = 3 * f;
    for (int i = 0; i < 3; ++i) {
      int h = 3 * f + i;
      int h_next = 3 * f + (i + 1) % 3;
      int v0 = V_cycle_F(f, i);
      int v1 = V_cycle_F(f, (i + 1) % 3);
      H0.row(h) << v0, v1;
      v_origin_H[h] = v0;
      f_left_H[h] = f;
      h_next_H[h] = h_next;
      // assign h_out for vertices if not already assigned
      // reassign if h is smaller than current h_out_V[v0]
      if (h_out_V[v0] > h) {
        h_out_V[v0] = h;
      }
    }
  }
  // Temporary containers for indices of +/- boundary half-edge
  std::vector<int> H_boundary_plus;
  std::unordered_set<int> H_boundary_minus;
  // find positive boundary half-edges
  // assign h_twin for interior half-edges
  for (int h = 0; h < H0.rows(); ++h) {
    // if h_twin_H[h] is already assigned, skip
    if (h_twin_H[h] != -1) {
      continue;
    }
    int h_twin = find_halfedge_index_of_twin(H0, h);
    if (h_twin == -1) {
      H_boundary_plus.push_back(h);
    } else {
      h_twin_H[h] = h_twin;
      h_twin_H[h_twin] = h;
    }
  }
  int Nh1 = H_boundary_plus.size();
  int Nh = Nh0 + Nh1;
  v_origin_H.conservativeResize(Nh);
  h_next_H.conservativeResize(Nh);
  h_twin_H.conservativeResize(Nh);
  f_left_H.conservativeResize(Nh);
  // define negative boundary half-edges
  // assign v_origin for negative boundary half-edges
  // assign h_twin for boundary half-edges
  for (int i = 0; i < Nh1; ++i) {
    int h = H_boundary_plus[i];
    int h_twin = Nh0 + i;
    // int v0 = H0(h, 0);
    int v1 = H0(h, 1);
    H_boundary_minus.insert(h_twin);
    v_origin_H[h_twin] = v1;
    h_twin_H[h] = h_twin;
    h_twin_H[h_twin] = h;
  }
  // enumerate boundaries b=0,1,...
  // assign h_right for boundaries
  // assign h_next for negative boundary half-edges
  // set f_left=-(b+1) for half-edges in boundary b
  while (!H_boundary_minus.empty()) {
    int b = h_negative_B.size();
    int h_negative_b = *H_boundary_minus.begin();
    h_negative_B.conservativeResize(b + 1);
    h_negative_B[b] = h_negative_b; // Assign new value
    int h = h_negative_b;
    // follow prev cycle along boundary b until we get back to h=h_negative_b
    do {
      int h_prev = h_twin_H[h];
      // rotate cw around origin of h until we find h_prev in boundary b
      // erase h from H_boundary_minus
      while (H_boundary_minus.find(h_prev) == H_boundary_minus.end()) {
        h_prev = h_twin_H[h_next_H[h_prev]];
      }
      h_next_H[h_prev] = h;
      h = h_prev;
      H_boundary_minus.erase(h);
      f_left_H[h] = -(b + 1);
    } while (h != h_negative_b);
  }
  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_right_F, h_negative_B);
}

VertexFaceTuple
he_samples_to_vf_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                         const Samplesi &v_origin_H, const Samplesi &h_next_H,
                         const Samplesi &h_twin_H, const Samplesi &f_left_H,
                         const Samplesi &h_right_F,
                         const Samplesi &h_negative_B) {

  int Nf = h_right_F.rows();
  Samples3i V_cycle_F = Samples3i(Nf, 3);
  for (int f = 0; f < Nf; ++f) {
    int h = h_right_F[f];
    for (int i = 0; i < 3; ++i) {
      V_cycle_F(f, i) = v_origin_H[h];
      h = h_next_H[h];
    }
  }
  return std::make_tuple(xyz_coord_V, V_cycle_F);
}

VertexEdgeFaceTuple
he_samples_to_vef_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                          const Samplesi &v_origin_H, const Samplesi &h_next_H,
                          const Samplesi &h_twin_H, const Samplesi &f_left_H,
                          const Samplesi &h_right_F,
                          const Samplesi &h_negative_B) {

  int Nf = h_right_F.rows();
  int Ne = v_origin_H.rows() / 2;
  Samples3i V_cycle_F = Samples3i(Nf, 3);
  Samples2i V_cycle_E = Samples2i(Ne, 2);
  std::set<std::vector<int>> setV_of_E;
  for (int f = 0; f < Nf; ++f) {
    int h0 = h_right_F[f];
    int h1 = h_next_H[h0];
    int h2 = h_next_H[h1];
    int v0 = v_origin_H[h0];
    int v1 = v_origin_H[h1];
    int v2 = v_origin_H[h2];
    V_cycle_F.row(f) << v0, v1, v2;
    std::vector<int> edge0 = {std::min(v0, v1), std::max(v0, v1)};
    std::vector<int> edge1 = {std::min(v1, v2), std::max(v1, v2)};
    std::vector<int> edge2 = {std::min(v2, v0), std::max(v2, v0)};
    // Insert edges into the set and update V_cycle_E
    if (setV_of_E.find(edge0) == setV_of_E.end()) {
      V_cycle_E.row(setV_of_E.size()) << edge0[0], edge0[1];
      setV_of_E.insert(edge0);
    }
    if (setV_of_E.find(edge1) == setV_of_E.end()) {
      V_cycle_E.row(setV_of_E.size()) << edge1[0], edge1[1];
      setV_of_E.insert(edge1);
    }
    if (setV_of_E.find(edge2) == setV_of_E.end()) {
      V_cycle_E.row(setV_of_E.size()) << edge2[0], edge2[1];
      setV_of_E.insert(edge2);
    }
  }
  return {xyz_coord_V, V_cycle_E, V_cycle_F};
}

VertexFaceTuple load_vf_samples_from_ply(const std::string &filepath,
                                         const bool preload_into_memory,
                                         const bool verbose) {
  std::streambuf *oldCoutStreamBuf = nullptr;
  std::ofstream nullStream;

  if (!verbose) {
    // Save the old buffer
    oldCoutStreamBuf = std::cout.rdbuf();

    // Redirect std::cout to /dev/null
    nullStream.open("/dev/null");
    std::cout.rdbuf(nullStream.rdbuf());
  }
  Samples3d xyz_coord_V;
  Samples3i V_cycle_F;
  std::cout << "..............................................................."
               ".........\n";
  std::cout << "Now Reading: " << filepath << std::endl;

  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;
  try {
    // For most files < 1gb, pre-loading the entire file upfront and wrapping it
    // into a stream is a net win for parsing speed, about 40% faster.
    if (preload_into_memory) {
      byte_buffer = read_file_binary(filepath);
      file_stream.reset(
          new memory_stream((char *)byte_buffer.data(), byte_buffer.size()));
    } else {
      file_stream.reset(new std::ifstream(filepath, std::ios::binary));
    }

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

    // Because most people have their own mesh types, tinyply treats parsed data
    // as structured/typed byte buffers.
    std::shared_ptr<tinyply::PlyData> vertices, normals, faces;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data.
    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      vertices =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }
    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 3);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    manual_timer read_timer;

    read_timer.start();
    file.read(*file_stream);
    read_timer.stop();

    const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
    std::cout << "\tparsing " << size_mb << "mb in " << parsing_time
              << " seconds [" << (size_mb / parsing_time) << " MBps]"
              << std::endl;

    if (vertices)
      std::cout << "\tRead " << vertices->count << " total vertices "
                << std::endl;
    if (faces)
      std::cout << "\tRead " << faces->count << " total faces (triangles) "
                << std::endl;

    // // convert to positions to Samples3d
    // const size_t numVerticesBytes = vertices->buffer.size_bytes();
    // xyz_coord_V.resize(vertices->count, 3);
    // std::memcpy(xyz_coord_V.data(), vertices->buffer.get(),
    // numVerticesBytes);
    // // convert faces to Samples3i
    // const size_t numFacesBytes = faces->buffer.size_bytes();
    // V_cycle_F.resize(faces->count, 3);
    // std::memcpy(V_cycle_F.data(), faces->buffer.get(), numFacesBytes);

    // Convert to positions to Samples3d
    const size_t numVertices = vertices->count;
    xyz_coord_V.resize(numVertices, 3);
    const double *vertexBuffer =
        reinterpret_cast<const double *>(vertices->buffer.get());
    for (size_t i = 0; i < numVertices; ++i) {
      xyz_coord_V(i, 0) = vertexBuffer[3 * i + 0]; // x
      xyz_coord_V(i, 1) = vertexBuffer[3 * i + 1]; // y
      xyz_coord_V(i, 2) = vertexBuffer[3 * i + 2]; // z
    }

    // Convert faces to Samples3i
    const size_t numFaces = faces->count;
    V_cycle_F.resize(numFaces, 3);
    const int *faceBuffer = reinterpret_cast<const int *>(faces->buffer.get());
    for (size_t i = 0; i < numFaces; ++i) {
      V_cycle_F(i, 0) = faceBuffer[3 * i + 0]; // vertex index 1
      V_cycle_F(i, 1) = faceBuffer[3 * i + 1]; // vertex index 2
      V_cycle_F(i, 2) = faceBuffer[3 * i + 2]; // vertex index 3
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return std::make_tuple(xyz_coord_V, V_cycle_F);
}

HalfEdgeTuple load_he_samples_from_ply(const std::string &filepath,
                                       const bool preload_into_memory,
                                       const bool verbose) {
  std::streambuf *oldCoutStreamBuf = nullptr;
  std::ofstream nullStream;

  if (!verbose) {
    // Save the old buffer
    oldCoutStreamBuf = std::cout.rdbuf();

    // Redirect std::cout to /dev/null
    nullStream.open("/dev/null");
    std::cout.rdbuf(nullStream.rdbuf());
  }

  Samples3d xyz_coord_V;
  Samplesi h_out_V;
  Samplesi v_origin_H;
  Samplesi h_next_H;
  Samplesi h_twin_H;
  Samplesi f_left_H;
  Samplesi h_right_F;
  Samplesi h_negative_B;

  std::cout << "..............................................................."
               ".........\n";
  std::cout << "Now Reading: " << filepath << std::endl;

  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;
  try {
    // For most files < 1gb, pre-loading the entire file upfront and wrapping it
    // into a stream is a net win for parsing speed, about 40% faster.
    if (preload_into_memory) {
      byte_buffer = read_file_binary(filepath);
      file_stream.reset(
          new memory_stream((char *)byte_buffer.data(), byte_buffer.size()));
    } else {
      file_stream.reset(new std::ifstream(filepath, std::ios::binary));
    }

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

    // Because most people have their own mesh types, tinyply treats parsed data
    // as structured/typed byte buffers.
    // std::shared_ptr<tinyply::PlyData> vertices, normals, faces;
    // std::shared_ptr<tinyply::PlyData> V, V_edge, F_edge, E_vertex, E_face,
    //     E_next, E_twin;
    std::shared_ptr<tinyply::PlyData> xyz_coord_V_, h_out_V_, v_origin_H_,
        h_next_H_, h_twin_H_, f_left_H_, h_right_F_, h_negative_B_;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data.
    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.

    // vertex
    // half_edge
    // face
    // boundary
    try {
      xyz_coord_V_ =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      h_out_V_ = file.request_properties_from_element("vertex", {"h"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      v_origin_H_ = file.request_properties_from_element("half_edge", {"v"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      h_next_H_ = file.request_properties_from_element("half_edge", {"n"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      h_twin_H_ = file.request_properties_from_element("half_edge", {"t"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      f_left_H_ = file.request_properties_from_element("half_edge", {"f"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      h_right_F_ = file.request_properties_from_element("face", {"h"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      h_negative_B_ = file.request_properties_from_element("boundary", {"h"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    manual_timer read_timer;

    read_timer.start();
    file.read(*file_stream);
    read_timer.stop();

    const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
    std::cout << "\tparsing " << size_mb << "mb in " << parsing_time
              << " seconds [" << (size_mb / parsing_time) << " MBps]"
              << std::endl;

    const size_t Nv = h_out_V_->count;
    const size_t Nh = v_origin_H_->count;
    const size_t Nf = h_right_F_->count;
    const size_t Nb = h_negative_B_->count;

    xyz_coord_V.resize(Nv, 3);
    h_out_V.resize(Nv);
    v_origin_H.resize(Nh);
    h_next_H.resize(Nh);
    h_twin_H.resize(Nh);
    f_left_H.resize(Nh);
    h_right_F.resize(Nf);
    h_negative_B.resize(Nb);

    std::cout << "\tRead " << Nv << " total vertices " << std::endl;

    std::cout << "\tRead " << Nh << " total half-edges " << std::endl;

    std::cout << "\tRead " << Nf << " total faces (triangles) " << std::endl;

    std::cout << "\tRead " << Nb << " total boundaries (connected components) "
              << std::endl;

    // // convert to positions to Eigen::Vector3d
    // {
    //   const size_t numVerticesBytes = V->buffer.size_bytes();
    //   mesh.V.resize(V->count);
    //   std::memcpy(mesh.V.data(), V->buffer.get(), numVerticesBytes);
    // }

    // {
    //   const size_t numEdgesBytes = V_edge->buffer.size_bytes();
    //   mesh.V_edge.resize(V_edge->count);
    //   std::memcpy(mesh.V_edge.data(), V_edge->buffer.get(), numEdgesBytes);
    // }

    // {
    //   const size_t numEdgesBytes = E_vertex->buffer.size_bytes();
    //   mesh.E_vertex.resize(E_vertex->count);
    //   std::memcpy(mesh.E_vertex.data(), E_vertex->buffer.get(),
    //   numEdgesBytes);
    // }

    // {
    //   const size_t numEdgesBytes = E_face->buffer.size_bytes();
    //   mesh.E_face.resize(E_face->count);
    //   std::memcpy(mesh.E_face.data(), E_face->buffer.get(), numEdgesBytes);
    // }

    // {
    //   const size_t numEdgesBytes = E_next->buffer.size_bytes();
    //   mesh.E_next.resize(E_next->count);
    //   std::memcpy(mesh.E_next.data(), E_next->buffer.get(), numEdgesBytes);
    // }

    // {
    //   const size_t numEdgesBytes = E_twin->buffer.size_bytes();
    //   mesh.E_twin.resize(E_twin->count);
    //   std::memcpy(mesh.E_twin.data(), E_twin->buffer.get(), numEdgesBytes);
    // }

    // // convert faces to...
    // {
    //   const size_t numFacesBytes = F_edge->buffer.size_bytes();
    //   mesh.F_edge.resize(F_edge->count);
    //   std::memcpy(mesh.F_edge.data(), F_edge->buffer.get(), numFacesBytes);
    // }
    // ply data to Samples3d,Samplesi
    const double *xyz_coord_V_buffer =
        reinterpret_cast<const double *>(xyz_coord_V_->buffer.get());
    const int *h_out_V_buffer =
        reinterpret_cast<const int *>(h_out_V_->buffer.get());
    const int *v_origin_H_buffer =
        reinterpret_cast<const int *>(v_origin_H_->buffer.get());
    const int *h_next_H_buffer =
        reinterpret_cast<const int *>(h_next_H_->buffer.get());
    const int *h_twin_H_buffer =
        reinterpret_cast<const int *>(h_twin_H_->buffer.get()); // h_twin
    const int *f_left_H_buffer =
        reinterpret_cast<const int *>(f_left_H_->buffer.get()); // f_left
    const int *h_right_F_buffer =
        reinterpret_cast<const int *>(h_right_F_->buffer.get()); // h_bound
    const int *h_negative_B_buffer =
        reinterpret_cast<const int *>(h_negative_B_->buffer.get()); // h_right
    for (size_t i = 0; i < Nv; ++i) {
      xyz_coord_V(i, 0) = xyz_coord_V_buffer[3 * i + 0]; // x
      xyz_coord_V(i, 1) = xyz_coord_V_buffer[3 * i + 1]; // y
      xyz_coord_V(i, 2) = xyz_coord_V_buffer[3 * i + 2]; // z
      h_out_V(i) = h_out_V_buffer[i];                    // h_out
    }
    for (size_t i = 0; i < Nh; ++i) {
      v_origin_H(i) = v_origin_H_buffer[i]; // v_origin
      h_next_H(i) = h_next_H_buffer[i];     // h_next
      h_twin_H(i) = h_twin_H_buffer[i];     // h_twin
      f_left_H(i) = f_left_H_buffer[i];     // f_left
    }
    for (size_t i = 0; i < Nf; ++i) {
      h_right_F(i) = h_right_F_buffer[i]; // h_bound
    }
    for (size_t i = 0; i < Nb; ++i) {
      h_negative_B(i) = h_negative_B_buffer[i]; // h_right
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_right_F, h_negative_B);
}

// void write_vf_samples_to_ply(Samples3d &xyz_coord_V,
// Samples3i &V_cycle_F,
//                              const std::string &output_directory,
//                              const std::string &filename,
//                              const bool useBinary)
void write_vf_samples_to_ply(Samples3d &xyz_coord_V, Samples3i &V_cycle_F,
                             const std::string &ply_path,
                             const bool use_binary) {

  // std::string ply_path = output_directory + "/" + filename;

  std::filebuf fb;
  fb.open(ply_path,
          use_binary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + ply_path);

  tinyply::PlyFile mesh_file;

  // mesh_file.add_properties_to_element(
  //     "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64, xyz_coord_V.rows(),
  //     reinterpret_cast<uint8_t *>(const_cast<double *>(xyz_coord_V.data())),
  //     tinyply::Type::INVALID, 0);

  // mesh_file.add_properties_to_element(
  //     "face", {"vertex_indices"}, tinyply::Type::INT32, V_cycle_F.rows(),
  //     reinterpret_cast<uint8_t *>(const_cast<int *>(V_cycle_F.data())),
  //     tinyply::Type::UINT8, V_cycle_F.cols());
  // Convert to row-major storage
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      xyz_coord_V_row_major = xyz_coord_V;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      V_cycle_F_row_major = V_cycle_F;

  mesh_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
      xyz_coord_V_row_major.rows(),
      reinterpret_cast<uint8_t *>(xyz_coord_V_row_major.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32,
      V_cycle_F_row_major.rows(),
      reinterpret_cast<uint8_t *>(V_cycle_F_row_major.data()),
      tinyply::Type::UINT8, V_cycle_F_row_major.cols());

  mesh_file.get_comments().push_back("MeshBrane vf_ply");

  // Write an ply file
  mesh_file.write(outstream, use_binary);
}

void write_he_samples_to_ply(
    const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
    const Samplesi &v_origin_H, const Samplesi &h_next_H,
    const Samplesi &h_twin_H, const Samplesi &f_left_H,
    const Samplesi &h_right_F, const Samplesi &h_negative_B,
    const std::string &ply_path, const bool use_binary) {

  std::filebuf fb;
  fb.open(ply_path,
          use_binary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + ply_path);

  tinyply::PlyFile mesh_file;

  // Convert to row-major storage
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      xyz_coord_V_row_major = xyz_coord_V;

  mesh_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
      xyz_coord_V_row_major.rows(),
      reinterpret_cast<uint8_t *>(xyz_coord_V_row_major.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "vertex", {"h"}, tinyply::Type::INT32, h_out_V.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_out_V.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "half_edge", {"v"}, tinyply::Type::INT32, v_origin_H.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(v_origin_H.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "half_edge", {"n"}, tinyply::Type::INT32, h_next_H.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_next_H.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "half_edge", {"t"}, tinyply::Type::INT32, h_twin_H.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_twin_H.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "half_edge", {"f"}, tinyply::Type::INT32, f_left_H.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(f_left_H.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "face", {"h"}, tinyply::Type::INT32, h_right_F.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_right_F.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "boundary", {"h"}, tinyply::Type::INT32, h_negative_B.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_negative_B.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.get_comments().push_back("MeshBrane he_ply");

  // Write an ply file
  mesh_file.write(outstream, use_binary);
}

////////////////////////////////////////////////////////////////
// MeshBuilder
////////////////////////////////////////////////////////////////
/////////////////
// Constructors /
/////////////////

MeshBuilder::MeshBuilder() : he_ply_path("") {}

MeshBuilder MeshBuilder::from_vf_ply(const std::string &ply_path,
                                     bool compute_he_stuff) {
  MeshBuilder mesh_converter;
  mesh_converter.vf_ply_path = ply_path;
  mesh_converter.vf_samples = load_vf_samples_from_ply(ply_path);
  if (compute_he_stuff) {
    mesh_converter.he_samples =
        vf_samples_to_he_samples(std::get<0>(mesh_converter.vf_samples),
                                 std::get<1>(mesh_converter.vf_samples));
  }
  return mesh_converter;
}
MeshBuilder MeshBuilder::from_vf_samples(const Samples3d &xyz_coord_V,
                                         const Samples3i &V_cycle_F,
                                         bool compute_he_stuff) {

  MeshBuilder mesh_converter;
  mesh_converter.vf_samples = std::make_tuple(xyz_coord_V, V_cycle_F);
  if (compute_he_stuff) {
    mesh_converter.he_samples =
        vf_samples_to_he_samples(xyz_coord_V, V_cycle_F);
  }
  return mesh_converter;
}
MeshBuilder MeshBuilder::from_he_ply(const std::string &ply_path,
                                     bool compute_vf_stuff) {
  MeshBuilder mesh_converter;
  mesh_converter.he_ply_path = ply_path;
  mesh_converter.he_samples = load_he_samples_from_ply(ply_path);
  if (compute_vf_stuff) {
    mesh_converter.vf_samples =
        he_samples_to_vf_samples(std::get<0>(mesh_converter.he_samples),
                                 std::get<1>(mesh_converter.he_samples),
                                 std::get<2>(mesh_converter.he_samples),
                                 std::get<3>(mesh_converter.he_samples),
                                 std::get<4>(mesh_converter.he_samples),
                                 std::get<5>(mesh_converter.he_samples),
                                 std::get<6>(mesh_converter.he_samples),
                                 std::get<7>(mesh_converter.he_samples));
  }
  return mesh_converter;
}
MeshBuilder MeshBuilder::from_he_samples(
    const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
    const Samplesi &v_origin_H, const Samplesi &h_next_H,
    const Samplesi &h_twin_H, const Samplesi &f_left_H,
    const Samplesi &h_right_F, const Samplesi &h_negative_B,
    bool compute_vf_stuff) {

  MeshBuilder mesh_converter;
  mesh_converter.he_samples =
      std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                      f_left_H, h_right_F, h_negative_B);
  if (compute_vf_stuff) {
    mesh_converter.vf_samples =
        he_samples_to_vf_samples(xyz_coord_V, h_out_V, v_origin_H, h_next_H,
                                 h_twin_H, f_left_H, h_right_F, h_negative_B);
  }
  return mesh_converter;
}
////////////
// Methods /
////////////
void MeshBuilder::write_vf_ply(const std::string &ply_path,
                               const bool use_binary) {
  write_vf_samples_to_ply(std::get<0>(vf_samples), std::get<1>(vf_samples),
                          ply_path, use_binary);
}

void MeshBuilder::write_he_ply(const std::string &ply_path,
                               const bool use_binary) {
  write_he_samples_to_ply(
      std::get<0>(he_samples), std::get<1>(he_samples), std::get<2>(he_samples),
      std::get<3>(he_samples), std::get<4>(he_samples), std::get<5>(he_samples),
      std::get<6>(he_samples), std::get<7>(he_samples), ply_path, use_binary);
}

VertexEdgeFaceTuple MeshBuilder::get_vef_samples() {
  return he_samples_to_vef_samples(
      std::get<0>(he_samples), std::get<1>(he_samples), std::get<2>(he_samples),
      std::get<3>(he_samples), std::get<4>(he_samples), std::get<5>(he_samples),
      std::get<6>(he_samples), std::get<7>(he_samples));
}

} // namespace mesh_io
} // namespace meshbrane
