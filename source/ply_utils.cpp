/**
 * @file ply_utils.cpp
 */

#define TINYPLY_IMPLEMENTATION
#include "ply_utils.hpp"
#include "tinyply.h"
#include <iostream>

int32_t get_index_of_twin(const std::vector<std::array<uint32_t, 2>> &E,
                          const uint32_t &e) {
  auto v0 = E[e][0];
  auto v1 = E[e][1];
  for (int32_t e_twin = 0; e_twin < E.size(); ++e_twin) {
    // Check if the edge in E is a twin of e
    if ((E[e_twin][0] == v1) && (E[e_twin][1] == v0)) {
      return e_twin; // Return the index of the twin edge
    }
  }
  return -1; // Return -1 if no twin edge is found
}

TriMeshData load_tri_mesh_data_from_ply(const std::string &filepath,
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

  TriMeshData mesh;
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

    // try {
    //   normals =
    //       file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    // } catch (const std::exception &e) {
    //   std::cerr << "tinyply exception: " << e.what() << std::endl;
    // }

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
    // if (normals)
    //   std::cout << "\tRead " << normals->count << " total vertex normals "
    //             << std::endl;
    if (faces)
      std::cout << "\tRead " << faces->count << " total faces (triangles) "
                << std::endl;

    // convert to positions to Eigen::Vector3d
    {
      const size_t numVerticesBytes = vertices->buffer.size_bytes();
      mesh.vertices.resize(vertices->count);
      std::memcpy(mesh.vertices.data(), vertices->buffer.get(),
                  numVerticesBytes);
    }

    // convert faces to std::array<uint32_t, 3>
    {
      const size_t numFacesBytes = faces->buffer.size_bytes();
      mesh.faces.resize(faces->count);
      std::memcpy(mesh.faces.data(), faces->buffer.get(), numFacesBytes);
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return mesh;
}

void write_tri_mesh_data_to_ply(TriMeshData &mesh_data,
                                const std::string &output_directory,
                                const std::string &filename, bool useBinary) {

  std::string filepath = output_directory + "/" + filename;

  std::filebuf fb;
  fb.open(filepath,
          useBinary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + filepath);

  tinyply::PlyFile mesh_file;

  mesh_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
      mesh_data.vertices.size(),
      reinterpret_cast<uint8_t *>(mesh_data.vertices.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::UINT32, mesh_data.faces.size(),
      reinterpret_cast<uint8_t *>(mesh_data.faces.data()), tinyply::Type::UINT8,
      3);

  mesh_file.get_comments().push_back("generated by tinyply 2.3");

  // Write an ply file
  mesh_file.write(outstream, useBinary);
}

HalfEdgeMeshData
buildHalfEdgeMeshDataFromTriMeshData(const TriMeshData &tm_data) {
  auto Nfaces = tm_data.faces.size();
  auto Nvertices = tm_data.vertices.size();
  auto Nedges = 3 * Nfaces;
  HalfEdgeMeshData he_data(Nvertices, Nedges, Nfaces);
  he_data.V = tm_data.vertices;

  auto &E_vertex = he_data.E_vertex;
  auto &E_face = he_data.E_face;
  auto &E_next = he_data.E_next;
  auto &E_twin = he_data.E_twin;
  auto &F_edge = he_data.F_edge;
  auto &V_edge = he_data.V_edge;

  auto &V = he_data.V;
  auto &F = tm_data.faces;
  std::vector<std::array<uint32_t, 2>> E(
      Nedges); // temp container for vertex indices of edges

  // Build everything but twin data
  for (size_t f = 0; f < Nfaces; ++f) {
    for (size_t i = 0; i < 3; ++i) {
      auto e = 3 * f + i;                // edge index
      auto e_next = 3 * f + (i + 1) % 3; // next edge index
      auto v0 = F[f][i];                 // first vertex index in edge
      auto v1 = F[f][(i + 1) % 3];       // second vertex index in edge

      E[e] = {v0, v1};
      E_vertex[e] = v1;
      E_face[e] = f;
      E_next[e] = e_next;
      F_edge[f] = e;
      V_edge[v0] = e;
    }
  }

  // Build twin data
  // for (size_t e = 0; e < Nedges; ++e) {
  //   if (E_twin[e] == -2) {
  //     auto v0 = E[e][0];
  //     auto v1 = E[e][1];
  //     auto e_twin = get_index_of_twin(E, {v1, v0});
  //     E_twin[e] = e_twin;
  //     if (e_twin != -1) {
  //       E_twin[e_twin] = e;
  //     }
  //   }
  // }
  for (size_t e = 0; e < Nedges; ++e) {
    if (E_twin[e] == -2) {
      auto e_twin = get_index_of_twin(E, e);
      E_twin[e] = e_twin;
      if (e_twin != -1) {
        E_twin[e_twin] = e;
      }
    }
  }

  return he_data;
}

void write_he_mesh_data_to_ply(HalfEdgeMeshData &mesh_data,
                               const std::string &output_directory,
                               const std::string &filename, bool useBinary,
                               const std::string &comment) {

  std::string filepath = output_directory + "/" + filename;

  std::filebuf fb;
  fb.open(filepath,
          useBinary ? std::ios::out | std::ios::binary : std::ios::out);
  std::ostream outstream(&fb);
  if (outstream.fail())
    throw std::runtime_error("failed to open " + filepath);

  tinyply::PlyFile mesh_file;

  mesh_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64, mesh_data.V.size(),
      reinterpret_cast<uint8_t *>(mesh_data.V.data()), tinyply::Type::INVALID,
      0);

  mesh_file.add_properties_to_element(
      "vertex", {"e"}, tinyply::Type::UINT32, mesh_data.V_edge.size(),
      reinterpret_cast<uint8_t *>(mesh_data.V_edge.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "face", {"e"}, tinyply::Type::UINT32, mesh_data.F_edge.size(),
      reinterpret_cast<uint8_t *>(mesh_data.F_edge.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "edge", {"v"}, tinyply::Type::UINT32, mesh_data.E_vertex.size(),
      reinterpret_cast<uint8_t *>(mesh_data.E_vertex.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "edge", {"f"}, tinyply::Type::UINT32, mesh_data.E_face.size(),
      reinterpret_cast<uint8_t *>(mesh_data.E_face.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "edge", {"n"}, tinyply::Type::UINT32, mesh_data.E_next.size(),
      reinterpret_cast<uint8_t *>(mesh_data.E_next.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "edge", {"t"}, tinyply::Type::INT32, mesh_data.E_twin.size(),
      reinterpret_cast<uint8_t *>(mesh_data.E_twin.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.get_comments().push_back(comment);

  // Write an ply file
  mesh_file.write(outstream, useBinary);
}

HalfEdgeMeshData load_he_mesh_data_from_ply(const std::string &filepath,
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

  HalfEdgeMeshData mesh;
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
    std::shared_ptr<tinyply::PlyData> V, V_edge, F_edge, E_vertex, E_face,
        E_next, E_twin;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data.
    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      V = file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      V_edge = file.request_properties_from_element("vertex", {"e"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      F_edge = file.request_properties_from_element("face", {"e"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      E_vertex = file.request_properties_from_element("edge", {"v"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      E_face = file.request_properties_from_element("edge", {"f"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      E_next = file.request_properties_from_element("edge", {"n"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      E_twin = file.request_properties_from_element("edge", {"t"});
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

    if (V)
      std::cout << "\tRead " << V->count << " total vertices " << std::endl;

    if (E_vertex)
      std::cout << "\tRead " << E_vertex->count << " total half-edges "
                << std::endl;

    if (F_edge)
      std::cout << "\tRead " << F_edge->count << " total faces (triangles) "
                << std::endl;

    // convert to positions to Eigen::Vector3d
    {
      const size_t numVerticesBytes = V->buffer.size_bytes();
      mesh.V.resize(V->count);
      std::memcpy(mesh.V.data(), V->buffer.get(), numVerticesBytes);
    }

    {
      const size_t numEdgesBytes = V_edge->buffer.size_bytes();
      mesh.V_edge.resize(V_edge->count);
      std::memcpy(mesh.V_edge.data(), V_edge->buffer.get(), numEdgesBytes);
    }

    {
      const size_t numEdgesBytes = E_vertex->buffer.size_bytes();
      mesh.E_vertex.resize(E_vertex->count);
      std::memcpy(mesh.E_vertex.data(), E_vertex->buffer.get(), numEdgesBytes);
    }

    {
      const size_t numEdgesBytes = E_face->buffer.size_bytes();
      mesh.E_face.resize(E_face->count);
      std::memcpy(mesh.E_face.data(), E_face->buffer.get(), numEdgesBytes);
    }

    {
      const size_t numEdgesBytes = E_next->buffer.size_bytes();
      mesh.E_next.resize(E_next->count);
      std::memcpy(mesh.E_next.data(), E_next->buffer.get(), numEdgesBytes);
    }

    {
      const size_t numEdgesBytes = E_twin->buffer.size_bytes();
      mesh.E_twin.resize(E_twin->count);
      std::memcpy(mesh.E_twin.data(), E_twin->buffer.get(), numEdgesBytes);
    }

    // convert faces to...
    {
      const size_t numFacesBytes = F_edge->buffer.size_bytes();
      mesh.F_edge.resize(F_edge->count);
      std::memcpy(mesh.F_edge.data(), F_edge->buffer.get(), numFacesBytes);
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return mesh;
}