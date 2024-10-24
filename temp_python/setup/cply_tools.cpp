/**
 * @file cply_tools.cpp
 */
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include <cply_tools.hpp>
// #include <pybind11/eigen.h>    // Eigen<->Numpy conversion
// #include <pybind11/pybind11.h> // PYBIND11_MODULE
#include <tuple>         // std::tuple
#include <unordered_set> // std::unordered_set
#include <vector>        // std::vector

// namespace py = pybind11;

////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////

INT_TYPE find_halfedge_index_of_twin(const Samples2i &H, const INT_TYPE &h) {
  auto v0 = H(h, 0);
  auto v1 = H(h, 1);
  for (INT_TYPE h_twin = 0; h_twin < H.rows(); ++h_twin) {
    if ((H(h_twin, 0) == v1) && (H(h_twin, 1) == v0)) {
      return h_twin; // Return the index of the twin edge
    }
  }
  return -1; // Return -1 if no twin edge is found
}

HalfEdgeSamples vf_samples_to_he_samples(const Samples3d &xyz_coord_V,
                                         const Samples3i &V_of_F) {

  auto Nv = xyz_coord_V.rows();
  auto Nf = V_of_F.rows();
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
  Samplesi h_bound_F = Samplesi(Nf);
  Samplesi h_right_B;
  // assign h_out for vertices to be minimum of outgoing half-edge indices
  // assign v_origin/f_left/h_next for half-edges in H0
  // assign h_bound for faces
  for (INT_TYPE f = 0; f < Nf; ++f) {
    h_bound_F[f] = 3 * f;
    for (INT_TYPE i = 0; i < 3; ++i) {
      INT_TYPE h = 3 * f + i;
      INT_TYPE h_next = 3 * f + (i + 1) % 3;
      INT_TYPE v0 = V_of_F(f, i);
      INT_TYPE v1 = V_of_F(f, (i + 1) % 3);
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
  std::vector<INT_TYPE> H_boundary_plus;
  std::unordered_set<INT_TYPE> H_boundary_minus;
  // find positive boundary half-edges
  // assign h_twin for interior half-edges
  for (INT_TYPE h = 0; h < H0.rows(); ++h) {
    // if h_twin_H[h] is already assigned, skip
    if (h_twin_H[h] != -1) {
      continue;
    }
    INT_TYPE h_twin = find_halfedge_index_of_twin(H0, h);
    if (h_twin == -1) {
      H_boundary_plus.push_back(h);
    } else {
      h_twin_H[h] = h_twin;
      h_twin_H[h_twin] = h;
    }
  }
  INT_TYPE Nh1 = H_boundary_plus.size();
  INT_TYPE Nh = Nh0 + Nh1;
  v_origin_H.conservativeResize(Nh);
  h_next_H.conservativeResize(Nh);
  h_twin_H.conservativeResize(Nh);
  f_left_H.conservativeResize(Nh);
  // define negative boundary half-edges
  // assign v_origin for negative boundary half-edges
  // assign h_twin for boundary half-edges
  for (INT_TYPE i = 0; i < Nh1; ++i) {
    INT_TYPE h = H_boundary_plus[i];
    INT_TYPE h_twin = Nh0 + i;
    // INT_TYPE v0 = H0(h, 0);
    INT_TYPE v1 = H0(h, 1);
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
    INT_TYPE b = h_right_B.size();
    INT_TYPE h_right_b = *H_boundary_minus.begin();
    h_right_B.conservativeResize(b + 1);
    h_right_B[b] = h_right_b; // Assign new value
    INT_TYPE h = h_right_b;
    // follow prev cycle along boundary b until we get back to h=h_right_b
    do {
      INT_TYPE h_prev = h_twin_H[h];
      // rotate cw around origin of h until we find h_prev in boundary b
      // erase h from H_boundary_minus
      while (H_boundary_minus.find(h_prev) == H_boundary_minus.end()) {
        h_prev = h_twin_H[h_next_H[h_prev]];
      }
      h_next_H[h_prev] = h;
      h = h_prev;
      H_boundary_minus.erase(h);
      f_left_H[h] = -(b + 1);
    } while (h != h_right_b);
  }
  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_bound_F, h_right_B);
}

VertexFaceSamples
he_samples_to_vf_samples(const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
                         const Samplesi &v_origin_H, const Samplesi &h_next_H,
                         const Samplesi &h_twin_H, const Samplesi &f_left_H,
                         const Samplesi &h_bound_F, const Samplesi &h_right_B) {

  INT_TYPE Nf = h_bound_F.rows();
  Samples3i V_of_F = Samples3i(Nf, 3);
  for (INT_TYPE f = 0; f < Nf; ++f) {
    INT_TYPE h = h_bound_F[f];
    for (INT_TYPE i = 0; i < 3; ++i) {
      V_of_F(f, i) = v_origin_H[h];
      h = h_next_H[h];
    }
  }
  return std::make_tuple(xyz_coord_V, V_of_F);
}

VertexFaceSamples load_vf_samples_from_ply(const std::string &filepath,
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
  Samples3i V_of_F;
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

    // // convert to positions to Samples3d
    // const size_t numVerticesBytes = vertices->buffer.size_bytes();
    // xyz_coord_V.resize(vertices->count, 3);
    // std::memcpy(xyz_coord_V.data(), vertices->buffer.get(),
    // numVerticesBytes);
    // // convert faces to Samples3i
    // const size_t numFacesBytes = faces->buffer.size_bytes();
    // V_of_F.resize(faces->count, 3);
    // std::memcpy(V_of_F.data(), faces->buffer.get(), numFacesBytes);

    // Convert to positions to Samples3d
    const size_t numVertices = vertices->count;
    xyz_coord_V.resize(numVertices, 3);
    const FLOAT_TYPE *vertexBuffer =
        reinterpret_cast<const FLOAT_TYPE *>(vertices->buffer.get());
    for (size_t i = 0; i < numVertices; ++i) {
      xyz_coord_V(i, 0) = vertexBuffer[3 * i + 0]; // x
      xyz_coord_V(i, 1) = vertexBuffer[3 * i + 1]; // y
      xyz_coord_V(i, 2) = vertexBuffer[3 * i + 2]; // z
    }

    // Convert faces to Samples3i
    const size_t numFaces = faces->count;
    V_of_F.resize(numFaces, 3);
    const INT_TYPE *faceBuffer =
        reinterpret_cast<const INT_TYPE *>(faces->buffer.get());
    for (size_t i = 0; i < numFaces; ++i) {
      V_of_F(i, 0) = faceBuffer[3 * i + 0]; // vertex index 1
      V_of_F(i, 1) = faceBuffer[3 * i + 1]; // vertex index 2
      V_of_F(i, 2) = faceBuffer[3 * i + 2]; // vertex index 3
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return std::make_tuple(xyz_coord_V, V_of_F);
}

HalfEdgeSamples load_he_samples_from_ply(const std::string &filepath,
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
  Samplesi h_bound_F;
  Samplesi h_right_B;

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
    std::shared_ptr<tinyply::PlyData> _xyz_coord_V, _h_out_V, _v_origin_H,
        _h_next_H, _h_twin_H, _f_left_H, _h_bound_F, _h_right_B;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data.
    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.

    // vertex
    // half_edge
    // face
    // boundary
    try {
      _xyz_coord_V =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _h_out_V = file.request_properties_from_element("vertex", {"h"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _v_origin_H = file.request_properties_from_element("half_edge", {"v"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _h_next_H = file.request_properties_from_element("half_edge", {"n"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _h_twin_H = file.request_properties_from_element("half_edge", {"t"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _f_left_H = file.request_properties_from_element("half_edge", {"f"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _h_bound_F = file.request_properties_from_element("face", {"h"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      _h_right_B = file.request_properties_from_element("boundary", {"h"});
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

    const size_t Nv = _h_out_V->count;
    const size_t Nh = _v_origin_H->count;
    const size_t Nf = _h_bound_F->count;
    const size_t Nb = _h_right_B->count;

    xyz_coord_V.resize(Nv, 3);
    h_out_V.resize(Nv);
    v_origin_H.resize(Nh);
    h_next_H.resize(Nh);
    h_twin_H.resize(Nh);
    f_left_H.resize(Nh);
    h_bound_F.resize(Nf);
    h_right_B.resize(Nb);

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
    const FLOAT_TYPE *xyz_coord_V_buffer =
        reinterpret_cast<const FLOAT_TYPE *>(_xyz_coord_V->buffer.get());
    const INT_TYPE *h_out_V_buffer =
        reinterpret_cast<const INT_TYPE *>(_h_out_V->buffer.get());
    const INT_TYPE *v_origin_H_buffer =
        reinterpret_cast<const INT_TYPE *>(_v_origin_H->buffer.get());
    const INT_TYPE *h_next_H_buffer =
        reinterpret_cast<const INT_TYPE *>(_h_next_H->buffer.get());
    const INT_TYPE *h_twin_H_buffer =
        reinterpret_cast<const INT_TYPE *>(_h_twin_H->buffer.get()); // h_twin
    const INT_TYPE *f_left_H_buffer =
        reinterpret_cast<const INT_TYPE *>(_f_left_H->buffer.get()); // f_left
    const INT_TYPE *h_bound_F_buffer =
        reinterpret_cast<const INT_TYPE *>(_h_bound_F->buffer.get()); // h_bound
    const INT_TYPE *h_right_B_buffer =
        reinterpret_cast<const INT_TYPE *>(_h_right_B->buffer.get()); // h_right
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
      h_bound_F(i) = h_bound_F_buffer[i]; // h_bound
    }
    for (size_t i = 0; i < Nb; ++i) {
      h_right_B(i) = h_right_B_buffer[i]; // h_right
    }

  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }

  if (!verbose) {
    // Restore old buffer
    std::cout.rdbuf(oldCoutStreamBuf);
  }

  return std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                         f_left_H, h_bound_F, h_right_B);
}

// void write_vf_samples_to_ply(Samples3d &xyz_coord_V, Samples3i &V_of_F,
//                              const std::string &output_directory,
//                              const std::string &filename,
//                              const bool useBinary)
void write_vf_samples_to_ply(Samples3d &xyz_coord_V, Samples3i &V_of_F,
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
  //     "face", {"vertex_indices"}, tinyply::Type::INT32, V_of_F.rows(),
  //     reinterpret_cast<uint8_t *>(const_cast<int *>(V_of_F.data())),
  //     tinyply::Type::UINT8, V_of_F.cols());
  // Convert to row-major storage
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      xyz_coord_V_row_major = xyz_coord_V;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      V_of_F_row_major = V_of_F;

  mesh_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64,
      xyz_coord_V_row_major.rows(),
      reinterpret_cast<uint8_t *>(xyz_coord_V_row_major.data()),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32, V_of_F_row_major.rows(),
      reinterpret_cast<uint8_t *>(V_of_F_row_major.data()),
      tinyply::Type::UINT8, V_of_F_row_major.cols());

  mesh_file.get_comments().push_back("MeshBrane vf_ply");

  // Write an ply file
  mesh_file.write(outstream, use_binary);
}

void write_he_samples_to_ply(
    const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
    const Samplesi &v_origin_H, const Samplesi &h_next_H,
    const Samplesi &h_twin_H, const Samplesi &f_left_H,
    const Samplesi &h_bound_F, const Samplesi &h_right_B,
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
      "face", {"h"}, tinyply::Type::INT32, h_bound_F.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_bound_F.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.add_properties_to_element(
      "boundary", {"h"}, tinyply::Type::INT32, h_right_B.rows(),
      reinterpret_cast<uint8_t *>(const_cast<int *>(h_right_B.data())),
      tinyply::Type::INVALID, 0);

  mesh_file.get_comments().push_back("MeshBrane he_ply");

  // Write an ply file
  mesh_file.write(outstream, use_binary);
}
////////////////////////////////////////////////////////////////
// testing
////////////////////////////////////////////////////////////////
struct Pet {
  Pet(const std::string &name) : name(name) {}
  void setName(const std::string &name_) { name = name_; }
  const std::string &getName() const { return name; }

  std::string name;
};
////////////////////////////////////////////////////////////////
// MeshConverter
////////////////////////////////////////////////////////////////
/////////////////
// Constructors /
/////////////////

MeshConverter::MeshConverter() : he_ply_path("") {}

MeshConverter MeshConverter::from_vf_ply(const std::string &ply_path,
                                         bool compute_he_stuff) {
  MeshConverter mesh_converter;
  mesh_converter.vf_ply_path = ply_path;
  mesh_converter.vf_samples = load_vf_samples_from_ply(ply_path);
  if (compute_he_stuff) {
    mesh_converter.he_samples =
        vf_samples_to_he_samples(std::get<0>(mesh_converter.vf_samples),
                                 std::get<1>(mesh_converter.vf_samples));
  }
  return mesh_converter;
}
MeshConverter MeshConverter::from_vf_samples(const Samples3d &xyz_coord_V,
                                             const Samples3i &V_of_F,
                                             bool compute_he_stuff) {

  MeshConverter mesh_converter;
  mesh_converter.vf_samples = std::make_tuple(xyz_coord_V, V_of_F);
  if (compute_he_stuff) {
    mesh_converter.he_samples = vf_samples_to_he_samples(xyz_coord_V, V_of_F);
  }
  return mesh_converter;
}
MeshConverter MeshConverter::from_he_ply(const std::string &ply_path,
                                         bool compute_vf_stuff) {
  MeshConverter mesh_converter;
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
MeshConverter MeshConverter::from_he_samples(
    const Samples3d &xyz_coord_V, const Samplesi &h_out_V,
    const Samplesi &v_origin_H, const Samplesi &h_next_H,
    const Samplesi &h_twin_H, const Samplesi &f_left_H,
    const Samplesi &h_bound_F, const Samplesi &h_right_B,
    bool compute_vf_stuff) {

  MeshConverter mesh_converter;
  mesh_converter.he_samples =
      std::make_tuple(xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H,
                      f_left_H, h_bound_F, h_right_B);
  if (compute_vf_stuff) {
    mesh_converter.vf_samples =
        he_samples_to_vf_samples(xyz_coord_V, h_out_V, v_origin_H, h_next_H,
                                 h_twin_H, f_left_H, h_bound_F, h_right_B);
  }
  return mesh_converter;
}
////////////
// Methods /
////////////
void MeshConverter::write_vf_ply(const std::string &ply_path,
                                 const bool use_binary) {
  write_vf_samples_to_ply(std::get<0>(vf_samples), std::get<1>(vf_samples),
                          ply_path, use_binary);
}

void MeshConverter::write_he_ply(const std::string &ply_path,
                                 const bool use_binary) {
  write_he_samples_to_ply(
      std::get<0>(he_samples), std::get<1>(he_samples), std::get<2>(he_samples),
      std::get<3>(he_samples), std::get<4>(he_samples), std::get<5>(he_samples),
      std::get<6>(he_samples), std::get<7>(he_samples), ply_path, use_binary);
}

////////////////////////////////////////////////////////////////
// Python bindings
////////////////////////////////////////////////////////////////

// PYBIND11_MODULE(cply_tools, m) {
//   m.doc() = "pybind11 cply_tools plugin"; // module docstring
//   m.def("vf_samples_to_he_samples", &vf_samples_to_he_samples,
//         "A function to compute half-edge data from vertices of faces");
//   py::class_<Pet>(m, "Pet")
//       .def(py::init<const std::string &>())
//       .def_readwrite("name", &Pet::name)
//       .def("setName", &Pet::setName)
//       .def("getName", &Pet::getName);
//   py::class_<MeshConverter>(m, "MeshConverter")
//       //   Constructors
//       .def(py::init<>())
//       .def_static("from_vf_ply", &MeshConverter::from_vf_ply,
//                   py::arg("ply_path"), py::arg("compute_he_stuff") = true)
//       .def_static("from_vf_samples", &MeshConverter::from_vf_samples,
//                   py::arg("xyz_coord_V"), py::arg("V_of_F"),
//                   py::arg("compute_he_stuff") = true)
//       .def_static("from_he_ply", &MeshConverter::from_he_ply,
//                   py::arg("ply_path"), py::arg("compute_vf_stuff") = true)
//       .def_static("from_he_samples", &MeshConverter::from_he_samples,
//                   py::arg("xyz_coord_V"), py::arg("h_out_V"),
//                   py::arg("v_origin_H"), py::arg("h_next_H"),
//                   py::arg("h_twin_H"), py::arg("f_left_H"),
//                   py::arg("h_bound_F"), py::arg("h_right_B"),
//                   py::arg("compute_vf_stuff") = true)
//       // Attributes
//       .def_readwrite("vf_ply_path", &MeshConverter::vf_ply_path)
//       .def_readwrite("vf_samples", &MeshConverter::vf_samples)
//       .def_readwrite("he_ply_path", &MeshConverter::he_ply_path)
//       .def_readwrite("he_samples", &MeshConverter::he_samples)
//       // Methods
//       .def("write_vf_ply", &MeshConverter::write_vf_ply, py::arg("ply_path"),
//            py::arg("use_binary") = true)
//       .def("write_he_ply", &MeshConverter::write_he_ply, py::arg("ply_path"),
//            py::arg("use_binary") = true);
// }

// py::class_<MeshConverter>(m, "MeshConverter").def(py::init<>());
//   .def_static("from_vf_ply", &MeshConverter::from_vf_ply,
//               py::arg("ply_path"), py::arg("compute_he_stuff") = true)
//   .def_static("from_vf_samples", &MeshConverter::from_vf_samples,
//               py::arg("xyz_coord_V"), py::arg("V_of_F"),
//               py::arg("compute_he_stuff") = true)
//   .def("vf_samples_to_he_samples",
//   &MeshConverter::vf_samples_to_he_samples) .def("vf_ply_data_to_samples",
//   &MeshConverter::vf_ply_data_to_samples) .def("write_vf_ply",
//   &MeshConverter::write_vf_ply, py::arg("ply_path"),
//        py::arg("use_binary") = true)