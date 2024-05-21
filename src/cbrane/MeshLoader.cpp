// MeshLoader.cpp

#include "MeshLoader.hpp"
#include "tinyply.h"
#include <fstream>
#include <stdexcept>

std::pair<std::vector<double3>, std::vector<int3>> MeshLoader::load_mesh_from_ply(const std::string& filename)
{
    std::ifstream ss(filename, std::ios::binary);
    if (ss.fail()) throw std::runtime_error("failed to open " + filename);

    tinyply::PlyFile file(ss);

    std::vector<double3> vertices;
    std::vector<int3> faces;

    const size_t numVerticesBytes = file.request_properties_from_element("vertex", { "x", "y", "z" }, vertices);
    const size_t numFacesBytes = file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);

    file.read(ss);

    if (vertices.empty() || faces.empty()) {
        throw std::runtime_error("file " + filename + " does not contain valid ply data");
    }

    return {vertices, faces};
}