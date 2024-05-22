// MeshLoader.cpp
#include "MeshLoader.hpp"

// Constructor stores the file path.
MeshLoader::MeshLoader(const std::string& filepath) : filepath_(filepath) {
    // Set the combinatorial mesh data.
    set_combinatorial_mesh_data();
}

std::vector<double3> MeshLoader::get_vertex_positions() {
    return V;
}

// Load calls load_face_vertex_list_from_ply with the stored file path.
FaceVertexList MeshLoader::load() {
    // Use the function from brane_utils to load the mesh from the PLY file.
    return load_face_vertex_list_from_ply(filepath_);
}

void MeshLoader::set_combinatorial_mesh_data() {
    // Set the combinatorial mesh data.
    FaceVertexList face_vertex_list = load_face_vertex_list_from_ply(filepath_);
    faceVertexList = face_vertex_list;
    V = face_vertex_list.vertices;
    F = face_vertex_list.faces;
    H = get_halfedge_vertex_indices();
}

std::vector<uint2> MeshLoader::get_halfedge_vertex_indices() {
    std::vector<uint2> halfedge_vertex_indices;
    for (size_t i = 0; i < F.size(); ++i) {
        uint3 face = F[i];
        for (size_t j = 0; j < 3; ++j) {
            uint2 halfedge_vertex_index = {face[j], face[(j + 1) % 3]};
            halfedge_vertex_indices.push_back(halfedge_vertex_index);
        }
    }
    return halfedge_vertex_indices;
}