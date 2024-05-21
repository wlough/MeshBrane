#include <tinyply.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <brane_utils.hpp>

class TriangulatedSurface {
public:
    std::vector<float> vertices;
    std::vector<int> faces;

    TriangulatedSurface(const std::string& filepath) {
        std::ifstream ss(filepath, std::ios::binary);
        if (ss.fail()) throw std::runtime_error("failed to open " + filepath);

        tinyply::PlyFile file;
        file.parse_header(ss);  // Use parse_header to initialize the PlyFile object

        std::shared_ptr<tinyply::PlyData> ply_vertices, ply_faces;
        try { ply_vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { ply_faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        file.read(ss);

        // Copy vertices
        const size_t numVerticesBytes = ply_vertices->buffer.size_bytes();
        vertices.resize(ply_vertices->count * 3);
        std::memcpy(vertices.data(), ply_vertices->buffer.get(), numVerticesBytes);

        // Copy faces
        const size_t numFacesBytes = ply_faces->buffer.size_bytes();
        faces.resize(ply_faces->count * 3);
        std::memcpy(faces.data(), ply_faces->buffer.get(), numFacesBytes);
    }
};

int main() {
    TriangulatedSurface surface("./data/ply_files/dumbbell.ply"); // Replace with your .ply file path

    // Now you can access the vertices and faces of the surface
    std::cout << "Number of vertices: " << surface.vertices.size() / 3 << "\n";
    std::cout << "Number of faces: " << surface.faces.size() / 3;

    return 0;
}