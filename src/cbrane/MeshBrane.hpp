#ifndef MeshBrane_hpp
#define MeshBrane_hpp

#include <vector>
#include "brane_utils.hpp" // Assuming this is where your Vertex, Face, and HalfEdge structs are defined

class Brane {
public:
    Brane(const std::vector<double3>& vertexPositions, const std::vector<uint3>& faceIndices);

    // Other methods...

private:
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
    std::vector<HalfEdge> halfEdges;

    void buildMesh(const std::vector<double3>& vertexPositions, const std::vector<uint3>& faceIndices);
};

#endif 