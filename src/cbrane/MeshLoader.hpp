#ifndef mesh_loader_hpp
#define mesh_loader_hpp

#include <vector>
#include <string>
#include <utility>
#include "brane_utils.hpp"

struct FaceVertexList {
    std::vector<double3> vertices;
    std::vector<uint3> faces;
};

class MeshLoader {
public:
    static FaceVertexList load(const std::string& filename);

private:
    static std::pair<std::vector<double3>, std::vector<uint3>> load_mesh_from_ply(const std::string& filename);
};

#endif 