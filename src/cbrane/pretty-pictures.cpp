#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

int main() {
    // Initialize Polyscope
    polyscope::init();

    // Load a mesh
    std::vector<std::array<double, 3>> vertexPositions = /* your vertex positions */;
    std::vector<std::array<size_t, 3>> faceIndices = /* your face indices */;

    // Register the mesh with Polyscope
    polyscope::registerSurfaceMesh("my mesh", vertexPositions, faceIndices);

    // Show the GUI
    polyscope::show();

    return 0;
}