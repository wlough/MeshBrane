#include "MeshBrane.hpp"
#include "MeshLoader.hpp"
#include "brane_utils.hpp"

int main() {
    // Use MeshLoader to load a .ply file
    auto [vertices, faces] = MeshLoader::loadPly("path/to/your/file.ply");

    // Use MeshBrane with the loaded data
    MeshBrane brane(vertices, faces);

    // Use a function from brane_utils
    brane_utils::someFunction(brane);

    // Rest of your code...
}