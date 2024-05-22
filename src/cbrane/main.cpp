/**
 * @file main.cpp
 * @brief main function for the cbrane project.
 */
#include "brane_utils.hpp"
#include "MeshLoader.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    std::vector<std::string> filepaths;
    filepaths.push_back("./data/ply_files/dumbbell_ultracoarse.ply");
    filepaths.push_back("./data/ply_files/dumbbell_coarse.ply");
    filepaths.push_back("./data/ply_files/dumbbell.ply");
    filepaths.push_back("./data/ply_files/dumbbell_fine.ply");
    filepaths.push_back("./data/ply_files/dumbbell_ultrafine.ply");
    std::vector<MeshLoader> mesh_loaders;
 
    for (size_t i = 0; i < filepaths.size(); ++i) {
        // FaceVertexList mesh = load_face_vertex_list_from_ply(filepaths[i]);
        // std::cout << "Number of vertices: " << mesh.vertices.size() << std::endl;
        // std::cout << "Number of faces: " << mesh.faces.size() << std::endl;
        mesh_loaders.push_back(MeshLoader(filepaths[i]));
        std::vector<double3> V = mesh_loaders[i].get_vertex_positions();
        std::cout << "Number of vertices: " << V.size();
    }
  
    return EXIT_SUCCESS;
}