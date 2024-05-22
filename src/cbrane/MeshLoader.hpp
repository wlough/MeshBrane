// MeshLoader.hpp
#ifndef MESHLOADER_HPP
#define MESHLOADER_HPP

#include <string>
#include "brane_utils.hpp" // FaceVertexList, load_face_vertex_list_from_ply

/**
 * @class MeshLoader
 * @brief Provides a way to load/save a mesh from a PLY file.
 */
class MeshLoader {
public:
    /**
     * @brief Constructor that takes the file path of the PLY file.
     * @param filepath The file path of the PLY file.
     */
    MeshLoader(const std::string& filepath);

    /**
     * @brief Loads the mesh from the PLY file and returns it as a FaceVertexList.
     * @return The loaded mesh as a FaceVertexList.
     */
    FaceVertexList load();
    /**
     * @brief Saves the mesh to a PLY file.
     * @return void
     */
    void save(const std::string& filepath);
    std::vector<double3> get_vertex_positions();
    

private:
    std::string filepath_; ///< The file path of the PLY file.
    std::vector<double3> V; ///< Vertex positions of the loaded mesh.
    std::vector<uint> V_hedge; ///< The half-edge index of each vertex.
    std::vector<uint3> F; ///< The indices of vertices in faces of the loaded mesh.
    std::vector<uint> F_hedge; ///< The half-edge index of each face.
    std::vector<uint2> H; ///< The indices of vertices in half-edges of the loaded mesh.
    std::vector<uint> H_face; ///< The face index of each half-edge.
    std::vector<uint> H_next; ///< The index of the next half-edge in each half-edge.
    std::vector<uint> H_twin; ///< The index of the twin half-edge in each half-edge.
    std::vector<uint> H_vertex; ///< The index of the vertex in each half-edge.
    FaceVertexList faceVertexList;
    HalfEdgeMesh halfEdgeMesh;
    void set_combinatorial_mesh_data();
    void set_face_vertex_list();
    void set_half_edge_mesh();
    std::vector<uint2> get_halfedge_vertex_indices();
    
};

#endif // MESHLOADER_HPP