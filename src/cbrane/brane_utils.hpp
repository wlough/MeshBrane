/**
 * @file brane_utils.hpp
 * @brief Utility functions and structures for the MeshBrane library.
 */

#ifndef brane_utils_hpp
#define brane_utils_hpp

#include <cmath> // std::sqrt
#include <Eigen/Dense> // Eigen::Vector3d
#include <vector>
#include <string> // std::string


/**
 * @struct double3
 *
 * @brief Components of 3D vector with double precision.
 */
struct double3 { double x, y, z; };

/**
 * @struct float4
 *
 * @brief rgba.
 */
struct float4 { float w, x, y, z; };

/**
 * @struct uint2
 *
 * @brief Indices of vertices in (half-)edges.
 */
struct uint2 { uint32_t x, y; };

/**
 * @struct uint3
 *
 * @brief Indices of vertices/edges in faces.
 */
struct uint3 { uint32_t x, y, z; };


/**
 * @struct FaceVertexList
 * 
 * @brief Simple face-vertex mesh.
 */
struct FaceVertexList {
    std::vector<double3> vertices;
    std::vector<uint3> faces;
};

/**
 * Half-edge mesh data structures.
 *
 */

struct HalfEdge;
struct Vertex;
struct Face;
// struct Boundary;

/**
 * @struct Vertex
 * @brief Vertex in a half-edge mesh.
 */
struct Vertex
{
    Eigen::Vector3d xyz;  // position
    Eigen::Vector3d normal;  // normal vector
    float4 rgba;  // color
    HalfEdge* halfedge;  // half-edge emanating from vertex 
    double mass;  // mass/area of the vertex
};

/**
 * @struct Face
 * @brief Face in a half-edge mesh.
 */
struct Face
{
    HalfEdge* halfedge;  // half-edge on the face
    Eigen::Vector3d normal;  // normal vector
};

/**
 * @struct HalfEdge
 * @brief Half-edge in a half-edge mesh.
*/
struct HalfEdge
{
    Vertex* vertex;  // vertex at the end of the half-edge
    HalfEdge* twin; // oppositely oriented adjacent half-edge
    HalfEdge* next; // next half-edge around the face
    HalfEdge* prev; // previous half-edge around the face
    Face* face; // face the half-edge borders
};

struct HalfEdgeMesh
{
    std::vector<Vertex*> vertices; // All vertices in the mesh
    std::vector<HalfEdge*> halfedges; // All half-edges in the mesh
    std::vector<Face*> faces; // All faces in the mesh
};

/**
 * 3D vector operations.
*/

/**
 * @brief Dot product 3D vectors.
 *
 * @return double
 */
inline double dot(const double3& v1, const double3& v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

/**
 * @brief Euclidean norm of 3D vector.
 *
 * @return double
 */
inline double norm(const double3& v)
{
    return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

/**
 * @brief Cross product of 3D vectors.
 *
 * @return double3
 */
inline double3 cross(const double3& v1, const double3& v2)
{
    return double3{
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    };
}

/**
 * @brief Scalar triple product of 3D vectors.
 *
 * @return double
 */
inline double triprod(const double3& a, const double3& b, const double3& c)
{
    return dot(a, cross(b, c));
}


/**
 * Mesh loading and processing utilities
*/

/**
 * @brief Load a face-vertex mesh from a .ply file.
 *
 * @param filepath Path to the .ply file.
 * @return FaceVertexList
 */
FaceVertexList load_face_vertex_list_from_ply(const std::string & filepath);

/**
 * @brief Builds list of indices of vertices in half-edges from face-vertex mesh.
*/
// std::vector<uint2> getHalfEdgeVertexIndices(const FaceVertexList & mesh);



#endif 
