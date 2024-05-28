/**
 * @file brane_utils.hpp
 * @brief Utility functions and structures for the MeshBrane library.
 */

#ifndef brane_utils_hpp
#define brane_utils_hpp

#include <Eigen/Dense> // Eigen::Vector3d
// #include <cmath>       // std::sqrt
#include <string>      // std::string
#include <vector>




/**
 * @struct VertexFaceList
 *
 * @brief Simple face-vertex mesh.
 */
struct VertexFaceList {
  std::vector<std::array<double, 3>> vertices;
  std::vector<std::array<uint32_t, 3>> faces;
};

/**
 * Half-edge mesh data structures.
 *
 */

struct HE_edge;
struct HE_vertex;
struct HE_face;

class Chart;
class Atlas;
class Brane;
class Chart;

class Atlas {
  std::vector<Chart*> charts;
}

struct HalfEdgeMesh {
  std::vector<HE_vertex *> vertices; // All vertices in the mesh
  std::vector<HE_edge *> halfedges;  // All half-edges in the mesh
  std::vector<HE_face *> faces;      // All faces in the mesh
};

// /**
//  * @struct HE_vertex
//  * @brief Vertex in a half-edge mesh.
//  */
// struct HE_vertex {
//   Eigen::Vector3d xyz; // position
//   HE_edge *edge;       // half-edge emanating from vertex
// };

// /**
//  * @struct HE_face
//  * @brief Face in a half-edge mesh.
//  */
// struct HE_face {
//   HE_edge *edge; // half-edge on the face
// };

// /**
//  * @struct HalfEdge
//  * @brief Half-edge in a half-edge mesh.
//  */
// struct HE_edge {
//   HE_vertex *vertex; // vertex at the end of the half-edge
//   HE_edge *twin;     // oppositely oriented adjacent half-edge
//   HE_edge *next;     // next half-edge around the face
//   HE_face *face;     // face the half-edge borders
//   // Method to get the previous half-edge
//   HE_edge *getPrev() {
//     HE_edge *edge = this->next;
//     while (edge->next != this) {
//       edge = edge->next;
//     }
//     return edge;
//   bool isBoundary() { return this->twin == nullptr; }
//   }
// };

/**
 * Mesh loading and processing utilities
 */

/**
 * @brief Load a face-vertex mesh from a .ply file.
 *
 * @param filepath Path to the .ply file.
 * @return FaceVertexList
 */
VertexFaceList load_face_vertex_list_from_ply(const std::string &filepath);

std::pair<std::vector<std::array<double, 3>>,
          std::vector<std::array<uint32_t, 3>>>
load_vertex_face_list_from_ply(const std::string &filepath);

#endif




// /**
//  * 3D vector operations.
//  */

// /**
//  * @brief Dot product 3D vectors.
//  *
//  * @return double
//  */
// inline double dot(const double3 &v1, const double3 &v2) {
//   return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
// }

// /**
//  * @brief Euclidean norm of 3D vector.
//  *
//  * @return double
//  */
// inline double norm(const double3 &v) {
//   return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
// }

// /**
//  * @brief Cross product of 3D vectors.
//  *
//  * @return double3
//  */
// inline double3 cross(const double3 &v1, const double3 &v2) {
//   return double3{v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
//                  v1.x * v2.y - v1.y * v2.x};
// }

// /**
//  * @brief Scalar triple product of 3D vectors.
//  *
//  * @return double
//  */
// inline double triprod(const double3 &a, const double3 &b, const double3 &c) {
//   return dot(a, cross(b, c));
// }
