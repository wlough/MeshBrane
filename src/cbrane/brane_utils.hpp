/**
 * @file brane_utils.hpp
 * @brief To be split into numdiff/pretty_pictures/ply_utils/etc...
 */

#ifndef brane_utils_hpp
#define brane_utils_hpp

#include <Eigen/Dense> // for Eigen::Vector3d
#include <chrono> // for std::chrono::high_resolution_clock and std::chrono::duration
#include <fstream>   // for std::ifstream
#include <istream>   // for std::istream
#include <stdexcept> // for std::runtime_error
#include <streambuf> // for std::streambuf
#include <string>    // for std::string
#include <vector>    // for std::vector

////////////////////////////////////////////
// mesh_data ///////////////////////////////
////////////////////////////////////////////

// /**
//  * @struct VertexFaceList
//  *
//  * @brief Simple face-vertex mesh.
//  */
// struct VertexFaceList {
//   std::vector<std::array<double, 3>> vertices;
//   std::vector<std::array<uint32_t, 3>> faces;
// };

// /**
//  * @struct HalfEdgeMesh
//  * @brief Half-edge mesh.
//  */
// struct HalfEdgeMesh {
//   HalfEdgeMesh() : vertices_(), edges_(), faces_() {}
//   HalfEdgeMesh(const std::vector<HE_vertex *> &vertices,
//                const std::vector<HE_edge *> &edges,
//                const std::vector<HE_face *> &faces)
//       : vertices_(vertices), edges_(edges), faces_(faces) {}
//   std::vector<HE_vertex *> vertices_; // All vertices in the mesh
//   std::vector<HE_edge *> edges_;      // All half-edges in the mesh
//   std::vector<HE_face *> faces_;      // All faces in the mesh
// };

// /**
//  * @struct HE_vertex
//  * @brief Vertex in a half-edge mesh.
//  */
// struct HE_vertex {
//   HE_vertex() : edge_(nullptr) {}
//   HE_vertex(const Eigen::Vector3d &xyz, HE_edge *edge)
//       : xyz_(xyz), edge_(edge) {}
//   Eigen::Vector3d xyz_; // position
//   HE_edge *edge_;       // half-edge emanating from vertex
// };

// /**
//  * @struct HE_face
//  * @brief Face in a half-edge mesh.
//  */
// struct HE_face {
//   HE_face() : edge_(nullptr) {}
//   HE_face(HE_edge *edge) : edge_(edge) {}
//   HE_edge *edge_; // half-edge on the face
// };

// /**
//  * @struct HalfEdge
//  * @brief Half-edge in a half-edge mesh.
//  */
// struct HE_edge {
//   HE_edge()
//       : vertex_(nullptr), twin_(nullptr), next_(nullptr), face_(nullptr) {}
//   HE_edge(HE_vertex *vertex, HE_edge *twin, HE_edge *next, HE_face *face)
//       : vertex_(vertex), twin_(twin), next_(next), face_(face) {}
//   HE_vertex *vertex_; // vertex at the end of the half-edge
//   HE_edge *twin_;     // oppositely oriented adjacent half-edge
//   HE_edge *next_;     // next half-edge around the face
//   HE_face *face_;     // face the half-edge borders
//   // Method to get the previous half-edge
//   HE_edge *getPrev() {
//     HE_edge *edge = this->next_;
//     while (edge->next_ != this) {
//       edge = edge->next_;
//     }
//     return edge;
//   }
//   bool isBoundary() { return this->twin_ == nullptr; }
// };

#endif
