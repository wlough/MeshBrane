/**
 * @file halfedge.hpp
 */

#ifndef halfedge_hpp
#define halfedge_hpp

#include "ply_utils.hpp" // for TriMeshData
#include <Eigen/Dense>   // for Eigen::Vector3d
#include <memory>        // for std::shared_ptr and std::weak_ptr
#include <vector>        // for std::vector

// struct HalfEdgeMeshData {
//   std::vector<Eigen::Vector3d> V;
//   std::vector<uint32_t> V_edge;
//   std::vector<std::array<uint32_t, 2>> E;
//   std::vector<uint32_t> E_vertex;
//   std::vector<uint32_t> E_face;
//   std::vector<uint32_t> E_next;
//   std::vector<int32_t> E_twin;
//   std::vector<std::array<uint32_t, 3>> F;
//   std::vector<uint32_t> F_edge;
// };

// HalfEdgeMeshData buildHalfEdgeMeshDataFromTriMeshData(const TriMeshData
// &data);

// Forward declarations
struct HE_vertex;
struct HE_edge;
struct HE_face;

struct HalfEdgeMesh {
  HalfEdgeMesh() : vertices_(), edges_(), faces_() {}
  HalfEdgeMesh(const std::vector<std::shared_ptr<HE_vertex>> &vertices,
               const std::vector<std::shared_ptr<HE_edge>> &edges,
               const std::vector<std::shared_ptr<HE_face>> &faces)
      : vertices_(vertices), edges_(edges), faces_(faces) {}
  std::vector<std::shared_ptr<HE_vertex>> vertices_;
  std::vector<std::shared_ptr<HE_edge>> edges_;
  std::vector<std::shared_ptr<HE_face>> faces_;
};

struct HE_vertex {
  HE_vertex() : edge_() {}
  HE_vertex(const Eigen::Vector3d &xyz, std::weak_ptr<HE_edge> edge)
      : xyz_(xyz), edge_(edge) {}
  Eigen::Vector3d xyz_;
  std::weak_ptr<HE_edge> edge_;
};

struct HE_face {
  HE_face() : edge_() {}
  HE_face(std::weak_ptr<HE_edge> edge) : edge_(edge) {}
  std::weak_ptr<HE_edge> edge_;
};

struct HE_edge {
  HE_edge() : vertex_(), twin_(), next_(), face_() {}
  HE_edge(std::weak_ptr<HE_vertex> vertex, std::weak_ptr<HE_edge> twin,
          std::weak_ptr<HE_edge> next, std::weak_ptr<HE_face> face)
      : vertex_(vertex), twin_(twin), next_(next), face_(face) {}
  std::weak_ptr<HE_vertex> vertex_;
  std::weak_ptr<HE_edge> twin_;
  std::weak_ptr<HE_edge> next_;
  std::weak_ptr<HE_face> face_;
  std::weak_ptr<HE_edge> getPrev() {
    auto edge = next_.lock();
    while (edge && edge->next_.lock() && edge->next_.lock().get() != this) {
      edge = edge->next_.lock();
    }
    return edge;
  }
  bool isBoundary() { return this->twin_.expired(); }
};

HalfEdgeMesh buildHalfEdgeMeshFromTriMeshData(const TriMeshData &data);

#endif // halfedge_hpp