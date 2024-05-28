#ifndef MeshBrane_hpp
#define MeshBrane_hpp

#include <Eigen/Dense> // Eigen::Vector3d
// #include <cmath>       // std::sqrt
#include <string> // std::string
#include <vector>
// #include "brane_utils.hpp" // Assuming this is where your Vertex, Face, and
// HalfEdge structs are defined

class Brane {
public:
  Brane(const HalfEdgeMesh &halfEdgeMesh, const BraneParams &braneParams);
  ~Brane();

private:
  HalfEdgeMesh halfEdgeMesh_;
  BraneParams braneParams_;
};

class BraneBuilder {
public:
  /**
   * @brief Constructor that takes the file path of the PLY file.
   * @param filepath The file path of the PLY file.
   */
  //   BraneBuilder(){};
  BraneBuilder(const std::string &filepath);

private:
  std::string filepath_;           ///< The file path of the PLY file.
  std::vector<Eigen::Vector3d> V_; ///< Vertex positions of the loaded mesh.
  std::vector<uint32_t> V_hedge_;  ///< The half-edge index of each vertex.
  std::vector<std::array<uint32_t, 3>>
      F_; ///< The indices of vertices in faces of the loaded mesh.
  std::vector<uint32_t> F_hedge_; ///< The half-edge index of each face.
  std::vector<std::array<uint32_t, 2>>
      H_; ///< The indices of vertices in half-edges of the loaded mesh.
  std::vector<uint32_t> H_face_; ///< The face index of each half-edge.
  std::vector<uint32_t>
      H_next_; ///< The index of the next half-edge in each half-edge.
  std::vector<int32_t>
      H_twin_; ///< The index of the twin half-edge in each half-edge.
  std::vector<uint32_t>
      H_vertex_; ///< The index of the vertex in each half-edge.
  HalfEdgeMesh halfEdgeMesh_;
  // void set_mesh_data_vecs();
  // void set_half_edge_mesh();
  // std::vector<uint2> get_halfedge_vertex_indices();
  void set_VHF_from_ply();
  void set_half_edges();
  void set_half_edge_mesh();
  int32_t get_twin_index(uint32_t i);
};

////////////////////////////////////////////
// Misc simulation stuff ///////////////////
////////////////////////////////////////////

struct BraneParams {
  double bendingModulus = 1.0;
  double splayModulus = 1.0;
  double spontaneousCurvature = 0.0;
  double preferredArea = 1.0;
  double preferredVolume = 1.0;
  double viscosity = 1.0;
  double tensionCoefficient = 1.0;
  double pressureCoefficient = 1.0;
  double dragCoefficientParallel = 1.0;
  double dragCoefficientPerp = 1.0;
};


////////////////////////////////////////////
// Half-Edge Mesh Data Structures //////////
////////////////////////////////////////////
struct HalfEdgeMesh {
  HalfEdgeMesh() : vertices_(), edges_(), faces_() {}
  HalfEdgeMesh(const std::vector<HE_vertex *> &vertices,
               const std::vector<HE_edge *> &edges,
               const std::vector<HE_face *> &faces)
      : vertices_(vertices), edges_(edges), faces_(faces) {}
  std::vector<HE_vertex *> vertices_; // All vertices in the mesh
  std::vector<HE_edge *> edges_;      // All half-edges in the mesh
  std::vector<HE_face *> faces_;      // All faces in the mesh
};

/**
 * @struct HE_vertex
 * @brief Vertex in a half-edge mesh.
 */
struct HE_vertex {
  HE_vertex() : edge_(nullptr) {}
  HE_vertex(const Eigen::Vector3d &xyz, HE_edge *edge)
      : xyz_(xyz), edge_(edge) {}
  Eigen::Vector3d xyz_; // position
  HE_edge *edge_;       // half-edge emanating from vertex
};

/**
 * @struct HE_face
 * @brief Face in a half-edge mesh.
 */
struct HE_face {
  HE_face() : edge_(nullptr) {}
  HE_face(HE_edge *edge) : edge_(edge) {}
  HE_edge *edge_; // half-edge on the face
};

/**
 * @struct HalfEdge
 * @brief Half-edge in a half-edge mesh.
 */
struct HE_edge {
  HE_edge()
      : vertex_(nullptr), twin_(nullptr), next_(nullptr), face_(nullptr) {}
  HE_edge(HE_vertex *vertex, HE_edge *twin, HE_edge *next, HE_face *face)
      : vertex_(vertex), twin_(twin), next_(next), face_(face) {}
  HE_vertex *vertex_; // vertex at the end of the half-edge
  HE_edge *twin_;     // oppositely oriented adjacent half-edge
  HE_edge *next_;     // next half-edge around the face
  HE_face *face_;     // face the half-edge borders
  // Method to get the previous half-edge
  HE_edge *getPrev() {
    HE_edge *edge = this->next_;
    while (edge->next_ != this) {
      edge = edge->next_;
    }
    return edge;
  }
  bool isBoundary() { return this->twin_ == nullptr; }
};

#endif // MeshBrane_hpp