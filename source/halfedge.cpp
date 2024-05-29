/**
 * @file halfedge.cpp
 */

#include "halfedge.hpp"

// HalfEdgeMeshData
// buildHalfEdgeMeshDataFromTriMeshData(const TriMeshData &tri_mesh_data) {

//   HalfEdgeMeshData half_edge_data;

//   half_edge_data.V = tri_mesh_data.vertices;
//   half_edge_data.F = tri_mesh_data.faces;

//   // Build the edge data
//   //   for (auto &face : half_edge_data.F) {
//   //     std::array<uint32_t, 3> face_edges;
//   //     for (size_t i = 0; i < 3; ++i) {
//   //       std::array<uint32_t, 2> edge;
//   //       edge[0] = face[i];
//   //       edge[1] = face[(i + 1) % 3];
//   //       half_edge_data.E.push_back(edge);
//   //       face_edges[i] = half_edge_data.E.size() - 1;
//   //     }
//   //     half_edge_data.F_edge.push_back(face_edges[0]);
//   //     half_edge_data.F_edge.push_back(face_edges[1]);
//   //     half_edge_data.F_edge.push_back(face_edges[2]);
//   //   }

//   return half_edge_data;
// }