#pragma once

/**
 * @file mesh_helpers.hpp
 * @brief Some helpers for the MatrixMesh class.
 */

#include "meshbrane/meshbrane_data_types.hpp"

namespace meshbrane {

/**
 * @brief Check that faces only contain vertex indices in range
 * `0 <= v < Nv` where `Nv=xyz_coord_V.rows()`
 *
 * @return bool
 */
bool VF_data_check_all_v_indices_in_range(Samples3d &xyz_coord_V,
                                          Samples3i &V_cycle_F);

/**
 * @brief Check that every vertex appears in at least one face.
 *
 * @return bool
 */
bool VF_data_check_no_isolated_vertices(Samples3d &xyz_coord_V,
                                        Samples3i &V_cycle_F);

bool VF_data_check_face_v_indices_unique(Samples3d &xyz_coord_V,
                                         Samples3i &V_cycle_F);

bool VF_data_check(Samples3d &xyz_coord_V, Samples3i &V_cycle_F);

bool min_HE_data_check_faces_are_triangles(
    Samples3d &xyz_coord_V, Samplesi &h_out_V, Samplesi &h_right_F,
    Samplesi &v_origin_H, Samplesi &f_left_H, Samplesi &h_next_H,
    Samplesi &h_twin_H, Samplesi &h_negative_B);

bool min_HE_data_check_twin_cycles_are_length_2(
    Samples3d &xyz_coord_V, Samplesi &h_out_V, Samplesi &h_right_F,
    Samplesi &v_origin_H, Samplesi &f_left_H, Samplesi &h_next_H,
    Samplesi &h_twin_H, Samplesi &h_negative_B);

bool min_HE_data_check(Samples3d &xyz_coord_V, Samplesi &h_out_V,
                       Samplesi &h_right_F, Samplesi &v_origin_H,
                       Samplesi &f_left_H, Samplesi &h_next_H,
                       Samplesi &h_twin_H, Samplesi &h_negative_B);

bool combinatorial_map_check(Samplesi &h_out_V, Samplesi &h_directed_E,
                             Samplesi &h_right_F, Samplesi &v_origin_H,
                             Samplesi &e_undirected_H, Samplesi &f_left_H,
                             Samplesi &h_next_H, Samplesi &h_twin_H,
                             Samplesi &h_negative_B);

} // namespace meshbrane
