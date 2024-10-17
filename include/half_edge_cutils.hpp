/**
 * @file half_edge_cutils.hpp
 */
#ifndef HALF_EDGE_CUTILS_HPP
#define HALF_EDGE_CUTILS_HPP

#include <data_types.hpp>

INT_TYPE find_halfedge_index_of_twin(const Samples2i &H, const INT_TYPE &h);

HalfEdgeSamples vf_samples_to_he_samples(const Samples3d &xyz_coord_V,
                                         const Samples3i &V_of_F);

#endif /* HALF_EDGE_CUTILS_HPP */