# half_edge_base_cython_utils.pxd

from libcpp.vector cimport vector
from cython cimport int, float
from typing import List, Tuple

cdef extern from "half_edge_base_cython_utils.py":
    # cdef int generate_H_next_h(int h, List[int] h_next_H)
    # cdef int generate_H_bound_f(int f, List[int] h_bound_F, List[int] h_next_H)
    # cdef List[int] find_h_comp_B(
    #     List[Tuple[float, float, float]] xyz_coord_V,
    #     List[int] h_out_V,
    #     List[int] v_origin_H,
    #     List[int] h_next_H,
    #     List[int] h_twin_H,
    #     List[int] f_left_H,
    #     List[int] h_bound_F
    # )
    cdef int get_index_of_twin(List[Tuple[int, int]] H, int h)
    cdef Tuple[
        List[Tuple[float, float, float]],
        List[int],
        List[int],
        List[int],
        List[int],
        List[int],
        List[int]
    ] vf_samples_to_he_samples(
        List[Tuple[float, float, float]] V,
        List[Tuple[int, int, int]] F
    )