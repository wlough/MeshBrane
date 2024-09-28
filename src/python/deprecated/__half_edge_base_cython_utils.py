import numpy as np
from cython.cimports import numpy as cnp
import cython

_NUMPY_INT_ = np.int32
_NUMPY_FLOAT_ = np.float64


@cython.cfunc
@cython.locals(
    xyz_coord_V=cnp.ndarray,
    h_out_V=cnp.ndarray,
    v_origin_H=cnp.ndarray,
    h_next_H=cnp.ndarray,
    h_twin_H=cnp.ndarray,
    f_left_H=cnp.ndarray,
    h_bound_F=cnp.ndarray,
    Nhedges=cython.int,
    h=cython.int,
    b=cython.int,
    h_start=cython.int,
)
def find_h_comp_B(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
):
    """
    Find a half-edge in each boundary of mesh
    """
    Nhedges = len(v_origin_H)
    # find and enumerate boundaries -1,-2,...
    H_need2visit = set([h for h in range(Nhedges) if f_left_H[h] < 0])
    _h_comp_B = []
    while H_need2visit:
        b = len(_h_comp_B)
        h_start = H_need2visit.pop()
        f_left_H[h_start] = -(b + 1)
        h = h_next_H[h_start]
        _h_comp_B.append(h)
        while h != h_start:
            H_need2visit.remove(h)
            f_left_H[h] = -(b + 1)
            h = h_next_H[h]
    h_comp_B = np.array(_h_comp_B, dtype=_NUMPY_INT_)
    return h_comp_B


@cython.cfunc
@cython.locals(
    H=cnp.ndarray,
    h=cython.int,
    hedge_twin=cnp.ndarray,
    ht_arr=cnp.ndarray,
)
def get_halfedge_index_of_twin(H, h):
    """
    Find the half-edge twin to h in the list of half-edges H.

    Parameters
    ----------
    H : np.ndarray
        List of half-edges [[v0, v1], ...]
    h : int
        Index of half-edge in H

    Returns
    -------
    h_twin : int
        Index of H[h_twin]=[v1,v0] in H, where H[h]=[v0,v1]. Returns -1 if twin not found.
    """
    hedge_twin = np.flip(H[h])
    ht_arr = np.where((hedge_twin[0] == H[:, 0]) * (hedge_twin[1] == H[:, 1]))[0]
    if ht_arr.size > 0:
        return ht_arr[0]
    return -1


@cython.cfunc
@cython.locals(
    xyz_coord_V=cnp.ndarray,
    vvv_of_F=cnp.ndarray,
    Nfaces=cython.int,
    Nvertices=cython.int,
    _Nhedges=cython.int,
    _H=cnp.ndarray,
    h_out_V=cnp.ndarray,
    _v_origin_H=cnp.ndarray,
    _h_next_H=cnp.ndarray,
    _f_left_H=cnp.ndarray,
    h_bound_F=cnp.ndarray,
    f=cython.int,
    i=cython.int,
    h=cython.int,
    h_next=cython.int,
    v0=cython.int,
    v1=cython.int,
    h_twin=cython.int,
    h_start=cython.int,
    bdry_count=cython.int,
    h_count=cython.int,
)
def vf_samples_to_he_samples(xyz_coord_V, vvv_of_F):
    Nfaces = len(vvv_of_F)
    Nvertices = len(xyz_coord_V)
    _Nhedges = 3 * Nfaces * 2
    _H = np.zeros((_Nhedges, 2), dtype=_NUMPY_INT_)
    h_out_V = -np.ones(Nvertices, dtype=_NUMPY_INT_)
    _v_origin_H = np.zeros(_Nhedges, dtype=_NUMPY_INT_)
    _h_next_H = -np.ones(_Nhedges, dtype=_NUMPY_INT_)
    _f_left_H = np.zeros(_Nhedges, dtype=_NUMPY_INT_)
    h_bound_F = np.zeros(Nfaces, dtype=_NUMPY_INT_)

    for f in range(Nfaces):
        h_bound_F[f] = 3 * f
        for i in range(3):
            h = 3 * f + i
            h_next = 3 * f + (i + 1) % 3
            v0 = vvv_of_F[f, i]
            v1 = vvv_of_F[f, (i + 1) % 3]
            _H[h] = [v0, v1]
            _v_origin_H[h] = v0
            _f_left_H[h] = f
            _h_next_H[h] = h_next
            if h_out_V[v0] == -1:
                h_out_V[v0] = h
    h_count = 3 * Nfaces
    need_twins = set([_ for _ in range(h_count)])
    need_next = set([0])
    need_next.pop()
    _h_twin_H = -2 * np.ones(_Nhedges, dtype=_NUMPY_INT_)  # -2 means not set
    while need_twins:
        h = need_twins.pop()
        if _h_twin_H[h] == -2:  # if twin not set
            h_twin = get_halfedge_index_of_twin(_H, h)  # returns -1 if twin not found
            if h_twin == -1:  # if twin not found
                h_twin = h_count
                h_count += 1
                v0, v1 = _H[h]
                _H[h_twin] = [v1, v0]
                _v_origin_H[h_twin] = v1
                need_next.add(h_twin)
                _h_twin_H[h] = h_twin
                _h_twin_H[h_twin] = h
                _f_left_H[h_twin] = -1
            else:
                _h_twin_H[h], _h_twin_H[h_twin] = h_twin, h
                need_twins.remove(h_twin)

    Nhedges = h_count
    v_origin_H = _v_origin_H[:Nhedges]
    h_next_H = _h_next_H[:Nhedges]
    f_left_H = _f_left_H[:Nhedges]
    h_twin_H = _h_twin_H[:Nhedges]
    while need_next:
        h = need_next.pop()
        h_next = h_twin_H[h]
        # rotate ccw around origin of twin until we find nex h on boundary
        while f_left_H[h_next] != -1:
            h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
        h_next_H[h] = h_next

    # find and enumerate boundaries -1,-2,...
    H_need2visit = set([h for h in range(Nhedges) if f_left_H[h] == -1])
    _h_comp_B = []
    while H_need2visit:
        b = len(_h_comp_B)
        h_start = H_need2visit.pop()
        f_left_H[h_start] = -(b + 1)
        h = h_next_H[h_start]
        _h_comp_B.append(h)
        while h != h_start:
            H_need2visit.remove(h)
            f_left_H[h] = -(b + 1)
            h = h_next_H[h]
    h_comp_B = np.array(_h_comp_B, dtype=_NUMPY_INT_)
    return (
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_comp_B,
    )
