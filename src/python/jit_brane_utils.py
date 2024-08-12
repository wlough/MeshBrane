from numba import jit, prange
import numpy as np
from numba import from_dtype, typeof
from numba.typed import (
    # List,
    Dict,
    # Tuple,
)
from numba.types import (
    # unicode_type,
    # boolean,
    int32,
    int64,
    float64,
    ListType,
    DictType,
    Array,
)

_NUMPY_INT_ = np.int64
_NUMPY_FLOAT_ = np.float64
_NUMBA_INT_ = from_dtype(_NUMPY_INT_)
_NUMBA_FLOAT_ = from_dtype(_NUMPY_FLOAT_)
# _NUMBA_INT_ = int64
# _NUMBA_FLOAT_ = float64
# import os
# # Enable Numba's debug mode
# os.environ['NUMBA_DEBUG'] = '1'

xyz_ply_numba_type = Array(
    from_dtype(
        np.dtype(
            [
                ("x", _NUMPY_FLOAT_),
                ("y", _NUMPY_FLOAT_),
                ("z", _NUMPY_FLOAT_),
            ]
        )
    ),
    1,
    "C",
)
face_ply_numba_type = Array(
    from_dtype(np.dtype([("vertex_indices", _NUMPY_INT_, (3,))])), 1, "C"
)
hedge_ply_numba_type = Array(
    from_dtype(np.dtype([("vertex_indices", _NUMPY_INT_, (2,))])), 1, "C"
)

vertex_index_numba_type = _NUMBA_INT_
halfedge_index_numba_type = _NUMBA_INT_
face_index_numba_type = _NUMBA_INT_
boundary_index_numba_type = _NUMBA_INT_
xyz_numba_type = Array(_NUMBA_FLOAT_, 1, "C")


xyz_coord_V_numba_type = DictType(vertex_index_numba_type, xyz_numba_type)
h_out_V_numba_type = DictType(vertex_index_numba_type, halfedge_index_numba_type)
v_origin_H_numba_type = DictType(halfedge_index_numba_type, vertex_index_numba_type)
h_next_H_numba_type = DictType(halfedge_index_numba_type, halfedge_index_numba_type)
h_twin_H_numba_type = DictType(halfedge_index_numba_type, halfedge_index_numba_type)
f_left_H_numba_type = DictType(halfedge_index_numba_type, face_index_numba_type)
h_bound_F_numba_type = DictType(face_index_numba_type, halfedge_index_numba_type)
h_comp_B_numba_type = DictType(boundary_index_numba_type, halfedge_index_numba_type)


def py2numba_dict(d, kt, vt, safe=True):
    n_keys = len(d)
    D = Dict.empty(kt, vt, n_keys=n_keys)
    if safe:
        try:
            kv = d.items()
        except TypeError:
            kv = enumerate(d)
        for _k, _v in kv:
            k = kt(_k)
            try:
                v = vt(_v)
            except NotImplementedError:
                v = _v
            D[k] = v
    else:
        try:
            D.update(d)
        except TypeError:
            kv = enumerate(d)
            for _k, _v in kv:
                k = kt(_k)
                try:
                    v = vt(_v)
                except NotImplementedError:
                    v = _v
                D[k] = v
    return D


def py2numba_half_edge_mesh_dicts(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    safe=True,
):
    d_kt_vt = [
        (xyz_coord_V, vertex_index_numba_type, xyz_numba_type),
        (h_out_V, vertex_index_numba_type, halfedge_index_numba_type),
        (v_origin_H, halfedge_index_numba_type, vertex_index_numba_type),
        (h_next_H, halfedge_index_numba_type, halfedge_index_numba_type),
        (h_twin_H, halfedge_index_numba_type, halfedge_index_numba_type),
        (f_left_H, halfedge_index_numba_type, face_index_numba_type),
        (h_bound_F, face_index_numba_type, halfedge_index_numba_type),
    ]
    numba_dicts = []

    for d, kt, vt in d_kt_vt:
        D = py2numba_dict(d, kt, vt, safe=safe)
        numba_dicts.append(D)
    return numba_dicts


@jit
def _rekey_half_edge_dicts(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
):
    Nv = len(xyz_coord_V)
    Nh = len(h_next_H)
    Nf = len(h_bound_F)
    old_V_new = sorted(xyz_coord_V.keys())
    new_V_old = {val: key for key, val in enumerate(old_V_new) if val != key}
    old_H_new = sorted(h_next_H.keys())
    new_H_old = {val: key for key, val in enumerate(old_H_new) if val != key}
    old_F_new = sorted(h_bound_F.keys())
    new_F_old = {val: key for key, val in enumerate(old_F_new) if val != key}
    for _k_old, _k_new in new_V_old.items():
        k_old = _k_old
        k_new = _k_new
        xyz = xyz_coord_V.pop(k_old)
        xyz_coord_V[k_new] = xyz
        h_out_v_old = h_out_V.pop(k_old)
        if h_out_v_old in new_H_old:
            h_out_v = new_H_old[h_out_v_old]
            h_out_V[k_new] = h_out_v
        else:
            h_out_V[k_new] = h_out_v_old
    for _k_old, _k_new in new_H_old.items():
        k_old = _k_old
        k_new = _k_new
        v_origin_h_old = v_origin_H.pop(k_old)
        if v_origin_h_old in new_V_old:
            v_origin_h = new_V_old[v_origin_h_old]
            v_origin_H[k_new] = v_origin_h
        else:
            v_origin_H[k_new] = v_origin_h_old
        h_next_h_old = h_next_H.pop(k_old)
        if h_next_h_old in new_H_old:
            h_next_h = new_H_old[h_next_h_old]
            h_next_H[k_new] = h_next_h
        else:
            h_next_H[k_new] = h_next_h_old
        h_twin_h_old = h_twin_H.pop(k_old)
        if h_twin_h_old in new_H_old:
            h_twin_h = new_H_old[h_twin_h_old]
            h_twin_H[k_new] = h_twin_h
        else:
            h_twin_H[k_new] = h_twin_h_old
        f_left_h_old = f_left_H.pop(k_old)
        if f_left_h_old in new_F_old:
            f_left_h = new_F_old[f_left_h_old]
            f_left_H[k_new] = f_left_h
        else:
            f_left_H[k_new] = f_left_h_old
    for _k_old, _k_new in new_F_old.items():
        k_old = _k_old
        k_new = _k_new
        h_bound_f_old = h_bound_F.pop(k_old)
        if h_bound_f_old in new_H_old:
            h_bound_f = new_H_old[h_bound_f_old]
            h_bound_F[k_new] = h_bound_f
        else:
            h_bound_F[k_new] = h_bound_f_old
    return (
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )


@jit
def rekey_half_edge_dicts(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    h_comp_B,
):
    """
    Reorders and updates keys to be contiguous integers starting
    """
    Nv = len(xyz_coord_V)
    Nh = len(h_next_H)
    Nf = len(h_bound_F)
    Nb = len(h_comp_B)
    old_V_new = sorted(xyz_coord_V.keys())
    new_V_old = {val: key for key, val in enumerate(old_V_new) if val != key}
    old_H_new = sorted(h_next_H.keys())
    new_H_old = {val: key for key, val in enumerate(old_H_new) if val != key}
    old_F_new = sorted(h_bound_F.keys())
    new_F_old = {val: key for key, val in enumerate(old_F_new) if val != key}
    ##
    # oldBkeys = [-b1, -b2, -b3, ...]
    # newBkeys = [-1, -2, -3, ...]
    oldBkeys = sorted(h_comp_B.keys(), reverse=True)
    newBkeys = np.array(
        [-(_key + 1) for _key, val in enumerate(old_Bkeys)], dtype=_NUMPY_INT_
    )
    new_B_old = {
        k_old: k_new for k_old, k_new in zip(oldBkeys, newBkeys) if k_old != k_new
    }
    new_F_old.update(new_B_old)
    for _k_old, _k_new in new_V_old.items():
        k_old = _k_old
        k_new = _k_new
        xyz = xyz_coord_V.pop(k_old)
        xyz_coord_V[k_new] = xyz
        h_out_v_old = h_out_V.pop(k_old)
        if h_out_v_old in new_H_old:
            h_out_v = new_H_old[h_out_v_old]
            h_out_V[k_new] = h_out_v
        else:
            h_out_V[k_new] = h_out_v_old
    for _k_old, _k_new in new_H_old.items():
        k_old = _k_old
        k_new = _k_new
        v_origin_h_old = v_origin_H.pop(k_old)
        if v_origin_h_old in new_V_old:
            v_origin_h = new_V_old[v_origin_h_old]
            v_origin_H[k_new] = v_origin_h
        else:
            v_origin_H[k_new] = v_origin_h_old
        h_next_h_old = h_next_H.pop(k_old)
        if h_next_h_old in new_H_old:
            h_next_h = new_H_old[h_next_h_old]
            h_next_H[k_new] = h_next_h
        else:
            h_next_H[k_new] = h_next_h_old
        h_twin_h_old = h_twin_H.pop(k_old)
        if h_twin_h_old in new_H_old:
            h_twin_h = new_H_old[h_twin_h_old]
            h_twin_H[k_new] = h_twin_h
        else:
            h_twin_H[k_new] = h_twin_h_old
        f_left_h_old = f_left_H.pop(k_old)
        if f_left_h_old in new_F_old:
            f_left_h = new_F_old[f_left_h_old]
            f_left_H[k_new] = f_left_h
        else:
            f_left_H[k_new] = f_left_h_old
    for _k_old, _k_new in new_F_old.items():
        k_old = _k_old
        k_new = _k_new
        h_bound_f_old = h_bound_F.pop(k_old)
        if h_bound_f_old in new_H_old:
            h_bound_f = new_H_old[h_bound_f_old]
            h_bound_F[k_new] = h_bound_f
        else:
            h_bound_F[k_new] = h_bound_f_old

    for _k_old, _k_new in new_B_old.items():
        k_old = _k_old
        k_new = _k_new
        h_comp_b_old = h_comp_B.pop(k_old)
        if h_comp_b_old in new_H_old:
            h_comp_b = new_H_old[h_comp_b_old]
            h_comp_B[k_new] = h_comp_b
        else:
            h_comp_B[k_new] = h_comp_b_old
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


@jit
def half_edge_arrays_to_dicts(
    xyz_coord_V_array,
    h_out_V_array,
    v_origin_H_array,
    h_next_H_array,
    h_twin_H_array,
    f_left_H_array,
    h_bound_F_array,
    h_comp_B_array,
):
    Nv = len(xyz_coord_V_array)
    Nh = len(h_next_H_array)
    Nf = len(h_bound_F_array)
    Nb = len(h_comp_B_array)
    xyz_coord_V = Dict.empty(vertex_index_numba_type, xyz_numba_type, n_keys=Nv)
    h_out_V = Dict.empty(vertex_index_numba_type, halfedge_index_numba_type, n_keys=Nv)
    v_origin_H = Dict.empty(
        halfedge_index_numba_type, vertex_index_numba_type, n_keys=Nh
    )
    h_next_H = Dict.empty(
        halfedge_index_numba_type, halfedge_index_numba_type, n_keys=Nh
    )
    h_twin_H = Dict.empty(
        halfedge_index_numba_type, halfedge_index_numba_type, n_keys=Nh
    )
    f_left_H = Dict.empty(halfedge_index_numba_type, face_index_numba_type, n_keys=Nh)
    h_bound_F = Dict.empty(face_index_numba_type, halfedge_index_numba_type, n_keys=Nf)
    h_comp_B = Dict.empty(
        boundary_index_numba_type, halfedge_index_numba_type, n_keys=Nb
    )
    for _k in range(Nv):
        k = vertex_index_numba_type(_k)
        xyz_coord_V[k] = xyz_coord_V_array[_k]
        h_out_V[k] = halfedge_index_numba_type(h_out_V_array[_k])
    for _k in range(Nh):
        k = halfedge_index_numba_type(_k)
        v_origin_H[k] = vertex_index_numba_type(v_origin_H_array[_k])
        h_next_H[k] = halfedge_index_numba_type(h_next_H_array[_k])
        h_twin_H[k] = halfedge_index_numba_type(h_twin_H_array[_k])
        f_left_H[k] = face_index_numba_type(f_left_H_array[_k])
    for _k in range(Nf):
        k = face_index_numba_type(_k)
        h_bound_F[k] = halfedge_index_numba_type(h_bound_F_array[_k])
    for _k in range(Nb):
        k = boundary_index_numba_type(-(_k + 1))
        h_comp_B[k] = halfedge_index_numba_type(h_comp_B_array[_k])
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


@jit(parallel=True)
def half_edge_dicts_to_arrays_old(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
):
    """this works"""
    Nv = len(xyz_coord_V)
    Nh = len(h_next_H)
    Nf = len(h_bound_F)
    xyz_coord_V_array = np.zeros((Nv, 3), dtype=_NUMPY_FLOAT_)
    h_out_V_array = np.zeros(Nv, dtype=_NUMPY_INT_)
    v_origin_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    h_next_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    h_twin_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    f_left_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    h_bound_F_array = np.zeros(Nf, dtype=_NUMPY_INT_)

    dic_V_arr = sorted(xyz_coord_V.keys())
    arr_V_dic = np.zeros(dic_V_arr[-1], dtype=_NUMPY_INT_)
    dic_H_arr = sorted(h_next_H.keys())
    arr_H_dic = np.zeros(dic_H_arr[-1], dtype=_NUMPY_INT_)
    dic_F_arr = sorted(h_bound_F.keys())
    arr_F_dic = np.zeros(dic_F_arr[-1], dtype=_NUMPY_INT_)

    for v_arr in prange(Nv):
        v_dic = dic_V_arr[v_arr]
        arr_V_dic[v_dic] = v_arr
        xyz_coord_V_array[v_arr] = xyz_coord_V[v_dic]
        h_out_V_array[v_arr] = h_out_V[v_dic]  #
    for h_arr in prange(Nh):
        h_dic = dic_H_arr[h_arr]
        arr_H_dic[h_dic] = h_arr
        v_origin_H_array[h_arr] = v_origin_H[h_dic]
        h_next_H_array[h_arr] = h_next_H[h_dic]
        h_twin_H_array[h_arr] = h_twin_H[h_dic]
        f_left_H_array[h_arr] = f_left_H[h_dic]
    for f_arr in prange(Nf):
        f_dic = dic_F_arr[f_arr]
        arr_F_dic[f_dic] = f_arr
        h_bound_F_array[f_arr] = h_bound_F[f_dic]

    return (
        xyz_coord_V_array,
        arr_H_dic[h_out_V_array],
        arr_V_dic[v_origin_H_array],
        arr_H_dic[h_next_H_array],
        arr_H_dic[h_twin_H_array],
        arr_F_dic[f_left_H_array],
        arr_H_dic[h_bound_F_array],
    )


@jit(parallel=True)
def half_edge_dicts_to_arrays(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    rekey=True,
):
    if rekey:
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        ) = rekey_half_edge_dicts(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
    Nv = len(xyz_coord_V)
    Nh = len(h_next_H)
    Nf = len(h_bound_F)
    xyz_coord_V_array = np.zeros((Nv, 3), dtype=_NUMPY_FLOAT_)
    h_out_V_array = np.zeros(Nv, dtype=_NUMPY_INT_)
    v_origin_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    h_next_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    h_twin_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    f_left_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
    h_bound_F_array = np.zeros(Nf, dtype=_NUMPY_INT_)

    dic_V_arr = sorted(xyz_coord_V.keys())
    # arr_V_dic = np.zeros(dic_V_arr[-1], dtype=_NUMPY_INT_)
    dic_H_arr = sorted(h_next_H.keys())
    # arr_H_dic = np.zeros(dic_H_arr[-1], dtype=_NUMPY_INT_)
    dic_F_arr = sorted(h_bound_F.keys())
    # arr_F_dic = np.zeros(dic_F_arr[-1], dtype=_NUMPY_INT_)

    for v_arr in prange(Nv):
        v_dic = dic_V_arr[v_arr]
        v_dic = vertex_index_numba_type(v_dic)
        xyz_coord_V_array[v_arr] = xyz_coord_V[v_dic]
        h_out_V_array[v_arr] = h_out_V[v_dic]  #
    for h_arr in prange(Nh):
        h_dic = dic_H_arr[h_arr]
        h_dic = halfedge_index_numba_type(h_dic)
        v_origin_H_array[h_arr] = v_origin_H[h_dic]
        h_next_H_array[h_arr] = h_next_H[h_dic]
        h_twin_H_array[h_arr] = h_twin_H[h_dic]
        f_left_H_array[h_arr] = f_left_H[h_dic]
    for f_arr in prange(Nf):
        f_dic = dic_F_arr[f_arr]
        f_dic = face_index_numba_type(f_dic)
        h_bound_F_array[f_arr] = h_bound_F[f_dic]

    return (
        xyz_coord_V_array,
        h_out_V_array,
        v_origin_H_array,
        h_next_H_array,
        h_twin_H_array,
        f_left_H_array,
        h_bound_F_array,
    )


@jit(
    f"{halfedge_index_numba_type}({vertex_index_numba_type}[:,:], {halfedge_index_numba_type})"
)
def jit_get_halfedge_index_of_twin(H, h):
    """
    Find the half-edge twin to h in the list of half-edges H.

    Parameters
    ----------
    H : list
        List of half-edges [[v0, v1], ...]
    h : int
        Index of half-edge in H

    Returns
    -------
    h_twin : int
        Index of H[h_twin]=[v1,v0] in H, where H[h]=[v0,v1]. Returns -1 if twin not found.
    """
    Nhedges = len(H)
    v0 = H[h][0]
    v1 = H[h][1]
    for h_twin in range(Nhedges):
        if H[h_twin][0] == v1 and H[h_twin][1] == v0:
            return halfedge_index_numba_type(h_twin)

    return halfedge_index_numba_type(-1)


@jit
def jit_vf_samples_to_he_samples(V, F):
    # (V, F) = source_samples
    Nfaces = len(F)
    Nvertices = len(V)
    xyz_coord_V = np.zeros((Nvertices, 3), dtype=_NUMPY_FLOAT_)
    xyz_coord_V[:] = V
    _Nhedges = 3 * Nfaces * 2
    _H = np.zeros((_Nhedges, 2), dtype=_NUMPY_INT_)
    h_out_V = -np.ones(Nvertices, dtype=_NUMPY_INT_)
    _v_origin_H = np.zeros(_Nhedges, dtype=_NUMPY_INT_)
    _h_next_H = -np.ones(_Nhedges, dtype=_NUMPY_INT_)
    _f_left_H = np.zeros(_Nhedges, dtype=_NUMPY_INT_)
    h_bound_F = np.zeros(Nfaces, dtype=_NUMPY_INT_)

    # h_count = 0
    for f in range(Nfaces):
        h_bound_F[f] = 3 * f
        for i in range(3):
            h = 3 * f + i
            h_next = 3 * f + (i + 1) % 3
            v0 = F[f][i]
            v1 = F[f][(i + 1) % 3]
            _H[h] = [v0, v1]
            _v_origin_H[h] = v0
            _f_left_H[h] = f
            _h_next_H[h] = h_next
            if h_out_V[v0] == -1:
                h_out_V[v0] = h
    h_count = 3 * Nfaces
    need_twins = set([_NUMBA_INT_(_) for _ in range(h_count)])
    need_next = set([_NUMBA_INT_(0)])
    need_next.pop()
    _h_twin_H = -2 * np.ones(_Nhedges, dtype=_NUMPY_INT_)  # -2 means not set
    while need_twins:
        h = need_twins.pop()
        if _h_twin_H[h] == -2:  # if twin not set
            h_twin = jit_get_halfedge_index_of_twin(
                _H, h
            )  # returns -1 if twin not found
            if h_twin == -1:  # if twin not found
                h_twin = _NUMBA_INT_(h_count)
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
    # H = _H[:Nhedges]
    v_origin_H = _v_origin_H[:Nhedges]
    h_next_H = _h_next_H[:Nhedges]
    f_left_H = _f_left_H[:Nhedges]
    h_twin_H = _h_twin_H[:Nhedges]
    # h_next_H.extend([-1] * len(need_next))
    while need_next:
        h = need_next.pop()
        h_next = h_twin_H[h]
        # rotate ccw around origin of twin until we find nex h on boundary
        while f_left_H[h_next] != -1:
            h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
        h_next_H[h] = h_next

    # find and enumerate boundaries -1,-2,...
    H_need2visit = set([_NUMBA_INT_(h) for h in range(Nhedges) if f_left_H[h] == -1])
    bdry_count = 0
    while H_need2visit:
        bdry_count += 1
        h_start = H_need2visit.pop()
        f_left_H[h_start] = -bdry_count
        h = h_next_H[h_start]
        while h != h_start:
            H_need2visit.remove(h)
            f_left_H[h] = -bdry_count
            h = h_next_H[h]

    return (
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )


#######################################################
@jit
def generate_H_next_h(h, h_next_H):
    """Generate half-edges in the face/boundary cycle containing half-edge h"""
    h_start = h
    while True:
        yield h
        h = h_next_H[h]
        if h == h_start:
            break


@jit
def generate_H_bound_f(f, h_bound_F, h_next_H):
    """Generate half-edges on the boundary of face f"""
    h = h_bound_F[f]
    h_start = h
    while True:
        yield h
        h = h_next_H[h]
        if h == h_start:
            break


@jit
def find_h_comp_B_array(
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
    self.h_bound_F
    self.generate_H_bound_f
    self.interior_boundary_contains_h
    self.h_twin_h
    self.f_left_h
    self.generate_H_next_h
    """
    h_comp_B = dict()
    complement_boundary_contains_H = set()
    F_need2check = set(f_left_H)  # set of faces that need to be checked
    while F_need2check:
        f = F_need2check.pop()
        for h in generate_H_bound_f(f, h_bound_F, h_next_H):
            if f_left_H[h_twin_H[h]] < 0:
                complement_boundary_contains_H.add(h_twin_H[h])
    while complement_boundary_contains_H:
        h = complement_boundary_contains_H.pop()
        bdry = f_left_H[h]
        h_comp_B[bdry] = h
        for h in generate_H_next_h(h, h_next_H):
            complement_boundary_contains_H.discard(h)
    Nb = len(h_comp_B)
    Bkeys = sorted(h_comp_B.keys(), reverse=True)
    h_comp_B_array = np.array([h_comp_B[b] for b in Bkeys], dtype=_NUMPY_INT_)
    return h_comp_B_array


#######################################################
# Laplacian operators
@jit(parallel=True)
def cotan_laplacian(
    Q, xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H, h_bound_F
):
    """
    Computes the cotan Laplacian of Q at each vertex
    """
    Nv = len(xyz_coord_V)
    lapQ = np.zeros_like(Q)
    for vi in prange(Nv):
        Atot = 0.0
        ri = xyz_coord_V[vi]
        qi = Q[vi]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        h_start = h_out_V[vi]
        hij = h_start
        while True:
            hijm1 = h_next_H[h_twin_H[hij]]
            hijp1 = h_twin_H[h_next_H[h_next_H[hij]]]
            vjm1 = v_origin_H[h_twin_H[hijm1]]
            vj = v_origin_H[h_twin_H[hij]]
            vjp1 = v_origin_H[h_twin_H[hijp1]]

            qj = Q[vj]

            rjm1 = xyz_coord_V[vjm1]
            rj = xyz_coord_V[vj]
            rjp1 = xyz_coord_V[vjp1]

            rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
            rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
            ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

            Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
            Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
            Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
            Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

            cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

            cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

            cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
            cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

            Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
            lapQ[vi] += (cot_thetam + cot_thetap) * (qj - qi) / 2
            hij = h_next_H[h_twin_H[hij]]
            if hij == h_start:
                break
        lapQ[vi] /= Atot

    return lapQ


@jit(parallel=True)
def belkin_laplacian(
    Q,
    s,
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
):
    """
    ...
    """
    Nv = len(xyz_coord_V)
    Nf = len(h_bound_F)
    lapQ = np.zeros_like(Q)
    for v in prange(Nv):
        xyz = xyz_coord_V[v]
        for f in range(Nf):
            h0 = h_bound_F[f]
            h1 = h_next_H[h0]
            h2 = h_next_H[h1]
            v0 = v_origin_H[h0]
            v1 = v_origin_H[h1]
            v2 = v_origin_H[h2]
            xyz0 = xyz_coord_V[v0]
            xyz1 = xyz_coord_V[v1]
            xyz2 = xyz_coord_V[v2]
            Avec = np.cross(xyz0, xyz1) + np.cross(xyz1, xyz2) + np.cross(xyz2, xyz0)
            Af = np.linalg.norm(Avec) / 2
            lapQ[v] += (
                (1 / (4 * np.pi * s**2))
                * (Af / 3)
                * np.exp(-np.linalg.norm(xyz - xyz0) ** 2 / (4 * s))
                * (Q[v0] - Q[v])
            )
            lapQ[v] += (
                (1 / (4 * np.pi * s**2))
                * (Af / 3)
                * np.exp(-np.linalg.norm(xyz - xyz1) ** 2 / (4 * s))
                * (Q[v1] - Q[v])
            )
            lapQ[v] += (
                (1 / (4 * np.pi * s**2))
                * (Af / 3)
                * np.exp(-np.linalg.norm(xyz - xyz2) ** 2 / (4 * s))
                * (Q[v2] - Q[v])
            )
    return lapQ
