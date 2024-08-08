from src.python.jit_brane import HalfEdgeMeshBase as jhem
from src.python.jit_brane import (
    vertex_index_numba_type,
    halfedge_index_numba_type,
    face_index_numba_type,
    boundary_index_numba_type,
    xyz_numba_type,
)
from src.python.half_edge_mesh import HalfEdgeMesh as hem
from numba.typed import Dict
from numba.types import DictType
import numpy as np

# from src.python.mesh_viewer import MeshViewer
import os

_TEST_DIR_ = "./output/jit_brane"
_NUM_VERTS_ = [
    # 162,
    642,
    2562,
    # 10242,
    # 40962,
    # 163842,
]  # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
_SURF_NAMES_ = [f"sphere_{N:06d}_he" for N in _NUM_VERTS_]


def make_output_dir(overwrite=False):

    # run_name = f"run_{n_run:06d}"
    # output_dir = f"{test_dir}/{run_name}"
    output_dir = _TEST_DIR_
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError(
            f"{output_dir} already exists. Choose a different output_dir, or Set overwrite=True"
        )
    os.system(f"mkdir -p {output_dir}")


def load_hem():
    return [
        hem.from_half_edge_ply(f"./data/ply/binary/{name}.ply") for name in _SURF_NAMES_
    ]


def dict2numba(d, kt, vt):
    D = Dict.empty(kt, vt)
    for k, v in d.items():
        D[k] = v
    return D


def hem_dicts2numbs(*py_dicts):
    # [
    #     xyz_coord_V,
    #     h_out_V,
    #     v_origin_H,
    #     h_next_H,
    #     h_twin_H,
    #     f_left_H,
    #     h_bound_F,
    # ] = py_dicts
    kv_types = [
        (vertex_index_numba_type, xyz_numba_type),
        (vertex_index_numba_type, halfedge_index_numba_type),
        (halfedge_index_numba_type, vertex_index_numba_type),
        (halfedge_index_numba_type, halfedge_index_numba_type),
        (halfedge_index_numba_type, halfedge_index_numba_type),
        (halfedge_index_numba_type, face_index_numba_type),
        (face_index_numba_type, halfedge_index_numba_type),
    ]
    numba_dicts = []
    for d, (kt, vt) in zip(py_dicts, kv_types):
        # D = Dict.empty(kt, vt)
        # numba_dicts.append(D)
        # for k,v in d.items():
        #     D[k]=v
        D = dict2numba(d, kt, vt)
        numba_dicts.append(D)
    return numba_dicts


M = load_hem()
m = M[0]
numba_dicts = hem_dicts2numbs(*m.data_dicts)
# %%
_xyz_coord_V = m._xyz_coord_V
_h_out_V = m._h_out_V
_v_origin_H = m._v_origin_H
_h_next_H = m._h_next_H
_h_twin_H = m._h_twin_H
_f_left_H = m._f_left_H
_h_bound_F = m._h_bound_F
kv_types = [
    (vertex_index_numba_type, xyz_numba_type),
    (vertex_index_numba_type, halfedge_index_numba_type),
    (halfedge_index_numba_type, vertex_index_numba_type),
    (halfedge_index_numba_type, halfedge_index_numba_type),
    (halfedge_index_numba_type, halfedge_index_numba_type),
    (halfedge_index_numba_type, face_index_numba_type),
    (face_index_numba_type, halfedge_index_numba_type),
]
kt, vt = kv_types[0]
dir(vt)
dir(vt.dtype)
vt.dtype.dtype
D = Dict.empty(kt, vt)
D[0] = np.array([1.0, 0.0, 2.0], dtype=vt.dtype.dtype)
# %%
# init_arrs = m.data_lists
# m.data_dicts
# jm = jhem(*init_dicts)
