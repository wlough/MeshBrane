from src.python.half_edge_base_utils import (
    vf_samples_to_he_samples,
    find_h_comp_B,
    make_half_edge_base_numba_utils,
)

make_half_edge_base_numba_utils()

# %%
import numpy as np
import os
from numba import jit, prange
from numba.pycc import CC
from numba.types import int32, float64

_NUMBA_INT_ = int32
_NUMPY_INT_ = np.int32
cc = CC()
# %%
from numba import typeof

# %%
import numpy as np
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.half_edge_base_numba_utils import (
    numba_vf_samples_to_he_samples,
    numba_find_h_comp_B,
)

# m = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/sphere_000642_he.ply")
m = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/neovius.ply")

V, F = m.xyz_array, m.V_of_F
F = np.int32(F)


(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    h_comp_B,
) = numba_vf_samples_to_he_samples(V, F)
args = (
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
)
# %%


def find_h_comp_B(
    _xyz_coord_V,
    _h_out_V,
    _v_origin_H,
    _h_next_H,
    _h_twin_H,
    _f_left_H,
    _h_bound_F,
):
    (
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    ) = (
        np.array(_xyz_coord_V, dtype=np.float64),
        np.array(_h_out_V, dtype=np.int32),
        np.array(_v_origin_H, dtype=np.int32),
        np.array(_h_next_H, dtype=np.int32),
        np.array(_h_twin_H, dtype=np.int32),
        np.array(_f_left_H, dtype=np.int32),
        np.array(_h_bound_F, dtype=np.int32),
    )
    return numba_find_h_comp_B(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )


B = find_h_comp_B(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
)
# %%
