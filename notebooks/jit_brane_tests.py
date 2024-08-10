from src.python.jit_brane import HalfEdgeMeshBase as jhem, HalfEdgeMeshBuilder
from src.python.ply_tools import VertTri2HalfEdgeConverter
from src.python.jit_brane_utils import (
    jit_vf_samples_to_he_samples,
    half_edge_arrays_to_dicts,
    py2numba_half_edge_mesh_dicts,
    #     halfedge_index_numba_type,
    #     rekey_half_edge_dicts,
    #     half_edge_dicts_to_arrays,
    #     half_edge_dicts_to_arrays2,
)
from src.python.half_edge_mesh import HalfEdgeMesh as hem
from numba import jit
import numpy as np

from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_test import HalfEdgeTestSphere
import os
from src.python.sphere_builder import cotan_laplacian, belkin_laplacian

_TEST_DIR_ = "./output/jit_brane"
_NUM_VERTS_ = [
    # 162,
    # 642,
    2562,
    # 10242,
    # 40962,
    # 163842,
]  # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
_SURF_NAMES_ = [f"sphere_{N:06d}_he" for N in _NUM_VERTS_]
_SURF_NAMES_ = ["sphere_002562_he", "sphere_010242_he", "sphere_163842_he", "annulus", "hex_sector", "neovius"]
plys = [f"./data/ply/binary/{name}.ply" for name in _SURF_NAMES_]
ply = plys[2]


def make_output_dir(overwrite=False):

    # run_name = f"run_{n_run:06d}"
    # output_dir = f"{test_dir}/{run_name}"
    output_dir = _TEST_DIR_
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError(f"{output_dir} already exists. Choose a different output_dir, or Set overwrite=True")
    os.system(f"mkdir -p {output_dir}")


def load_hem():
    return [hem.from_half_edge_ply(f"./data/ply/binary/{name}.ply") for name in _SURF_NAMES_]


def load_jhem():
    return [HalfEdgeMeshBuilder.from_half_edge_ply(f"./data/ply/binary/{name}.ply") for name in _SURF_NAMES_]


#
# M = load_hem()
# jM = load_jhem()
# jm = jM[0]
jm = HalfEdgeMeshBuilder.from_half_edge_ply(ply)
tm = HalfEdgeTestSphere.from_half_edge_ply(ply, 1.0)
# %%
Q = jm.xyz_array + 0.0
data_arrays = jm.data_arrays
lapQcotan = cotan_laplacian(Q, *data_arrays)
lapQbelkin = belkin_laplacian(Q, 0.01, *data_arrays)

jlapQcotan = jm.cotan_laplacian(Q)
jlapQobelkin = jm.obelkin_laplacian(Q, 0.01)
jlapQpbelkin = jm.pbelkin_laplacian(Q, 0.01)
# jlapQbelkin1 = jm.belkin_laplacian(Q, .01)
tlapQcotan = tm.cotan_laplacian(Q)
tlapQbelkin = tm.belkin_laplacian(Q, 0.01)
#
# lapQbelkin-lapQbelkin1
# %timeit jm.cotan_laplacian(Q)
# %timeit jm.cotan_laplacian1(Q)
