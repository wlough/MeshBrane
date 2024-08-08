from src.python.jit_brane import HalfEdgeMeshBase as jhem
from src.python.jit_brane import py2numba_half_edge_mesh_dicts
from src.python.half_edge_mesh import HalfEdgeMesh as hem
from numba import jit
import numpy as np


np.int64(1)
# from src.python.mesh_viewer import MeshViewer
import os

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
_SURF_NAMES_ = ["sphere_002562_he", "annulus", "hex_sector", "neovius"]


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


M = load_hem()
m = M[0]
init_dicts = py2numba_half_edge_mesh_dicts(*m.data_dicts)
jm = jhem(*init_dicts)
# jm.run_tests()

# dat = [np.array(_) for _ in m.data_lists]
# mv = MeshViewer(*dat)
# mv.plot()
V,H,F = jm.patch_from_seed_vertex(3)
# %%
d1=jm.xyz_coord_V.copy()
d1[13]
d1[13]=2*d1[13]
jm.xyz_coord_V[13]
# %%



Q = jm.xyz_array
lapQcotan=jm.cotan_laplacian(Q)
lapQcotan1=jm.cotan_laplacian1(Q)
np.linalg.norm(lapQcotan-lapQcotan1)
jm.data_lists

%timeit jm.cotan_laplacian(Q)
%timeit jm.cotan_laplacian1(Q)

#

#

#

#

#

#

#


# %%
