from src.python.jit_brane import HalfEdgeMeshBase, HalfEdgeMeshBuilder
from src.python.half_edge_mesh import HalfEdgeMesh
import numpy as np
from src.python.mesh_viewer import MeshViewer
import os


_TEST_DIR_ = "./output/jit_brane"
_NUM_VERTS_ = [
    12,
    42,
    162,
    642,
    2562,
    10242,
    40962,
    163842,
    655362,
    2621442,
]
_SURF_NAMES_ = [f"sphere_{N:06d}_he" for N in _NUM_VERTS_]
_SURF_NAMES_ = [
    "sphere_002562_he",
    "sphere_010242_he",
    "sphere_040962_he",
    "sphere_163842_he",
    "annulus",
    "hex_sector",
    "neovius",
]
_PLYS_ = [f"./data/ply/binary/{name}.ply" for name in _SURF_NAMES_]
_NPZS_ = [f"./data/half_edge_arrays/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_]


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


_M = HalfEdgeMeshBuilder.load_test_spheres()
# %%

m = _M[6]
F = m.V_of_F


Nv, Nf = m.num_vertices, m.num_faces
Af = m.area_F()
Ab = m.barcell_area_V()
# Av = m.vorcell_area_V()
# Am = m.meyercell_area_V()
s = np.mean(Ab)
Q = m.xyz_array

lapQc = m.cotan_laplacian(Q)
lapQb = m.pbelkin_laplacian(Q, s)
lapQg = m.pguckenberger_laplacian(Q)
