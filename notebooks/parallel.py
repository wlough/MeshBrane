import multiprocessing as mp
from src.python.ply_tools import SphereFactory, VertTri2HalfEdgeConverter
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.half_edge_test import HalfEdgeTestSphere as TestSurf
import numpy as np

_TEST_DIR_ = "./output/parallel_sphere_tests"
_NUM_VERTS_ = [162, 642, 2562, 10242, 40962, 163842]
# [12, 42, 162, 642, 2562, 10242, 40962, 163842]
_SURF_NAMES_ = [f"unit_sphere_{N:06d}_he" for N in _NUM_VERTS_]
_SURF_PARAMS_ = [1.0]


def get_M(
    run_name="multi0",
    test_dir=_TEST_DIR_,
    surf_names=_SURF_NAMES_,
    surface_params=_SURF_PARAMS_,
    overwrite=False,
):
    # fixed_heat_param_vals = [0.0025, 0.01, 0.025]
    M = []
    for _ in range(len(surf_names)):
        surf = surf_names[_]
        print("\n" + surf + "\n" + len(surf) * "-")
        ply = f"./data/ply/binary/{surf}.ply"
        m = TestSurf.from_half_edge_ply(ply, 1.0)
        M.append(m)
    print("Done")
    return M
