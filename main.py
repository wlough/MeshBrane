from src.python.mesh import (
    HalfEdgeMesh,
    # HalfEdgeCurve,
    # HalfEdgeLoop,
    # Boundary2D,
)


# from src.python.mesh.viewer import MeshViewer

# from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
ply_path = "./data/half_edge_base/ply/neovius_he.ply"


m = HalfEdgeMesh.load(ply_path=ply_path)
# mv = MeshViewer(m)
# mv.plot()
he_samples = m.he_samples
vf_samples = (m.xyz_coord_V, m.V_of_F)
H = m.V_of_H
h = 0
hh = np.array(range(m.num_half_edges), dtype="int32")
# %%
from src.python.half_edge_utils import get_halfedge_index_of_twin, vf_samples_to_he_samples
# from src.python.half_edge_base_utils import get_halfedge_index_of_twin as get_halfedge_index_of_twin_numba
from src.python.half_edge_base_utils import V_of_F as get_V_of_F_numba
from src.python.half_edge_base_utils import find_h_right_B as find_h_right_B_numba
from src.python.half_edge_base_utils import (
    vf_samples_to_he_samples as vf_samples_to_he_samples_numba,
)

ht = get_halfedge_index_of_twin(H, h)

# ht_numba = get_halfedge_index_of_twin(H, h)
# V_of_F = get_V_of_F(*he_samples)
# V_of_F_numba = get_V_of_F_numba(*he_samples)
#
# h_right_B = find_h_right_B(*he_samples[:-1])
# h_right_B_numba = find_h_right_B_numba(*he_samples[:-1])
#
he_cpp = vf_samples_to_he_samples(*vf_samples)
he_numba = vf_samples_to_he_samples_numba(*vf_samples)
[_.shape for _ in he_cpp]

def testfun():
    he_cpp = vf_samples_to_he_samples(*vf_samples)
    return 0

%timeit testfun()
%timeit vf_samples_to_he_samples_numba(*vf_samples)
