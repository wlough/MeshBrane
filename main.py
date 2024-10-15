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
# %%
from src.cpp.half_edge_utils import (
    get_V_of_F,
    get_halfedge_index_of_twin,
    find_h_right_B,
    vf_samples_to_he_samples,
)
from src.python.half_edge_base_utils import V_of_F as get_V_of_F_numba
from src.python.half_edge_base_utils import find_h_right_B as find_h_right_B_numba
from src.python.half_edge_base_utils import (
    vf_samples_to_he_samples as vf_samples_to_he_samples_numba,
)

ht = get_halfedge_index_of_twin(H, h)

V_of_F = get_V_of_F(*he_samples)
V_of_F_numba = get_V_of_F_numba(*he_samples)

h_right_B = find_h_right_B(*he_samples[:-1])
h_right_B_numba = find_h_right_B_numba(*he_samples[:-1])


_B = [set(m.generate_H_next_h(h)) for h in h_right_B]
_B_numba = [set(m.generate_H_next_h(h)) for h in h_right_B_numba]
[sum([x == y for y in _B]) for x in _B_numba]
B = set()
B_numba = set()
for _ in _B:
    B |= _
for _ in _B_numba:
    B_numba |= _

B_numba == B
set(h_right_B) - set(h_right_B_numba)
set(h_right_B_numba) - set(h_right_B)

# %timeit find_h_right_B(*he_samples[:-1])
# %timeit find_h_right_B_numba(*he_samples[:-1])


# %timeit get_V_of_F(*he_samples)
# %timeit V_of_F(*he_samples)
