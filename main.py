from src.python.mesh import (
    HalfEdgeMesh,
    # HalfEdgeCurve,
    # HalfEdgeLoop,
    # Boundary2D,
)


from src.python.mesh.viewer import MeshViewer

# from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
ply_path = "./data/half_edge_base/ply/neovius_he.ply"


m = HalfEdgeMesh.load(ply_path=ply_path)

he_samples = m.he_samples
vf_samples = (m.xyz_coord_V, m.V_of_F)
V_of_H = m.V_of_H
h = 13
# %%
from src.python.utilities import (
    vf_samples_to_he_samples,
)

# from src.python.half_edge_base_utils import get_halfedge_index_of_twin as get_halfedge_index_of_twin_numba
from src.python.half_edge_base_utils import V_of_F as get_V_of_F_numba
from src.python.half_edge_base_utils import find_h_right_B as find_h_right_B_numba
from src.python.half_edge_base_utils import (
    vf_samples_to_he_samples as vf_samples_to_he_samples_numba,
)

he_cpp = vf_samples_to_he_samples(*vf_samples)
he_numba = vf_samples_to_he_samples_numba(*vf_samples)

(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    h_right_B,
) = he_cpp

m_cpp = HalfEdgeMesh(*he_cpp)
m_numba = HalfEdgeMesh(*he_numba)

mv = MeshViewer(m)
mv_cpp = MeshViewer(m_cpp, show_half_edges=True)
rgba_boundary_face = mv_cpp.colors["magenta80"]
rgba_boundary_face = mv_cpp.colors["magenta80"]
# rgba_boundary_half_edge
# rgba_boundary_face
mv_cpp.update(rgba_boundary_face=rgba_boundary_face)
mv_cpp.plot()

# %timeit vf_samples_to_he_samples(*vf_samples)
# %timeit vf_samples_to_he_samples_numba(*vf_samples)
