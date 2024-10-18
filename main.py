from src.python.mesh import HalfEdgeMesh
from src.python.mesh.cply_tools import vf_samples_to_he_samples, Pet, MeshConverter
import numpy as np


ply_path = "./data/half_edge_base/ply/annulus_he.ply"
# ply_path = "./data/half_edge_base/ply/neovius_he.ply"
# vf_ply_path = "./data/ply/ascii/annulus.ply"
vf_ply_path = "./data/half_edge_base/ply/annulus_vf.ply"
# vf_ply_path = "./data/half_edge_base/ply/dumbbell_vf.ply"
m = HalfEdgeMesh.load(ply_path=ply_path)
# c = MeshConverter.from_he_ply(
#     ply_path,
# )
c = MeshConverter.from_vf_ply(
    vf_ply_path,
)


# %timeit HalfEdgeMesh.load(ply_path=ply_path)
# %timeit MeshConverter.from_he_ply(ply_path)
#

he_samples = m.he_samples
vf_samples = (m.xyz_coord_V, m.V_of_F)
(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    h_right_B,
) = he_samples

che_samples = c.he_samples
(
    cxyz_coord_V,
    ch_out_V,
    cv_origin_H,
    ch_next_H,
    ch_twin_H,
    cf_left_H,
    ch_bound_F,
    ch_right_B,
) = che_samples
cvf_samples = c.vf_samples
(cV, cF) = cvf_samples
# cm = HalfEdgeMesh(*che_samples)

# %%
test_ply_path = "./data/half_edge_base/ply/test_vf.ply"
c.write_vf_ply(test_ply_path, False)
# cm = HalfEdgeMesh.from_vf_ply(ply_path=test_ply_path)
cc = MeshConverter.from_vf_ply(
    test_ply_path,
)
[np.linalg.norm((a - b).ravel()) for a, b in zip(c.he_samples, cc.he_samples)]
[np.linalg.norm((a - b).ravel()) for a, b in zip(c.vf_samples, cc.vf_samples)]
# %%
from src.python.mesh.viewer import MeshViewer

# mv = MeshViewer(m)
cmv = MeshViewer(cm, show_half_edges=True)
rgba_boundary_face = cmv.colors["magenta80"]
rgba_boundary_face = cmv.colors["magenta80"]
# rgba_boundary_half_edge
# rgba_boundary_face
cmv.update(rgba_boundary_face=rgba_boundary_face)
cmv.plot()

# %timeit vf_samples_to_he_samples(*vf_samples)
# %timeit vf_samples_to_he_samples_numba(*vf_samples)
# %%
