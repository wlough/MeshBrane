from src.python.half_edge_base_mesh import Brane
from src.python.half_edge_base_viewer import MeshViewer
import numpy as np

vf_ply = "./data/ply/binary/torus_003072_vf.ply"
he_ply = "./data/ply/binary/torus_003072_he.ply"
he_ply = "./data/half_edge_base/ply/unit_sphere_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_003072_he.ply"

# he_ply = "./data/half_edge_base/ply/dumbbell_he.ply"
# he_ply = "./data/half_edge_base/ply/neovius_he.ply"
brane_kwargs = {
    "length_reg_stiffness": 1e-9,
    "area_reg_stiffness": 1e-3,
    "volume_reg_stiffness": 1e1,
    "bending_modulus": 1e1,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": 1e3,
}
b = Brane.from_half_edge_ply(he_ply, **brane_kwargs)
# b.xyz_coord_V+=.001*(np.random.rand(*b.xyz_coord_V.shape)-.5)
b._xyz_coord_V[:, 0] = b._xyz_coord_V[:, 0] * 0.9
Fb = b.compute_bending_force()
# Fb /= np.max(np.linalg.norm(Fb, axis=-1))
n = b.normal_other_weighted_V()
# %%
# b.flip_non_delaunay()
mv = MeshViewer(b, show_half_edges=True, show_wireframe_surface=False)
# mv2 = MeshViewer(
#     b,
#     # show_wireframe_surface=False,
#     # show_face_colored_surface=True,
#     # show_vertex_colored_surface=False,
#     # show_vertices=True,
#     # show_half_edges=True,
#     # target_faces=5000,
# )
mv.add_vector_field(b.xyz_coord_V, Fb, rgba=None, name="Fb")
# mv.add_vector_field(b.xyz_coord_V, .1*n, rgba=None, name="Fb")
# mv2.vector_field_data
mv.plot()
# %%
V, F = b.xyz_coord_V, b.V_of_F


# def fix_VF(V, F, target_faces=1000, boundary_vertex_deletion=True):
# """
# Decimate a mesh represented by vertices and faces.
# """
import numpy as np
import pyvista as pv

num_faces = F.shape[0]
num_vertices = V.shape[0]

F_pv = np.zeros((num_faces, 4), dtype="int32")
F_pv[:, 0] = 3
F_pv[:, 1:] = F
F_pv = F_pv.ravel()
# Create a PyVista mesh
M = pv.PolyData(V, F_pv)
Mclean = M.clean(
    point_merging=True,
    tolerance=1e-6,
    lines_to_points=True,
    polys_to_lines=True,
    strips_to_polys=True,
    absolute=False,
)
M.plot(show_edges=True)
Mclean.plot(show_edges=True)
Mclean = Mclean.smooth(n_iter=20, relaxation_factor=0.01)
# Extract simplified vertices and faces
Vsimp = np.array(Mclean.points)
Fsimp = Mclean.faces.reshape(-1, 4)[:, 1:]
bc = Brane.from_vf_data(Vsimp, Fsimp)
# %%

m1._gaussian_curvature_v(0)
m1.gaussian_curvature_v(0)
# %%
# m = HalfEdgeMesh.from_half_edge_ply(he_ply)
m = m1
H, K, lapH, n = m.compute_curvature_data()
H0 = 0
Kb = 0.1
Fbend = -2 * Kb * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))
Av = m.barcell_area_V()
FbendA = Fbend * Av
# %%
# mv = MeshViewer(*m.data_arrays)
# mv.plot()
scale_vec = 10.5
Fn = FbendA
# vec = scale_vec * np.einsum("i,ij->ij", Fn, n)
vec = 0.1 * n
# vec=scale_vec*n
vfdat = [m.xyz_coord_V, vec]
mv_kwargs = {
    "vector_field_data": [vfdat],
    # "V_rgba": V_rgba,
    # "color_by_V_rgba": True,
    # "E_rgba": E_rgba,
}
mv = MeshViewer(*m.data_arrays, **mv_kwargs)
mv.plot()
# %%
# from src.python.half_edge_mesh import HalfEdgeMesh
