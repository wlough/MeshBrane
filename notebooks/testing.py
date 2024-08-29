from src.python.half_edge_base_mesh import Brane
from src.python.half_edge_base_viewer import MeshViewer
import numpy as np

vf_ply = "./data/ply/binary/torus_003072_vf.ply"
he_ply = "./data/ply/binary/torus_003072_he.ply"
he_ply = "./data/half_edge_base/ply/unit_sphere_002562_he.ply"
he_ply = "./data/half_edge_base/ply/oblate_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_003072_he.ply"

# he_ply = "./data/half_edge_base/ply/dumbbell_he.ply"
# he_ply = "./data/half_edge_base/ply/neovius_he.ply"
# R = 32
# kBT = 0.2
# tau = 1.28e5
# bending_modulus = 20 * kBT
# A = 4 * np.pi * R**2
#
# spontaneous_volume = 4 * np.pi * R**3 / 3
# length_reg_stiffness = 80 * kBT
# area_reg_stiffness = 6.43e6 * kBT / A
# volume_reg_stiffness = 1.6e7 * kBT / R**3
# linear_drag_coeff = 0.4 * kBT * tau / R**2
R = 1.0
kBT = 0.2 / 32
tau = 1.28e5
bending_modulus = 20 * kBT
A = 4 * np.pi * R**2

spontaneous_volume = 4 * np.pi * R**3 / 3
length_reg_stiffness = 80 * kBT
area_reg_stiffness = 6.43e6 * kBT / A
volume_reg_stiffness = 1.6e7 * kBT / R**3
linear_drag_coeff = 0.4 * kBT * tau / R**2
brane_kwargs = {
    "length_reg_stiffness": length_reg_stiffness,
    "area_reg_stiffness": area_reg_stiffness,
    "volume_reg_stiffness": volume_reg_stiffness,
    "bending_modulus": bending_modulus,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": linear_drag_coeff,
    # "spontaneous_face_area"=spontaneous_face_area,
    # "spontaneous_volume": spontaneous_volume,
}
# brane_kwargs = {
#     "length_reg_stiffness": 1e-9,
#     "area_reg_stiffness": 1e-2,
#     "volume_reg_stiffness": 1e2,
#     "bending_modulus": 1e1,
#     "splay_modulus": 1.0,
#     "spontaneous_curvature": 0.0,
#     "linear_drag_coeff": 1e3,
# }


b = Brane.from_half_edge_ply(he_ply, **brane_kwargs)
b._xyz_coord_V[:, 2] = b.xyz_coord_V[:, 2] * 0.8

Fb = b.Fbend_analytic()
Fa = b.Farea_harmonic()
Fv = b.Fvolume_harmonic()
Ft = b.Ftether()
F = Fb + Fa + Fv + Ft
dt = 1e-2
Dxyz_coord_V = dt * F / b.linear_drag_coeff
# %%
#
# b.euler_step(1e-2)
mv = MeshViewer(
    b,
    show_wireframe_surface=True,
    show_face_colored_surface=False,
    show_vertex_colored_surface=False,
    show_vertices=False,
    show_half_edges=False,
    show_plot_axes=True,
    figsize=(480, 480),
)
# mv.add_vector_field(b.xyz_coord_V, 1.1 * valvec, rgba=(1, 0, 0, 1), name="valence")
# mv.add_vector_field(b.xyz_coord_V, 50*(n_lap-n_other), rgba=(1, 0, 0, 1), name="n-n")
# mv.add_vector_field(b.xyz_coord_V, Fb, rgba=None, name="Fb", mask_points=1)
# mv.add_vector_field(b.xyz_coord_V, Fv, rgba=None, name="Fv", mask_points=1)
# mv.add_vector_field(b.xyz_coord_V, Fa, rgba=None, name="Fa", mask_points=1)
# mv.add_vector_field(b.xyz_coord_V, Fl_OG, rgba=None, name="Fl", mask_points=1)
dt = 1e-2
for iter in range(100):
    b.euler_step(1e-2)
    # Fb = b.Fbend_analytic()
    # Fa = b.Farea_harmonic()
    # Fv = b.Fvolume_harmonic()
    # Ft = b.Ftether()
    # F = Fb + Fa + Fv + Ft
    # F = Fb + Fa + Fv + Ft
    # Dxyz_coord_V = dt * F / b.linear_drag_coeff
    # mv.add_vector_field(b.xyz_coord_V, Dxyz_coord_V, rgba=None, name="Fl", mask_points=1)
    # mv2.vector_field_data
    mv.plot(save=True, show=False)
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
