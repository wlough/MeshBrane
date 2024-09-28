# os.path.exists
# os.path.join
# os.path.relpath
# os.getcwd
# os.chdir
# os.path.basename

from python.half_edge_mesh import HalfEdgeMeshBase
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
import numpy as np

vf_ply = "./data/ply/binary/torus_003072_vf.ply"
he_ply = "./data/ply/binary/torus_003072_he.ply"
he_ply = "./data/half_edge_base/ply/unit_sphere_002562_he.ply"
he_ply = "./data/half_edge_base/ply/oblate_002562_he.ply"
# he_ply = "./data/half_edge_base/ply/oblate_003072_he.ply"

he_ply = "./data/half_edge_base/ply/dumbbell_fine_he.ply"
he_ply = "./data/half_edge_base/ply/neovius_he.ply"
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
A = 4 * 3.14159 * R**2

spontaneous_volume = 4 * 3.14159 * R**3 / 3
length_reg_stiffness = 80 * kBT
area_reg_stiffness = 6.43e6 * kBT / A
volume_reg_stiffness = 1.6e7 * kBT / R**3
linear_drag_coeff = 0.4 * kBT * tau / R**2

spontaneous_face_area = 1.0
brane_kwargs = {
    "length_reg_stiffness": length_reg_stiffness,
    "area_reg_stiffness": area_reg_stiffness,
    "volume_reg_stiffness": volume_reg_stiffness,
    "bending_modulus": bending_modulus,
    "splay_modulus": 1.0,
    "spontaneous_curvature": 0.0,
    "linear_drag_coeff": linear_drag_coeff,
    "spontaneous_face_area": spontaneous_face_area,
    "spontaneous_volume": spontaneous_volume,
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
def regularize_by_shifts(m):
    Nv = m.num_vertices
    V = np.zeros_like(m.xyz_coord_V)
    for i in range(Nv):
        if m.boundary_contains_v(i):
            continue

        valence = 0
        for h in m.generate_H_out_v_clockwise(i):
            valence += 1
            V[i] += m.xyz_coord_v(m.v_head_h(h))
        V[i] /= valence
    m.xyz_coord_V = V


m0 = HalfEdgeMeshBase.from_he_ply(he_ply)
m = HalfEdgeMeshBase.from_he_ply(he_ply)
# b = Brane.from_he_ply(he_ply, **brane_kwargs)
mv = MeshViewer(m)
mv.plot()
# b._xyz_coord_V[:, 2] = b.xyz_coord_V[:, 2] * 0.8

# Fb = b.Fbend_analytic()
# Fa = b.Farea_harmonic()
# Fv = b.Fvolume_harmonic()
# Ft = b.Ftether()
# F = Fb + Fa + Fv + Ft
# dt = 1e-2
# Dxyz_coord_V = dt * F / b.linear_drag_coeff
# %%
V, F = m.xyz_coord_V, m.V_of_F
m1 = HalfEdgeMeshBase.from_vf_data(V, F)
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
from python.half_edge_mesh import HalfEdgeMeshBase
from src.python.half_edge_base_ply_tools import (
    VertTri2HalfEdgeMeshConverter,
    MeshConverterBase,
)

# he_ply0 = "./data/half_edge_base/ply/dumbbell_he.ply"
# he_ply = "./data/half_edge_base/ply/dumbbell_he_test.ply"
# vf_ply = "./data/half_edge_base/ply/dumbbell_vf.ply"
# he_ply0 = "./data/half_edge_base/ply/annulus_he.ply"
# he_ply = "./data/half_edge_base/ply/annulus_he_test.ply"
# vf_ply = "./data/half_edge_base/ply/annulus_vf.ply"
# m = HalfEdgeMeshBase.from_half_edge_ply(he_ply0)
# V, F = m.xyz_coord_V, m.V_of_F
#
#
# c = MeshConverterBase.from_no_boundary_he_ply(he_ply0, compute_vf_stuff=True)
# c.write_he_ply(he_ply)
# # dir(c.he_ply_data)
#
# c.he_ply_data.elements
MeshConverterBase.update_no_boundary_he_plys()
