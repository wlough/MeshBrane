import numpy as np
from src.model import Brane
from src.utils import load_mesh_from_ply

# import matplotlib.pyplot as plt
# from src.pretty_pictures import polyscope_plots as pp
from src.pretty_pictures import mayavi_plots as mp

# from src.numdiff import (index_of_nested,)


# %%
file_path = "./data/ply_files/dumbbell_ultracoarse.ply"
vertices, faces = load_mesh_from_ply(file_path)
b = Brane(vertices, faces)
bflip = Brane(vertices, faces)
# bflip.flip_bad_edges()
# bflip.get_bad_hedges()
# # %%
# h = 207
# h = 60
# h = int(len(b.H_label) * np.random.rand(1)[0])
# bflip.edge_flip(h)
# bflip.edge_flip(h)
# for bb in [b, bflip]:
#     ht = bb.twin(h)
#     bb.H_rgb[h] = np.array([1.0, 0.0, 0.0])
#     bb.H_rgb[ht] = np.array([0.0, 0.0, 1.0])
#     bb.V_rgb[bb.v_of_h(h)] = np.array([1.0, 0.0, 0.0])
#     bb.V_rgb[bb.v_of_h(ht)] = np.array([0.0, 0.0, 1.0])
#     f1 = bb.f_of_h(h)
#     f2 = bb.f_of_h(ht)
#     bb.F_rgb[f1] = np.array([1.0, 0.0, 0.0])
#     bb.F_rgb[f2] = np.array([0.0, 0.0, 1.0])
#
# bflip.V_rgb[0] = np.array([0.0, 1.0, 0.0])
# bflip.H_rgb[bflip.h_of_v(0)] = np.array([0.0, 1.0, 0.0])
# mp.brane_plot(
#     bflip,
#     show_surface=True,
#     show_halfedges=True,
#     show_edges=False,
#     show_vertices=True,
#     show_normals=False,
#     show_tangant1=False,
#     show_tangent2=False,
#     show_plot_axes=False,
# )


# %%

f = []
bb = bflip
# for _ in range(1):
Lpref = bb.avereage_hedge_length()
for v in bb.V_label:
    # val = bb.valence(v)
    # print(f"v={v}, val={val}")
    # F = bb.F_reg_length(v, Lpref)
    F = F_reg_length(bb, v, Lpref)
    normF = np.linalg.norm(F)
    f.append(normF)
    # bb.V_pq[v, :3] += F


mp.brane_plot(
    bflip,
    show_surface=True,
    show_halfedges=True,
    show_edges=False,
    show_vertices=True,
    show_normals=False,
    show_tangant1=False,
    show_tangent2=False,
    show_plot_axes=False,
)
# f.index(max(f))


def F_reg_length(self, v, Lpref):
    Ke = 1e-2
    xyz = self.vertex_position(v)
    neighbors = self.v_adjacent_to_v(v)
    N = len(neighbors)
    F = np.zeros(3)

    for _v0 in range(0, N):
        v0 = neighbors[_v0]
        xyz0 = self.vertex_position(v0)
        r = xyz - xyz0
        L = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
        print(v0 - v)
        gradL = r / L
        F += -Ke * (L - Lpref) * gradL / Lpref

    return F
