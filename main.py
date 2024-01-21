import numpy as np
from src.model import Brane
from src.utils import load_mesh_from_ply

# import matplotlib.pyplot as plt
# from src.pretty_pictures import polyscope_plots as pp
from src.pretty_pictures import mayavi_plots as mp

# from src.numdiff import (index_of_nested,)


# %%
file_path = "./data/ply_files/oblate_ultracoarse.ply"
vertices, faces = load_mesh_from_ply(file_path)
b = Brane(vertices, faces)
bflip = Brane(vertices, faces)
# bflip.flip_bad_edges()
h = 207
# h = 60

bflip.edge_flip(h)
for bb in [b, bflip]:
    ht = bb.twin(h)
    bb.H_rgb[h] = np.array([1.0, 0.0, 0.0])
    bb.H_rgb[ht] = np.array([0.0, 0.0, 1.0])
    bb.V_rgb[bb.v_of_h(h)] = np.array([1.0, 0.0, 0.0])
    bb.V_rgb[bb.v_of_h(ht)] = np.array([0.0, 0.0, 1.0])
    f1 = bb.f_of_h(h)
    f2 = bb.f_of_h(ht)
    bb.F_rgb[f1] = np.array([1.0, 0.0, 0.0])
    bb.F_rgb[f2] = np.array([0.0, 0.0, 1.0])
# %%
for h in b.H_label:
    v = b.v_of_h(h)
    val = b.valence(v)
    print(f"h={h}, valence={val}")
# %%
mp.brane_plot(
    b,
    show_surface=True,
    show_halfedges=True,
    show_edges=False,
    show_vertices=True,
    show_normals=False,
    show_tangant1=False,
    show_tangent2=False,
    show_plot_axes=False,
)


# %%


import sympy as sp

from src.symdiff import dot, cross

x1, x2, x3 = sp.symbols("x1:4")
y1, y2, y3 = sp.symbols("y1:4")
z1, z2, z3 = sp.symbols("z1:4")

r1 = sp.Array([x1, y1, z1])
r2 = sp.Array([x2, y2, z2])
r3 = sp.Array([x3, y3, z3])

# Avec = cross(r2-r1,r3-r1).applyfunc(lambda _ : _.expand())
Avec = cross(r1, r2) + cross(r2, r3) + cross(r3, r1)

normAvec = sp.sqrt(dot(Avec, Avec))
gradA = sp.derive_by_array(normAvec, r1)
# gradA2 = Avec/
