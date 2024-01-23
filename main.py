import numpy as np
from src.model import Brane
from src.utils import load_mesh_from_ply

# import matplotlib.pyplot as plt
# from src.pretty_pictures import polyscope_plots as pp
from src.pretty_pictures import mayavi_plots as mp

from src.numdiff import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    jitdot,
    jitnorm,
    jitcross,
)

# out_path = "./output"
# hash("heeey")

# %%
file_path = "./data/ply_files/oblate_ultracoarse.ply"
vertices, faces = load_mesh_from_ply(file_path)
Ke = 1e-2
Ka = 1e-3
Kc = 1e-4
Kb = 1e0
Ks = 1e-1
zeta = 1e0
dt = 1e-2
params = np.array([Ke, Ka, Kc, Kb, Ks, zeta, dt])
b = Brane(vertices, faces, params)

# %%

# %%
mp.brane_plot(
    b,
    show_surface=True,
    show_halfedges=True,
    show_edges=False,
    show_vertices=True,
    show_normals=True,
    show_tangent1=True,
    show_tangent2=True,
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
