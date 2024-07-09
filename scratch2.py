from src.python.ply_tools import SphereBuilder
from src.python.half_edge_mesh import HalfEdgeMesh

b = SphereBuilder()
b.refine(convert_to_half_edge=False)
# %%
num_faces = [len(F) for F in b.F]
num_faces = [20, 80, 320, 1280, 5120, 20480, 81920, 327680, 1310720, 5242880]
num_vertices = [b.num_vertices(_) for _ in range(len(b.F))]
num_vertices = [12, 42, 162, 642, 2562, 10242, 40962, 163842, 655362, 2621442]
num_edges = [Nf + Nv - 2 for Nf, Nv in zip(num_faces, num_vertices)]
num_edges = [30, 120, 480, 1920, 7680, 30720, 122880, 491520, 1966080, 7864320]
SphereBuilder.build_test_plys(num_refine=5)
# %%

from python.sym_tools import SymTorus, SymSphere
from src.python.half_edge_mesh import HalfEdgeMesh
import sympy as sp
import numpy as np

# b = DoughnutFactory()
# b.refine(convert_to_half_edge=False)
torus = SymTorus()
sphere = SymSphere()
t = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/torus_003072_he.ply")
s = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/unit_sphere_02562.ply")
iso_torus = np.array([torus.implicit_fun(*xyz) for xyz in t.xyz_array])
iso_sphere = np.array([sphere.implicit_fun(*xyz) for xyz in s.xyz_array])


theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 200)
th, ph = theta[3], phi[15]
xyz = sphere.parametric_fun(th, ph)
sphere.xyz_thetaphi
sphere.gaussian_curvature_fun(th, ph)
# %%
H = sp.Array(
    [
        [
            [x_i.diff(phi_j).diff(phi_k) for phi_k in sphere.thetaphi]
            for phi_j in sphere.thetaphi
        ]
        for x_i in sphere.xyz_thetaphi
    ]
)
H - sphere.hessian
