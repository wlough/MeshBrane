from src.python.ply_tools import VertTri2HalfEdgeConverter, SphereBuilder
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_ops import CotanLaplaceOperator
from src.python.rigid_body import exp_so3

import numpy as np

#
# ex = 3
#
# N_big = 3 * 2**ex
# N_small = 2**ex
# rad_small = 1 / 3
# rad_big = 1
#
# V = []
# for b in range(N_big + 1):
#     phi_big = 2 * np.pi * b / N_big
#     for s in range(N_small + 1):
#         phi_small = 2 * np.pi * s / N_small
#         x = np.cos(phi_big) * (rad_big + np.cos(phi_small) * rad_small)
#         y = np.sin(phi_big) * (rad_big + np.cos(phi_small) * rad_small)
#         z = np.sin(phi_small) * rad_small
#         V.append(np.array([x, y, z]))
#
# for ia, a in enumerate(V):
#     print(f"---------------")
#     count = 0
#     for ib, b in enumerate(V):
#         if np.linalg.norm(a - b) < 1e-5:
#             count += 1
#     print(f"{ia=}, {count=}")
# F = []
# for b in range(N_big):
#     for s in range(N_small):
#         i1 = b * (N_small + 1) + s
#         i2 = (b + 1) * (N_small + 1) + s
#         i3 = b * (N_small + 1) + (s + 1)
#         i4 = (b + 1) * (N_small + 1) + (s + 1)
#         F.append([i1, i3, i4])
#         F.append([i1, i4, i2])
#
#
# m = HalfEdgeMesh.from_vert_face_list(V, F)
#
# mv = MeshViewer(*m.data_lists, show_vertices=True)
#
# mv.plot()
#


def VF_torus(p):

    N_big = 3 * 2**p
    N_small = 2**p
    rad_small = 1 / 3
    rad_big = 1

    V = []
    F = []
    for b in range(N_big):
        phi_big = 2 * np.pi * b / N_big
        bp1 = (b + 1) % N_big
        for s in range(N_small):
            sp1 = (s + 1) % N_small
            phi_small = 2 * np.pi * s / N_small
            x = np.cos(phi_big) * (rad_big + np.cos(phi_small) * rad_small)
            y = np.sin(phi_big) * (rad_big + np.cos(phi_small) * rad_small)
            z = np.sin(phi_small) * rad_small
            V.append(np.array([x, y, z]))
            b_s = b * N_small + s
            b_sp1 = b * N_small + sp1
            bp1_s = bp1 * N_small + s
            bp1_sp1 = bp1 * N_small + sp1
            F.append([b_s, bp1_sp1, bp1_s])
            F.append([b_s, b_sp1, bp1_sp1])
    return V, F


VF = [VF_torus(p) for p in [2, 3, 4, 5]]
V, F = VF_torus(5)
[len(V) for V, F in VF]
m = HalfEdgeMesh.from_vert_face_list(V, F)

mv = MeshViewer(*m.data_lists, show_vertices=True)

mv.plot()
