import numpy as np
import sympy as sp
import polyscope as ps
from src.model import Brane, testBrane
from src.utils import (
    make_implicit_surface_mesh,
    make_sample_mesh,
    load_mesh_from_ply,
    save_mesh_to_ply,
    make_trisurface_patch,
    make_quadsurface_patch,
)
import matplotlib.pyplot as plt
from src.pretty_pictures import (
    polyscope_list_plot,
)
from src.numdiff import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    exp_so3,
    log_so3,
    exp_quaternion,
    log_quaternion,
    jitcross,
    mul_se3_quaternion,
    inv_se3_quaternion,
    #     mul_quaternion,
    #     inv_quaternion,
    rotate_by_quaternion,
    se3_quaternion_to_matrix,
    se3_matrix_to_quaternion,
    #     quaternion_to_matrix2,
    log_se3_quaternion,
    exp_se3_quaternion,
    exp_se3,
    #     exp_se3_slow,
    log_so3_quaternion,
    safe_log_so3_quaternion,
    log_unit_quaternion,
    index_of_nested,
)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay
from copy import deepcopy
from mayavi import mlab

from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, inv

# %%
# file_path = "./data/ply_files/mem3dg_oblate.ply"
# vertices, faces = load_mesh_from_ply(file_path)
vertices, faces, normals = make_sample_mesh("torus")
b = Brane(vertices, faces)
pq = b.regularize_mesh_acm_quat()
b2 = Brane(vertices, faces)
b2.V_pq = pq
polyscope_list_plot(branes=[b, b2])
b.V_scalar
# %%
v_c = 13
V_pq, F = b.build_minimesh_from_psi(v_c)
V = V_pq[:, :3]

mini_b = Brane(V, F)
# %%
branes = [b]
b.V_scalar = np.zeros_like(b.V_pq[:, 0]) + 0.1
pq_point_clouds = []
polyscope_list_plot(pq_point_clouds=pq_point_clouds, branes=branes)

# %%

vertices, faces = make_trisurface_patch()

# %%
# vertices, faces = make_quadsurface_patch(Nfaces=7)
b = Brane(vertices, faces)
b.frame_the_mesh(vertices)
b = testBrane(vertices, faces)

bdry = np.array([vertices[b.H_vertex[h]] for h in b.H_label if b.H_isboundary[h]])
bad_verts = np.array([vertices[_] for _ in [1, 5]])
surf = {"vertices": b.vertices, "faces": b.faces}
point_clouds = [{"points": bad_verts}]
point_clouds = [
    {"points": np.array([_]), "name": f"{i}"} for i, _ in enumerate(vertices)
]
point_clouds = [{"points": np.array([b.vertices[_] for _ in face])} for face in b.faces]
polyscope_list_plot(surfaces=[surf], point_clouds=point_clouds)


for h in b.H_label:
    hn = b.H_next[h]
    hnn = b.H_next[hn]
    hnnn = b.H_next[hnn]
    hnp = b.H_prev[hn]

    f1 = b.H_face[h]
    h2 = b.F_hedge[f1]
    f2 = b.H_face[h2]
    if b.H_isboundary[h]:
        # print(f"diff={h -hnp}, h={h}")
        print(b.halfedges[h])
    # print(f"bdry={not b.H_isboundary[h]}, f={f1}")

    # print(f"h2_is_bdry={b.H_isboundary[h2]}")
    # print(f1 - f2)

b.V_label
b.halfedges[17]


# %%
# def get_combinatorial_mesh_data(self):
# """og is the one with _"""
self = b
V_label = self.V_label
H_label = self.H_label
F_label = self.F_label
halfedges = self.halfedges

H_isboundary = self.H_isboundary
faces = self.faces.copy()
####################
# vertices
V_hedge = -np.ones_like(V_label)  # outgoing halfedge
####################
# faces
F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
####################
# halfedges
H_vertex = -np.ones_like(H_label)  # vertex it points to
H_face = -np.ones_like(H_label)  # face it belongs to
H_next = -np.ones_like(
    H_label
)  # next halfedge inside the face (ordered counter-clockwise)
H_prev = -np.ones_like(
    H_label
)  # previous halfedge inside the face (ordered counter-clockwise)
H_twin = -np.ones_like(H_label)  # opposite halfedge
####################

# assign each face a halfedge
# assign each interior halfedge previous/next halfedge
# assign each interior halfedge a face
for f in F_label:
    face = faces[f]
    N_v_of_f = len(face)
    hedge0 = np.array([face[0], face[1]])
    # h0 = halfedges_list.index(hedge0)
    h0 = index_of_nested(halfedges, hedge0)
    F_hedge[f] = h0  # assign each face a halfedge
    for _ in range(N_v_of_f):
        # for each vertex in face, get the indices of the
        # previous/next vertex
        _p1 = (_ + 1) % N_v_of_f
        _m1 = (_ - 1) % N_v_of_f
        vm1 = face[_m1]
        v0 = face[_]
        vp1 = face[_p1]
        # get outgoing halfedge
        hedge = np.array([v0, vp1])
        # h = halfedges_list.index(hedge)
        h = index_of_nested(halfedges, hedge)
        # get incident halfedge
        hedge_prev = np.array([vm1, v0])
        # h_prev = halfedges_list.index(hedge_prev)
        h_prev = index_of_nested(halfedges, hedge_prev)
        # assign previous/next halfedge
        H_prev[h] = h_prev
        H_next[h_prev] = h
        # assign face to halfedge
        H_face[h] = f
        print(h)

        hedge_twin = np.array([vp1, v0])
        # h = halfedges_list.index(hedge)
        h_t = index_of_nested(halfedges, hedge_twin)
        H_twin[h] = h_t
        H_twin[h_t] = h

# self.prev(18)
# self.twin(18)
# self.H_isboundary[18]
# self.halfedges[18]
#
# self.prev(19)
# self.twin(19)
# self.H_isboundary[19]
# self.halfedges[19]
# assign each halfedge a twin halfedge
# h_next = self.twin(self.prev(h_next))
# h_next
# index_of_nested(halfedges, np.array([4, 5], dtype=np.int32))
# assign each halfedge a vertex
# assign each vertex a halfedge
# assign each boundary halfedge previous/next halfedge
print("*******************")
for h in H_label:
    # if h == 19:
    #     break
    v0, v1 = halfedges[h]
    # hedge_twin = np.array([v1, v0])
    # h_twin = index_of_nested(halfedges, hedge_twin)
    # H_twin[h] = h_twin
    H_vertex[h] = v1
    if V_hedge[v0] == -1:
        V_hedge[v0] = h

    if H_isboundary[h]:
        print(f"-------------------")
        print(h)
        # print(f"h={h}")
        # h_t = h_twin
        # print(f"h_t={h_t}")
        # h_tp = H_prev[h_t]
        # print(f"h_tp={h_tp}")
        # h_tpt = H_twin[h_tp]
        # print(f"h_tpt={h_tpt}")
        # h_tptp = H_prev[h_tpt]
        # print(f"h_tptp={h_tptp}")
        # h_tptpt = H_twin[h_tptp]
        # print(f"h_tptpt={h_tptpt}")
        # h_tptptp = H_prev[h_tptpt]
        # print(f"h_tptptp={h_tptptp}")
        # h_next = H_twin[h_tptptp]
        # print(f"h_next={h_next}")
        # H_next[h] = h_next
        # H_prev[h_next] = h
        h_next = self.twin(h)
        while True:
            # h_p = self.prev(h_t)
            # h_t = self.twin(h_p)
            h_next = H_prev[h_next]  # self.prev(h_next)
            h_next = H_twin[h_next]  # self.twin(h_next)
            if H_isboundary[h_next]:
                break
        # H_next[h] = h_next
        # H_prev[h_next] = h
        # print(f"h={h}")
        # h_t = self.twin(h)
        # print(f"h_t={h_t}, bdry={H_isboundary[h_t]}")
        # h_pt = self.prev(h_t)
        # print(f"h_pt={h_pt}, bdry={H_isboundary[h_pt]}")
        # h_tpt = self.twin(h_pt)
        # print(f"h_tpt={h_tpt}, bdry={H_isboundary[h_tpt]}")
        # h_ptpt = self.prev(h_tpt)
        # print(f"h_ptpt={h_ptpt}, bdry={H_isboundary[h_ptpt]}")
        # h_tptpt = self.twin(h_ptpt)
        # print(f"h_tptpt={h_tptpt}, bdry={H_isboundary[h_tptpt]}")
        # h_ptptpt = self.prev(h_tptpt)
        # print(f"h_ptptpt={h_ptptpt}, bdry={H_isboundary[h_ptptpt]}")
        # h_tptptpt = self.twin(h_ptptpt)
        # h_next = h_tptptpt
        print(f"h_next={h_next}, bdry={H_isboundary[h_next]}")
        H_next[h] = h_next
        H_prev[h_next] = h
        print(f"-------------------")

# return (
#     V_hedge,
#     H_vertex,
#     H_face,
#     H_next,
#     H_prev,
#     H_twin,
#     F_hedge,
# )
