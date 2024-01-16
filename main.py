import numpy as np
import sympy as sp
import polyscope as ps
from src.model import Brane
from src.utils import (
    make_implicit_surface_mesh,
    make_sample_mesh,
    load_mesh_from_ply,
    save_mesh_to_ply,
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

# %%
v_c = 13
V_pq, F = b.build_minimesh_from_psi(v_c)
V = V_pq[:, :3]

mini_b = Brane(V, F)
# %%
branes = [b]
b.V_scalar = np.zeros_like(b.V_label)
pq_point_clouds = []
ps_structures = polyscope_list_plot(pq_point_clouds=pq_point_clouds, branes=branes)

# %%
v_c = 13
V_pq, F = b.build_minimesh_from_psi(v_c)
V = V_pq[:, :3]

# %%
# self = b
vertices, faces = V, F
# def label_half_edges(self, vertices, faces):
# """
# faces must be ordered counter-clockwise!
# """

####################
# vertices
Nvertices = len(vertices)
V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
V_hedge = -np.ones_like(V_label)  # outgoing halfedge
####################
# faces
Nfaces = len(faces)
F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)
F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
####################
# halfedges
# Nedges =
# Nhedges = 3 * Nfaces
halfedges = []
H_label = []  # np.array([_ for _ in range(Nhedges)], dtype=np.int32)
interior_hedge_labels = []
H_vertex = []  # -np.ones_like(H_label)  # vertex it points to
H_face = []  # -np.ones_like(H_label)  # face it belongs to  ***
H_next = []  # -np.ones_like(
# H_label
# )  # next halfedge inside the face (ordered counter-clockwise)
H_prev = []  # -np.ones_like(
# H_label
# )  # previous halfedge inside the face (ordered counter-clockwise)
H_twin = []  # -np.ones_like(H_label)  # opposite halfedge
# H_isboundary = []
####################
# save and label halfedges
for f in F_label:
    face = faces[f]
    N_v_of_f = len(face)
    for _ in range(N_v_of_f):
        _next = np.mod(_ + 1, N_v_of_f)  # index shift to get next
        v0 = face[_]  #
        v1 = face[_next]
        hedge = [v0, v1]
        hedge_twin = [v1, v0]

        try:
            h = halfedges.index(hedge)
        except Exception:
            halfedges.append(hedge)
            h = halfedges.index(hedge)

        try:
            interior_hedge_labels.index(h)
        except Exception:
            interior_hedge_labels.append(h)

        try:
            h_twin = halfedges.index(hedge_twin)
        except Exception:
            halfedges.append(hedge_twin)

Nhedges = len(halfedges)
H_label = np.ones(Nhedges, dtype=np.int32)
H_vertex = np.ones(Nhedges, dtype=np.int32)
H_face = np.ones(Nhedges, dtype=np.int32)
H_next = np.ones(Nhedges, dtype=np.int32)
H_prev = np.ones(Nhedges, dtype=np.int32)
H_twin = np.ones(Nhedges, dtype=np.int32)


# fill halfedges that are in the faces
for f in F_label:
    face = faces[f]
    N_v_of_f = len(face)
    h0 = len(halfedges)  # halfedges.__len__()  # label of 1st hedge in face
    F_hedge[f] = h0  #

    for _ in range(N_v_of_f):
        _next = np.mod(_ + 1, N_v_of_f)  # index shift to get next
        _prev = np.mod(_ - 1, N_v_of_f)  # index shift to get prev
        v0 = face[_]  #
        v1 = face[_next]
        h = h0 + _
        h_prev = h0 + _prev
        h_next = h0 + _next
        hedge = [v0, v1]
        halfedges.append(hedge)
        H_label.append(h)
        H_vertex.append(v1)
        H_face.append(f)
        H_prev.append(h_prev)
        H_next.append(h_next)
        if V_hedge[v0] == -1:
            V_hedge[v0] = h


# %%
#############################################################
# save halfedges that are in the faces
for f in F_label:
    face = faces[f]
    N_v_of_f = len(face)
    h0 = len(halfedges)  # halfedges.__len__()  # label of 1st hedge in face
    F_hedge[f] = h0  #

    for _ in range(N_v_of_f):
        _next = np.mod(_ + 1, N_v_of_f)  # index shift to get next
        _prev = np.mod(_ - 1, N_v_of_f)  # index shift to get prev
        v0 = face[_]  #
        v1 = face[_next]
        h = h0 + _
        h_prev = h0 + _prev
        h_next = h0 + _next
        hedge = [v0, v1]
        halfedges.append(hedge)
        H_label.append(h)
        H_vertex.append(v1)
        H_face.append(f)
        H_prev.append(h_prev)
        H_next.append(h_next)
        if V_hedge[v0] == -1:
            V_hedge[v0] = h

# save halfedges that are on the boundaries
# and fill H_twin for interior halfedges
Ninterior_hedges = len(halfedges)  # halfedges.__len__()
H_isboundary = Ninterior_hedges * [False]
for h in range(Ninterior_hedges):
    hedge = halfedges[h]
    hedge_twin = [hedge[1], hedge[0]]
    try:
        h_twin = halfedges.index(hedge_twin)
        H_twin.append(h_twin)
    except Exception:
        halfedges.append(hedge_twin)
        h_twin = halfedges.index(hedge_twin)

        H_label.append(h_twin)
        H_vertex.append(hedge_twin[1])
        H_face.append(-1)  # -1=boundary face
        # H_prev.append(h_prev)
        # H_next.append(h_next)

        H_twin.append(h_twin)

N_hedges = len(halfedges)  # halfedges.__len__()

# get next/prev on boundary
for h in range(Ninterior_hedges, N_hedges):
    hedge = halfedges[h]
    hedge_twin = [hedge[1], hedge[0]]
    h_twin = halfedges.index(hedge_twin)
    H_twin.append(h_twin)
    H_isboundary.append(True)

    h_t = h_twin
    h_tp = H_prev[h_t]

    h_tpt = H_twin[h_tp]
    h_tptp = H_prev[h_tpt]

    h_tptpt = H_twin[h_tptp]
    h_tptptp = H_prev[h_tptpt]

    # h_next = H_twin[hh]
    hedge_tptptp = halfedges[h_tptptp]
    hedge_next = [hedge_tptptp[1], hedge_tptptp[0]]
    h_next = halfedges.index(hedge_next)

    H_next.append(h_next)

    hh = H_next[h_twin]

    hh = H_twin[hh]
    hh = H_next[hh]

    hh = H_twin[hh]
    hh = H_next[hh]

    # h_prev = H_twin[hh]
    hhedge = halfedges[hh]
    hedge_prev = [hhedge[1], hhedge[0]]
    h_prev = halfedges.index(hedge_prev)
    H_prev.append(h_prev)

halfedges = np.array(halfedges, dtype=np.int32)
# return (
#     V_label,
#     V_hedge,
#     np.array(halfedges, dtype=np.int32),
#     np.array(H_label, dtype=np.int32),
#     np.array(H_vertex, dtype=np.int32),
#     np.array(H_face, dtype=np.int32),
#     np.array(H_next, dtype=np.int32),
#     np.array(H_prev, dtype=np.int32),
#     np.array(H_twin, dtype=np.int32),
#     np.array(H_isboundary),
#     F_label,
#     F_hedge,
# )

len(halfedges)
