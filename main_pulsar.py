import numpy as np
import sympy as sp
from importlib import reload

from numdiff import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    exp_so3,
    log_so3,
    exp_quaternion,
    log_quaternion,
    jitcross,
    multiply_se3_quaternion,
    inverse_se3_quaternion,
    mul_quaternion,
    inv_quaternion,
    rotate_by_quaternion,
    se3_quaternion_to_matrix,
    se3_matrix_to_quaternion,
    quaternion_to_matrix2,
    log_se3_quaternion,
    exp_se3_quaternion,
    exp_se3,
    exp_se3_slow,
)

# from skimage.measure import marching_cubes
from branes.model import (
    make_implicit_surface_mesh,
    make_sample_mesh,
    load_sample_mesh,
    # brane,
    FramedBrane,
    # get_face_data,
    mayavi_mesh_plot,
    transpose_csr,
    get_face_data,
    get_area_weighted_vertex_normals,
    polyscope_mesh_plot,
    polyscope_multimesh_plot,
    default_color_dict,
    example_multimesh_plot,
)


from scipy.sparse import csr_matrix
from mayavi import mlab

import polyscope as ps
import dill

# with open("./scratch/test_brane.pickle", "wb") as _f:
#     dill.dump(b, _f, recurse=True)

from scipy.linalg import expm, logm


# %%
vertices, faces, normals = make_sample_mesh("dumbbell2")


b = FramedBrane(vertices, faces, normals)

vertices, edges, faces, frames = (
    b.position_vectors(),
    b.edges,
    b.faces,
    b.orthogonal_matrices(),
)


# example_multimesh_plot(b)

# %%
brane = b
vertices, edges, faces, frames = (
    brane.position_vectors(),
    brane.edges,
    brane.faces,
    brane.orthogonal_matrices(),
)
framed_vertices = brane.framed_vertices
Nedges = len(edges)
psi = np.zeros((Nedges, 6))
# def get_psi(brane)

for e in range(Nedges):
    v0, v1 = edges[e]
    pq0, pq1 = framed_vertices[v0], framed_vertices[v1]
    pq0_inv = inverse_se3_quaternion(pq0)
    pq = pq0  # multiply_se3_quaternion(pq0_inv, pq1)
    psi[e] = log_se3_quaternion(pq)


# %%

ex, ey, ez = np.eye(3)
i, j, k = np.eye(4)[1:]
theta_vec = 0.25 * np.pi * ez + np.random.rand(3)
x, y, z = np.random.rand(3)  # 1.0, 2.0, 3.0
r = x * ex + y * ey + z * ez
_r = x * i + y * j + z * k
Q = exp_so3(theta_vec)
q = exp_quaternion(theta_vec)
q_inv = inv_quaternion(q)


qr1 = mul_quaternion(q, mul_quaternion(_r, q_inv))

qr2 = mul_quaternion(mul_quaternion(q, _r), q_inv)

qr = rotate_by_quaternion(q, r)

Qr = Q @ r
qr - Qr
