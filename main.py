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
vertices, faces = make_quadsurface_patch(Nfaces=7)
b = testBrane(vertices, faces)
bdry = np.array([vertices[b.H_vertex[h]] for h in b.H_label if b.H_isboundary[h]])
b.H_face
surf = {"vertices": vertices, "faces": faces}
cloud = {"points": bdry}
polyscope_list_plot(surfaces=[surf], point_clouds=[cloud])


def frame_the_mesh(self, vertices):
    F_label = self.F_label
    Nfaces = len(F_label)
    F_area_vectors = np.zeros((Nfaces, 3))
    # vertices = self.V_pq[:, :3]

    for _f in range(Nfaces):
        f = F_label[_f]
        h = self.F_hedge[f]
        hn = self.H_next[h]
        hp = self.H_prev[h]

        v0 = self.H_vertex[hp]
        v1 = self.H_vertex[h]
        v2 = self.H_vertex[hn]

        u1 = vertices[v1] - vertices[v0]
        u2 = vertices[v2] - vertices[v1]

        F_area_vectors[_f] = jitcross(u1, u2)
    # F_area_vectors = self.get_face_area_vectors()

    # vertices = self.V_pq[:, :3]

    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    # ez = np.array([0.0, 0.0, 1.0])
    Nverts = len(vertices)
    # framed_vertices = np.zeros((Nverts, 7))
    # matrices = np.zeros((Nverts, 3, 3))
    framed_vertices = np.zeros((Nverts, 7))
    for i in range(Nverts):
        F = self.f_adjacent_to_v(i)
        n = np.zeros(3)
        for f in F:
            n += F_area_vectors[f]
        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)

        cross_with_ey = np.sqrt(n[2] ** 2 + n[0] ** 2) > 1e-6
        if cross_with_ey:
            e1 = jitcross(ey, n)
        else:
            e1 = jitcross(ex, n)
        e1 /= np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
        e2 = jitcross(n, e1)

        R = np.zeros((3, 3))
        R[:, 0] = e1
        R[:, 1] = e2
        R[:, 2] = n
        framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
        framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
        framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])

        framed_vertices[i, :3] = vertices[i, :]
    return framed_vertices


# %%


matrix = A.copy()
is_single = False
matrix = np.asarray(matrix, dtype=float)

if matrix.ndim not in [2, 3] or matrix.shape[-2:] != (3, 3):
    raise ValueError(
        "Expected `matrix` to have shape (3, 3) or "
        "(N, 3, 3), got {}".format(matrix.shape)
    )

if matrix.ndim == 2:
    matrix = matrix[None, :, :]
    is_single = True

num_rotations = 1
decision_matrix = np.empty((num_rotations, 4))
decision_matrix[:, :3] = matrix.diagonal(axis1=1, axis2=2)
decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
choices = decision_matrix.argmax(axis=1)

quat = np.empty((num_rotations, 4))

ind = np.nonzero(choices != 3)[0]
i = choices[ind]
j = (i + 1) % 3
k = (j + 1) % 3

quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

ind = np.nonzero(choices == 3)[0]
quat[ind] = [1, 0, 0, 0]

quat /= np.linalg.norm(quat, axis=1)[:, None]

if is_single:
    return cls(quat[0], normalized=True, copy=False)
else:
    return cls(quat, normalized=True, copy=False)
