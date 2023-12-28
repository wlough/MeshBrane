import numpy as np
import sympy as sp
import polyscope as ps
from numdiff import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    exp_so3,
    log_so3,
    exp_quaternion,
    log_quaternion,
    jitcross,
)

# from skimage.measure import marching_cubes
from branes.model import (
    make_implicit_surface_mesh,
    make_sample_mesh,
    brane,
    FramedBrane,
    get_face_data,
)

# from scipy.linalg import expm, logm


vertices, faces, normals = make_sample_mesh("torus")

b = FramedBrane(vertices, faces, normals)
for e in range(len(b.edges) - 1):
    ep = b.edges[e]
    em = np.flip(b.edges[e + 1])
    print()
# %%
vertices = b.framed_vertices
faces = b.faces
Nvertices = len(vertices)
Nfaces = len(faces)
edges_list = []

# Afe ###############
Afe_data_list = []  # [-1,1,...]
Afe_indices_list = []  #
# Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

# Aev ###############
# Aev_data_list = []  # [-1,1,...]
Aev_indices_list = []  # vertex indices
# Aev_indptr = []  # [0,2,4,...]

for f in range(Nfaces):
    face = faces[f]
    for _v in range(3):
        vm = face[_v]
        vp = face[np.mod(_v + 1, 3)]
        edge_p = [vm, vp]
        edge_m = [vp, vm]
        try:  # is negative edge already in edges?
            e_m = edges_list.index(edge_m)
        except Exception:  # if not, then add it
            edges_list.append(edge_m)
            e = len(edges_list) - 1
            Afe_indices_list.append(e)
            Afe_data_list.append(-1)
            Aev_indices_list.append(vp)
            Aev_indices_list.append(vm)
        try:  # is positive edge already in edges?
            e_p = edges_list.index(edge_p)
        except Exception:  # if neither, add positive edge to edges
            edges_list.append(edge_p)
            e = len(edges_list) - 1
            Afe_indices_list.append(e)
            Afe_data_list.append(1)
            Aev_indices_list.append(vm)
            Aev_indices_list.append(vp)


Afe_data = np.array(Afe_data_list, dtype=np.int32)
Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
edges = np.array(edges_list, dtype=np.int32)

# len(edges)
# %%
# b = brane(sample_surface_name="torus")
# # b.
# V, F = b.vertices, b.faces
# #
# ps.init()
#
# ps_mesh = ps.register_surface_mesh("my mesh", V, F)
# ps.show()
# N = 100
