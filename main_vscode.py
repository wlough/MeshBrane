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
    FramedVertex,
    get_face_data,
    frame_the_mesh,
)

# from scipy.linalg import expm, logm


# %%
vertices, faces, normals = make_sample_mesh("torus")
matrices = frame_the_mesh(vertices, faces, normals)

Nverts = len(vertices)
framed_vertices_data = np.zeros((Nverts, 7))
framed_vertices_data[:, :3] = vertices
framed_vertices_data[:, 3:] = np.array([matrix_to_quaternion(R) for R in matrices])
framed_vertices = [FramedVertex(*f) for f in framed_vertices_data]
# for v in framed_vertices:
#     print(v.orthogonal_matrix())
b = FramedBrane(framed_vertices, faces)
# %%


FramedVertex_init_dict = {
    "x": 0.0,
    "y": 0.0,
    "z": 1.0,
    "qw": 0.5,
    "qx": 0.0,
    "qy": 0.0,
    "qz": np.sqrt(0.75),
}
v = FramedVertex(**FramedVertex_init_dict)
v.quaternion_to_matrix()

# %%
b = brane(sample_surface_name="torus")
# b.
V, F = b.vertices, b.faces
#
ps.init()

ps_mesh = ps.register_surface_mesh("my mesh", V, F)
ps.show()
N = 100
