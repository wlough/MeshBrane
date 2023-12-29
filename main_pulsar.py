import numpy as np
import sympy as sp

# from numdiff import (
#     # matrix_to_quaternion,
#     # quaternion_to_matrix,
#     # exp_so3,
#     # log_so3,
#     # exp_quaternion,
#     # log_quaternion,
#     # jitcross,
# )

# from skimage.measure import marching_cubes
from branes.model import (
    make_implicit_surface_mesh,
    make_sample_mesh,
    # brane,
    FramedBrane,
    # get_face_data,
    mayavi_mesh_plot,
    transpose_csr,
)

from scipy.sparse import csr_matrix
from mayavi import mlab

# import polyscope as ps

# from scipy.linalg import expm, logm


vertices, faces, normals = make_sample_mesh("torus")
# vertices, faces, normals = make_implicit_surface_mesh(
#     implicit_fun_str, xyz_minmax, Nxyz
# )

b = FramedBrane(vertices, faces, normals)

vertices, edges, faces, frames = (
    b.position_vectors(),
    b.edges,
    b.faces,
    b.orthogonal_matrices(),
)


# %%
vertex = 13

v_of_e_of_v = b.vertices_adjacent_to_vertex(vertex)
_normals = 0.1 * np.array([normals[_] for _ in v_of_e_of_v])
_vertices = np.array([vertices[_] for _ in v_of_e_of_v])

V, F = b.mini_mesh(vertex)

fig_kwargs = {
    "vertices": V,
    "edges": edges,
    "faces": F,
    "frames": None,  # frames,
    "show": True,
    "save": False,
    "fig_path": None,
    "plot_vertices": True,
    "plot_edges": True,
    "plot_faces": True,
    # "vector_field_data": {
    #     "vectors": _normals,
    #     "positions": _vertices,
    #     "color": (0.7057, 0.0156, 0.1502),
    # },
}


#
mayavi_mesh_plot(**fig_kwargs)
# data_T, indices_T, indptr_T = transpose_csr(b.Afe_data,b.Afe_indices,b.Afe_indptr)
# %%
vertices = b.position_vectors()
edges = b.edges
faces = b.faces
Nvertices = len(vertices)
Nedges = len(edges)
Nfaces = len(faces)

# Afe_data = b.Afe_data
# Afe_indices = b.Afe_indices
# Afe_indptr = b.Afe_indptr

Afv_data = b.Afv_data
Afv_indices = b.Afv_indices
Afv_indptr = b.Afv_indptr

# Aev_data = b.Aev_data
# Aev_indices = b.Aev_indices
# Aev_indptr = b.Aev_indptr

# csr_matrix(np.random.rand(100).reshape((10,10)))
# data, indices, indptr = Afe_data, Afe_indices, Afe_indptr
# Nrows, Ncolums = Nfaces, Nedges


# data, indices, indptr = Afv_data, Afv_indices, Afv_indptr
# Nrows, Ncolums = Nfaces, Nvertices

Nrows, Ncolums = 103, 219
mat = csr_matrix(np.random.rand(Nrows * Ncolums).reshape((Nrows, Ncolums)))
data, indices, indptr = mat.data, mat.indices, mat.indptr


#
A = csr_matrix((data, indices, indptr), shape=(Nrows, Ncolums))
AT = A.T.tocsr()


AT_data, AT_indices, AT_indptr = transpose_csr(data, indices, indptr)

AT.data - AT_data
AT.indices - AT_indices
AT.indptr - AT_indptr
# AT.indptr - np.array([0, *AT_indptr[:-1]])

# %%
_AT = csr_matrix((AT_data, AT_indices, AT_indptr), shape=(Ncolums, Nrows))


np.linalg.norm(np.ravel((_AT - AT).todense()))
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
