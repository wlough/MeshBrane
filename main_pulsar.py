import numpy as np
import sympy as sp

from numdiff import (
    #     matrix_to_quaternion,
    #     quaternion_to_matrix,
    #     exp_so3,
    #     log_so3,
    #     exp_quaternion,
    #     log_quaternion,
    jitcross,
    multiply_se3_quaternion,
    inverse_se3_quaternion,
    #     mul_quaternion,
    #     inv_quaternion,
    #     rotate_by_quaternion,
    #     se3_quaternion_to_matrix,
    #     se3_matrix_to_quaternion,
    #     quaternion_to_matrix2,
    log_se3_quaternion,
    exp_se3_quaternion,
    #     exp_se3,
    #     exp_se3_slow,
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
    polyscope_plot,
    polyscope_plot2,
)

# from scipy.linalg import expm, logm

from scipy.sparse import csr_matrix
import polyscope as ps

# %%
vertices, faces, normals = make_sample_mesh("oblate")


b = FramedBrane(vertices, faces, normals)
# example_multimesh_plot(b)
vertices, edges, faces, frames = (
    b.position_vectors(),
    b.edges,
    b.faces,
    b.orthogonal_matrices(),
)
psi_on_edges = b.get_psi_on_edges()


# %%
def mean_pose(self, vertex_list):
    """computes the SE3-valued mean of the euclidean transformations
    associated with vertices in vertex_list"""
    iters = 40
    v_initial = 0
    framed_vertices = self.framed_vertices

    Nsamps = len(vertex_list)
    mu_g = framed_vertices[vertex_list[v_initial]]

    for iter in range(iters):
        mu_g_inv = inverse_se3_quaternion(mu_g)
        Psi = np.zeros(6)
        for i in vertex_list:
            g = framed_vertices[i]
            mu_g_inv_g = multiply_se3_quaternion(mu_g_inv, g)
            # mu_g_inv_g /= np.linalg.norm(mu_g_inv_g)
            Psi += log_se3_quaternion(mu_g_inv_g) / Nsamps
            # Psi += log_se3_quaternion(multiply_se3_quaternion(g, mu_g_inv)) / Nsamps
        mu_g = multiply_se3_quaternion(mu_g, exp_se3_quaternion(Psi))
        # mu_g = multiply_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
    return mu_g


nan_list = []
f_list = [1037, 1038]  # [_ for _ in range(len(b.faces))]
pq = np.zeros((len(f_list), 7))
for n_f in range(len(f_list)):
    f = f_list[n_f]
    face = faces[f]
    G = np.array([b.framed_vertices[v] for v in face])
    mu_g = G[0]
    mu_g_inv = inverse_se3_quaternion(mu_g)
    mu_g_inv_g = multiply_se3_quaternion(mu_g_inv, mu_g)

    pq_mean = mean_pose(b, face)
    pq[n_f] = pq_mean

# pq = np.array([*pq,*pq_R])
polyscope_plot2(vertices, faces, frames=None, pq=pq)
# ps.remove_surface_mesh("my_mesh")

# registered_surfaces = ps.get_surface_meshes()
ps.remove_all_structures()


# %%


def myplot(brane, vertex_list=None):
    if vertex_list is None:
        vertex_list = [0, 93, 260, 500]
    vertices, faces = brane.position_vectors(), brane.faces
    frames = brane.orthogonal_matrices()
    surfaces_list = []
    brane_dict = {
        "vertices": vertices,
        "faces": faces,
        # "frames": frames,
    } | default_color_dict

    pq = np.zeros((len(vertex_list), 7))
    for _v in range(len(vertex_list)):
        vertex = vertex_list[_v]
        ring = brane.vertices_adjacent_to_vertex(vertex)
        pq_mean = brane.mean_pose(ring)
        pq[_v] = pq_mean

        v_of_e_of_v = np.array([vertex, *ring])
        mini_vertices, mini_faces = brane.mini_mesh(vertex)
        mini_frames = np.array([frames[_] for _ in v_of_e_of_v])

        mini_brane_dict = {
            "vertices": mini_vertices,
            "faces": mini_faces,
            "frames": mini_frames,
            "face_color": (0.0, 0.0, 1.0),
            "edge_color": (1.0, 0.0, 0.0),
            "vertex_color": (1.0, 0.0, 0.0),
            "normal_color": (1.0, 0.0, 0.0),
            "tangent_color": (1.0, 0.0, 0.0),
            "face_alpha": 1.0,
        }
        surfaces_list.append(mini_brane_dict)

    surfaces_list.append(brane_dict)

    polyscope_plot(surfaces_list, pq)


# %%
myplot(b)
