import numpy as np
import sympy as sp
import polyscope as ps
from src.model import Brane, FramedBrane, HalfEdgeMesh
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
    #     matrix_to_quaternion,
    #     quaternion_to_matrix,
    #     exp_so3,
    #     log_so3,
    #     exp_quaternion,
    #     log_quaternion,
    jitcross,
    mul_se3_quaternion,
    inv_se3_quaternion,
    #     mul_quaternion,
    #     inv_quaternion,
    #     rotate_by_quaternion,
    se3_quaternion_to_matrix,
    #     se3_matrix_to_quaternion,
    #     quaternion_to_matrix2,
    log_se3_quaternion,
    exp_se3_quaternion,
    #     exp_se3,
    #     exp_se3_slow,
)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

# %%

# file_path = "./data/ply_files/mem3dg_oblate.ply"
# vertices, faces = load_mesh_from_ply(file_path)

vertices, faces, normals = make_sample_mesh("torus")

# fb = FramedBrane(vertices, faces, normals)
# %%


def laplacian_smooth(_vertices, _faces, lambda_factor=0.5, iterations=10):
    vertices = _vertices.copy()
    faces = _faces.copy()
    # Create adjacency matrix
    n = len(vertices)
    adjacency_matrix = csr_matrix((n, n))
    for face in faces:
        for i in range(3):
            adjacency_matrix[face[i], face[(i + 1) % 3]] = 1
            adjacency_matrix[face[(i + 1) % 3], face[i]] = 1

    # Create degree matrix
    degree_matrix = csr_matrix((n, n))
    degree_matrix.setdiag(adjacency_matrix.sum(axis=1))

    # Create Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Perform smoothing
    for _ in range(iterations):
        for i in range(3):
            b = degree_matrix.dot(vertices[:, i])
            vertices[:, i] = spsolve(
                laplacian_matrix + lambda_factor * degree_matrix, b
            )

    return vertices


vertices_laplacian_smooth = laplacian_smooth(
    vertices, faces, lambda_factor=0.5, iterations=10
)


def delaunay_smooth(_vertices, _faces):
    # Flatten the vertices to 2D if they are 3D
    vertices = _vertices.copy()
    faces = _faces.copy()
    if vertices.shape[1] == 3:
        vertices = vertices[:, :2]

    # Perform Delaunay triangulation
    tri = Delaunay(vertices)

    # Return the new faces
    return tri.points, tri.simplices


vertices_delaunay_smooth, faces_delaunay_smooth = delaunay_smooth(vertices, faces)
# %%
b = Brane(vertices, faces)
b_laplacian_smooth = b  # Brane(vertices_laplacian_smooth, faces)
b_delaunay_smooth = Brane(vertices, faces_delaunay_smooth)

# %%
branes = [b, b_laplacian_smooth, b_delaunay_smooth]
polyscope_list_plot(branes=branes)
# polyscope_list_plot(brane=b_laplacian_smooth)
# polyscope_list_plot(brane=b_delaunay_smooth)
# %%


# %%


b.V_scalar = b.get_angle_defects()
R = 0.7  # big radius
r = 0.7 / 3.0  # small radius

Rout = R + r
Rin = R - r
Kout = 1 / (Rout * r)
Kin = 1 / (Rin * r)
# %%
self = b
Nverts = len(self.pq)
verts_of_verts = []
# defects = np.zeros(Nverts)
K = np.zeros(Nverts)
# V = self.V_index
for v0 in range(Nverts):
    # p0 = self.pq[v, :3]
    h_start = self.V_hedge[v0]
    defect = 2 * np.pi
    area = 0.0

    h = h_start
    v = self.H_vertex[h]
    e2 = self.pq[v, :3] - self.pq[v0, :3]
    norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
    h = self.H_next[self.H_twin[h]]

    while True:
        e1 = e2
        norm_e1 = norm_e2
        v = self.H_vertex[h]  # 2nd vert
        e2 = self.pq[v, :3] - self.pq[v0, :3]
        norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
        # e1_cross_e2 = jitcross(e1, e2)
        # norm_e1_cross_e2 = np.sqrt(
        #     e1_cross_e2[0] ** 2 + e1_cross_e2[1] ** 2 + e1_cross_e2[2] ** 2
        # )
        # sin_angle = norm_e1_cross_e2 / (norm_e1 * norm_e2)
        cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
            norm_e1 * norm_e2
        )
        angle = np.arccos(cos_angle)
        face_area = 0.5 * norm_e1 * norm_e2 * np.sin(angle)
        # print(f"angle={np.round(angle / np.pi, 3)}")
        # print(f"farea={np.round(face_area, 3)}")

        defect -= angle
        area += face_area / 3

        h = self.H_next[self.H_twin[h]]
        if h == h_start:
            break
    K[v0] = defect / area
    print(f"area={np.round(area, 6)}")
    print(f"angle defect={np.round(defect/np.pi, 6)}")


# K
# plt.plot(K)
# %%
# example_multimesh_plot(b)
# vertices, edges, faces, frames = (
#     b.position_vectors(),
#     b.edges,
#     b.faces,
#     b.orthogonal_matrices(),
# )
# psi_on_edges = b.get_psi_on_edges()
vertices, edges, faces, frames = (
    b.vertices,
    b.halfedges,
    b.faces,
    b.orthogonal_matrices(),
)
polyscope_mesh_plot(vertices, faces, frames=None)
# %%
PQ = np.random.rand(7)
PQ[3:] /= np.linalg.norm(PQ[3:])
b.rigid_transform(PQ)
vertices, edges, faces, frames = (
    b.vertices,
    b.halfedges,
    b.faces,
    b.orthogonal_matrices(),
)
polyscope_mesh_plot(vertices, faces, frames=None)


# %%
def mean_pose(self, vertex_list):
    """computes the SE3-valued mean of the euclidean transformations
    associated with vertices in vertex_list"""
    iters = 3
    v_initial = 0
    framed_vertices = self.pq

    Nsamps = len(vertex_list)
    mu_g = framed_vertices[vertex_list[v_initial]]

    for iter in range(iters):
        mu_g_inv = inv_se3_quaternion(mu_g)
        Psi = np.zeros(6)
        for i in vertex_list:
            g = framed_vertices[i]
            g[3:] /= np.linalg.norm(g[3:])
            mu_g_inv_g = mul_se3_quaternion(mu_g_inv, g)
            mu_g_inv_g[3:] /= np.linalg.norm(mu_g_inv_g[3:])

            Psi += log_se3_quaternion(mu_g_inv_g) / Nsamps
            # Psi += log_se3_quaternion(mul_se3_quaternion(g, mu_g_inv)) / Nsamps
        mu_g = mul_se3_quaternion(mu_g, exp_se3_quaternion(Psi))
        # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
    return mu_g


nan_list = []
f_list = [_ for _ in range(len(b.faces))]  # [158, 159]  #
pq = np.zeros((len(f_list), 7))
for n_f in range(len(f_list)):
    f = f_list[n_f]
    face = faces[f]
    G = np.array([b.pq[v] for v in face])
    mu_g = G[0]
    mu_g_inv = inv_se3_quaternion(mu_g)
    mu_g_inv_g = mul_se3_quaternion(mu_g_inv, mu_g)

    pq_mean = mean_pose(b, face)
    xx = np.linalg.norm(pq_mean - mu_g)
    if xx > 1:
        # pq_mean = mu_g
        nan_list.append(face)
    pq[n_f] = pq_mean

    # print(xx)

# pq = np.array([*pq,*pq_R])
polyscope_plot2(vertices, faces, frames=None, pq=pq)
# ps.remove_surface_mesh("my_mesh")

# registered_surfaces = ps.get_surface_meshes()
ps.remove_all_structures()

pq = [b.pq[_] for _ in nan_list[0]]

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
