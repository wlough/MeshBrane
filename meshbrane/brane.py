import alphashape
import numpy as np
from numdiff import fib_sphere, jitcross  # , my_vectorize_args, my_vectorize

# from skimage.measure import marching_cubes
from mayavi import mlab
import os
from numba import njit

# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix  # , csc_matrix


# ######################################################
# # Initialization functions
# ######################################################
# # @njit
# def get_face_data(vertices, faces, surface_com):
#     """
#     computes what are hopefully outward pointing unit normal vectors
#     and directed area vectors of the faces. Reorders vertices of each face to
#     match unit normal direction.
#     """
#     Nfaces = len(faces)
#     face_normals = np.zeros((Nfaces, 3))
#     face_areas = np.zeros((Nfaces, 3))
#     face_centroids = np.zeros((Nfaces, 3))
#     for f in range(Nfaces):
#         fv0, fv1, fv2 = faces[f]
#         v0_xyz = vertices[fv0]
#         v1_xyz = vertices[fv1]
#         v2_xyz = vertices[fv2]
#
#         # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
#         f_normal = (
#             jitcross(v0_xyz, v1_xyz)
#             + jitcross(v1_xyz, v2_xyz)
#             + jitcross(v2_xyz, v0_xyz)
#         )
#         f_area = 0.5 * f_normal
#
#         f_normal /= np.sqrt(f_normal @ f_normal)
#         face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
#         face_centroids[f] = face_com
#         n_dot_dr = f_normal @ (face_com - surface_com)
#         if n_dot_dr > 0:
#             face_normals[f, :] = f_normal
#             face_areas[f, :] = f_area
#         else:
#             faces[f, :] = np.array([fv1, fv0, fv2])
#             face_normals[f, :] = -f_normal
#             face_areas[f, :] = -f_area
#     return faces, face_centroids, face_normals, face_areas
#
#
# # @njit
# def get_edge_data(faces):
#     """
#     Computes vertex indices of edges, and edge indices of faces
#     """
#     edges_sets = []
#     edges = []
#     Nfaces = len(faces)
#     edges_of_faces = np.empty((Nfaces, 3), dtype=np.int32)
#
#     for nface in range(Nfaces):
#         v0, v1, v2 = faces[nface]
#         e01 = {v0, v1}
#         e12 = {v1, v2}
#         e20 = {v2, v0}
#
#         for nedge, edge in enumerate((e01, e12, e20)):
#             add_edge = not edge in edges_sets
#             if add_edge:
#                 edges_sets.append(edge)
#                 edges.append(list(edge))
#                 edges_of_faces[nface, nedge] = len(edges) - 1
#             else:
#                 edges_of_faces[nface, nedge] = edges_sets.index(edge)
#
#     return np.array(edges), edges_of_faces
#
#
# # @njit
# def get_ev_adjacency_csr(vertices, edges):
#     """
#     A.data -- nonzero matrix elements
#     A.indices -- column indices (vertex indices) for nonzero matrix elements
#     A.indptr --- pointers for nonzero matrix elements and their column indices
#     *** len(A.indptr) = #rows+1
#
#     Nvertices = len(vertices)
#     Nedges = len(edges)
#     A0 = np.zeros((Nedges, Nvertices), dtype=np.int32)
#     for e in range(Nedges):
#         v0, v1 = edges[e]
#         A0[e, v0] = -1
#         A0[e, v1] = 1
#     return A0
#     """
#     Nedges = len(edges)
#     Nvertices = len(vertices)
#     indices = edges.ravel()
#     data = np.ravel(Nedges * [[-1, 1]])
#     indptr = [2 * _ for _ in range(Nedges + 1)]
#     A0 = csr_matrix((data, indices, indptr), shape=(Nedges, Nvertices))
#
#     return A0
#
#
# # @njit
# def get_fe_adjacency_dense(edges, faces, edges_of_faces):
#     # Nvertices = len(vertices)
#     Nedges = len(edges)
#     Nfaces = len(faces)
#     A1 = np.zeros((Nfaces, Nedges), dtype=np.int32)  # most
#     for f in range(Nfaces):
#         e0, e1, e2 = edges_of_faces[f]
#         v0, v1, v2 = faces[f]
#         pos = [[v0, v1], [v1, v2], [v2, v0]]
#         neg = [[v1, v0], [v2, v1], [v0, v2]]
#
#         for e in edges_of_faces[f]:
#             ev = list(edges[e])
#             if ev in pos:
#                 # index = pos.index(ev)
#                 A1[f, e] = 1
#             elif ev in neg:
#                 # index = neg.index(ev)
#                 A1[f, e] = -1
#             else:
#                 raise ValueError
#             # if ev in pos:
#             #     index = pos.index(ev)
#             #     A1[f, e] = 1
#             # elif ev in neg:
#             #     index = neg.index(ev)
#             #     A1[f, e] = -1
#             # else:
#             #     raise ValueError
#             # pos.pop(index)
#             # neg.pop(index)
#     return A1
#
#
# # @njit
# def get_fe_adjacency_csr_data(edges, faces, edges_of_faces):
#     # Nvertices = len(vertices)
#     Nfaces = len(faces)
#     data = np.zeros(3 * Nfaces, dtype=np.int32)
#     indices = edges_of_faces.ravel()  # np.zeros(3 * Nfaces, dtype=np.int32)
#     indptr = np.zeros(Nfaces + 1, dtype=np.int32)
#     # indptr = [3 * _ for _ in range(Nedges + 1)]
#     for f in range(Nfaces):
#         indptr[f + 1] = 3 * (f + 1)
#         e0, e1, e2 = edges_of_faces[f]
#         v0, v1, v2 = faces[f]
#         pos = [[v0, v1], [v1, v2], [v2, v0]]
#         neg = [[v1, v0], [v2, v1], [v0, v2]]
#
#         # for e in edges_of_faces[f]:
#         for n_e in range(3):
#             e = edges_of_faces[f, n_e]
#             ev = list(edges[e])
#             if ev in pos:
#                 data[3 * f + n_e] = 1
#             elif ev in neg:
#                 data[3 * f + n_e] = -1
#             else:
#                 raise ValueError
#     return data, indices, indptr
#
#
# def get_fe_adjacency_csr(edges, faces, edges_of_faces):
#     data, indices, indptr = get_fe_adjacency_csr_data(edges, faces, edges_of_faces)
#     Nfaces, Nedges = len(faces), len(edges)
#     A1 = csr_matrix((data, indices, indptr), shape=(Nfaces, Nedges))
#     return A1
#
#
# def get_fv_adjacency_csr(vertices, faces):
#     Nfaces, Nvertices = len(faces), len(vertices)
#     indices = faces.ravel()
#     indptr = [3 * f for f in range(Nfaces + 1)]
#     data = np.ones(3 * Nfaces)
#
#     Afv = csr_matrix((data, indices, indptr), shape=(Nfaces, Nvertices))
#     return Afv
#
#
# def get_faces_of_vertices(vertices, faces):
#     Nfaces = len(faces)
#     Nvertices = len(vertices)
#     faces_of_vertices = []
#     for v in range(Nvertices):
#         faces_of_vertices.append([])
#         for f in range(Nfaces):
#             if v in faces[f]:
#                 faces_of_vertices[-1].append(f)
#     return faces_of_vertices
#
#
# def get_area_weighted_vertex_normals(vertices, faces_of_vertices, face_areas):
#     """
#     computes unit normal vectors at vertices.
#     """
#     Nvertices = len(vertices)
#     vertex_normals = np.zeros((Nvertices, 3))
#     for v in range(Nvertices):
#         # Nfaces_of_v = len(faces_of_vertices[v])
#         # Nfaces_of_v = max([1,len(faces_of_vertices[v])])
#         for f in faces_of_vertices[v]:
#             vertex_normals[v] += face_areas[f]
#         normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
#         if normal_norm > 0:
#             vertex_normals[v] /= normal_norm
#     return vertex_normals
#
#
# ######################################################
# ######################################################


######################################################
# Simulation functions
######################################################
@njit
def update_face_data(vertices, faces):
    """
    computes outward pointing unit normal vectors
    and directed area vectors of the faces.
    """
    Nfaces = len(faces)
    face_normals = np.zeros((Nfaces, 3))
    face_areas = np.zeros((Nfaces, 3))
    face_centroids = np.zeros((Nfaces, 3))
    for f in range(Nfaces):
        fv0, fv1, fv2 = faces[f]
        v0_xyz = vertices[fv0]
        v1_xyz = vertices[fv1]
        v2_xyz = vertices[fv2]

        # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
        f_normal = (
            jitcross(v0_xyz, v1_xyz)
            + jitcross(v1_xyz, v2_xyz)
            + jitcross(v2_xyz, v0_xyz)
        )
        f_area = 0.5 * f_normal

        f_normal /= np.sqrt(f_normal @ f_normal)
        face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
        face_centroids[f] = face_com
        face_normals[f, :] = f_normal
        face_areas[f, :] = f_area
    return face_centroids, face_normals, face_areas


@njit
def update_vertex_normals(vertices, Avf_indices, Avf_indptr, face_areas):
    """
    computes unit normal vectors at vertices.
    """
    Nvertices = len(vertices)

    vertex_normals = np.zeros((Nvertices, 3))
    for v in range(Nvertices):
        faces_of_v = Avf_indices[Avf_indptr[v] : Avf_indptr[v + 1]]
        for f in faces_of_v:
            vertex_normals[v] += face_areas[f]
        normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
        if normal_norm > 0:
            vertex_normals[v] /= normal_norm
    return vertex_normals


def get_y_of_x_csr(Axy_csr):
    # Aef_csr = self.Aef_csr
    indices, indptr = Axy_csr.indices, Axy_csr.indptr
    Nx = len(indptr) - 1
    x_of_y = []
    for v in range(Nx):
        x_of_y.append(indices[indptr[v] : indptr[v + 1]])

    return x_of_y


######################################################
######################################################


class meshbrane(object):
    """docstring for meshbrane"""

    def __init__(self, Npts=89, alpha=1.0):
        ###################################################
        big_sphere = fib_sphere(int(5 * Npts / 10))
        medium_sphere = fib_sphere(int(3 * Npts / 10))
        medium_sphere *= 0.513
        medium_sphere = np.array(
            [xyz - np.array([1.01, 0.0, 0.0]) for xyz in medium_sphere]
        )

        small_sphere = fib_sphere(int(2 * Npts / 10))
        small_sphere *= 0.2513
        small_sphere = np.array(
            [xyz - np.array([1.51, 0.0, 0.0]) for xyz in small_sphere]
        )

        point_cloud = np.array([*big_sphere, *medium_sphere, *small_sphere])

        alpha_shape = alphashape.alphashape(point_cloud, alpha)
        vertices = np.array(alpha_shape.vertices)
        Nvertices = len(vertices)
        faces = np.array(alpha_shape.faces)
        vertices_com = np.einsum("vx->x", vertices) / Nvertices
        faces, face_centroids, face_normals, face_areas = get_face_data(
            vertices, faces, vertices_com
        )
        edges, edges_of_faces = get_edge_data(faces)

        Aev_csr = get_ev_adjacency_csr(vertices, edges)
        Afe_csr = get_fe_adjacency_csr(edges, faces, edges_of_faces)
        Afv_csr = get_fv_adjacency_csr(vertices, faces)
        Ave_csr = Aev_csr.T.tocsr()
        Aef_csr = Afe_csr.T.tocsr()
        Avf_csr = Afv_csr.T.tocsr()

        faces_of_vertices = get_y_of_x_csr(Avf_csr)
        vertex_normals = get_area_weighted_vertex_normals(
            vertices, faces_of_vertices, face_areas
        )
        ###################################################
        self.alpha_shape = alpha_shape
        ###################################################
        ###################################################
        ###################################################
        os.system("rm -r ./scratch/temp_images")
        os.system("mkdir ./scratch/temp_images")
        ###################################################
        # stuff you need to update each timestep
        self.vertices = vertices
        self.vertices_com = vertices_com
        self.face_centroids = face_centroids
        self.face_normals = face_normals
        self.face_areas = face_areas
        self.vertex_normals = vertex_normals

        ###################################################
        # stuff you need to update after adding/removing edge/face/vertex
        self.edges = edges
        self.faces = faces
        self.Aev_csr = Aev_csr
        self.Afe_csr = Afe_csr
        self.Ave_csr = Ave_csr
        self.Aef_csr = Aef_csr
        self.Afv_csr = Afv_csr
        self.Avf_csr = Avf_csr

        ###################################################
        # figure stuff
        self.face_color = (0.0, 0.2667, 0.1059)
        self.edge_color = (1.0, 0.498, 0.0)
        self.vertex_color = (0.7057, 0.0156, 0.1502)
        self.normal_color = (0.2298, 0.2987, 0.7537)

    def wiggly_timestep(self, dt=0.03):
        normal_drift = 0.1
        vertices = self.vertices
        vertex_normals = self.vertex_normals
        new_vertices = np.zeros_like(vertices)
        Nvertices = len(vertices)

        for v in range(Nvertices):
            new_vertices[v] = (
                vertices[v]
                + dt * 2 * (np.random.rand() - 0.5 + normal_drift) * vertex_normals[v]
            )

        self.vertices = new_vertices

    def update_after_timestep(self):
        vertices, faces, Avf_csr = self.vertices, self.faces, self.Avf_csr
        ###########
        Nvertices = len(vertices)
        vertices_com = np.einsum("vx->x", vertices) / Nvertices
        face_centroids, face_normals, face_areas = update_face_data(vertices, faces)
        # faces_of_vertices = get_y_of_x_csr(Avf_csr)
        Avf_indices, Avf_indptr = Avf_csr.indices, Avf_csr.indptr
        vertex_normals = update_vertex_normals(
            vertices, Avf_indices, Avf_indptr, face_areas
        )
        ############
        self.vertices_com = vertices_com
        self.face_centroids = face_centroids
        self.face_normals = face_normals
        self.face_areas = face_areas
        self.vertex_normals = vertex_normals

    def plot_mesh(
        self,
        show=True,
        save=False,
        fig_path=None,
        plot_vertices=True,
        plot_edges=True,
        plot_faces=True,
        plot_face_normals=False,
        plot_vertex_normals=False,
    ):
        vertices = self.vertices
        faces = self.faces
        face_centroids = self.face_centroids
        face_normals = self.face_normals
        vertex_normals = self.vertex_normals
        face_color = self.face_color
        edge_color = self.edge_color
        vertex_color = self.vertex_color
        # normal_color = self.normal_color
        # figsize = (2180, 2180)
        figsize = (720, 720)
        ##########################################################
        if show:
            mlab.options.offscreen = False
        else:
            mlab.options.offscreen = True

        vertex_X, vertex_Y, vertex_Z = vertices.T
        face_X, face_Y, face_Z = face_centroids.T
        face_nX, face_nY, face_nZ = face_normals.T
        vertex_nX, vertex_nY, vertex_nZ = vertex_normals.T
        # bgcolor = (1.0, 1.0, 1.0)
        # fgcolor = (0.0, 0.0, 0.0)
        figsize = (2180, 2180)
        title = f"Membrane mesh"
        fig = mlab.figure(title, size=figsize)  # , bgcolor=bgcolor, fgcolor=fgcolor)
        if plot_edges:
            mem_edges = mlab.triangular_mesh(
                vertex_X,
                vertex_Y,
                vertex_Z,
                faces,
                # opacity=0.4,
                color=edge_color,
                # representation="wireframe"
                representation="mesh",
                # representation="surface"
                # representation="fancymesh",
                tube_radius=0.02,
                tube_sides=3,
            )
        if plot_faces:
            mem_faces = mlab.triangular_mesh(
                vertex_X,
                vertex_Y,
                vertex_Z,
                faces,
                opacity=0.4,
                color=face_color,
                # representation="wireframe"
                # representation="mesh",
                representation="surface"
                # representation="fancymesh",
                # tube_radius=None
            )
        if plot_vertices:
            mem_vertices = mlab.points3d(
                *vertices.T, mode="sphere", scale_factor=0.075, color=vertex_color
            )

        if plot_face_normals:
            f_normals = mlab.quiver3d(
                face_X, face_Y, face_Z, face_nX, face_nY, face_nZ, color=face_color
            )

        if plot_vertex_normals:
            v_normals = mlab.quiver3d(
                vertex_X,
                vertex_Y,
                vertex_Z,
                vertex_nX,
                vertex_nY,
                vertex_nZ,
                color=vertex_color,
            )

        if show:
            mlab.show()
        if save:
            mlab.savefig(fig_path, figure=fig, size=figsize)
        mlab.close(all=True)

        # mlab.close(all=True)

    def movie(self):
        image_type = "png"
        image_directory = "./scratch/temp_images"
        movie_directory = "./scratch/temp_images/movie.mp4"
        os.system(
            "ffmpeg "
            # frame rate (Hz)
            + "-r 20 "
            # frame size (width x height)
            + "-s 1080x720 "
            # input files
            + "-i "
            + image_directory
            + f"/img_%04d.{image_type} "
            # video codec
            + "-vcodec libx264 "
            # video quality, lower means better
            + "-crf 25 "
            # pixel format
            + "-pix_fmt yuv420p "
            # output file
            + movie_directory
        )

    def wiggly_sim(self, T=1, dt=0.01, figure_kwargs=None):
        if figure_kwargs is None:
            figure_kwargs = {"show": False, "save": True}
        t = 0
        image_count = 0
        while t < T:
            self.wiggly_timestep(dt=dt)
            self.update_after_timestep()
            fig_path = f"./scratch/temp_images/img_{image_count:0>4}.png"
            figure_kwargs["fig_path"] = fig_path
            self.plot_mesh(**figure_kwargs)
            image_count += 1
            t += dt

        self.movie()


brane = meshbrane(Npts=100, alpha=1.0)

figure_kwargs = {
    "show": False,
    "save": True,
    "plot_vertices": True,
    "plot_edges": True,
    "plot_faces": True,
    "plot_face_normals": False,
    "plot_vertex_normals": False,
}

brane.wiggly_sim(T=0.2, dt=0.01, figure_kwargs=figure_kwargs)

# %%
vertices = brane.vertices
edges = brane.edges
Avf = brane.Avf_csr
Ave = brane.Ave_csr
v = 13

faces_of_vertices = get_y_of_x_csr(Avf)
edges_of_vertices = get_y_of_x_csr(Ave)

faces_of_v = faces_of_vertices[v]
edges_of_v = edges_of_vertices[v]

# _e = 2
# e = edges_of_v[2]
# edge = edges[e]
#

Nev = len(edges_of_v)
cell_vertex_indices = np.zeros(Nev, dtype=np.int32)

for _e in range(Nev):
    e = edges_of_v[_e]
    edge = edges[e]
    edge = edges[e]
    ev_sgn = Ave[v, e]
    if ev_sgn == 1:
        # v_cell = sum(edge) - v
        v_cell = edge[0]
    elif ev_sgn == -1:
        # v_cell = sum(edge) - v
        v_cell = edge[1]
    cell_vertex_indices[_e] = v_cell
cell_vertices = np.array([vertices[v] for v in cell_vertex_indices])
vertex = vertices[v]
# %%
# def plot_mesh_cell(self, show=True, save=False, fig_path=None):
self = brane
show = True
save = False
fig_path = None
vertices = self.vertices
faces = self.faces
face_centroids = self.face_centroids
face_normals = self.face_normals
vertex_normals = self.vertex_normals
face_color = self.face_color
edge_color = self.edge_color
vertex_color = self.vertex_color
normal_color = self.normal_color
figsize = (2180, 2180)
##########################################################
if show:
    mlab.options.offscreen = False
else:
    mlab.options.offscreen = True

cell_X, cell_Y, cell_Z = cell_vertices.T

vertex_X, vertex_Y, vertex_Z = vertices.T
face_X, face_Y, face_Z = face_centroids.T
face_nX, face_nY, face_nZ = face_normals.T
vertex_nX, vertex_nY, vertex_nZ = vertex_normals.T
# bgcolor = (1.0, 1.0, 1.0)
# fgcolor = (0.0, 0.0, 0.0)
figsize = (2180, 2180)
title = f"Membrane mesh"
fig = mlab.figure(title, size=figsize)  # , bgcolor=bgcolor, fgcolor=fgcolor)
mem_edges = mlab.triangular_mesh(
    vertex_X,
    vertex_Y,
    vertex_Z,
    faces,
    # opacity=0.4,
    color=edge_color,
    # representation="wireframe"
    representation="mesh",
    # representation="surface"
    # representation="fancymesh",
    tube_radius=0.03,
)
mem_faces = mlab.triangular_mesh(
    vertex_X,
    vertex_Y,
    vertex_Z,
    faces,
    opacity=0.4,
    color=face_color,
    # representation="wireframe"
    # representation="mesh",
    representation="surface"
    # representation="fancymesh",
    # tube_radius=None
)
mem_vertices = mlab.points3d(
    *vertices.T, mode="sphere", scale_factor=0.1, color=vertex_color
)

cell_vertices = mlab.points3d(
    *cell_vertices.T, mode="sphere", scale_factor=0.1, color=normal_color
)

# f_normals = mlab.quiver3d(
#     face_X,
#     face_Y,
#     face_Z,
#     face_nX,
#     face_nY,
#     face_nZ,
# )
# v_normals = mlab.quiver3d(
#     vertex_X,
#     vertex_Y,
#     vertex_Z,
#     vertex_nX,
#     vertex_nY,
#     vertex_nZ,
#     color=normal_color,
# )

if show:
    mlab.show()
if save:
    mlab.savefig(fig_path, figure=fig, size=figsize)
mlab.close(all=True)

# mlab.close(all=True)
