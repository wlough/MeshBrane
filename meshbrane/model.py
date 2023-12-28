import numpy as np
from numba import njit
from numdiff import jitcross
from scipy.sparse import csr_matrix
from meshbrane.init_functions import (
    random_init,
    get_face_data,
    get_edge_data,
    get_ev_adjacency_csr,
    get_fe_adjacency_csr,
    get_fv_adjacency_csr,
    get_area_weighted_vertex_normals,
    implicit_init,
)
from meshbrane.simulation_functions import (
    get_y_of_x_csr,
    update_face_data,
    update_vertex_normals,
)
from mayavi import mlab
import os
from copy import deepcopy
from alphashape import alphashape as ashp


class meshbrane(object):
    """
    docstring for meshbrane
    random_args = {"Npts": 89, "alpha": 1.0}
    implicit_kwargs = {""}
    """

    def __init__(self, random_args=None, implicit_args=None):
        if implicit_args is None:
            alpha_shape = random_init(**random_args)
            self.alpha_shape = alpha_shape
            vertices = np.array(alpha_shape.vertices)
            faces = np.array(alpha_shape.faces)
        else:
            # vertices, faces = implicit_init(implicit_expr, implicit_vars)
            vertices, faces = implicit_init(**implicit_args)
        ###############################
        Nvertices = len(vertices)
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
        # self.vertices = vertices
        # self.vertices_com = vertices_com
        self.edges = edges
        self.faces = faces
        # self.face_centroids = face_centroids
        # self.face_normals = face_normals
        # self.face_areas = face_areas
        # self.vertex_normals = vertex_normals
        self.Aev = Aev_csr
        self.Afe = Afe_csr
        self.Ave = Ave_csr
        self.Aef = Aef_csr
        self.Afv = Afv_csr
        self.Avf = Avf_csr
        ###################################################

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
        vertices, faces, Avf_csr = self.vertices, self.faces, self.Avf
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


class sparsebrane(object):
    """
    docstring for sparsebrane
    random_args = {"Npts": 89, "alpha": 1.0}
    implicit_kwargs = {""}
    """

    def __init__(
        self, init_type=None, random_args=None, implicit_args=None, cloud_args=None
    ):
        if init_type == "random":
            alpha_shape = random_init(**random_args)
            self.alpha_shape = alpha_shape
            vertices = np.array(alpha_shape.vertices)
            faces = np.array(alpha_shape.faces)
        elif init_type == "implicit":
            vertices, faces = implicit_init(**implicit_args)
        elif init_type == "cloud":
            alpha_shape = ashp(**cloud_args)
            self.alpha_shape = alpha_shape
            vertices = np.array(alpha_shape.vertices)
            faces = np.array(alpha_shape.faces)
        ###############################
        Nvertices = len(vertices)
        surface_com = np.einsum("vx->x", vertices) / Nvertices
        faces, face_centroids, face_normals, face_areas = jit_face_data(
            vertices, faces, surface_com
        )
        Afe, Aev, edges = self.get_boundary_ops_csr(vertices, faces)
        Afv = get_Afv(vertices, faces)
        Aef = Afe.T.tocsr()
        Ave = Aev.T.tocsr()
        Avf = Afv.T.tocsr()

        vertex_normals = jit_area_weighted_vertex_normals(
            vertices, Avf.indices, Avf.indptr, face_areas
        )
        ###########################
        ###########################
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.Afe = Afe
        self.Aef = Aef
        self.Aev = Aev
        self.Ave = Ave

        self.face_centroids = face_centroids
        self.face_normals = face_normals
        self.vertex_normals = vertex_normals
        ###########################
        ###########################
        # figure stuff
        os.system("rm -r ./scratch/temp_images")
        os.system("mkdir ./scratch/temp_images")
        self.face_color = (0.0, 0.2667, 0.1059)
        self.edge_color = (1.0, 0.498, 0.0)
        self.vertex_color = (0.7057, 0.0156, 0.1502)
        self.normal_color = (0.2298, 0.2987, 0.7537)

    def orient_faces(self, vertices, faces_old):
        """
        computes what are hopefully outward pointing unit normal vectors
        and directed area vectors of the faces. Reorders vertices of each face to
        match unit normal direction.
        """
        Nvertices = len(vertices)
        surface_com = np.einsum("vx->x", vertices) / Nvertices
        Nfaces = len(faces_old)
        faces = np.zeros((Nfaces, 3), dtype=np.int32)
        for f in range(Nfaces):
            fv0, fv1, fv2 = faces_old[f]
            v0_xyz = vertices[fv0]
            v1_xyz = vertices[fv1]
            v2_xyz = vertices[fv2]

            # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
            f_normal = (
                jitcross(v0_xyz, v1_xyz)
                + jitcross(v1_xyz, v2_xyz)
                + jitcross(v2_xyz, v0_xyz)
            )
            face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
            n_dot_dr = f_normal @ (face_com - surface_com)
            if n_dot_dr > 0:
                faces[f, :] = np.array([fv0, fv1, fv2])
            else:
                faces[f, :] = np.array([fv1, fv0, fv2])
        return faces

    def get_fv_adjacency_csr(self, vertices, faces):
        Nfaces, Nvertices = len(faces), len(vertices)
        indices = faces.ravel()
        indptr = [3 * f for f in range(Nfaces + 1)]
        data = np.ones(3 * Nfaces)

        Afv = csr_matrix((data, indices, indptr), shape=(Nfaces, Nvertices))
        return Afv

    def get_edges_from_faces(self, faces):
        """
        Computes vertex indices of edges from vertex indices of faces
        """
        Nfaces = len(faces)
        Nedges = 0

        vm_edge = []
        vp_edge = []

        edge_indices = {}
        _eij = [[0, 1], [1, 2], [2, 0]]

        for nface in range(Nfaces):
            v0, v1, v2 = faces[nface]
            e01 = [v0, v1]
            e12 = [v1, v2]
            e20 = [v2, v0]

            for nedge, edge in enumerate((e01, e12, e20)):
                add_edge = not edge in edges_sets
                if add_edge:
                    edges_sets.append(edge)
                    edges.append(list(edge))

        return np.array(edges)

    def get_fe_csr_data(self, faces):
        """
        Computes vertex indices of edges, and edge indices of faces
        """
        # Nvertices = len(vertices)
        Nfaces = len(faces)
        # Nedges = 0
        edges = []
        Afe_data_list = []  # [-1,1,...]
        Afe_indices_list = []  #
        Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

        for f in range(Nfaces):
            face = faces[f]
            for _v in range(3):
                vm = face[_v]
                vp = face[np.mod(_v + 1, 3)]
                edge_p = [vm, vp]
                edge_m = [vp, vm]
                try:  # is negative edge already in edges?
                    e = edges.index(edge_m)
                    fe_orientation = -1
                except Exception:
                    try:  # is positive edge already in edges?
                        e = edges.index(edge_p)
                        fe_orientation = 1
                    except Exception:  # if neither, add positive edge to edges
                        edges.append(edge_p)
                        e = len(edges) - 1
                        fe_orientation = 1

                Afe_indices_list.append(e)
                Afe_data_list.append(fe_orientation)

            # Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
        return Afe_data_list, Afe_indices_list, Afe_indptr, edges

    def get_boundary_ops_csr(self, vertices, faces):
        Nvertices = len(vertices)
        (
            Afe_data,
            Afe_indices,
            Aev_indices,
            edges,
        ) = jit_boundary_ops_csr_data(vertices, faces)
        # edges = np.array(edges, dtype=np.int32)
        # Afe ###############
        Nfaces = len(faces)
        # Afe_data = np.array(Afe_data_list, dtype=np.int32)  # [-1,1,...]
        # Afe_indices = np.array(Afe_indices_list, dtype=np.int32)  #
        Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

        # Aev ###############
        Nedges = len(edges)
        Aev_data = np.array(Nedges * [-1, 1], dtype=np.int32)  # [-1,1,...]
        # Aev_indices = np.array(Aev_indices_list, dtype=np.int32)  # vertex indices
        Aev_indptr = np.array(
            [2 * _ for _ in range(Nedges + 1)], dtype=np.int32
        )  # [0,2,4,...]

        Afe = csr_matrix((Afe_data, Afe_indices, Afe_indptr), shape=(Nfaces, Nedges))
        Aev = csr_matrix((Aev_data, Aev_indices, Aev_indptr), shape=(Nedges, Nvertices))

        return Afe, Aev, edges

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

    def plot_part(self, part_vertices=None, part_faces=None):
        vertices = self.vertices
        faces = self.faces
        face_centroids = self.face_centroids
        # face_normals = self.face_normals
        # vertex_normals = self.vertex_normals
        face_color = self.face_color
        edge_color = self.edge_color
        vertex_color = self.vertex_color
        # normal_color = self.normal_color
        # figsize = (2180, 2180)
        figsize = (720, 720)
        ##########################################################
        mlab.options.offscreen = False

        vertex_X, vertex_Y, vertex_Z = vertices.T
        face_X, face_Y, face_Z = face_centroids.T
        # face_nX, face_nY, face_nZ = face_normals.T
        # vertex_nX, vertex_nY, vertex_nZ = vertex_normals.T
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
            # opacity=0.6,
            color=edge_color,
            # representation="wireframe"
            representation="mesh",
            # representation="surface"
            # representation="fancymesh",
            tube_radius=0.02,
            tube_sides=3,
        )

        # part_edges = mlab.triangular_mesh(
        #     *part_vertices.T,
        #     part_faces,
        #     opacity=1.0,
        #     color=edge_color,
        #     # representation="wireframe"
        #     representation="mesh",
        #     # representation="surface"
        #     # representation="fancymesh",
        #     tube_radius=0.02,
        #     tube_sides=3,
        # )

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

        # part_faces = mlab.triangular_mesh(
        #     *part_vertices.T,
        #     part_faces,
        #     opacity=1.0,
        #     color=face_color,
        #     # representation="wireframe"
        #     # representation="mesh",
        #     representation="surface"
        #     # representation="fancymesh",
        #     # tube_radius=None
        # )

        # mem_vertices = mlab.points3d(
        #     *vertices.T,
        #     mode="sphere",
        #     scale_factor=0.075,
        #     color=vertex_color,
        #     opacity=0.6,
        # )

        part_vertices = mlab.points3d(
            *part_vertices.T,
            mode="sphere",
            scale_factor=0.075,
            color=vertex_color,
            opacity=1,
        )

        mlab.show()


@njit
def jit_boundary_ops_csr_data(vertices, faces):
    """
    Computes edges and boundary operators from vertices and faces
    """
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
                e = edges_list.index(edge_m)
                fe_orientation = -1
            except Exception:
                try:  # is positive edge already in edges?
                    e = edges_list.index(edge_p)
                    fe_orientation = 1
                except Exception:  # if neither, add positive edge to edges
                    edges_list.append(edge_p)
                    e = len(edges_list) - 1
                    fe_orientation = 1
                    Aev_indices_list.append(vm)
                    Aev_indices_list.append(vp)

            Afe_indices_list.append(e)
            Afe_data_list.append(fe_orientation)

    Afe_data = np.array(Afe_data_list, dtype=np.int32)
    Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
    Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
    edges = np.array(edges_list, dtype=np.int32)
    return Afe_data, Afe_indices, Aev_indices, edges


@njit
def jit_face_data(vertices, faces, surface_com):
    """
    computes what are hopefully outward pointing unit normal vectors
    and directed area vectors of the faces. Reorders vertices of each face to
    match unit normal direction.
    """
    # Nvertices = len(vertices)
    # surface_com = np.einsum("vx->x", vertices)

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
        n_dot_dr = f_normal @ (face_com - surface_com)
        if n_dot_dr > 0:
            # faces[f, :] = np.array([fv0, fv1, fv2])
            face_normals[f, :] = f_normal
            face_areas[f, :] = f_area
        else:
            faces[f, :] = np.array([fv1, fv0, fv2])
            face_normals[f, :] = -f_normal
            face_areas[f, :] = -f_area
    return faces, face_centroids, face_normals, face_areas


@njit
def jit_area_weighted_vertex_normals(vertices, Avf_indices, Avf_indptr, face_areas):
    """
    computes unit normal vectors at vertices.
    Avf_indices, Avf_indptr
    """
    Nvertices = len(vertices)
    vertex_normals = np.zeros((Nvertices, 3))
    for v in range(Nvertices):
        # Nfaces_of_v = len(faces_of_vertices[v])
        # Nfaces_of_v = max([1,len(faces_of_vertices[v])])
        faces_of_v = Avf_indices[Avf_indptr[v] : Avf_indptr[v + 1]]
        for f in faces_of_v:
            vertex_normals[v] += face_areas[f]
        normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
        if normal_norm > 0:
            vertex_normals[v] /= normal_norm
    return vertex_normals


@njit
def jit_y_of_x_csr(Axy_indices, Axy_indptr):
    # indices, indptr = Axy_csr.indices, Axy_csr.indptr
    Nx = len(Axy_indptr) - 1
    x_of_y = []
    for nx in range(Nx):
        x_of_y.append(Axy_indices[Axy_indptr[nx] : Axy_indptr[nx + 1]])

    return x_of_y


def get_Afv(vertices, faces):
    Nfaces, Nvertices = len(faces), len(vertices)
    indices = faces.ravel()
    indptr = [3 * f for f in range(Nfaces + 1)]
    data = np.ones(3 * Nfaces)

    Afv = csr_matrix((data, indices, indptr), shape=(Nfaces, Nvertices))
    return Afv


# def jit_data_for_reinit():


def collapse_edge(V, E, F, e):
    # f_boundary_coeffs = Afe_coeffs[Afe_indptr[f] : Afe_indptr[f + 1]]
    # f_boundary_indices = Afe_indices[Afe_indptr[f] : Afe_indptr[f + 1]]
    adjacent_faces = 1

    # adjacent_faces = Aff_indices[Aff_indptr[f] : Aff_indptr[f + 1]]


###########
