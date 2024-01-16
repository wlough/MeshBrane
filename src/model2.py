from numba import float64, int32
from numba.experimental import jitclass
import numpy as np
from src.numdiff import (
    jitcross,
    transpose_csr,
    quaternion_to_matrix,
    inv_se3_quaternion,
    mul_se3_quaternion,
    log_se3_quaternion,
    exp_se3_quaternion,
    inv_quaternion,
    mul_quaternion,
    log_unit_quaternion,
    exp_unit_quaternion,
)

# from mayavi import mlab


framed_brane_spec = [
    # ("vertices", float64[:, :]),
    # ("normals", float64[:, :]),
    ("faces", int32[:, :]),
    ("edges", int32[:, :]),
    ("half_edges", int32[:, :]),
    ("pq", float64[:, :]),
    ("Afe_data", int32[:]),
    ("Afe_indices", int32[:]),
    ("Afe_indptr", int32[:]),
    ("Aev_data", int32[:]),
    ("Aev_indices", int32[:]),
    ("Aev_indptr", int32[:]),
    ("Afv_data", int32[:]),
    ("Afv_indices", int32[:]),
    ("Afv_indptr", int32[:]),
    ("Aef_data", int32[:]),
    ("Aef_indices", int32[:]),
    ("Aef_indptr", int32[:]),
    ("Ave_data", int32[:]),
    ("Ave_indices", int32[:]),
    ("Ave_indptr", int32[:]),
    ("Avf_data", int32[:]),
    ("Avf_indices", int32[:]),
    ("Avf_indptr", int32[:]),
    ("psi", float64[:, :, :]),
]


# @jitclass(framed_brane_spec)
class FramedBrane:
    def __init__(self, vertices, faces, normals):
        ##################
        self.faces = faces
        self.Afv_indices = faces.ravel()
        self.Afv_indptr = np.array(
            [3 * f for f in range(len(faces) + 1)], dtype=np.int32
        )
        self.Afv_data = np.ones(3 * len(faces), dtype=np.int32)
        (
            self.edges,
            self.Afe_data,
            self.Afe_indices,
            self.Afe_indptr,
            self.Aev_data,
            self.Aev_indices,
            self.Aev_indptr,
        ) = self.get_adjacencies()
        self.Aef_data, self.Aef_indices, self.Aef_indptr = transpose_csr(
            self.Afe_data, self.Afe_indices, self.Afe_indptr
        )
        self.Ave_data, self.Ave_indices, self.Ave_indptr = transpose_csr(
            self.Aev_data, self.Aev_indices, self.Aev_indptr
        )
        self.Avf_data, self.Avf_indices, self.Avf_indptr = transpose_csr(
            self.Afv_data, self.Afv_indices, self.Afv_indptr
        )
        ####################################################################
        self.pq = self.frame_the_mesh(vertices, normals)

    def frame_the_mesh(self, vertices, normals):
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        # ez = np.array([0.0, 0.0, 1.0])
        Nverts = len(vertices)
        # framed_vertices = np.zeros((Nverts, 7))
        # matrices = np.zeros((Nverts, 3, 3))
        framed_vertices = np.zeros((Nverts, 7))
        for i in range(Nverts):
            cross_with_ey = np.sqrt(normals[i, 2] ** 2 + normals[i, 0] ** 2) > 1e-6
            if cross_with_ey:
                e1 = jitcross(ey, normals[i])
            else:
                e1 = jitcross(ex, normals[i])
            e1 /= np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            e2 = jitcross(normals[i], e1)

            R = np.zeros((3, 3))
            R[:, 0] = e1
            R[:, 1] = e2
            R[:, 2] = normals[i]
            framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
            framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])

            framed_vertices[i, :3] = vertices[i, :]
        return framed_vertices

    def get_face_data(self, vertices, faces, surface_com):
        """
        computes what are hopefully outward pointing unit normal vectors
        and directed area vectors of the faces. Reorders vertices of each face to
        match unit normal direction.
        """
        # faces = faces_old
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
                face_normals[f, :] = f_normal
                face_areas[f, :] = f_area
            else:
                faces[f, :] = np.array([fv1, fv0, fv2])
                face_normals[f, :] = -f_normal
                face_areas[f, :] = -f_area
        return faces, face_centroids, face_normals, face_areas

    def get_adjacencies(self):
        # vertices = self.framed_vertices
        faces = self.faces
        # Nvertices = len(vertices)
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
                    edges_list.index(edge_m)
                except Exception:  # if not, then add it
                    edges_list.append(edge_m)
                    e = len(edges_list) - 1
                    Afe_indices_list.append(e)
                    Afe_data_list.append(-1)
                    Aev_indices_list.append(vp)
                    Aev_indices_list.append(vm)
                try:  # is positive edge already in edges?
                    edges_list.index(edge_p)
                except Exception:  # if neither, add positive edge to edges
                    edges_list.append(edge_p)
                    e = len(edges_list) - 1
                    Afe_indices_list.append(e)
                    Afe_data_list.append(1)
                    Aev_indices_list.append(vm)
                    Aev_indices_list.append(vp)

        Afe_data = np.array(Afe_data_list, dtype=np.int32)
        Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
        Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

        edges = np.array(edges_list, dtype=np.int32)
        Nedges = len(edges)

        Aev_data = np.array(Nedges * [-1, 1], dtype=np.int32)  # [-1,1,...]
        Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
        Aev_indptr = np.array(
            [2 * _ for _ in range(Nedges + 1)], dtype=np.int32
        )  # [0,2,4,...]

        return (
            edges,
            Afe_data,
            Afe_indices,
            Afe_indptr,
            Aev_data,
            Aev_indices,
            Aev_indptr,
        )

    def get_edges(self, faces):
        # vertices = self.framed_vertices
        # faces = self.faces
        # Nvertices = len(vertices)
        Nfaces = len(faces)
        edges_list = []
        halfedges_list = []

        for f in range(Nfaces):
            face = faces[f]
            for _v in range(3):
                vm = face[_v]
                vp = face[np.mod(_v + 1, 3)]
                edge_p = [vm, vp]
                edge_m = [vp, vm]

                try:  # is negative edge already in edges?
                    halfedges_list.index(edge_m)
                except Exception:  # if not, then add it
                    halfedges_list.append(edge_m)
                try:  # is positive edge already in edges?
                    halfedges_list.index(edge_p)
                except Exception:  # if neither, add positive edge to edges
                    halfedges_list.append(edge_p)

                try:  # is negative edge already in edges?
                    edges_list.index(edge_m)
                except Exception:
                    try:  # is positive edge already in edges?
                        edges_list.index(edge_p)
                    except Exception:  # if neither, add positive edge to edges
                        edges_list.append(edge_p)

        edges = np.array(edges_list, dtype=np.int32)
        halfedges = np.array(halfedges_list, dtype=np.int32)

        return edges, halfedges

    def position_vectors(self):
        return self.pq[:, :3]

    def unit_quaternions(self):
        return self.pq[:, 3:]

    def orthogonal_matrices(self):
        Q = self.pq[:, 3:]
        Nv = len(Q)
        R = np.zeros((Nv, 3, 3))
        for v in range(Nv):
            q = Q[v]
            R[v] = quaternion_to_matrix(q)
        return R

    def get_y_of_x_csr(self, Axy_indices, Axy_indptr):
        # indices, indptr = Axy_csr.indices, Axy_csr.indptr
        Nx = len(Axy_indptr) - 1
        x_of_y = []
        for nx in range(Nx):
            x_of_y.append(Axy_indices[Axy_indptr[nx] : Axy_indptr[nx + 1]])

        return x_of_y

    def get_edges_of_faces(self):
        """e_of_f, v_of_e"""
        # indices, indptr = Axy_csr.indices, Axy_csr.indptr
        Afe_indices, Afe_indptr = self.Afe_indices, self.Afe_indptr
        Nfaces = len(Afe_indptr) - 1
        e_of_f = []
        for f in range(Nfaces):
            e_of_f.append(Afe_indices[Afe_indptr[f] : Afe_indptr[f + 1]])

        return e_of_f

    def edges_of_face(self, face):
        """face = index of face"""
        return self.Afe_indices[self.Afe_indptr[face] : self.Afe_indptr[face + 1]]

    def faces_of_edge(self, edge):
        """face = index of face"""
        return self.Aef_indices[self.Aef_indptr[edge] : self.Aef_indptr[edge + 1]]

    def faces_of_vertex(self, vertex):
        """face = index of face"""
        return self.Avf_indices[self.Avf_indptr[vertex] : self.Avf_indptr[vertex + 1]]

    def vertices_of_face(self, face):
        """face = index of face"""
        return self.Afv_indices[self.Afv_indptr[face] : self.Afv_indptr[face + 1]]

    def edges_of_vertex(self, vertex):
        """face = index of face"""
        return self.Ave_indices[self.Ave_indptr[vertex] : self.Ave_indptr[vertex + 1]]

    def vertices_of_edge(self, edge):
        """face = index of face"""
        return self.Aev_indices[self.Aev_indptr[edge] : self.Aev_indptr[edge + 1]]

    def vertices_adjacent_to_vertex(self, vertex):
        """face = index of face"""

        e_of_v = self.edges_of_vertex(vertex)
        v_of_e_of_v = []  # [b.edges[_] for _ in e_of_v]

        for e in e_of_v:
            edge = self.edges[e]
            for v in edge:
                add_v = (not v in v_of_e_of_v) and v != vertex
                if add_v:
                    v_of_e_of_v.append(v)
        return v_of_e_of_v

    def mini_mesh(self, v):
        faces = self.faces
        vertices = self.position_vectors()
        # V, E, F = [], [], []
        F = self.faces_of_vertex(v)
        V = [v]
        for f in F:
            face = faces[f]
            for _v in face:
                add_v = not _v in V
                if add_v:
                    V.append(_v)

        Nfaces = len(F)
        Nvertices = len(V)
        mini_faces = np.zeros((Nfaces, 3), dtype=np.int32)
        mini_vertices = np.zeros((Nvertices, 3))
        for _ in range(Nvertices):
            _v = V[_]
            mini_vertices[_] = vertices[_v]
        for _ in range(Nfaces):
            f = F[_]
            face = faces[f]
            mini_faces[_, 0] = V.index(face[0])
            mini_faces[_, 1] = V.index(face[1])
            mini_faces[_, 2] = V.index(face[2])
        return mini_vertices, mini_faces

    def get_psi_on_edges(self):
        pq = self.pq
        edges = self.edges
        Nedges = len(edges)
        psi = np.zeros((Nedges, 6))
        for e in range(Nedges):
            v0, v1 = edges[e]
            pq0, pq1 = pq[v0], pq[v1]
            pq0_inv = inv_se3_quaternion(pq0)
            pq = mul_se3_quaternion(pq0_inv, pq1)
            psi[e] = log_se3_quaternion(pq)
        return psi

    # def get_initial_psi(self):
    #     pq = self.pq
    #     Nverts = len(pq)
    #     edges = self.edges
    #     psi = np.zeros((Nverts, 2, 6))
    #     for v in range(Nverts):
    #
    #         # v0, v1 = edges[e]
    #         # pq0, pq1 = pq[v0], pq[v1]
    #         # pq0_inv = inv_se3_quaternion(pq0)
    #         # pq = mul_se3_quaternion(pq0_inv, pq1)
    #         # psi[e] = log_se3_quaternion(pq)
    #     return psi

    def next(self, ij):
        "faces=orbits of next map"
        return ij

    def mean_pose(self, vertex_list):
        """computes the SE3-valued mean of the euclidean transformations
        associated with vertices in vertex_list"""
        iters = 4
        v_initial = 0
        framed_vertices = self.pq

        Nsamps = len(vertex_list)
        mu_g = framed_vertices[vertex_list[v_initial]]

        for iter in range(iters):
            mu_g_inv = inv_se3_quaternion(mu_g)
            Psi = np.zeros(6)
            for i in vertex_list:
                g = framed_vertices[i]
                mu_g_inv_g = mul_se3_quaternion(mu_g_inv, g)
                Psi += log_se3_quaternion(mu_g_inv_g) / Nsamps
                # Psi += log_se3_quaternion(mul_se3_quaternion(g, mu_g_inv)) / Nsamps
            mu_g = mul_se3_quaternion(mu_g, exp_se3_quaternion(Psi))
            # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
        return mu_g

    # def regularize_mesh(self):
    #


Brane_spec = [
    ("vertices", float64[:, :]),
    ("faces", int32[:, :]),
    ("halfedges", int32[:, :]),
    ("V_index", int32[:]),
    ("V_hedge", int32[:]),
    ("H_index", int32[:]),
    ("H_vertex", int32[:]),
    ("H_face", int32[:]),
    ("H_next", int32[:]),
    ("H_prev", int32[:]),
    ("H_twin", int32[:]),
    ("F_index", int32[:]),
    ("F_hedge", int32[:]),
    ("pq", float64[:, :]),
    ("V_rgb", float64[:, :]),
    ("F_rgb", float64[:, :]),
    ("V_scalar", float64[:]),
    ("F_scalar", float64[:]),
    # ("name", str),
]


@jitclass(Brane_spec)
class Brane:
    def __init__(self, vertices, faces):
        # make fun to check counterclockwise faces
        # self.vertices = vertices
        # self.faces = faces
        # Nverts = len(vertices)
        # self.pq = np.zeros((Nverts, 7))
        # self.pq[:, :3] = vertices
        # self.name = name
        (
            self.V_index,
            self.V_hedge,
            self.halfedges,
            self.H_index,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_index,
            self.F_hedge,
        ) = self.get_halfedge_data(vertices, faces)
        self.faces = faces
        ##############

        self.pq = self.frame_the_mesh(vertices)
        self.V_scalar = self.get_Gaussian_curvature()

    def get_halfedge_data(self, vertices, faces):
        """
        faces must be ordered counter-clockwise!
        """
        ####################
        # vertices
        Nvertices = len(vertices)
        V_index = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        V_hedge = -np.ones_like(V_index)  # outgoing halfedge
        ####################
        # faces
        Nfaces = len(faces)
        F_index = np.array([_ for _ in range(Nfaces)], dtype=np.int32)
        F_hedge = -np.ones_like(F_index)  # one of the halfedges bounding it
        ####################
        # halfedges
        Nhedges = 3 * Nfaces
        halfedges_list = []
        H_index = np.array([_ for _ in range(Nhedges)], dtype=np.int32)
        H_vertex = -np.ones_like(H_index)  # vertex it points to
        H_face = -np.ones_like(H_index)  # face it belongs to  ***
        H_next = -np.ones_like(
            H_index
        )  # next halfedge inside the face (ordered counter-clockwise)
        H_prev = -np.ones_like(
            H_index
        )  # previous halfedge inside the face (ordered counter-clockwise)
        H_twin = -np.ones_like(H_index)  # opposite halfedge
        ####################

        # h = 0

        for f in F_index:
            face = faces[f]
            v0, v1, v2 = faces[f]
            hedge01 = [face[0], face[1]]
            hedge12 = [face[1], face[2]]
            hedge20 = [face[2], face[0]]

            hedge10 = [face[1], face[0]]
            hedge21 = [face[2], face[1]]
            hedge02 = [face[0], face[2]]

            try:  # is hedge01 indexed
                h01 = halfedges_list.index(hedge01)
                h10 = halfedges_list.index(hedge10)
            except Exception:
                halfedges_list.append(hedge01)
                halfedges_list.append(hedge10)
                h01 = halfedges_list.index(hedge01)
                h10 = halfedges_list.index(hedge10)

            try:  # is hedge12 indexed
                h12 = halfedges_list.index(hedge12)
                h21 = halfedges_list.index(hedge21)
            except Exception:
                halfedges_list.append(hedge12)
                halfedges_list.append(hedge21)
                h12 = halfedges_list.index(hedge12)
                h21 = halfedges_list.index(hedge21)

            try:  # is hedge20 indexed
                h20 = halfedges_list.index(hedge20)
                h02 = halfedges_list.index(hedge02)
            except Exception:
                halfedges_list.append(hedge20)
                halfedges_list.append(hedge02)
                h20 = halfedges_list.index(hedge20)
                h02 = halfedges_list.index(hedge02)

            # Each vertex references one outgoing halfedge, i.e. a halfedge that starts at this vertex.
            if V_hedge[v0] == -1:
                V_hedge[v0] = h01
            if V_hedge[v1] == -1:
                V_hedge[v1] = h12
            if V_hedge[v2] == -1:
                V_hedge[v2] = h20

            # Each face references one of the halfedges bounding it.
            F_hedge[f] = h01

            # Each halfedge provides a handle to...
            # the vertex it points to
            H_vertex[h01] = v1
            H_vertex[h12] = v2
            H_vertex[h20] = v0
            # the face it belongs to
            H_face[h01] = f
            H_face[h12] = f
            H_face[h20] = f
            # the next/previous halfedge inside the face (ordered counter-clockwise)
            H_next[h01] = h12
            H_next[h12] = h20
            H_next[h20] = h01
            H_prev[h01] = h20
            H_prev[h12] = h01
            H_prev[h20] = h12
            # the opposite halfedge
            H_twin[h01] = h10
            H_twin[h12] = h21
            H_twin[h20] = h02

        halfedges = np.array(halfedges_list, dtype=np.int32)
        return (
            V_index,
            V_hedge,
            halfedges,
            H_index,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            F_index,
            F_hedge,
        )

    def v_adjacent_to_v(self, v):
        """
        gets vertices adjacent to v in counterclockwise order
        """
        h_start = self.V_hedge[v]
        neighbors = []

        h = h_start
        while True:
            neighbors.append(self.H_vertex[h])
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return np.array(neighbors, dtype=np.int32)

    def f_adjacent_to_v(self, v):
        """
        gets faces adjacent to v in counterclockwise order
        """
        h_start = self.V_hedge[v]
        neighbors = []

        h = h_start
        while True:
            neighbors.append(self.H_face[h])
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return neighbors

    def get_face_area_vectors(self):
        F_index = self.F_index
        Nfaces = len(F_index)
        F_area_vectors = np.zeros((Nfaces, 3))
        vertices = self.pq[:, :3]

        for _f in range(Nfaces):
            f = F_index[_f]
            h = self.F_hedge[f]
            hn = self.H_next[h]
            hp = self.H_prev[h]

            v0 = self.H_vertex[hp]
            v1 = self.H_vertex[h]
            v2 = self.H_vertex[hn]

            u1 = vertices[v1] - vertices[v0]
            u2 = vertices[v2] - vertices[v1]

            F_area_vectors[_f] = jitcross(u1, u2)
        # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return F_area_vectors

    def get_vertex_normal(self, v):
        F = self.f_adjacent_to_v(v)
        n = np.zeros(3)

        for f in F:
            n += self.get_face_area_vector(f)

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    def frame_the_mesh(self, vertices):
        F_index = self.F_index
        Nfaces = len(F_index)
        F_area_vectors = np.zeros((Nfaces, 3))
        # vertices = self.pq[:, :3]

        for _f in range(Nfaces):
            f = F_index[_f]
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

        # vertices = self.pq[:, :3]

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

    def orthogonal_matrices(self):
        Q = self.pq[:, 3:]
        Nv = len(Q)
        R = np.zeros((Nv, 3, 3))
        for v in range(Nv):
            q = Q[v]
            R[v] = quaternion_to_matrix(q)
        return R

    def rcm_pose(self, vertex_list):
        """computes the SE3-valued Riemannian center of mass of the euclidean transformations associated with vertices in vertex_list"""
        iters = 4
        pq = self.pq
        Nsamps = len(vertex_list)
        G = np.zeros((Nsamps, 7))
        # G[:] = np.array([pq[i] for i in vertex_list])
        ##########################################
        g0 = pq[vertex_list[0]]  # G[0]
        g0_inv = inv_se3_quaternion(g0)
        for i in range(Nsamps):
            G[i] = mul_se3_quaternion(g0_inv, pq[vertex_list[i]])
        ##########################################
        mu_g = np.array([G[0, 0], G[0, 1], G[0, 2], G[0, 3], G[0, 4], G[0, 5], G[0, 6]])
        # mu_g = np.zeros(7)  # sum(G) / Nsamps
        # for g in G:
        #     mu_g += g / Nsamps
        # mu_g[3:] /= np.sqrt(mu_g[3] ** 2 + mu_g[4] ** 2 + mu_g[5] ** 2 + mu_g[6] ** 2)

        for iter in range(iters):
            mu_g_inv = inv_se3_quaternion(mu_g)
            Psi = np.zeros(6)
            for g in G:
                mu_g_inv_g = mul_se3_quaternion(mu_g_inv, g)
                Psi += log_se3_quaternion(mu_g_inv_g) / Nsamps
            mu_g = mul_se3_quaternion(mu_g, exp_se3_quaternion(Psi))
            # for g in G:
            #     g_mu_g_inv = mul_se3_quaternion(g, mu_g_inv)
            #     Psi += log_se3_quaternion(g_mu_g_inv) / Nsamps
            # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
        ##########################################
        mu_g = mul_se3_quaternion(g0, mu_g)
        ##########################################
        return mu_g

    def acm_pose(self, vertex_list):
        """computes the affine center of mass of the euclidean transformations associated with vertices in vertex_list"""
        G = self.pq
        Nsamps = len(vertex_list)
        g0 = G[vertex_list[0]]
        mu_g = np.zeros_like(g0)
        for i in vertex_list:
            g = G[i]
            mu_g += g / Nsamps
        mu_g[3:] /= np.sqrt(mu_g[3] ** 2 + mu_g[4] ** 2 + mu_g[5] ** 2 + mu_g[6] ** 2)
        return mu_g

    def rcm_quaternion(self, vertex_list):
        """computes the unit quaternion-valued Riemannian center of mass of the euclidean transformations associated with vertices in vertex_list"""
        iters = 4

        G = self.pq[:, 3:]
        Nsamps = len(vertex_list)
        g0 = G[vertex_list[0]]
        mu_g = np.zeros_like(g0)
        mu_g[:] = g0
        for iter in range(iters):
            mu_g_inv = inv_quaternion(mu_g)
            Psi = np.zeros(3)
            for i in vertex_list:
                g = G[i]
                mu_g_inv_g = mul_quaternion(mu_g_inv, g)
                Psi += log_unit_quaternion(mu_g_inv_g) / Nsamps
                # Psi += log_se3_quaternion(mul_se3_quaternion(g, mu_g_inv)) / Nsamps
            mu_g = mul_quaternion(mu_g, exp_unit_quaternion(Psi))
            # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
        return mu_g

    def rigid_transform(self, PQ):
        Nverts = len(self.pq)
        for i in range(Nverts):
            self.pq[i] = mul_se3_quaternion(PQ, self.pq[i])
        # self.vertices = self.pq[:, :3]

    def angle_defect(self, v):
        """
        2*pi - sum_f (angle_f)
        """
        p0 = self.pq[v, :3]
        h_start = self.V_hedge[v]
        defect = 2 * np.pi

        h = h_start
        v = self.H_vertex[h]
        e2 = self.pq[v, :3] - p0
        norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
        h = self.H_next[self.H_twin[h]]

        while True:
            e1 = e2
            norm_e1 = norm_e2
            v = self.H_vertex[h]  # 2nd vert
            e2 = self.pq[v, :3] - p0
            norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
            cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
                norm_e1 * norm_e2
            )
            defect -= np.arccos(cos_angle)

            h = self.H_next[self.H_twin[h]]
            if h == h_start:
                break

        return defect

    def get_angle_defects(self):
        """
        2*pi - sum_f (angle_f)
        """
        Nverts = len(self.pq)
        defects = np.zeros(Nverts)
        # V = self.V_index
        for v0 in range(Nverts):
            # p0 = self.pq[v, :3]
            h_start = self.V_hedge[v0]
            defects[v0] = 2 * np.pi

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
                cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
                    norm_e1 * norm_e2
                )
                defects[v0] -= np.arccos(cos_angle)

                h = self.H_next[self.H_twin[h]]
                if h == h_start:
                    break

        return defects

    def get_Gaussian_curvature(self):
        """
        2*pi - sum_f (angle_f)
        """
        Nverts = len(self.pq)
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

                defect -= angle
                area += 0.5 * norm_e1 * norm_e2 * np.sin(angle) / 3

                h = self.H_next[self.H_twin[h]]
                if h == h_start:
                    break
            K[v0] = defect / area

        return K

    def regularize_mesh_rcm(self):
        pq = np.zeros_like(self.pq)

        Nverts = len(pq)
        for v in range(Nverts):
            V = self.v_adjacent_to_v(v)
            # Nsamps = len(V)
            # G = np.array([PQ[_] for _ in V])
            pq[v] = self.rcm_pose(V)
        return pq

    def regularize_mesh_acm(self):
        pq = np.zeros_like(self.pq)

        Nverts = len(pq)
        for v in range(Nverts):
            V = self.v_adjacent_to_v(v)
            # Nsamps = len(V)
            # G = np.array([PQ[_] for _ in V])
            pq[v] = self.acm_pose(V)
        return pq

    def regularize_mesh_acm_quat(self):
        pq = np.zeros_like(self.pq)

        Nverts = len(pq)
        for v in range(Nverts):
            V = self.v_adjacent_to_v(v)
            # Nsamps = len(V)
            # G = np.array([PQ[_] for _ in V])
            pq[v] = self.acm_pose(V)
            pq[v, 3:] = self.rcm_quaternion(V)
        return pq


HalfEdgeMesh_spec = [
    ("vertices", float64[:, :]),
    ("faces", int32[:, :]),
    ("halfedges", int32[:, :]),
    ("V_index", int32[:]),
    ("V_hedge", int32[:]),
    ("H_index", int32[:]),
    ("H_vertex", int32[:]),
    ("H_face", int32[:]),
    ("H_next", int32[:]),
    ("H_prev", int32[:]),
    ("H_twin", int32[:]),
    ("F_index", int32[:]),
    ("F_hedge", int32[:]),
    ("pq", float64[:, :]),
]


# @jitclass(HalfEdgeMesh_spec)
class HalfEdgeMesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        (
            self.V_index,
            self.V_hedge,
            self.halfedges,
            self.H_index,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_index,
            self.F_hedge,
        ) = self.get_halfedge_data(vertices, faces)
        ##############
        self.pq = self.frame_the_mesh()

    def get_halfedge_data(self, vertices, faces):
        """
        faces must be ordered counter-clockwise!
        """
        ####################
        # vertices
        Nvertices = len(vertices)
        V_index = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        V_hedge = -np.ones_like(V_index)  # outgoing halfedge
        ####################
        # faces
        Nfaces = len(faces)
        F_index = np.array([_ for _ in range(Nfaces)], dtype=np.int32)
        F_hedge = -np.ones_like(F_index)  # one of the halfedges bounding it
        ####################
        # halfedges
        Nhedges = 3 * Nfaces
        halfedges_list = []
        H_index = np.array([_ for _ in range(Nhedges)], dtype=np.int32)
        H_vertex = -np.ones_like(H_index)  # vertex it points to
        H_face = -np.ones_like(H_index)  # face it belongs to  ***
        H_next = -np.ones_like(
            H_index
        )  # next halfedge inside the face (ordered counter-clockwise)
        H_prev = -np.ones_like(
            H_index
        )  # previous halfedge inside the face (ordered counter-clockwise)
        H_twin = -np.ones_like(H_index)  # opposite halfedge
        ####################

        # h = 0

        for f in F_index:
            face = faces[f]
            v0, v1, v2 = faces[f]
            hedge01 = [face[0], face[1]]
            hedge12 = [face[1], face[2]]
            hedge20 = [face[2], face[0]]

            hedge10 = [face[1], face[0]]
            hedge21 = [face[2], face[1]]
            hedge02 = [face[0], face[2]]

            try:  # is hedge01 indexed
                h01 = halfedges_list.index(hedge01)
                h10 = halfedges_list.index(hedge10)
            except Exception:
                halfedges_list.append(hedge01)
                halfedges_list.append(hedge10)
                h01 = halfedges_list.index(hedge01)
                h10 = halfedges_list.index(hedge10)

            try:  # is hedge12 indexed
                h12 = halfedges_list.index(hedge12)
                h21 = halfedges_list.index(hedge21)
            except Exception:
                halfedges_list.append(hedge12)
                halfedges_list.append(hedge21)
                h12 = halfedges_list.index(hedge12)
                h21 = halfedges_list.index(hedge21)

            try:  # is hedge20 indexed
                h20 = halfedges_list.index(hedge20)
                h02 = halfedges_list.index(hedge02)
            except Exception:
                halfedges_list.append(hedge20)
                halfedges_list.append(hedge02)
                h20 = halfedges_list.index(hedge20)
                h02 = halfedges_list.index(hedge02)

            # Each vertex references one outgoing halfedge, i.e. a halfedge that starts at this vertex.
            if V_hedge[v0] == -1:
                V_hedge[v0] = h01
            if V_hedge[v1] == -1:
                V_hedge[v1] = h12
            if V_hedge[v2] == -1:
                V_hedge[v2] = h20

            # Each face references one of the halfedges bounding it.
            F_hedge[f] = h01

            # Each halfedge provides a handle to...
            # the vertex it points to
            H_vertex[h01] = v1
            H_vertex[h12] = v2
            H_vertex[h20] = v0
            # the face it belongs to
            H_face[h01] = f
            H_face[h12] = f
            H_face[h20] = f
            # the next/previous halfedge inside the face (ordered counter-clockwise)
            H_next[h01] = h12
            H_next[h12] = h20
            H_next[h20] = h01
            H_prev[h01] = h20
            H_prev[h12] = h01
            H_prev[h20] = h12
            # the opposite halfedge
            H_twin[h01] = h10
            H_twin[h12] = h21
            H_twin[h20] = h02

        halfedges = np.array(halfedges_list, dtype=np.int32)
        return (
            V_index,
            V_hedge,
            halfedges,
            H_index,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            F_index,
            F_hedge,
        )

    def v_adjacent_to_v(self, v):
        """
        gets vertices adjacent to v in counterclockwise order
        """
        h_start = self.V_hedge[v]
        neighbors = []

        h = h_start
        while True:
            neighbors.append(self.H_vertex[h])
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return neighbors

    def f_adjacent_to_v(self, v):
        """
        gets faces adjacent to v in counterclockwise order
        """
        h_start = self.V_hedge[v]
        neighbors = []

        h = h_start
        while True:
            neighbors.append(self.H_face[h])
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return neighbors

    def get_face_area_vector(self, f):
        h = self.F_hedge[f]
        hn = self.H_next[h]
        hp = self.H_prev[h]

        v0 = self.H_vertex[hp]
        v1 = self.H_vertex[h]
        v2 = self.H_vertex[hn]

        u1 = self.vertices[v1] - self.vertices[v0]
        u2 = self.vertices[v2] - self.vertices[v1]

        A = jitcross(u1, u2)
        # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return A

    def get_face_area_vectors(self):
        F_index = self.F_index
        Nfaces = len(F_index)
        F_area_vectors = np.zeros((Nfaces, 3))

        for _f in range(Nfaces):
            f = F_index[_f]
            h = self.F_hedge[f]
            hn = self.H_next[h]
            hp = self.H_prev[h]

            v0 = self.H_vertex[hp]
            v1 = self.H_vertex[h]
            v2 = self.H_vertex[hn]

            u1 = self.vertices[v1] - self.vertices[v0]
            u2 = self.vertices[v2] - self.vertices[v1]

            F_area_vectors[_f] = jitcross(u1, u2)
        # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return F_area_vectors

    def get_vertex_normal(self, v):
        F = self.f_adjacent_to_v(v)
        n = np.zeros(3)

        for f in F:
            n += self.get_face_area_vector(f)

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    def frame_the_mesh(self):
        F_area_vectors = self.get_face_area_vectors()
        vertices = self.vertices

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
            R[:, 2] = n[i]
            framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
            framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])

            framed_vertices[i, :3] = vertices[i, :]
        return framed_vertices


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
