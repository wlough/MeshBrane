from numba import float64, int32, boolean
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

Brane_spec = [
    ("vertices", float64[:, :]),
    ("faces", int32[:, :]),
    ("halfedges", int32[:, :]),
    ("V_label", int32[:]),
    ("V_hedge", int32[:]),
    ("H_label", int32[:]),
    ("H_vertex", int32[:]),
    ("H_face", int32[:]),
    ("H_next", int32[:]),
    ("H_prev", int32[:]),
    ("H_twin", int32[:]),
    ("H_isboundary", boolean[:]),
    ("F_label", int32[:]),
    ("F_hedge", int32[:]),
    ("V_pq", float64[:, :]),
    ("V_rgb", float64[:, :]),
    ("F_rgb", float64[:, :]),
    ("V_scalar", float64[:]),
    ("F_scalar", float64[:]),
    # ("name", str),
    ("H_tangent_components", float64[:, :]),
    ("H_psi", float64[:, :]),
]


@jitclass(Brane_spec)
class Brane:
    def __init__(self, vertices, faces):
        # make fun to check counterclockwise faces
        # self.faces, self.F_label = self.check_faces(faces, vertices)
        (
            self.vertices,
            self.V_label,
            self.faces,
            self.F_label,
        ) = self.label_vertices_faces(vertices, faces)

        self.halfedges, self.H_label, self.H_isboundary = self.build_label_halfedges(
            vertices, faces
        )

        (
            self.V_hedge,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_hedge,
        ) = self.get_combinatorial_mesh_data()

        # self.vertices = vertices
        # self.faces = faces
        # Nverts = len(vertices)
        # self.V_pq = np.zeros((Nverts, 7))
        # self.V_pq[:, :3] = vertices
        # self.name = name
        # (
        #     self.V_label,
        #     self.V_hedge,
        #     self.halfedges,
        #     self.H_label,
        #     self.H_vertex,
        #     self.H_face,
        #     self.H_next,
        #     self.H_prev,
        #     self.H_twin,
        #     self.F_label,
        #     self.F_hedge,
        # ) = self.get_halfedge_data(vertices, faces)
        # (
        #     self.V_label,
        #     self.V_hedge,
        #     self.halfedges,
        #     self.H_label,
        #     self.H_vertex,
        #     self.H_face,
        #     self.H_next,
        #     self.H_prev,
        #     self.H_twin,
        #     self.H_isboundary,
        #     self.F_label,
        #     self.F_hedge,
        # ) = self.label_half_edges(vertices, faces)

        # print("get_halfedge_data")
        # self.faces = faces
        ##############

        self.V_pq = self.frame_the_mesh(vertices)
        # self.V_scalar = self.get_Gaussian_curvature()
        self.H_psi, self.H_tangent_components = self.get_initial_edge_tangents()

    def label_vertices_faces(self, vertices_in, faces_in):
        vertices = vertices_in.copy()
        faces = faces_in.copy()
        Nvertices = len(vertices)
        Nfaces = len(faces)
        V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)

        return vertices, V_label, faces, F_label

    def build_label_halfedges(self, vertices, faces):
        halfedges = []
        H_isboundary = []
        H_label = []  # np.array([_ for _ in range(Nhedges)], dtype=np.int32)
        interior_hedge_labels = set()
        ####################
        # save and label halfedges
        for face in faces:
            # face = faces[f]
            N_v_of_f = len(face)
            for _ in range(N_v_of_f):
                _next = np.mod(_ + 1, N_v_of_f)  # index shift to get next
                v0 = face[_]  #
                v1 = face[_next]
                hedge = [v0, v1]
                hedge_twin = [v1, v0]
                try:
                    h = halfedges.index(hedge)
                except Exception:
                    halfedges.append(hedge)
                    h = halfedges.index(hedge)
                interior_hedge_labels.add(h)
                # try:
                #     interior_hedge_labels.index(h)
                # except Exception:
                #     interior_hedge_labels.append(h)
                try:
                    halfedges.index(hedge_twin)
                except Exception:
                    halfedges.append(hedge_twin)

        Nhedges = len(halfedges)

        for h in range(Nhedges):
            H_isboundary.append(h not in interior_hedge_labels)
            H_label.append(h)

        return (
            np.array(halfedges, dtype=np.int32),
            np.array(H_label, dtype=np.int32),
            np.array(H_isboundary),
        )

    def get_combinatorial_mesh_data(self):
        V_label = self.V_label
        H_label = self.H_label
        F_label = self.F_label
        halfedges = self.halfedges
        halfedges_list = []  # halfedges.tolist()  # list(halfedges)
        for hedge in halfedges:
            v0, v1 = hedge
            halfedges_list.append([v0, v1])
        H_isboundary = self.H_isboundary
        faces = self.faces
        ####################
        # vertices
        V_hedge = -np.ones_like(V_label)  # outgoing halfedge
        ####################
        # faces
        F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
        ####################
        # halfedges
        H_vertex = -np.ones_like(H_label)  # vertex it points to
        H_face = -np.ones_like(H_label)  # face it belongs to  ***
        H_next = -np.ones_like(
            H_label
        )  # next halfedge inside the face (ordered counter-clockwise)
        H_prev = -np.ones_like(
            H_label
        )  # previous halfedge inside the face (ordered counter-clockwise)
        H_twin = -np.ones_like(H_label)  # opposite halfedge
        ####################

        # assign each face a halfedge
        # assign each interior halfedge previous/next halfedge
        # assign each interior halfedge a face
        for f in F_label:
            face = faces[f]
            N_v_of_f = len(face)
            hedge0 = [face[0], face[1]]
            h0 = halfedges_list.index(hedge0)
            F_hedge[f] = h0  # assign each face a halfedge
            for _ in range(N_v_of_f):
                _p1 = np.mod(_ + 1, N_v_of_f)  # index shift to get next
                _m1 = np.mod(_ - 1, N_v_of_f)  # index shift to get prev
                vm1 = face[_m1]
                v0 = face[_]  #
                vp1 = face[_p1]

                hedge = [v0, vp1]
                hedge_prev = [vm1, v0]
                h = halfedges_list.index(hedge)
                h_prev = halfedges_list.index(hedge_prev)
                H_prev[h] = h_prev  # assign previous/next halfedge
                H_next[h_prev] = h
                H_face[h] = f

        # assign each halfedge a twin halfedge
        # assign each halfedge a vertex
        # assign each vertex a halfedge
        # assign each boundary halfedge previous/next halfedge
        for h in H_label:
            v0, v1 = halfedges[h]
            hedge_twin = [v1, v0]
            h_twin = halfedges_list.index(hedge_twin)
            H_twin[h] = h_twin
            H_vertex[h] = v1
            if V_hedge[v0] == -1:
                V_hedge[v0] = h

            if H_isboundary[h]:
                h_t = h_twin
                h_tp = H_prev[h_t]

                h_tpt = H_twin[h_tp]
                h_tptp = H_prev[h_tpt]

                h_tptpt = H_twin[h_tptp]
                h_tptptp = H_prev[h_tptpt]

                h_next = H_twin[h_tptptp]
                H_next[h] = h_next
                H_prev[h_next] = h

        return (
            V_hedge,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            F_hedge,
        )

    # def reverse_face(self, f):
    #

    def label_half_edges(self, vertices, faces):
        """
        faces must be ordered counter-clockwise!
        """

        ####################
        # vertices
        Nvertices = len(vertices)
        V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        V_hedge = -np.ones_like(V_label)  # outgoing halfedge
        ####################
        # faces
        Nfaces = len(faces)
        F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)
        F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
        ####################
        # halfedges
        # Nedges =
        # Nhedges = 3 * Nfaces
        halfedges_list = []
        H_label = []  # np.array([_ for _ in range(Nhedges)], dtype=np.int32)
        H_vertex = []  # -np.ones_like(H_label)  # vertex it points to
        H_face = []  # -np.ones_like(H_label)  # face it belongs to  ***
        H_next = []  # -np.ones_like(
        # H_label
        # )  # next halfedge inside the face (ordered counter-clockwise)
        H_prev = []  # -np.ones_like(
        # H_label
        # )  # previous halfedge inside the face (ordered counter-clockwise)
        H_twin = []  # -np.ones_like(H_label)  # opposite halfedge
        # H_isboundary = []
        ####################

        # save halfedges that are in the faces
        for f in F_label:
            face = faces[f]
            N_v_of_f = len(face)
            h0 = len(
                halfedges_list
            )  # halfedges_list.__len__()  # label of 1st hedge in face
            F_hedge[f] = h0  #

            for _ in range(N_v_of_f):
                _next = np.mod(_ + 1, N_v_of_f)  # index shift to get next
                _prev = np.mod(_ - 1, N_v_of_f)  # index shift to get prev
                v0 = face[_]  #
                v1 = face[_next]
                h = h0 + _
                h_prev = h0 + _prev
                h_next = h0 + _next
                hedge = [v0, v1]
                halfedges_list.append(hedge)
                H_label.append(h)
                H_vertex.append(v1)
                H_face.append(f)
                H_prev.append(h_prev)
                H_next.append(h_next)
                if V_hedge[v0] == -1:
                    V_hedge[v0] = h

        # save halfedges that are on the boundaries
        Ninterior_hedges = len(halfedges_list)  # halfedges_list.__len__()
        H_isboundary = Ninterior_hedges * [False]
        for h in range(Ninterior_hedges):
            hedge = halfedges_list[h]
            hedge_twin = [hedge[1], hedge[0]]
            try:
                h_twin = halfedges_list.index(hedge_twin)
                H_twin.append(h_twin)
            except Exception:
                halfedges_list.append(hedge_twin)
                h_twin = halfedges_list.index(hedge_twin)
                H_twin.append(h_twin)

        N_hedges = len(halfedges_list)  # halfedges_list.__len__()
        for h in range(Ninterior_hedges, N_hedges):
            hedge = halfedges_list[h]
            hedge_twin = [hedge[1], hedge[0]]
            h_twin = halfedges_list.index(hedge_twin)
            H_twin.append(h_twin)
            H_isboundary.append(True)

            hh = h_twin
            hh = H_prev[hh]

            hh = H_twin[hh]
            hh = H_prev[hh]

            hh = H_twin[hh]
            hh = H_prev[hh]

            # h_next = H_twin[hh]
            hhedge = halfedges_list[hh]
            hedge_next = [hhedge[1], hhedge[0]]
            h_next = halfedges_list.index(hedge_next)

            H_next.append(h_next)

            hh = H_next[h_twin]

            hh = H_twin[hh]
            hh = H_next[hh]

            hh = H_twin[hh]
            hh = H_next[hh]

            # h_prev = H_twin[hh]
            hhedge = halfedges_list[hh]
            hedge_prev = [hhedge[1], hhedge[0]]
            h_prev = halfedges_list.index(hedge_prev)
            H_prev.append(h_prev)

        halfedges = np.array(halfedges_list, dtype=np.int32)
        return (
            V_label,
            V_hedge,
            halfedges,
            np.array(H_label, dtype=np.int32),
            np.array(H_vertex, dtype=np.int32),
            np.array(H_face, dtype=np.int32),
            np.array(H_next, dtype=np.int32),
            np.array(H_prev, dtype=np.int32),
            np.array(H_twin, dtype=np.int32),
            np.array(H_isboundary),
            F_label,
            F_hedge,
        )

    def get_halfedge_data(self, vertices, faces):
        """
        faces must be ordered counter-clockwise!
        """
        ####################
        # vertices
        Nvertices = len(vertices)
        V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        V_hedge = -np.ones_like(V_label)  # outgoing halfedge
        ####################
        # faces
        Nfaces = len(faces)
        F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)
        F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
        ####################
        # halfedges
        Nhedges = 3 * Nfaces
        halfedges_list = []
        H_label = np.array([_ for _ in range(Nhedges)], dtype=np.int32)
        H_vertex = -np.ones_like(H_label)  # vertex it points to
        H_face = -np.ones_like(H_label)  # face it belongs to  ***
        H_next = -np.ones_like(
            H_label
        )  # next halfedge inside the face (ordered counter-clockwise)
        H_prev = -np.ones_like(
            H_label
        )  # previous halfedge inside the face (ordered counter-clockwise)
        H_twin = -np.ones_like(H_label)  # opposite halfedge
        ####################

        # h = 0

        for f in F_label:
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
            V_label,
            V_hedge,
            halfedges,
            H_label,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            F_label,
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
        F_label = self.F_label
        Nfaces = len(F_label)
        F_area_vectors = np.zeros((Nfaces, 3))
        vertices = self.V_pq[:, :3]

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

    def orthogonal_matrices(self):
        Q = self.V_pq[:, 3:]
        Nv = len(Q)
        R = np.zeros((Nv, 3, 3))
        for v in range(Nv):
            q = Q[v]
            R[v] = quaternion_to_matrix(q)
        return R

    def rcm_pose(self, vertex_list):
        """computes the SE3-valued Riemannian center of mass of the euclidean transformations associated with vertices in vertex_list"""
        iters = 4
        pq = self.V_pq
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
        G = self.V_pq
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

        G = self.V_pq[:, 3:]
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
        Nverts = len(self.V_pq)
        for i in range(Nverts):
            self.V_pq[i] = mul_se3_quaternion(PQ, self.V_pq[i])
        # self.vertices = self.V_pq[:, :3]

    def angle_defect(self, v):
        """
        2*pi - sum_f (angle_f)
        """
        p0 = self.V_pq[v, :3]
        h_start = self.V_hedge[v]
        defect = 2 * np.pi

        h = h_start
        v = self.H_vertex[h]
        e2 = self.V_pq[v, :3] - p0
        norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
        h = self.H_next[self.H_twin[h]]

        while True:
            e1 = e2
            norm_e1 = norm_e2
            v = self.H_vertex[h]  # 2nd vert
            e2 = self.V_pq[v, :3] - p0
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
        Nverts = len(self.V_pq)
        defects = np.zeros(Nverts)
        # V = self.V_label
        for v0 in range(Nverts):
            # p0 = self.V_pq[v, :3]
            h_start = self.V_hedge[v0]
            defects[v0] = 2 * np.pi

            h = h_start
            v = self.H_vertex[h]
            e2 = self.V_pq[v, :3] - self.V_pq[v0, :3]
            norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
            h = self.H_next[self.H_twin[h]]

            while True:
                e1 = e2
                norm_e1 = norm_e2
                v = self.H_vertex[h]  # 2nd vert
                e2 = self.V_pq[v, :3] - self.V_pq[v0, :3]
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
        Nverts = len(self.V_pq)
        # defects = np.zeros(Nverts)
        K = np.zeros(Nverts)
        # V = self.V_label
        for v0 in range(Nverts):
            # p0 = self.V_pq[v, :3]
            h_start = self.V_hedge[v0]
            defect = 2 * np.pi
            area = 0.0

            h = h_start
            v = self.H_vertex[h]
            e2 = self.V_pq[v, :3] - self.V_pq[v0, :3]
            norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
            h = self.H_next[self.H_twin[h]]

            while True:
                e1 = e2
                norm_e1 = norm_e2
                v = self.H_vertex[h]  # 2nd vert
                e2 = self.V_pq[v, :3] - self.V_pq[v0, :3]
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
        pq = np.zeros_like(self.V_pq)

        Nverts = len(pq)
        for v in range(Nverts):
            V = self.v_adjacent_to_v(v)
            # Nsamps = len(V)
            # G = np.array([PQ[_] for _ in V])
            pq[v] = self.rcm_pose(V)
        return pq

    def regularize_mesh_acm(self):
        pq = np.zeros_like(self.V_pq)

        Nverts = len(pq)
        for v in range(Nverts):
            V = self.v_adjacent_to_v(v)
            # Nsamps = len(V)
            # G = np.array([PQ[_] for _ in V])
            pq[v] = self.acm_pose(V)
        return pq

    def regularize_mesh_acm_quat(self):
        pq = np.zeros_like(self.V_pq)

        Nverts = len(pq)
        for v in range(Nverts):
            V = self.v_adjacent_to_v(v)
            # Nsamps = len(V)
            # G = np.array([PQ[_] for _ in V])
            pq[v] = self.acm_pose(V)
            pq[v, 3:] = self.rcm_quaternion(V)
        return pq

    def face_normal(self, f):
        h0 = self.F_hedge[f]
        h1 = self.H_next[h0]
        h2 = self.H_next[h1]

        v0 = self.H_vertex[h0]
        v1 = self.H_vertex[h1]
        v2 = self.H_vertex[h2]

        e1 = self.V_pq[v1, :3] - self.V_pq[v0, :3]
        e2 = self.V_pq[v2, :3] - self.V_pq[v1, :3]

        e3 = jitcross(e1, e2)
        e3 /= np.sqrt(e3[0] ** 2 + e3[1] ** 2 + e3[2] ** 2)

        return e3

    def v_of_f(self, f):
        h_start = self.F_hedge[f]
        V = []

        h = h_start
        while True:
            V.append(self.H_vertex[h])
            h = self.H_next[h]
            if h == h_start:
                break

        return np.array(V, dtype=np.int32)

    def get_initial_edge_tangents(self):
        # H_label = self.H_label
        Nhedges = len(self.H_label)
        H_tangent_components = np.zeros((Nhedges, 2))
        H_psi = np.zeros((Nhedges, 6))

        for h in range(Nhedges):
            hp = self.H_prev[h]
            v0 = self.H_vertex[hp]
            v1 = self.H_vertex[h]
            pq0 = self.V_pq[v0]
            pq1 = self.V_pq[v1]
            pq0inv = inv_se3_quaternion(pq0)
            pq01 = mul_se3_quaternion(pq0inv, pq1)
            psi01 = log_se3_quaternion(pq01)

            ell01 = psi01[:3]
            the01 = psi01[3:]
            H_psi[h] = psi01
            H_tangent_components[h] = ell01[:2]

        return H_psi, H_tangent_components

    def build_minimesh_from_psi(self, v_c):
        pq_c = self.V_pq[v_c]  # center vertex pose

        poses = [[*pq_c]]
        faces = []

        h_start = self.V_hedge[v_c]
        h = h_start  # halfedge from center to boundary vertex
        v_new = 0
        while True:
            v_new += 1
            faces.append([0, v_new, v_new + 1])

            pq_c2b = exp_se3_quaternion(
                self.H_psi[h]
            )  # center to boundary transformation
            # pq_b = mul_se3_quaternion(pq_c, pq_c2b)  # boundary pose
            poses.append([*mul_se3_quaternion(pq_c, pq_c2b)])
            h = self.H_twin[self.H_prev[h]]  # halfedge from center to boundary vertex
            if h == h_start:
                break

        faces[-1][-1] = 1  # relabel vertex of last vertex of last face
        F = np.array(faces, dtype=np.int32)
        V_pq = np.array(poses, dtype=np.float64)
        return V_pq, F

    def build_submesh(self, v_c):
        # psi_c2b = np.zeros(6)
        # pq_c = self.V_pq[v_c]  # center vertex pose
        # V_label = [v_c]
        # V_hedge =
        # H_label =
        # H_vertex =
        # H_face =
        # H_next =
        # H_prev =
        # H_twin =
        # F_label = []
        # F_hedge =

        return 1
