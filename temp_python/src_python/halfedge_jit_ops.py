class CotanLaplacian(LaplaceTestOperator):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh

    def apply(self, Q, *args):
        """
        Computes the cotan Laplacian of Q at each vertex
        """
        Nv = self.mesh.num_vertices
        lapQ = np.zeros_like(Q)
        for vi in range(Nv):
            Atot = 0.0
            ri = self.mesh.xyz_coord_v(vi)
            qi = Q[vi]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            for hij in self.mesh.generate_H_out_v_clockwise(vi):
                hijm1 = self.mesh.h_next_h(self.mesh.h_twin_h(hij))
                hijp1 = self.mesh.h_twin_h(self.mesh.h_prev_h(hij))
                vjm1 = self.mesh.v_head_h(hijm1)
                vj = self.mesh.v_head_h(hij)
                vjp1 = self.mesh.v_head_h(hijp1)

                qj = Q[vj]

                rjm1 = self.mesh.xyz_coord_v(vjm1)
                rj = self.mesh.xyz_coord_v(vj)
                rjp1 = self.mesh.xyz_coord_v(vjp1)

                rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
                rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
                ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
                rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

                Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
                Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
                Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
                Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
                Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

                cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

                cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

                cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
                cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

                Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
                lapQ[vi] += (cot_thetam + cot_thetap) * (qj - qi) / 2
            lapQ[vi] /= Atot

        return lapQ


########################################################
from numba import njit, float64, int32, boolean, prange
from numba.experimental import jitclass
import numpy as np
from src.numdiff import (
    jitcross,
    jitdot,
    jitnorm,
    triprod,
    transpose_csr,
    quaternion_to_matrix,
    matrix_to_quaternion,
    inv_se3_quaternion,
    mul_se3_quaternion,
    log_se3_quaternion,
    exp_se3_quaternion,
    inv_quaternion,
    mul_quaternion,
    log_unit_quaternion,
    exp_unit_quaternion,
    rotate_by_quaternion,
    index_of_nested,
)

Brane_spec = [
    ###########################
    # Sim data
    ("V", float64[:, :, :]),
    ###########################
    # mesh data
    ("vertices", float64[:, :]),
    ("V_pq", float64[:, :]),
    ("V_hedge", int32[:]),
    ("halfedges", int32[:, :]),
    ("H_vertex", int32[:]),
    ("H_face", int32[:]),
    ("H_next", int32[:]),
    ("H_prev", int32[:]),
    ("H_twin", int32[:]),
    # ("H_isboundary", boolean[:]),
    ("faces", int32[:, :]),
    ("F_hedge", int32[:]),
    ###########################
    # physical parameters
    ("bending_modulus", float64),
    ("splay_modulus", float64),
    ("Kvolume", float64),
    ("Karea", float64),
    ("Klength", float64),
    ("linear_drag_coeff", float64),
    ("spontaneous_curvature", float64),
    ###########################
    # geometric stuff
    ("total_volume", float64),
    ("total_area", float64),
    ("preferred_total_volume", float64),
    ("preferred_total_area", float64),
    ("preferred_cell_area", float64),
    ("preferred_face_area", float64),
    ("preferred_cell_volume", float64),
    ("preferred_edge_length", float64),
    ("L_elements", float64[:]),
    ("L_indices", int32[:, :]),
    ("P", float64[:, :, :]),
    ("B", float64[:, :, :]),
    ("N", float64[:, :]),
    ###########################
    # visualization stuff
    ("V_vector_data", float64[:, :]),
    ("V_rgb", float64[:, :]),
    ("H_rgb", float64[:, :]),
    ("F_rgb", float64[:, :]),
    ("V_normal_rgb", float64[:, :]),
    ("V_tangent1_rgb", float64[:, :]),
    ("V_tangent2_rgb", float64[:, :]),
    ("V_radius", float64[:]),
    ("V_scalar", float64[:]),
    ("H_scalar", float64[:]),
    ("F_scalar", float64[:]),
    ("F_opacity", float64),
    ("H_opacity", float64),
    ("V_opacity", float64),
    ("F_alpha", float64[:]),
    ("H_alpha", float64[:]),
    ("V_alpha", float64[:]),
    ###########################
]


@jitclass(Brane_spec)
class Brane:
    def __init__(
        self,
        vertices,
        faces,
        length_reg_stiffness,
        area_reg_stiffness,
        volume_reg_stiffness,
        bending_modulus,
        splay_modulus,
        spontaneous_curvature,
        linear_drag_coeff,
        V_hedge=None,
        halfedges=None,
        H_vertex=None,
        H_face=None,
        H_next=None,
        H_prev=None,
        H_twin=None,
        F_hedge=None,
        # reinit,
    ):
        """to do: make fun to check counterclockwise faces

        Each vertex knows
        -one outgoing halfedge
            V_hedge
        Each face knows
        -one of its bounding halfedges
            F_hedge
        Each halfedge knows
        -the vertex it points to
            H_vertex
        -face it belongs to
            H_face
        -next halfedge inside the face (ordered counter-clockwise)
            H_next
        -previous halfedge inside the face (ordered counter-clockwise)
            H_prev
        -twin/opposite halfedge
            H_twin
        -if it's contained in the boundary of the mesh
            H_isboundary

        """
        ###########################
        if halfedges is None:
            (
                self.vertices,
                self.V_hedge,
                self.halfedges,
                self.H_vertex,
                self.H_face,
                self.H_next,
                self.H_prev,
                self.H_twin,
                self.faces,
                self.F_hedge,
            ) = self._get_combinatorial_mesh_data(vertices, faces)
        else:
            self.vertices = vertices.copy()
            self.V_hedge = V_hedge.copy()
            self.halfedges = halfedges.copy()
            self.H_vertex = H_vertex.copy()
            self.H_face = H_face.copy()
            self.H_next = H_next.copy()
            self.H_prev = H_prev.copy()
            self.H_twin = H_twin.copy()
            self.faces = faces.copy()
            self.F_hedge = F_hedge.copy()

        self.V_pq = self._frame_the_mesh(
            self.vertices,
            self.faces,
            self.halfedges,
            self.V_hedge,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_hedge,
        )
        Nv = len(self.V_pq)
        self.V = np.zeros((3, Nv, 3))
        self.V[0] = self.V_pq[:, :3]
        ###########################
        # physical parameters
        self.bending_modulus = bending_modulus
        self.splay_modulus = splay_modulus
        self.Kvolume = volume_reg_stiffness
        self.Karea = area_reg_stiffness
        self.Klength = length_reg_stiffness
        self.linear_drag_coeff = linear_drag_coeff
        self.spontaneous_curvature = spontaneous_curvature
        # self.params = params
        ###########################
        # geometric stuff
        (
            self.preferred_edge_length,
            self.preferred_cell_area,
            self.preferred_face_area,
            self.preferred_total_volume,
            self.preferred_total_area,
        ) = self._preferred_geometric_defaults(
            vertices,
            self.V_hedge,
            self.H_vertex,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.faces,
            self.F_hedge,
        )
        self.total_volume = self._total_volume(self.vertices, self.faces)
        self.total_area = self._total_area(
            self.vertices,
            self.V_hedge,
            self.H_vertex,
            self.H_prev,
            self.H_twin,
        )
        ###########################
        # visualization stuff
        (
            self.V_vector_data,
            self.V_rgb,
            self.V_normal_rgb,
            self.V_tangent1_rgb,
            self.V_tangent2_rgb,
            self.V_radius,
            self.H_rgb,
            self.F_rgb,
            self.F_opacity,
            self.H_opacity,
            self.V_opacity,
            self.F_scalar,
            self.H_scalar,
            self.V_scalar,
        ) = self._visual_defaults(self.vertices, self.halfedges, self.faces)
        self.F_alpha = self.F_opacity * np.ones(len(self.faces))
        self.H_alpha = self.H_opacity * np.ones(len(self.halfedges))
        self.V_alpha = self.V_opacity * np.ones(len(self.V_pq))
        #############################################################
        Nflips = self.delaunay_regularize_by_flips()

        tol = 1e-9
        self.L_elements, self.L_indices = self.get_meyer_weighted_heat_laplacian(tol)
        self.P = np.zeros((Nv, 3, 3))
        self.B = np.zeros((Nv, 3, 3))
        self.N = np.zeros((Nv, 3))

    ###########################################################################
    ###########################################################################
    # initialization functions #
    ############################
    # these don't call any other class functions and are safe to use before any
    # class attributes have been assigned
    ###########################################################################

    def _get_halfedges(self, vertices, faces):
        """Builds halfedges from vertices and faces and determines whether the halfedge is
        contained in the boundary of the mesh."""
        halfedges = []
        H_isboundary = []
        ####################
        # save and label halfedges
        h = 0
        for face in faces:
            N_v_of_f = len(face)
            for _ in range(N_v_of_f):
                # index shift to get next
                _next = (_ + 1) % N_v_of_f
                v0 = face[_]  #
                v1 = face[_next]
                hedge = [v0, v1]
                halfedges.append(hedge)
                H_isboundary.append(False)
                h += 1

        for hedge in halfedges:
            v0, v1 = hedge
            hedge_twin = [v1, v0]
            try:
                halfedges.index(hedge_twin)
            except Exception:
                halfedges.append(hedge_twin)
                H_isboundary.append(True)
                h += 1

        return (
            np.array(halfedges, dtype=np.int32),
            np.array(H_isboundary),
        )

    def _get_combinatorial_mesh_data(self, vertices_in, faces_in):
        """."""
        vertices = vertices_in.copy()
        faces = faces_in.copy()
        Nvertices = len(vertices)
        Nfaces = len(faces)
        halfedges, H_isboundary = self._get_halfedges(vertices, faces)
        Nhalfedges = len(halfedges)

        #
        # H_isboundary = self.H_isboundary
        # faces = self.faces.copy()
        ####################
        # vertices
        V_hedge = -np.ones(Nvertices, dtype=np.int32)  # outgoing halfedge
        ####################
        # faces
        F_hedge = -np.ones(Nfaces, dtype=np.int32)  # one of the halfedges bounding it
        ####################
        # halfedges
        H_vertex = -np.ones(Nhalfedges, dtype=np.int32)  # vertex it points to
        H_face = -np.ones_like(H_vertex)  # face it belongs to
        # next/previous halfedge inside the face (ordered counter-clockwise)
        H_next = -np.ones_like(H_vertex)
        H_prev = -np.ones_like(H_vertex)
        H_twin = -np.ones_like(H_vertex)  # opposite halfedge
        ####################

        # assign each face a halfedge
        # assign each interior halfedge previous/next halfedge
        # assign each interior halfedge a face
        # assign each halfedge a twin halfedge
        for f in range(Nfaces):
            face = faces[f]
            N_v_of_f = len(face)
            hedge0 = np.array([face[0], face[1]])
            h0 = index_of_nested(halfedges, hedge0)
            F_hedge[f] = h0  # assign each face a halfedge
            for _ in range(N_v_of_f):
                # for each vertex in face, get the indices of the
                # previous/next vertex
                _p1 = (_ + 1) % N_v_of_f
                _m1 = (_ - 1) % N_v_of_f
                vm1 = face[_m1]
                v0 = face[_]
                vp1 = face[_p1]
                # get outgoing halfedge
                hedge = np.array([v0, vp1])
                h = index_of_nested(halfedges, hedge)
                # get incident halfedge
                hedge_prev = np.array([vm1, v0])
                h_prev = index_of_nested(halfedges, hedge_prev)
                # assign previous/next halfedge
                H_prev[h] = h_prev
                H_next[h_prev] = h
                # assign face to halfedge
                H_face[h] = f

                hedge_twin = np.array([vp1, v0])
                h_t = index_of_nested(halfedges, hedge_twin)
                H_twin[h] = h_t
                H_twin[h_t] = h

        # assign each halfedge a vertex
        # assign each vertex a halfedge
        # assign each boundary halfedge previous/next halfedge
        for h in range(Nhalfedges):
            v0, v1 = halfedges[h]
            H_vertex[h] = v1
            if V_hedge[v0] == -1:
                V_hedge[v0] = h
            if H_isboundary[h]:
                h_next = H_twin[h]
                while True:
                    h_next = H_twin[H_prev[h_next]]
                    if H_isboundary[h_next]:
                        break
                H_next[h] = h_next
                H_prev[h_next] = h

        return (
            vertices,
            V_hedge,
            halfedges,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            faces,
            F_hedge,
        )

    def _frame_the_mesh(
        self,
        vertices,
        faces,
        halfedges,
        V_hedge,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        F_hedge,
    ):
        Nfaces = len(faces)
        F_area_vectors = np.zeros((Nfaces, 3))

        for f in range(Nfaces):
            h = F_hedge[f]
            hn = H_next[h]
            hp = H_prev[h]

            v0 = H_vertex[hp]
            v1 = H_vertex[h]
            v2 = H_vertex[hn]

            u1 = vertices[v1] - vertices[v0]
            u2 = vertices[v2] - vertices[v1]

            F_area_vectors[f] = jitcross(u1, u2)

        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        Nverts = len(vertices)
        V_pq = np.zeros((Nverts, 7))
        for i in range(Nverts):
            F = self._f_adjacent_to_v(i, V_hedge, H_face, H_prev, H_twin)
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
            V_pq[i, 3:] = matrix_to_quaternion(R)

            V_pq[i, :3] = vertices[i, :]
        return V_pq

    def _f_adjacent_to_v(self, v, V_hedge, H_face, H_prev, H_twin):
        """
        gets faces adjacent to v in counterclockwise order
        """
        h_start = V_hedge[v]
        neighbors = []

        h = h_start
        while True:
            neighbors.append(H_face[h])
            h = H_prev[h]
            h = H_twin[h]
            if h == h_start:
                break

        return neighbors

    def _average_cell_area(
        self,
        vertices,
        V_hedge,
        H_vertex,
        H_prev,
        H_twin,
    ):
        A = 0.0
        Nvertices = len(vertices)
        for v in range(Nvertices):
            r = vertices[v]
            Av = 0.0
            h = V_hedge[v]
            h_start = h
            while True:
                v1 = H_vertex[h]
                r1 = vertices[v1]
                h = H_prev[h]
                h = H_twin[h]
                v2 = H_vertex[h]
                r2 = vertices[v2]
                a = jitcross(r, r1) / 2 + jitcross(r1, r2) / 2 + jitcross(r2, r) / 2
                Av += np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) / 3

                if h == h_start:
                    break

            A += Av / Nvertices
        return A

    def _total_area(
        self,
        vertices,
        V_hedge,
        H_vertex,
        H_prev,
        H_twin,
    ):
        A = 0.0
        Nvertices = len(vertices)
        for v in range(Nvertices):
            r = vertices[v]
            Av = 0.0
            h = V_hedge[v]
            h_start = h
            while True:
                v1 = H_vertex[h]
                r1 = vertices[v1]
                h = H_prev[h]
                h = H_twin[h]
                v2 = H_vertex[h]
                r2 = vertices[v2]
                a = jitcross(r, r1) / 2 + jitcross(r1, r2) / 2 + jitcross(r2, r) / 2
                Av += np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) / 3

                if h == h_start:
                    break

            A += Av
        return A

    def _average_hedge_length(self, vertices, H_twin, H_vertex):
        L = 0.0
        Nh = len(H_twin)
        for h in range(Nh):
            ht = H_twin[h]
            v1 = H_vertex[ht]
            v2 = H_vertex[h]
            xyz1 = vertices[v1]
            xyz2 = vertices[v2]
            u = xyz2 - xyz1
            L += np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) / Nh
        return L

    def _average_signed_volume_of_faces(self, vertices, faces):
        vol = 0.0
        Nfaces = len(faces)
        for face in faces:
            v0, v1, v2 = face
            p0 = vertices[v0]
            p1 = vertices[v1]
            p2 = vertices[v2]
            vol_f = triprod(p0, p1, p2) / 6
            vol += vol_f / Nfaces
        return vol

    def _total_volume(self, vertices, faces):
        vol = 0.0
        for face in faces:
            v0, v1, v2 = face
            p0 = vertices[v0]
            p1 = vertices[v1]
            p2 = vertices[v2]
            vol_f = triprod(p0, p1, p2) / 6
            vol += vol_f
        return abs(vol)

    def _preferred_geometric_defaults(
        self,
        vertices,
        V_hedge,
        H_vertex,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ):
        Nf = len(faces)
        Nv = len(vertices)
        volume = self._total_volume(vertices, faces)
        area = self._total_area(
            vertices,
            V_hedge,
            H_vertex,
            H_prev,
            H_twin,
        )
        Rv = (3 * volume / (4 * np.pi)) ** (1 / 3)
        Ra = np.sqrt(area / (4 * np.pi))
        w = 0.75
        R = (1 - w) * Ra + w * Rv
        preferred_total_area = 4 * np.pi * R**2
        preferred_face_area = preferred_total_area / Nf
        preferred_cell_area = preferred_face_area * Nf / Nv
        preferred_total_volume = 4 * np.pi * R**3 / 3
        preferred_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))

        # preferred_edge_length = self._average_hedge_length(vertices, H_twin, H_vertex)
        # preferred_cell_area = self._average_cell_area(
        #     vertices,
        #     V_hedge,
        #     H_vertex,
        #     H_prev,
        #     H_twin,
        # )
        # preferred_total_volume = self._total_volume(vertices, faces)
        # preferred_total_area = self._total_area(
        #     vertices,
        #     V_hedge,
        #     H_vertex,
        #     H_prev,
        #     H_twin,
        # )
        return (
            preferred_edge_length,
            preferred_cell_area,
            preferred_face_area,
            preferred_total_volume,
            preferred_total_area,
        )

    def _visual_defaults(self, V, H, F):
        face_color = np.array([0.0, 0.63335, 0.05295])
        F_opacity = 0.8

        hedge_color = np.array([1.0, 0.498, 0.0])
        H_opacity = 1.0
        # hedge_radius = 0.0025

        vertex_color = np.array([1.0, 0.498, 0.0])  # np.array([0.7057, 0.0156, 0.1502])
        V_opacity = 1.0
        vertex_radius = 0.025

        normal_color = np.array([0.0, 0.0, 0.0])  # (1.0, 0.0, 0.0)
        tangent_color1 = np.array([0.7057, 0.0156, 0.1502])  # (1.0, 0.0, 0.0)
        tangent_color2 = np.array([0.2298, 0.2987, 0.7537])

        Nverts = len(V)
        Nhedges = len(H)
        Nfaces = len(F)

        V_rgb = np.zeros((Nverts, 3))
        V_normal_rgb = np.zeros((Nverts, 3))
        V_tangent1_rgb = np.zeros((Nverts, 3))
        V_tangent2_rgb = np.zeros((Nverts, 3))
        V_radius = np.zeros(Nverts)
        for _ in range(Nverts):
            V_rgb[_] = vertex_color
            V_normal_rgb[_] = normal_color
            V_tangent1_rgb[_] = tangent_color1
            V_tangent2_rgb[_] = tangent_color2
            V_radius[_] = vertex_radius
        H_rgb = np.zeros((Nhedges, 3))
        # H_radius = np.zeros(Nhedges)
        for _ in range(Nhedges):
            H_rgb[_] = hedge_color
            # H_radius[_] = hedge_radius
        F_rgb = np.zeros((Nfaces, 3))
        for _ in range(Nfaces):
            F_rgb[_] = face_color
        V_scalar = np.zeros(Nverts)
        H_scalar = np.zeros(Nhedges)
        F_scalar = np.zeros(Nfaces)
        V_vector_data = np.zeros((Nverts, 3))
        ##########################
        return (
            V_vector_data,
            V_rgb,
            V_normal_rgb,
            V_tangent1_rgb,
            V_tangent2_rgb,
            V_radius,
            H_rgb,
            # H_radius,
            F_rgb,
            F_opacity,
            H_opacity,
            V_opacity,
            F_scalar,
            H_scalar,
            V_scalar,
        )

    def _params(self):
        kBT = 0.2
        R = 9.0

    ###########################################################################
    # mesh navigation functions
    # using only H_twin/H_next/H_vertex/H_face, V_hedge, F_hedge
    ############################
    def face(self, f):
        """consistent with initial face assignment"""
        face = []
        h = self.H_twin[self.H_next[self.H_twin[self.F_hedge[f]]]]
        h_start = h
        while True:
            face.append(self.H_vertex[h])
            h = self.H_next[h]
            if h == h_start:
                break
        return np.array(face, dtype=np.int32)

    def halfedge(self, h):
        v1 = self.H_vertex[h]
        v0 = self.H_vertex[self.H_twin[h]]
        hedge = np.array([v0, v1], dtype=np.int32)
        return hedge

    def prev(self, h):
        h_next = h
        while True:
            h_prev = h_next
            h_next = self.H_next[h_prev]
            if h_next == h:
                break
        return h_prev

    ###########################################################################
    # helper functions
    ############################
    def V_one_ring_neighbors(self, i):
        """"""
        Vi = []
        h_start = self.V_hedge[i]
        hij = h_start
        while True:
            Vi.append(self.H_vertex[hij])
            hij = self.H_twin[self.H_prev[hij]]
            if hij == h_start:
                break
        return Vi

    ###########################################################################
    # timestepping
    ############################
    def timestep(self):
        Nv = len(self.V_pq)
        A = np.zeros(Nv)
        for v in range(Nv):
            A[v] = self.meyercell_area(v)

    def weighted_drag_coeffs_step(self, dt):
        Nv = len(self.V_pq)
        dVp = np.zeros((Nv, 3))
        Fl = self.Ftether()
        Fa, Fv = self.Fa_Fv()
        Fb = self.Fbend_mixed()
        F = Fb + Fl + Fa + Fv
        gamma = self.linear_drag_coeff
        for v in range(Nv):
            A = self.meyercell_area(v)
            drag_coeff = gamma * A
            dVp[v] = dt * F[v] / drag_coeff

        return dVp

    ###########################################################################
    # Forces
    ############################
    def Fa_Fv(self):
        Nv = len(self.V_pq)
        Fv = np.zeros((Nv, 3))
        V0 = self.preferred_total_volume
        V = self.volume_of_mesh()
        Kv = self.Kvolume
        Fa = np.zeros((Nv, 3))
        Ka = self.Karea
        A0 = self.preferred_face_area

        for i in range(Nv):
            ri = self.V_pq[i, :3]
            h_start = self.V_hedge[i]
            hk = h_start
            while True:
                vk = self.H_vertex[hk]
                hkp1 = self.H_twin[self.H_prev[hk]]
                vkp1 = self.H_vertex[hkp1]
                rk = self.V_pq[vk, :3]
                rkp1 = self.V_pq[vkp1, :3]
                Fv[i] += -Kv * ((V - V0) / V0) * jitcross(rk, rkp1)
                ############
                vecAikkp1 = (
                    jitcross(ri, rk) + jitcross(rk, rkp1) + jitcross(rkp1, ri)
                ) / 2
                Aikkp1 = np.sqrt(
                    vecAikkp1[0] ** 2 + vecAikkp1[1] ** 2 + vecAikkp1[2] ** 2
                )
                nikkp1 = vecAikkp1 / Aikkp1
                Fa[i] += -Ka * ((Aikkp1 - A0) / (2 * A0)) * jitcross(nikkp1, rkp1 - rk)
                ############
                hk = hkp1
                if hk == h_start:
                    break
        return Fa, Fv

    def Utether_OG(self, s, _a=None):
        l0 = self.preferred_edge_length
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.Klength
        Dl = 0.8 * l0
        normDs = np.abs(s - l0)
        if normDs > Dl / 4 and normDs < Dl / 2:
            # U = a * Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
            U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
        else:
            U = 0.0
        return U

    def Utether(self, s, _a=None):
        l0 = self.preferred_edge_length
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.Klength
        Dl = 0.8 * l0
        normDs = np.abs(s - l0)
        if normDs <= Dl / 4:
            U = 0.0
        elif normDs < Dl / 2:
            U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
        else:
            U = np.inf
        return U

    def Utether1(self, s, _a=None):
        l0 = self.preferred_edge_length
        Dl = 0.8 * l0
        lmin = l0 - Dl / 2
        lrep = l0 - Dl / 4
        latt = l0 + Dl / 4
        lmax = l0 + Dl / 2
        if _a is None:
            a = l0
        else:
            a = _a
        Kl = self.Klength
        if s > latt:
            Uatt = Kl * np.exp(a / (latt - s)) / (lmax - s)
        else:
            Uatt = 0.0
        if s < lrep:
            Urep = Kl * np.exp(a / (s - lrep)) / (s - lmin)
        else:
            Urep = 0.0

        U = Urep + Uatt
        return U

    def Ftether_OG(self, _a=None):
        Nv = len(self.V_pq)
        Fl = np.zeros((Nv, 3))
        l0 = self.preferred_edge_length
        Kl = self.Klength
        if _a is None:
            a = l0
        else:
            a = _a
        Dl = 0.8 * l0

        for i in range(Nv):
            ri = self.V_pq[i, :3]
            h_start = self.V_hedge[i]
            hk = h_start
            while True:
                vk = self.H_vertex[hk]
                rk = self.V_pq[vk, :3]
                Drki = ri - rk
                s = np.sqrt(Drki[0] ** 2 + Drki[1] ** 2 + Drki[2] ** 2)
                Ds = s - l0
                normDs = np.abs(Ds)
                if normDs > Dl / 4 and normDs < Dl / 2:
                    # U = a * Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    Fl[i] += -(
                        (1 / (Dl / 2 - normDs) + a / (normDs - Dl / 4) ** 2)
                        * U
                        * (Ds / normDs)
                        * Drki
                        / s
                    )
                else:
                    pass
                ############
                hk = self.H_twin[self.H_prev[hk]]
                if hk == h_start:
                    break
        return Fl

    def Ftether(self, _a=None):
        Nv = len(self.V_pq)
        Fl = np.zeros((Nv, 3))
        l0 = self.preferred_edge_length
        Kl = self.Klength
        if _a is None:
            a = l0
        else:
            a = _a
        Dl = 0.8 * l0

        for i in range(Nv):
            ri = self.V_pq[i, :3]
            h_start = self.V_hedge[i]
            hk = h_start
            while True:
                vk = self.H_vertex[hk]
                rk = self.V_pq[vk, :3]
                Drki = ri - rk
                s = np.sqrt(Drki[0] ** 2 + Drki[1] ** 2 + Drki[2] ** 2)
                Ds = s - l0
                normDs = np.abs(Ds)
                if normDs <= Dl / 4:
                    pass
                else:
                    U = Kl * np.exp(-a / (normDs - Dl / 4)) / (Dl / 2 - normDs)
                    Fl[i] += -(
                        (1 / (Dl / 2 - normDs) + a / (normDs - Dl / 4) ** 2)
                        * U
                        * (Ds / normDs)
                        * Drki
                        / s
                    )
                ############
                hk = self.H_twin[self.H_prev[hk]]
                if hk == h_start:
                    break
        return Fl

    def Fb(self, Nsmooth=0, _a=None):
        a = self.preferred_edge_length
        Kb = self.bending_modulus
        H, K = self.get_angle_weighted_arc_curvatures()
        H0 = self.spontaneous_curvature
        for _ in range(Nsmooth):
            H = self.gaussian_smooth_samples(H, 1, a)
            K = self.gaussian_smooth_samples(K, 1, a)
        Nv = len(H)
        lapH = self.cotan_laplacian(H)
        F = np.zeros((Nv, 3))

        Fn = -2 * Kb * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))

        for v in range(Nv):
            n = self.other_weighted_vertex_normal(v)
            Av = self.vorcell_area(v)
            F[v] = Fn[v] * n * Av
        return F

    def Fbend_mixed(self):
        Nv = len(self.V_pq)
        F = np.zeros((Nv, 3))
        Kb = self.bending_modulus
        K = self.get_gaussian_curvature_meyer()
        H = np.zeros_like(K)
        H0 = self.spontaneous_curvature
        Hn = self.slow_smoothed_laplacian(self.V_pq[:, :3]) / 2
        n = np.zeros((Nv, 3))
        A = np.zeros(Nv)
        for i in range(Nv):
            n[i] = self.other_weighted_vertex_normal(i)
            A[i] = self.meyercell_area(i)
            H[i] = n[i, 0] * Hn[i, 0] + n[i, 1] * Hn[i, 1] + n[i, 2] * Hn[i, 2]
        lapH = self.slow_smoothed_laplacian(H)

        Fn = -2 * Kb * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))

        for i in range(Nv):
            F[i] = Fn[i] * n[i] * A[i]
        return F

    ###########################################################################
    # Curvature #
    ##########################
    def gaussian_curvature_normal(self, v):
        Psi = np.zeros(3)
        ri = self.V_pq[v, :3]
        h_start = self.V_hedge[v]
        hj = h_start
        while True:
            hjp1 = self.H_twin[self.H_prev[hj]]
            hjm1 = self.H_next[self.H_twin[hj]]
            vj = self.H_vertex[hj]
            vjp1 = self.H_vertex[hjp1]
            vjm1 = self.H_vertex[hjm1]
            rjp1 = self.V_pq[vjp1, :3]
            rjm1 = self.V_pq[vjm1, :3]
            rj = self.V_pq[vj, :3]
            Drj = rj - ri
            Drjm1 = rjm1 - ri
            Drjp1 = rjp1 - ri
            Drjm1_cross_Drj = jitcross(Drjm1, Drj)
            Drj_cross_Drjp1 = jitcross(Drj, Drjp1)
            normDrj = np.sqrt(Drj[0] ** 2 + Drj[1] ** 2 + Drj[2] ** 2)
            normDrjm1_cross_Drj = np.sqrt(
                Drjm1_cross_Drj[0] ** 2
                + Drjm1_cross_Drj[1] ** 2
                + Drjm1_cross_Drj[2] ** 2
            )
            Drj_cross_Drjp1 = np.sqrt(
                Drj_cross_Drjp1[0] ** 2
                + Drj_cross_Drjp1[1] ** 2
                + Drj_cross_Drjp1[2] ** 2
            )
            sin_Dphi = (
                normDrj
                * triprod(Drjm1, Drj, Drjp1)
                / (normDrjm1_cross_Drj * Drj_cross_Drjp1)
            )
            Dphi = np.arcsin(sin_Dphi)
            Psi += Dphi * Drj / normDrj
            hj = hjp1
            if hj == h_start:
                break
        A = self.vorcell_area(v)
        Kn = Psi / (2 * A)
        return Kn

    def arc_curvature(self, h):
        hij = h
        hji = self.H_twin[h]
        vi = self.H_vertex[hji]
        vj = self.H_vertex[hij]
        ri = self.V_pq[vi, :3]
        rj = self.V_pq[vj, :3]
        # ni = self.area_weighted_vertex_normal(vi)
        # nj = self.area_weighted_vertex_normal(vj)
        ni = self.quat_normal_vector(vi)
        rij = rj - ri
        kappa = (
            2
            * (ni[0] * rij[0] + ni[1] * rij[1] + ni[2] * rij[2])
            / (rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2])
        )
        return kappa

    def get_angle_weighted_arc_curvatures(self):
        Nv = len(self.V_pq)
        H = np.zeros(Nv)
        K = np.zeros(Nv)
        for vi in range(Nv):
            ri = self.V_pq[vi, :3]
            # ni = self.quat_normal_vector(vi)
            ni = self.other_weighted_vertex_normal(vi)
            kappai = 0.0
            kappa2i = 0.0
            h_start = self.V_hedge[vi]
            hj = h_start
            while True:
                hjp1 = self.H_twin[self.H_prev[hj]]
                hjm1 = self.H_next[self.H_twin[hj]]
                vj = self.H_vertex[hj]
                vjp1 = self.H_vertex[hjp1]
                vjm1 = self.H_vertex[hjm1]
                rjp1 = self.V_pq[vjp1, :3]
                rjm1 = self.V_pq[vjm1, :3]
                rj = self.V_pq[vj, :3]
                # ni += jitcross(ri, rj)+jitcross(rj, rjp1)+jitcross(rjp1, ri)
                Drj = rj - ri
                Drj_dot_Drj = Drj[0] ** 2 + Drj[1] ** 2 + Drj[2] ** 2
                Drjm1 = rjm1 - ri
                Drjm1_dot_Drjm1 = Drjm1[0] ** 2 + Drjm1[1] ** 2 + Drjm1[2] ** 2
                Drjp1 = rjp1 - ri
                Drjp1_dot_Drjp1 = Drjp1[0] ** 2 + Drjp1[1] ** 2 + Drjp1[2] ** 2
                ni_dot_Drj = ni[0] * Drj[0] + ni[1] * Drj[1] + ni[2] * Drj[2]
                ni_dot_Drjp1 = ni[0] * Drjp1[0] + ni[1] * Drjp1[1] + ni[2] * Drjp1[2]
                ni_dot_Drjm1 = ni[0] * Drjm1[0] + ni[1] * Drjm1[1] + ni[2] * Drjm1[2]
                tjp1 = (Drjp1 - ni_dot_Drjp1 * ni) / np.sqrt(
                    Drjp1_dot_Drjp1 - ni_dot_Drjp1**2
                )
                tjm1 = (Drjm1 - ni_dot_Drjm1 * ni) / np.sqrt(
                    Drjm1_dot_Drjm1 - ni_dot_Drjm1**2
                )
                tjp1_dot_tjm1 = (
                    tjp1[0] * tjm1[0] + tjp1[1] * tjm1[1] + tjp1[2] * tjm1[2]
                )
                kappaj = 2 * ni_dot_Drj / Drj_dot_Drj
                Dthetaj = np.arccos(tjp1_dot_tjm1) / 2
                kappai += kappaj * Dthetaj / (2 * np.pi)
                kappa2i += kappaj**2 * Dthetaj / (2 * np.pi)

                hj = hjp1
                if hj == h_start:
                    break
            H[vi] = kappai
            K[vi] = 3 * kappai**2 - 2 * kappa2i

        return H, K

    def principal_curvatures(self, H, K):
        try:
            N = len(H)
            kappa = np.zeros((N, 2))
            for i in range(N):
                kappap = H[i] + np.sqrt(H[i] ** 2 - K[i])
                kappam = H[i] - np.sqrt(H[i] ** 2 - K[i])
                if abs(kappap) > abs(kappam):
                    kappa = np.array([kappap, kappam])
                else:
                    kappa = np.array([kappam, kappap])
        except:
            kappa = np.zeros(2)
            if abs(kappap) > abs(kappam):
                kappa = np.array([kappap, kappam])
            else:
                kappa = np.array([kappam, kappap])

        return kappa

    def get_meyer_masses(self):
        Nv = len(self.vertices)
        M = np.zeros(Nv)
        for v in range(Nv):
            M[v] = self.meyercell_area(v)
        return M

    def heat_laplacian_weight(self, x, y, h):
        X = self.V_pq[x, :3]
        Y = self.V_pq[y, :3]
        dxy = jitnorm(Y - X)
        Wxy = np.exp(-(dxy**2) / (4 * h)) / (4 * np.pi * h**2)
        return Wxy

    def get_meyer_weighted_heat_laplacian(self, tol=1e-9):

        index_list = []
        matrix_element_list = []
        Nv = len(self.V_pq)
        for x in range(Nv):
            Mx = self.meyercell_area(x)
            for y in range(Nv):
                My = self.meyercell_area(y)
                Wxy = self.heat_laplacian_weight(x, y, Mx)
                Lxy = Wxy * My
                if Lxy > tol:
                    index_list.append([x, y])
                    matrix_element_list.append(Lxy)

        return np.array(matrix_element_list), np.array(index_list, dtype=np.int32)

    def get_heat_laplacian(self, h, tol=1e-9):

        index_list = []
        matrix_element_list = []
        Nv = len(self.V_pq)
        for x in range(Nv):
            for y in range(Nv):
                My = self.meyercell_area(y)
                Wxy = self.heat_laplacian_weight(x, y, h)
                Lxy = Wxy * My
                if Lxy > tol:
                    index_list.append([x, y])
                    matrix_element_list.append(Lxy)

        return np.array(matrix_element_list), np.array(index_list, dtype=np.int32)

    def get_TM_projection(self):
        Nv = len(self.V_pq)

        P = np.zeros((Nv, 3, 3))
        for _ in range(len(self.L_elements)):
            x, y = self.L_indices[_]
            Lxy = self.L_elements[_]
            rx, ry = self.V_pq[x, :3], self.V_pq[y, :3]
            P[x] += 0.5 * Lxy * np.outer(ry - rx, ry - rx)
        return P

    def get_other_weighted_vertex_normals(self):
        """Weights for Computing Vertex Normals from Facet Normals Max99"""
        Nvertices = len(self.V_pq)
        N = np.zeros((Nvertices, 3))
        for x in range(Nvertices):
            N[x] = self.other_weighted_vertex_normal(x)
        return N

    def get_heat_unit_normals(self):
        Nvertices = len(self.V_pq)
        N = np.zeros((Nvertices, 3))
        P = self.P
        for x in range(Nvertices):
            D, U = np.linalg.eigh(P[x])
            N[x] = U[:, 0]
        return N

    def get_curvature_matrix(self):
        Nv = len(self.V_pq)

        B = np.zeros((Nv, 3, 3))
        for _ in range(len(self.L_elements)):
            x, y = self.L_indices[_]
            Lxy = self.L_elements[_]
            rx, ry = self.V_pq[x, :3], self.V_pq[y, :3]
            nx, ny = self.N[x], self.N[y]
            B[x] += -0.5 * Lxy * np.outer(ry - rx, ny - nx)
        return B

    def flip_normals(self):
        Nv = len(self.V_pq)

        for x in range(Nv):
            rx = self.V_pq[x, :3]
            nx = self.N[x]
            # nx_dot_stuff = 0.0
            hxy1 = self.V_hedge[x]
            hxy2 = self.H_twin[self.H_prev[hxy1]]
            y1 = self.H_vertex[hxy1]
            y2 = self.H_vertex[hxy2]
            ry1 = self.V_pq[y1, :3]
            ry2 = self.V_pq[y2, :3]
            n_approx = jitcross(rx, ry1) + jitcross(ry1, ry2) + jitcross(ry2, rx)
            nx_dot_n_approx = jitdot(nx, n_approx)
            if nx_dot_n_approx < 0:
                self.N[x] *= -1

    ###########################################################################
    # cotan stuff #
    ##########################
    def meyercell_area(self, v):
        """Meyer's mixed area of cell dual to vertex v"""
        Atot = 0.0
        ri = self.V_pq[v, :3]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        h_start = self.V_hedge[v]
        hij = h_start
        while True:
            vj = self.H_vertex[hij]
            rj = self.V_pq[vj, :3]
            hijp1 = self.H_twin[self.H_prev[hij]]
            vjp1 = self.H_vertex[hijp1]
            rjp1 = self.V_pq[vjp1, :3]

            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
            rjp1_ri = rjp1[0] * ri[0] + rjp1[1] * ri[1] + rjp1[2] * ri[2]

            normDrij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
            # normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
            normDrjjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
            normDrjp1i = np.sqrt(rjp1_rjp1 - 2 * rjp1_ri + ri_ri)
            cos_thetajijp1 = (ri_ri + rj_rjp1 - ri_rj - rjp1_ri) / (
                normDrij * normDrjp1i
            )
            cos_thetajp1ji = (rj_rj + rjp1_ri - rj_rjp1 - ri_rj) / (
                normDrij * normDrjjp1
            )
            cos_thetaijp1j = (rjp1_rjp1 + ri_rj - rj_rjp1 - rjp1_ri) / (
                normDrjp1i * normDrjjp1
            )
            if cos_thetajijp1 < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += (
                    np.sqrt(
                        semiP
                        * (semiP - normDrij)
                        * (semiP - normDrjjp1)
                        * (semiP - normDrjp1i)
                    )
                    / 2
                )
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 4
            elif cos_thetajp1ji < 0 or cos_thetaijp1j < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += (
                    np.sqrt(
                        semiP
                        * (semiP - normDrij)
                        * (semiP - normDrjjp1)
                        * (semiP - normDrjp1i)
                    )
                    / 4
                )
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 8
            else:
                cot_thetaijp1j = cos_thetaijp1j / np.sqrt(1 - cos_thetaijp1j**2)
                cot_thetajp1ji = cos_thetajp1ji / np.sqrt(1 - cos_thetajp1ji**2)
                Atot += (
                    normDrij**2 * cot_thetaijp1j / 8
                    + normDrjp1i**2 * cot_thetajp1ji / 8
                )

            hij = hijp1
            if hij == h_start:
                break

        return Atot

    def vorcell_area(self, v):
        """area of cell dual to vertex v"""
        Atot = 0.0
        r = self.V_pq[v, :3]
        r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_twin[self.H_prev[h]]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]

            r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
            r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
            r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
            r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

            normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
            normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
            normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
            cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
            cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)

            cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
            cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
            Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8

            if h == h_start:
                break

        return Atot

    def cotan_laplacian(self, Y):
        """computes the laplacian of Y at each vertex"""
        Nv = self.V_pq.shape[0]
        lapY = np.zeros_like(Y)
        for vi in range(Nv):
            Atot = 0.0
            ri = self.V_pq[vi, :3]
            yi = Y[vi]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            h_start = self.V_hedge[vi]
            hij = h_start
            while True:
                hijm1 = self.H_next[self.H_twin[hij]]
                hijp1 = self.H_twin[self.H_prev[hij]]
                vjm1 = self.H_vertex[hijm1]
                vj = self.H_vertex[hij]
                vjp1 = self.H_vertex[hijp1]

                yj = Y[vj]

                rjm1 = self.V_pq[vjm1, :3]
                rj = self.V_pq[vj, :3]
                rjp1 = self.V_pq[vjp1, :3]

                rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
                rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
                ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
                rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

                Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
                Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
                Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
                Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
                Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

                cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

                cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

                cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
                cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

                Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
                lapY[vi] += (cot_thetam + cot_thetap) * (yj - yi) / 2

                hij = hijp1

                if hij == h_start:
                    break
            lapY[vi] /= Atot

        return lapY

    def smoothed_laplacian(self, Y):
        """computes the laplacian of Y at each vertex"""

        Nv = len(self.V_pq)
        lapY = np.zeros_like(Y)
        for vi in range(Nv):
            ri = self.V_pq[vi, :3]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            Ai = self.meyercell_area(vi)
            h_start = self.V_hedge[vi]
            hij = h_start
            while True:
                vj = self.H_vertex[hij]
                hijm1 = self.H_next[self.H_twin[hij]]
                hijp1 = self.H_twin[self.H_prev[hij]]
                vjm1 = self.H_vertex[hijm1]
                vjp1 = self.H_vertex[hijp1]
                rjm1 = self.V_pq[vjm1, :3]
                rj = self.V_pq[vj, :3]
                rjp1 = self.V_pq[vjp1, :3]
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                vecAijjp1 = (
                    jitcross(ri, rj) + jitcross(rj, rjp1) + jitcross(rjp1, ri)
                ) / 2
                vecAijm1j = (
                    jitcross(ri, rjm1) + jitcross(rjm1, rj) + jitcross(rj, ri)
                ) / 2
                Aijjp1 = np.sqrt(
                    vecAijjp1[0] ** 2 + vecAijjp1[1] ** 2 + vecAijjp1[2] ** 2
                )
                Aijm1j = np.sqrt(
                    vecAijm1j[0] ** 2 + vecAijm1j[1] ** 2 + vecAijm1j[2] ** 2
                )
                lapY[vi] += (
                    (Aijjp1 + Aijm1j)
                    * np.exp(-(ri_ri - 2 * ri_rj + rj_rj) / (4 * Ai))
                    * (Y[vj] - Y[vi])
                    / (12 * np.pi * Ai**2)
                )
                hij = hijp1
                if hij == h_start:
                    break
        return lapY

    def slow_smoothed_laplacian(self, Y):
        """computes the laplacian of Y at each vertex"""

        Nv = len(self.V_pq)
        Nf = len(self.faces)
        lapY = np.zeros_like(Y)
        for vi in range(Nv):
            ri = self.V_pq[vi, :3]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            Ai = self.meyercell_area(vi)
            for f in range(Nf):
                vj, vk, vl = self.faces[f]
                rj = self.V_pq[vj, :3]
                rk = self.V_pq[vk, :3]
                rl = self.V_pq[vl, :3]
                vecAf = (jitcross(rj, rk) + jitcross(rk, rl) + jitcross(rl, rj)) / 2
                Af = np.sqrt(vecAf[0] ** 2 + vecAf[1] ** 2 + vecAf[2] ** 2)
                for vm in self.faces[f]:
                    rm = self.V_pq[vm, :3]
                    rm_rm = rm[0] ** 2 + rm[1] ** 2 + rm[2] ** 2
                    ri_rm = ri[0] * rm[0] + ri[1] * rm[1] + ri[2] * rm[2]
                    Drim_sqr = ri_ri - 2 * ri_rm + rm_rm
                    lapY[vi] += (
                        (Af / (12 * np.pi * Ai**2))
                        * np.exp(-Drim_sqr / (4 * Ai))
                        * (Y[vm] - Y[vi])
                    )

        return lapY

    def mean_curvature_vector_cot(self, v):
        Atot = 0.0
        r = self.V_pq[v, :3]
        Hvec = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v0 = self.H_vertex[h]
            r0 = self.V_pq[v0, :3]
            vm1 = self.H_vertex[self.H_next[self.H_twin[h]]]
            rm1 = self.V_pq[vm1, :3]
            h = self.H_twin[self.H_prev[h]]
            vp1 = self.H_vertex[h]
            rp1 = self.V_pq[vp1, :3]

            ua1 = r0 - rm1
            ua2 = r - rm1
            alpha = np.arccos(jitdot(ua1, ua2) / (jitnorm(ua1) * jitnorm(ua2)))

            ub1 = r - rp1
            ub2 = r0 - rp1
            beta = np.arccos(jitdot(ub1, ub2) / (jitnorm(ub1) * jitnorm(ub2)))

            Hvec += (1 / np.tan(alpha) + 1 / np.tan(beta)) * (r0 - r) / 2
            Atot += (1 / np.tan(alpha) + 1 / np.tan(beta)) * jitdot(r0 - r, r0 - r) / 8

            if h == h_start:
                break
        return Hvec / (2 * Atot)

    def mean_curvature_cot(self, v):
        Atot = 0.0
        r = self.V_pq[v, :3]
        Hvec = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v0 = self.H_vertex[h]
            r0 = self.V_pq[v0, :3]
            vm1 = self.H_vertex[self.H_next[self.H_twin[h]]]
            rm1 = self.V_pq[vm1, :3]
            h = self.H_twin[self.H_prev[h]]
            vp1 = self.H_vertex[h]
            rp1 = self.V_pq[vp1, :3]

            ua1 = r0 - rm1
            ua2 = r - rm1
            alpha = np.arccos(jitdot(ua1, ua2) / (jitnorm(ua1) * jitnorm(ua2)))

            ub1 = r - rp1
            ub2 = r0 - rp1
            beta = np.arccos(jitdot(ub1, ub2) / (jitnorm(ub1) * jitnorm(ub2)))

            Hvec += (1 / np.tan(alpha) + 1 / np.tan(beta)) * (r0 - r) / 2
            Atot += (1 / np.tan(alpha) + 1 / np.tan(beta)) * jitdot(r0 - r, r0 - r) / 8

            if h == h_start:
                break

        Hvec /= 2 * Atot
        n = self.area_weighted_vertex_normal(v)
        H = jitdot(n, Hvec)
        return H

    def mean_curvature(self, v):
        Atot = 0.0
        r = self.V_pq[v, :3]
        r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
        Hvec = np.zeros(3)
        n = np.zeros(3)
        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_twin[self.H_prev[h]]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]

            r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
            r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
            r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
            r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
            # u1_u1 = r1_r1 - 2 * r_r1 + r_r
            # u2_u2 = r2_r2 - 2 * r1_r2 + r1_r1
            # u3_u3 = r_r - 2 * r2_r + r2_r2

            # u1 = r1 - r
            # u2 = r2 - r1
            # u3 = r - r2

            normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
            normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
            normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
            cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
            cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
            # sin_alpha = np.sqrt(1-cos_alpha**2)
            # sin_beta = np.sqrt(1-cos_beta**2)
            cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
            cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
            # alpha = np.arccos(jitdot(u2, u1) / (normu2 * normu1))
            # beta = np.arccos(jitdot(u3, u2) / (normu3 * normu2))

            Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
            Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
            n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)
            # Atot += (1 / np.tan(alpha) + 1 / np.tan(beta)) * jitdot(r - r0, r - r0) / 8

            if h == h_start:
                break

        Hvec /= 2 * Atot
        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        # n = self.area_weighted_vertex_normal(v)
        H = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
        return H

    def get_mean_curvature(self):
        """ """
        Nv = self.V_pq.shape[0]
        H = np.zeros(Nv)

        for v in range(Nv):
            Atot = 0.0
            r = self.V_pq[v, :3]
            r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            Hvec = np.zeros(3)
            n = np.zeros(3)
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_twin[self.H_prev[h]]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
                r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
                r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
                r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
                r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

                normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
                normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
                normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
                cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
                cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
                cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
                cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)

                Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
                Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
                n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)

                if h == h_start:
                    break

            Hvec /= 2 * Atot
            n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
        return H

    def get_mean_curvature_vector(self):
        """ """
        Nv = self.V_pq.shape[0]
        Hvec = np.zeros((Nv, 3))

        for v in range(Nv):
            Atot = 0.0
            r = self.V_pq[v, :3]
            r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            # Hvec = np.zeros(3)
            # n = np.zeros(3)
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_twin[self.H_prev[h]]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
                r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
                r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
                r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
                r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

                normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
                normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
                normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
                cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
                cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
                cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
                cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)

                Hvec[v] += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
                Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
                # n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)

                if h == h_start:
                    break

            Hvec[v] /= 2 * Atot
            # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            # H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
        return Hvec

    def get_gaussian_curvature(self):
        """gaussian curvature using voroni cell areas"""
        Nv = self.V_pq.shape[0]
        K = np.zeros(Nv)

        for v in range(Nv):
            Atot = 0.0
            r = self.V_pq[v, :3]
            r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            defect = 2 * np.pi
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_twin[self.H_prev[h]]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
                r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
                r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
                r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
                r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

                normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
                normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
                normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
                cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
                cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
                cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
                cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
                cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
                gamma = np.arccos(cos_gamma)
                defect -= gamma
                Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8

                if h == h_start:
                    break

            K[v] = defect / Atot
        return K

    def get_curvatures(self):
        """ """
        Nv = self.V_pq.shape[0]
        H = np.zeros(Nv)
        K = np.zeros(Nv)

        for v in range(Nv):
            Atot = 0.0
            r = self.V_pq[v, :3]
            r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            Hvec = np.zeros(3)
            n = np.zeros(3)
            defect = 2 * np.pi
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_twin[self.H_prev[h]]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
                r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
                r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
                r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
                r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

                normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
                normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
                normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
                cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
                cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
                cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
                cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
                cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)

                defect -= np.arccos(cos_gamma)
                Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
                Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
                n += jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)

                if h == h_start:
                    break

            Hvec /= 2 * Atot
            n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
            K[v] = defect / Atot
        return H, K

    def get_gaussian_curvature_meyer(self):
        """gaussian curvature using voroni cell areas"""
        Nv = self.V_pq.shape[0]
        K = np.zeros(Nv)

        for v in range(Nv):
            Atot = self.meyercell_area(v)
            r = self.V_pq[v, :3]
            r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            defect = 2 * np.pi
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_twin[self.H_prev[h]]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
                r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
                r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
                r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
                r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

                normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
                # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
                normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # jitnorm(u3)
                # cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
                # cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
                cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
                # cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
                # cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
                gamma = np.arccos(cos_gamma)
                defect -= gamma
                # Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8

                if h == h_start:
                    break

            K[v] = defect / Atot
        return K

    ###########################################################################
    # geometric computations #
    ##########################

    def barcell_area(self, v):
        """area of cell dual to vertex v"""
        r = self.V_pq[v, :3]
        A = 0.0
        h = self.V_hedge[v]
        h_start = h
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]
            A_face_vec = (
                jitcross(r, r1) / 2 + jitcross(r1, r2) / 2 + jitcross(r2, r) / 2
            )
            A_face = np.sqrt(
                A_face_vec[0] ** 2 + A_face_vec[1] ** 2 + A_face_vec[2] ** 2
            )
            A += A_face / 3

            if h == h_start:
                break
        return A

    def get_total_area(self):
        vertices = self.V_pq[:, :3]
        Nv = self.V_pq.shape[0]
        V_hedge = self.V_hedge
        H_vertex = self.H_vertex
        H_prev = self.H_prev
        H_twin = self.H_twin
        A = 0.0
        for v in range(Nv):
            r = vertices[v]
            Av = 0.0
            h = V_hedge[v]
            h_start = h
            while True:
                v1 = H_vertex[h]
                r1 = vertices[v1]
                h = H_prev[h]
                h = H_twin[h]
                v2 = H_vertex[h]
                r2 = vertices[v2]
                a = jitcross(r, r1) / 2 + jitcross(r1, r2) / 2 + jitcross(r2, r) / 2
                Av += np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) / 3

                if h == h_start:
                    break

            A += Av
        return A

    def volume_of_mesh(self):
        """computes total mesh volume as the sum of signed volume of faces"""
        Nf = len(self.faces)
        vol = 0.0
        for f in range(Nf):
            v0, v1, v2 = self.faces[f]
            p0 = self.V_pq[v0, :3]
            p1 = self.V_pq[v1, :3]
            p2 = self.V_pq[v2, :3]
            vol += triprod(p0, p1, p2) / 6
        return vol

    def face_area_vector(self, f):
        """directed area of face f"""
        A = np.zeros(3)

        h_start = self.F_hedge[f]
        h = h_start
        while True:
            h_next = self.H_next[h]
            i = self.H_vertex[h]
            i_next = self.H_vertex[h_next]
            A += 0.5 * jitcross(self.V_pq[i, :3], self.V_pq[i_next, :3])
            h = h_next
            if h == h_start:
                break

        return A

    def angle_defect(self, v):
        """
        2*pi - sum_f (angle_f)
        """
        r = self.V_pq[v, :3]
        h = self.V_hedge[v]
        h_start = h
        defect = 2 * np.pi
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]
            e1 = r1 - r
            e2 = r2 - r
            norm_e1 = np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
            cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
                norm_e1 * norm_e2
            )
            defect -= np.arccos(cos_angle)
            if h == h_start:
                break

        return defect

    def get_angle_defects(self):
        """
        2*pi - sum_f (angle_f)
        """
        Nverts = len(self.V_pq)
        defects = np.zeros(Nverts)
        for v0 in range(Nverts):
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

    def get_gaussian_curvature2(self):
        """
        2*pi - sum_f (angle_f)
        """
        Nv = len(self.V_pq)
        # defects = np.zeros(Nverts)
        K = np.zeros(Nv)
        # Nv = len(self.V_pq)
        # for v0 in range(Nv):
        # v0 = 0
        # K = np.random.rand(Nverts)
        # while True:
        #     # p0 = self.V_pq[v, :3]
        #     h_start = self.V_hedge[v0]
        #     defect = 2 * np.pi
        #     area = 0.0
        #
        #     h = h_start
        #     v = self.H_vertex[h]
        #     e2 = self.V_pq[v, :3] - self.V_pq[v0, :3]
        #     norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
        #     h = self.H_next[self.H_twin[h]]
        #
        #     while True:
        #         e1 = e2
        #         norm_e1 = norm_e2
        #         v = self.H_vertex[h]  # 2nd vert
        #         e2 = self.V_pq[v, :3] - self.V_pq[v0, :3]
        #         norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
        #         # e1_cross_e2 = jitcross(e1, e2)
        #         # norm_e1_cross_e2 = np.sqrt(
        #         #     e1_cross_e2[0] ** 2 + e1_cross_e2[1] ** 2 + e1_cross_e2[2] ** 2
        #         # )
        #         # sin_angle = norm_e1_cross_e2 / (norm_e1 * norm_e2)
        #         cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
        #             norm_e1 * norm_e2
        #         )
        #         angle = np.arccos(cos_angle)
        #
        #         defect -= angle
        #         area += 0.5 * norm_e1 * norm_e2 * np.sin(angle) / 3
        #
        #         h = self.H_next[self.H_twin[h]]
        #         if h == h_start:
        #             break
        #     K[v0] = defect / area
        for v in range(Nv):
            K[v] = self.gaussian_curvature(v)

        return K

    def gaussian_curvature(self, v):
        K = self.angle_defect(v)
        K /= self.barcell_area(v)
        return K

    ###########################################################################
    # mesh mutation/regularization functions #
    ###############################
    def orientation_check(self, h):
        """checks if faces adjacent to an edge have consistent normals"""

        fl = self.H_face[h]
        fr = self.H_face[self.H_twin[h]]
        il, jl, kl = self.faces[fl]
        ir, jr, kr = self.faces[fr]
        ril, rjl, rkl = self.V_pq[il, :3], self.V_pq[jl, :3], self.V_pq[kl, :3]
        rir, rjr, rkr = self.V_pq[ir, :3], self.V_pq[jr, :3], self.V_pq[kr, :3]
        Al = jitcross(ril, rjl) + jitcross(rjl, rkl) + jitcross(rkl, ril)
        Ar = jitcross(rir, rjr) + jitcross(rjr, rkr) + jitcross(rkr, rir)
        Al_Ar = Al[0] * Ar[0] + Al[1] * Ar[1] + Al[2] * Ar[2]
        is_good = Al_Ar > 0
        return is_good

    def volume_length_quality_metric(self, f):
        v0, v1, v2 = self.faces[f]
        r0 = self.V_pq[v0, :3]
        r1 = self.V_pq[v1, :3]
        r2 = self.V_pq[v2, :3]
        Av = (jitcross(r0, r1) + jitcross(r1, r2) + jitcross(r2, r0)) / 2
        A = jitnorm(Av)
        e01 = r1 - r0
        e12 = r2 - r1
        e20 = r0 - r2

        e_rms = (
            np.sqrt(
                e01[0] ** 2
                + e01[1] ** 2
                + e01[2] ** 2
                + e12[0] ** 2
                + e12[1] ** 2
                + e12[2] ** 2
                + e20[0] ** 2
                + e20[1] ** 2
                + e20[2] ** 2
            )
            / 3
        )

        a = 4 * np.sqrt(3) * A / (3 * e_rms**2)
        return a

    def edge_flip(self, h):
        r"""
        h/ht can not be on boundary!
        keeps fa
               v2                           v2
             /    \                       /  |  \
            /      \                     /   |   \
           /h2    h1\                   /h2  |  h1\
          /    f1    \                 /     |     \
         /            \               /  f1  |  f2  \
        /      h       \             /       |       \
       v3--------------v1  |----->  v3      h|ht     v1
        \      ht      /             \       |       /
         \            /               \      |      /
          \    f2    /                 \     |     /
           \h3    h4/                   \h3  |  h4/
            \      /                     \   |   /
             \    /                       \  |  /
               v4                           v4
        """
        ht = self.H_twin[h]
        h1 = self.H_next[h]
        h2 = self.H_prev[h]
        h3 = self.H_next[ht]
        h4 = self.H_prev[ht]
        f1 = self.H_face[h]
        f2 = self.H_face[ht]
        v1 = self.H_vertex[h4]
        v2 = self.H_vertex[h1]
        v3 = self.H_vertex[h2]
        v4 = self.H_vertex[h3]
        self.faces[f1] = np.array([v2, v3, v4], dtype=np.int32)
        self.faces[f2] = np.array([v4, v1, v2], dtype=np.int32)
        self.F_hedge[f1] = h2
        self.F_hedge[f2] = h4
        self.halfedges[h] = np.array([v4, v2], dtype=np.int32)
        self.halfedges[ht] = np.array([v2, v4], dtype=np.int32)
        self.H_next[h] = h2
        self.H_prev[h2] = h
        self.H_next[h2] = h3
        self.H_prev[h3] = h2
        self.H_next[h3] = h
        self.H_prev[h] = h3  #
        self.H_next[ht] = h4
        self.H_prev[h4] = ht
        self.H_next[h4] = h1
        self.H_prev[h1] = h4
        self.H_next[h1] = ht
        self.H_prev[ht] = h1
        self.H_face[h3] = f1
        self.H_face[h1] = f2
        self.H_vertex[h] = v2
        self.H_vertex[ht] = v4
        self.V_hedge[v3] = h3
        self.V_hedge[v1] = h1
        self.V_hedge[v2] = h2
        self.V_hedge[v4] = h4

    def monte_wants_flip(self, h):
        Ka = self.Karea
        A0 = self.preferred_face_area
        l0 = self.preferred_edge_length
        ht = self.H_twin[h]
        h1 = self.H_next[h]
        h2 = self.H_prev[h]
        h3 = self.H_next[ht]
        h4 = self.H_prev[ht]

        v1 = self.H_vertex[h4]
        v2 = self.H_vertex[h1]
        v3 = self.H_vertex[h2]
        v4 = self.H_vertex[h3]
        r1 = self.V_pq[v1, :3]
        r2 = self.V_pq[v2, :3]
        r3 = self.V_pq[v3, :3]
        r4 = self.V_pq[v4, :3]
        vecA123 = (jitcross(r1, r2) + jitcross(r2, r3) + jitcross(r3, r1)) / 2
        vecA134 = (jitcross(r1, r3) + jitcross(r3, r4) + jitcross(r4, r1)) / 2
        vecA234 = (jitcross(r2, r3) + jitcross(r3, r4) + jitcross(r4, r2)) / 2
        vecA124 = (jitcross(r1, r2) + jitcross(r2, r4) + jitcross(r4, r1)) / 2
        A123 = np.sqrt(vecA123[0] ** 2 + vecA123[1] ** 2 + vecA123[2] ** 2)
        A134 = np.sqrt(vecA134[0] ** 2 + vecA134[1] ** 2 + vecA134[2] ** 2)
        A234 = np.sqrt(vecA234[0] ** 2 + vecA234[1] ** 2 + vecA234[2] ** 2)
        A124 = np.sqrt(vecA124[0] ** 2 + vecA124[1] ** 2 + vecA124[2] ** 2)

        Upre = (Ka / (2 * A0)) * ((A123 - A0) ** 2 + (A134 - A0) ** 2)
        Upos = (Ka / (2 * A0)) * ((A234 - A0) ** 2 + (A124 - A0) ** 2)

        Dr13 = self.V_pq[v1, :3] - self.V_pq[v3, :3]
        Dr24 = self.V_pq[v2, :3] - self.V_pq[v4, :3]
        L13 = np.sqrt(Dr13[0] ** 2 + Dr13[1] ** 2 + Dr13[2] ** 2)
        L24 = np.sqrt(Dr24[0] ** 2 + Dr24[1] ** 2 + Dr24[2] ** 2)
        Upre += self.Utether(L13, l0)
        Upos += self.Utether(L24, l0)
        flip_it = Upos < Upre
        return flip_it

    def do_the_monte_flips(self):
        Nflips = 0
        Nh = len(self.halfedges)
        for h in range(Nh):
            flip_it = self.is_flippable(h) and self.monte_wants_flip(h)
            if flip_it:
                self.edge_flip(h)
                Nflips += 1
        return Nflips

    def flip_helps_valence(self, h):
        r"""
        Returns True if flipping edge 'h' decreases variance
        of the valence of the four vertices illustrated below.
        \sum_{i=1}^4 valence(i)^2/4 - (\sum_{i=1}^4 valence(i))^2/4
        This favors vertices with valence=6 and is equivalent
        returning True if the flip decreases the energy
        \sum_{i in all vertices} (valence(i)-6)^2.
          v2             v2
         / \            /|\
       v3---v1  |---> v3 | v1
         \ /            \|/
          v4             v4
        """

        ht = self.H_twin[h]
        h1 = self.H_next[h]
        h3 = self.H_next[ht]
        v1 = self.H_vertex[h]
        v2 = self.H_vertex[h1]
        v3 = self.H_vertex[ht]
        v4 = self.H_vertex[h3]

        val1 = self.valence(v1)
        val2 = self.valence(v2)
        val3 = self.valence(v3)
        val4 = self.valence(v4)

        flip_it = val1 - val2 + val3 - val4 > 2

        return flip_it

    def is_delaunay(self, h):
        r"""
        checks if edge is locally delaunay
          v2
          /|\
        v3 | v1
          \|/
           v4
        """
        vi = self.H_vertex[self.H_next[self.H_twin[h]]]
        vj = self.H_vertex[h]
        vk = self.H_vertex[self.H_next[h]]
        vl = self.H_vertex[self.H_prev[h]]

        pij = self.V_pq[vj, :3] - self.V_pq[vi, :3]
        pil = self.V_pq[vl, :3] - self.V_pq[vi, :3]
        pkj = self.V_pq[vj, :3] - self.V_pq[vk, :3]
        pkl = self.V_pq[vl, :3] - self.V_pq[vk, :3]

        pij_pil = pij[0] * pil[0] + pij[1] * pil[1] + pij[2] * pil[2]
        pkl_pkj = pkl[0] * pkj[0] + pkl[1] * pkj[1] + pkl[2] * pkj[2]
        normpij = np.sqrt(pij[0] ** 2 + pij[1] ** 2 + pij[2] ** 2)
        normpil = np.sqrt(pil[0] ** 2 + pil[1] ** 2 + pil[2] ** 2)
        normpkj = np.sqrt(pkj[0] ** 2 + pkj[1] ** 2 + pkj[2] ** 2)
        normpkl = np.sqrt(pkl[0] ** 2 + pkl[1] ** 2 + pkl[2] ** 2)

        alphai = np.arccos(pij_pil / (normpij * normpil))
        alphak = np.arccos(pkl_pkj / (normpkl * normpkj))

        return alphai + alphak <= np.pi

    def is_flippable_old(self, h):
        r"""
          vj
          /|\
        vk | vi
          \|/
           vl
        """
        hlj = h
        hjk = self.H_next[hlj]
        hkl = self.H_next[hjk]

        hjl = self.H_twin[hlj]
        hli = self.H_next[hjl]
        hij = self.H_next[hli]

        hkj = self.H_twin[hjk]
        fkj = self.H_face[hkj]
        hji = self.H_twin[hij]
        fji = self.H_face[hji]

        hlk = self.H_twin[hkl]
        flk = self.H_face[hlk]
        hil = self.H_twin[hli]
        fil = self.H_face[hil]

        flippable = (fkj != fji) and (flk != fil)

        return flippable

    def is_flippable(self, h):
        """
        edge flip hlj-->hki is allowed unless vi and vk are already neighbors
          vj
          /|\
        vk | vi
          \|/
           vl
        """
        hlj = h
        hjk = self.H_next[hlj]
        hjl = self.H_twin[hlj]
        hli = self.H_next[hjl]
        flippable = True

        vj = self.H_vertex[hlj]
        vk = self.H_vertex[hjk]

        him = self.H_twin[hli]
        while True:
            him = self.H_twin[self.H_prev[him]]
            vm = self.H_vertex[him]
            if vm == vk:
                flippable = False
                break
            if vm == vj:
                break

        return flippable

    def shift_vertex_towards_barycenter(self, v, weight):
        """Translates vertex in the direction of the barycenter of
        its neighbors. weight=1 means all the way to the barycenter."""
        r0 = self.V_pq[v, :3].copy()
        h = self.V_hedge[v]
        h_start = h
        r = np.zeros(3)
        val = 0
        while True:
            vb = self.H_vertex[h]
            r += self.V_pq[vb, :3]
            val += 1
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break
        r /= val
        self.V_pq[v, :3] = weight * r + (1 - weight) * r0

    def regularize_by_flips(self):
        Nh = len(self.halfedges)
        Nflips = 0
        for h in range(Nh):
            flip_it = self.flip_helps_valence(h) & self.is_flippable(h)
            if flip_it:
                self.edge_flip(h)
                Nflips += 1
        return Nflips

    def regularize_by_shifts(self, weight):
        Nv = self.V_pq.shape[0]
        for v in range(Nv):
            self.shift_vertex_towards_barycenter(v, weight)

    def smooth_samples(self, samples_in, weight, iters):
        """shifts samples towards the avereage value of their neighbors"""
        samples = samples_in.copy()
        Nv = self.V_pq.shape[0]
        for iter in range(iters):
            for v in range(Nv):
                h = self.V_hedge[v]
                h_start = h
                samp = np.zeros_like(samples[v])
                val = 0
                while True:
                    vb = self.H_vertex[h]
                    samp += samples[vb]
                    val += 1
                    h = self.H_twin[self.H_prev[h]]
                    if h == h_start:
                        break
                samp /= val
                samples[v] = weight * samp + (1 - weight) * samples[v]
        return samples

    def delaunay_regularize_by_flips(self):
        Nh = len(self.halfedges)
        Nflips = 0
        for h in range(Nh):
            is_del = self.is_delaunay(h)
            if not is_del:
                flippable = self.is_flippable(h)
                if flippable:
                    self.edge_flip(h)
                    Nflips += 1
        return Nflips

    def gaussian_smooth_samples(self, samples_in, iters, a):
        """shifts samples towards the avereage value of their neighbors"""
        samples = samples_in.copy()
        Nv = self.V_pq.shape[0]
        for iter in range(iters):
            # samples0 = samples_in.copy()
            for v in range(Nv):
                r0 = self.V_pq[v, :3]
                r0_r0 = r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
                h = self.V_hedge[v]
                h_start = h
                samp = samples[v]
                W = 1
                while True:
                    vb = self.H_vertex[h]
                    r = self.V_pq[vb, :3]
                    r_r0 = r[0] * r0[0] + r[1] * r0[1] + r[2] * r0[2]
                    r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
                    w = np.exp(-(r_r - 2 * r_r0 + r0_r0) / a**2)
                    samp += w * samples[vb]
                    W += w
                    h = self.H_twin[self.H_prev[h]]
                    if h == h_start:
                        break

                samples[v] = samp / W
        return samples

    ###########################################################################
    # forces and time evolution #
    #############################

    def Flj(self, v):
        r0 = self.V_pq[v, :3]
        F = np.zeros(3)
        L0 = self.preferred_edge_length
        eps = self.Klength
        A = 2 ** (-1 / 6) * L0
        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v = self.H_vertex[h]
            R = self.V_pq[v, :3] - r0
            normR = jitnorm(R)
            A_normR = A / normR
            Dg = (-24 * eps / A) * (2 * A_normR**13 - A_normR**7)
            F += -Dg * R / normR
            h = self.H_twin[self.H_prev[h]]

            if h == h_start:
                break
        return F

    def Ulj(self, R):
        L0 = self.preferred_edge_length
        eps = self.Klength
        A = 2 ** (-1 / 6) * L0

        normR = jitnorm(R)
        A_normR = A / normR
        g = 4 * eps * (A_normR**12 - A_normR**6)

        return g

    def length_reg_force(self, v):
        """E ~ 1/2*Ke*(L-L0)**2/L0"""
        Ke = self.Klength
        L0 = self.preferred_edge_length
        xyz = self.V_pq[v, :3]
        neighbors = self.v_adjacent_to_v(v)
        N = len(neighbors)
        F = np.zeros(3)

        for _v0 in range(0, N):
            v0 = neighbors[_v0]
            xyz0 = self.V_pq[v0, :3]
            r = xyz - xyz0
            L = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            gradL = r / L
            # if L >1e-12:
            #     gradL = r / L
            # else:
            #     gradL = r
            F += -Ke * (L - L0) * gradL / L0

        return F

    def LOCAL_area_reg_force(self, v):
        """E ~ 1/2*Ka*(A-A0)**2"""
        A0 = self.preferred_cell_area
        Ka = self.Karea
        r = self.V_pq[v, :3]
        F = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]

            Avec = (jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)) / 2
            A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3

            gradA = jitcross(Avec, r2 - r1) / (2 * A)

            F += -Ka * (A - A0) * gradA / A0

            if h == h_start:
                break
        return F

    def area_reg_force(self, v):
        """E ~ 1/2*Ka*(A-A0)**2"""
        r = self.V_pq[v, :3]
        F = np.zeros(3)
        area0 = self.preferred_total_area
        area = self.total_area
        Ka = self.Karea

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]

            Avec = (jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)) / 2
            A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3

            gradA = jitcross(Avec, r2 - r1) / (2 * A)

            F += -Ka * (area - area0) * gradA / area0

            if h == h_start:
                break
        return F

    def volume_reg_force(self, v):
        """."""
        vol0 = self.preferred_total_volume
        vol = self.total_volume
        Kv = self.Kvolume

        F = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            v1 = self.H_vertex[h]
            r1 = self.V_pq[v1, :3]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            r2 = self.V_pq[v2, :3]

            # grad_vol = jitcross(r1, r2) / (2 * A)

            F += -Kv * (vol - vol0) * jitcross(r1, r2) / (18 * vol0)

            if h == h_start:
                break
        return F

    def forward_euler_reg_step(self, dt):
        Nv = len(self.V_pq)
        linear_drag_coeff = self.linear_drag_coeff
        self.total_volume = self.volume_of_mesh()
        self.total_area = self.get_total_area()

        for v in range(Nv):
            F = np.zeros(3)
            # F += self.length_reg_force(v)
            F += self.area_reg_force(v)
            F += self.volume_reg_force(v)
            F += self.Flj(v)
            self.V_pq[v, :3] = self.V_pq[v, :3] + dt * F / linear_drag_coeff

        # for v in range(Nv):
        #     self.V_pq[v, 3:] = self.get_new_quat_dumb(v)

    def get_new_euler_state(self, dt):
        # Nv = len(self.V_pq)
        # vertices = np.zeros((Nv, 3))
        linear_drag_coeff = self.linear_drag_coeff
        self.total_volume = self.volume_of_mesh()
        self.total_area = self.get_total_area()
        Fb = self.Fbend()
        Fl = self.Flength()
        Fa = self.Farea()
        Fv = self.Fvolume()
        F = Fb + Fl + Fa + Fv
        vertices = self.V_pq[:, :3] + dt * F / linear_drag_coeff

        # for v in range(Nv):
        #     self.V_pq[v, 3:] = self.get_new_quat_dumb(v)
        success = True
        return vertices, success

    def Fbend(self, Nsmooth=0):
        """from Tu"""
        a = self.preferred_edge_length
        Kbend = self.bending_modulus
        H, K = self.get_angle_weighted_arc_curvatures()
        for _ in range(Nsmooth):
            H = self.gaussian_smooth_samples(H, 1, a)
            K = self.gaussian_smooth_samples(K, 1, a)
        Nv = H.shape[0]
        lapH = self.cotan_laplacian(H)
        F = np.zeros((Nv, 3))

        Fn = -2 * Kbend * (lapH + 2 * H * (H**2 - K))

        for v in range(Nv):
            n = self.other_weighted_vertex_normal(v)
            Av = self.vorcell_area(v)
            F[v] = Fn[v] * n * Av
        return F

    def Flength(self):
        Nv = len(self.V_pq)
        F = np.zeros((Nv, 3))
        Ke = self.Klength
        L0 = self.preferred_edge_length
        for v in range(Nv):
            r = self.V_pq[v, :3]
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v0 = self.H_vertex[h]
                r0 = self.V_pq[v0, :3]
                u = r - r0
                L = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
                gradL = u / L
                F[v] += -Ke * (L - L0) * gradL / L0
                h = self.H_twin[self.H_prev[h]]
                if h == h_start:
                    break
        return F

    def Farea(self):
        """local cell area refulation"""
        Nv = len(self.V_pq)
        F = np.zeros((Nv, 3))
        # area0 = self.preferred_total_area
        # area = self.total_area
        A0 = self.preferred_cell_area
        Ka = self.Karea
        for v in range(Nv):
            r = self.V_pq[v, :3]
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_prev[h]
                h = self.H_twin[h]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                Avec = (jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)) / 2
                A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3

                gradA = jitcross(Avec, r2 - r1) / (2 * A)

                # F[v] += -Ka * (area - area0) * gradA / area0
                F[v] += -Ka * (A - A0) * gradA / A0

                if h == h_start:
                    break
        return F

    def Fvolume(self):
        Nv = len(self.V_pq)
        F = np.zeros((Nv, 3))
        vol0 = self.preferred_total_volume
        vol = self.volume_of_mesh()
        Kv = self.Kvolume
        for v in range(Nv):
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                h = self.H_prev[h]
                h = self.H_twin[h]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]

                # grad_vol = jitcross(r1, r2) / (2 * A)

                F[v] += -Kv * (vol - vol0) * jitcross(r1, r2) / (18 * vol0)

                if h == h_start:
                    break
        return F

    ###########################################################################
    ###########################################################################
    # helper functions #
    ####################################

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

    def h_out_from_v(self, v):
        """
        gets outgoing halfedges adjacent to v in counterclockwise order
        """
        neighbors = []

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            neighbors.append(h)
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return np.array(neighbors, dtype=np.int32)

    def h_in_to_v(self, v):
        """
        gets halfedges incident to v in counterclockwise order
        """
        neighbors = []

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            neighbors.append(self.H_twin[h])
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return np.array(neighbors, dtype=np.int32)

    def v_adjacent_to_v(self, v):
        """
        gets vertices adjacent to v in counterclockwise order
        """
        neighbors = []

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            neighbor = self.H_vertex[h]
            neighbors.append(neighbor)
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break

        return np.array(neighbors, dtype=np.int32)

    def valence(self, v):
        h_start = self.V_hedge[v]
        val = 0
        h = h_start
        while True:
            val += 1
            h = self.H_prev[h]
            h = self.H_twin[h]
            if h == h_start:
                break
        return val

    def area_weighted_vertex_normal(self, v):
        """."""
        n = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            f = self.H_face[h]
            n += self.face_area_vector(f)

            h = self.H_prev[h]
            h = self.H_twin[h]

            if h == h_start:
                break

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    def other_weighted_vertex_normal(self, v):
        """Weights for Computing Vertex Normals from Facet Normals Max99"""
        n = np.zeros(3)
        ri = self.V_pq[v, :3]
        h_start = self.V_hedge[v]
        hj = h_start
        while True:
            hjp1 = self.H_twin[self.H_prev[hj]]
            vj = self.H_vertex[hj]
            vjp1 = self.H_vertex[hjp1]
            Drj = self.V_pq[vj, :3] - ri
            Drjp1 = self.V_pq[vjp1, :3] - ri
            Drj_dot_Drj = Drj[0] ** 2 + Drj[1] ** 2 + Drj[2] ** 2
            Drjp1_dot_Drjp1 = Drjp1[0] ** 2 + Drjp1[1] ** 2 + Drjp1[2] ** 2

            n += jitcross(Drj, Drjp1) / (Drj_dot_Drj * Drjp1_dot_Drjp1)
            hj = hjp1

            if hj == h_start:
                break

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    def quat_normal_vector(self, v):
        qw, qx, qy, qz = self.V_pq[v, 3:]
        n = np.array(
            [
                2 * qw * qy + 2 * qx * qz,
                2 * qy * qz - 2 * qw * qx,
                qw**2 - qx**2 - qy**2 + qz**2,
            ]
        )
        return n

    def get_new_quat_dumb(self, v):
        qw, qx, qy, qz = self.V_pq[v, 3:]
        Q = np.zeros((3, 3))
        Q[:, 2] = self.other_weighted_vertex_normal(v)

        Q[0, 0] = qw**2 + qx**2 - qy**2 - qz**2
        Q[1, 0] = 2 * qw * qz + 2 * qx * qy
        Q[2, 0] = 2 * qx * qz - 2 * qw * qy
        Q[:, 0] -= jitdot(Q[:, 2], Q[:, 0]) * Q[:, 2]
        Q[:, 0] /= jitnorm(Q[:, 0])
        Q[:, 1] = jitcross(Q[:, 2], Q[:, 0])

        q = matrix_to_quaternion(Q)
        return q

    ###########################################################################
    # simulation/data management/visualization functions #
    ############################

    def rigid_transform(self, PQ):
        Nverts = len(self.V_pq)
        for i in range(Nverts):
            self.V_pq[i] = mul_se3_quaternion(PQ, self.V_pq[i])

    def shifted_hedge_vectors(self):
        """halfedge vector shifted toward face centroid for visualization"""
        Nh = len(self.halfedges)
        vecs = np.zeros((Nh, 3))
        points = np.zeros((Nh, 3))
        for h in range(Nh):
            points[h, :], vecs[h, :] = self.shifted_hedge_vector(h)
        return points, vecs

    def shifted_hedge_vector(self, h):
        shift_to_center = 0.15
        v0 = self.H_vertex[self.H_prev[h]]
        v1 = self.H_vertex[h]
        v2 = self.H_vertex[self.H_next[h]]
        com = (self.V_pq[v0, :3] + self.V_pq[v1, :3] + self.V_pq[v2, :3]) / 3
        p = shift_to_center * com + (1 - shift_to_center) * self.V_pq[v0, :3]
        u = (1 - shift_to_center) * (self.V_pq[v1, :3] - self.V_pq[v0, :3])
        return p, u

    def pack_mesh_data(self):
        """V.astype(int)"""
        V_pq = self.V_pq
        V_hedge = self.V_hedge

        halfedges = self.halfedges
        H_vertex = self.H_vertex
        H_face = self.H_face
        H_next = self.H_next
        H_prev = self.H_prev
        H_twin = self.H_twin

        faces = self.faces
        F_hedge = self.F_hedge

        Nvertices = len(V_pq)
        Nhalfedges = len(halfedges)
        Nfaces = len(faces)
        nv = 8
        nh = 7
        nf = 4
        Nvdat = nv * Nvertices
        Nhdat = nh * Nhalfedges
        Nfdat = nf * Nfaces
        Ndata = Nvdat + Nhdat + Nfdat

        V = np.zeros(Nvdat)
        H = np.zeros(Nhdat)
        F = np.zeros(Nfdat)
        VHF = np.zeros(Ndata + 6)

        V[: 7 * Nvertices] = V_pq.T.ravel()
        V[7 * Nvertices :] = V_hedge
        VHF[:Nvdat] = V

        H[: 2 * Nhalfedges] = halfedges.T.ravel()
        H[2 * Nhalfedges : 3 * Nhalfedges] = H_vertex
        H[3 * Nhalfedges : 4 * Nhalfedges] = H_face
        H[4 * Nhalfedges : 5 * Nhalfedges] = H_next
        H[5 * Nhalfedges : 6 * Nhalfedges] = H_prev
        H[6 * Nhalfedges : 7 * Nhalfedges] = H_twin
        VHF[Nvdat : Nvdat + Nhdat] = H

        F[: 3 * Nfaces] = faces.T.ravel()
        F[3 * Nfaces :] = F_hedge
        VHF[Nvdat + Nhdat : Nvdat + Nhdat + Nfdat] = F
        VHF[-6:] = np.array([Nvertices, nv, Nhalfedges, nh, Nfaces, nf])

        return VHF

    def unpack_mesh_data(self, VHF):
        """V.astype(int)"""

        Nvertices, nv, Nhalfedges, nh, Nfaces, nf = VHF[-6:].astype(np.int32)
        Nvdat = nv * Nvertices
        Nhdat = nh * Nhalfedges
        Nfdat = nf * Nfaces
        V = VHF[:Nvdat]
        H = VHF[Nvdat : Nvdat + Nhdat].astype(np.int32)
        F = VHF[Nvdat + Nhdat : Nvdat + Nhdat + Nfdat].astype(np.int32)

        V_pq = np.zeros((Nvertices, 7))
        # V_pq = np.array([V[_ * Nvertices : (_ + 1) * Nvertices] for _ in range(7)])
        V_pq[:, 0] = V[0 * Nvertices : (0 + 1) * Nvertices]
        V_pq[:, 1] = V[1 * Nvertices : (1 + 1) * Nvertices]
        V_pq[:, 2] = V[2 * Nvertices : (2 + 1) * Nvertices]
        V_pq[:, 3] = V[3 * Nvertices : (3 + 1) * Nvertices]
        V_pq[:, 4] = V[4 * Nvertices : (4 + 1) * Nvertices]
        V_pq[:, 5] = V[5 * Nvertices : (5 + 1) * Nvertices]
        V_pq[:, 6] = V[6 * Nvertices : (6 + 1) * Nvertices]

        # V_pq = V[: 7 * Nvertices].reshape((Nvertices, 7)).astype(np.float64)
        V_hedge = V[7 * Nvertices :].astype(np.int32)

        # halfedges = H[: 2 * Nhalfedges].reshape((Nhalfedges, 2))
        halfedges = np.zeros((Nhalfedges, 2), dtype=np.int32)
        halfedges[:, 0] = H[0 * Nhalfedges : (0 + 1) * Nhalfedges]
        halfedges[:, 1] = H[1 * Nhalfedges : (1 + 1) * Nhalfedges]
        H_vertex = H[2 * Nhalfedges : 3 * Nhalfedges]
        H_face = H[3 * Nhalfedges : 4 * Nhalfedges]
        H_next = H[4 * Nhalfedges : 5 * Nhalfedges]
        H_prev = H[5 * Nhalfedges : 6 * Nhalfedges]
        H_twin = H[6 * Nhalfedges : 7 * Nhalfedges]

        # faces = F[: 3 * Nfaces].reshape((Nfaces, 3))
        faces = np.zeros((Nfaces, 3), dtype=np.int32)
        faces[:, 0] = F[0 * Nfaces : (0 + 1) * Nfaces]
        faces[:, 1] = F[1 * Nfaces : (1 + 1) * Nfaces]
        faces[:, 2] = F[2 * Nfaces : (2 + 1) * Nfaces]
        F_hedge = F[3 * Nfaces :]

        return (
            V_pq,
            V_hedge,
            halfedges,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            faces,
            F_hedge,
        )

    def pack_visual_data(self):
        """V.astype(int)"""
        V_rgb = self.V_rgb
        V_normal_rgb = self.V_normal_rgb
        V_tangent1_rgb = self.V_tangent1_rgb
        V_tangent2_rgb = self.V_tangent2_rgb
        V_radius = self.V_radius

        H_rgb = self.H_rgb

        F_rgb = self.F_rgb

        F_opacity = self.F_opacity
        H_opacity = self.H_opacity
        V_opacity = self.V_opacity

        Nvertices = len(V_rgb)
        Nhalfedges = len(H_rgb)
        Nfaces = len(F_rgb)
        nv = 13
        nh = 3
        nf = 3
        Nvdat = nv * Nvertices
        Nhdat = nh * Nhalfedges
        Nfdat = nf * Nfaces
        Ndata = Nvdat + Nhdat + Nfdat

        V = np.zeros(Nvdat)
        H = np.zeros(Nhdat)
        F = np.zeros(Nfdat)
        VHF = np.zeros(Ndata + 9)

        V[0 * 3 * Nvertices : (0 + 1) * 3 * Nvertices] = V_rgb.T.ravel()
        V[1 * 3 * Nvertices : (1 + 1) * 3 * Nvertices] = V_normal_rgb.T.ravel()
        V[2 * 3 * Nvertices : (2 + 1) * 3 * Nvertices] = V_tangent1_rgb.T.ravel()
        V[3 * 3 * Nvertices : (3 + 1) * 3 * Nvertices] = V_tangent2_rgb.T.ravel()
        V[4 * 3 * Nvertices :] = V_radius
        VHF[:Nvdat] = V

        H[0 * 3 * Nhalfedges : (0 + 1) * 3 * Nhalfedges] = H_rgb.T.ravel()
        VHF[Nvdat : Nvdat + Nhdat] = H

        F[0 * 3 * Nfaces : (0 + 1) * 3 * Nfaces] = F_rgb.T.ravel()

        VHF[Nvdat + Nhdat : Nvdat + Nhdat + Nfdat] = F
        VHF[-9:] = np.array(
            [F_opacity, H_opacity, V_opacity, Nvertices, nv, Nhalfedges, nh, Nfaces, nf]
        )

        return VHF

    def unpack_visual_data(self, VHF):
        """V.astype(int)"""
        F_opacity, H_opacity, V_opacity = VHF[-9:-6]
        Nvertices, nv, Nhalfedges, nh, Nfaces, nf = VHF[-6:].astype(np.int32)
        Nvdat = nv * Nvertices
        Nhdat = nh * Nhalfedges
        Nfdat = nf * Nfaces
        V = VHF[:Nvdat]
        H = VHF[Nvdat : Nvdat + Nhdat]
        F = VHF[Nvdat + Nhdat : Nvdat + Nhdat + Nfdat]

        V_rgb = np.zeros((Nvertices, 3))
        V_normal_rgb = np.zeros((Nvertices, 3))
        V_tangent1_rgb = np.zeros((Nvertices, 3))
        V_tangent2_rgb = np.zeros((Nvertices, 3))
        # V_radius = np.zeros(Nvertices)

        V_rgb[:, 0] = V[0 * Nvertices : (0 + 1) * Nvertices]
        V_rgb[:, 1] = V[1 * Nvertices : (1 + 1) * Nvertices]
        V_rgb[:, 2] = V[2 * Nvertices : (2 + 1) * Nvertices]

        V_normal_rgb[:, 0] = V[3 * Nvertices : (3 + 1) * Nvertices]
        V_normal_rgb[:, 1] = V[4 * Nvertices : (4 + 1) * Nvertices]
        V_normal_rgb[:, 2] = V[5 * Nvertices : (5 + 1) * Nvertices]

        V_tangent1_rgb[:, 0] = V[6 * Nvertices : (6 + 1) * Nvertices]
        V_tangent1_rgb[:, 1] = V[7 * Nvertices : (7 + 1) * Nvertices]
        V_tangent1_rgb[:, 2] = V[8 * Nvertices : (8 + 1) * Nvertices]

        V_tangent2_rgb[:, 0] = V[9 * Nvertices : (9 + 1) * Nvertices]
        V_tangent2_rgb[:, 1] = V[10 * Nvertices : (10 + 1) * Nvertices]
        V_tangent2_rgb[:, 2] = V[11 * Nvertices : (11 + 1) * Nvertices]

        V_radius = V[(11 + 1) * Nvertices :]

        H_rgb = np.zeros((Nhalfedges, 3))
        H_rgb[:, 0] = H[0 * Nhalfedges : (0 + 1) * Nhalfedges]
        H_rgb[:, 1] = H[1 * Nhalfedges : (1 + 1) * Nhalfedges]
        H_rgb[:, 2] = H[2 * Nhalfedges : (2 + 1) * Nhalfedges]

        F_rgb = np.zeros((Nfaces, 3))
        F_rgb[:, 0] = F[0 * Nfaces : (0 + 1) * Nfaces]
        F_rgb[:, 1] = F[1 * Nfaces : (1 + 1) * Nfaces]
        F_rgb[:, 2] = F[2 * Nfaces : (2 + 1) * Nfaces]

        return (
            V_rgb,
            V_normal_rgb,
            V_tangent1_rgb,
            V_tangent2_rgb,
            V_radius,
            H_rgb,
            F_rgb,
            F_opacity,
            H_opacity,
            V_opacity,
        )

    ###########################################################################
    # framed brane functions #
    ##########################
