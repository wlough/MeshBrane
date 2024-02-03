from numba import float64, int32, boolean
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
    ("Kbend", float64),
    ("Ksplay", float64),
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
    ("preferred_cell_volume", float64),
    ("preferred_edge_length", float64),
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
        ###########################
        # physical parameters
        self.Kbend = bending_modulus
        self.Ksplay = splay_modulus
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
        self.total_volume = self.preferred_total_volume
        self.total_area = self.preferred_total_area
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

        #############################################################

    ###########################################################################
    ###########################################################################
    # initialization functions #
    ############################
    # these don't call any other class functions and are safe to use before any
    # class attributes have been assigned
    ###########################################################################

    def _get_halfedges(self, vertices, faces):
        """Builds halfedges from vertices and faces, assigns an integer-valued
        label/index to each halfedge, and determines whether the halfedge is
        contained in the boundary of the mesh."""
        halfedges = []
        H_isboundary = []
        # H_lab = []
        ####################
        # save and label halfedges
        h = 0
        for face in faces:
            # face = faces[f]
            N_v_of_f = len(face)
            for _ in range(N_v_of_f):
                # index shift to get next
                _next = (_ + 1) % N_v_of_f
                v0 = face[_]  #
                v1 = face[_next]
                hedge = [v0, v1]
                halfedges.append(hedge)
                H_isboundary.append(False)
                # H_lab.append(h)
                h += 1

        for hedge in halfedges:
            v0, v1 = hedge
            hedge_twin = [v1, v0]
            try:
                halfedges.index(hedge_twin)
            except Exception:
                halfedges.append(hedge_twin)
                H_isboundary.append(True)
                # H_lab.append(h)
                h += 1

        return (
            np.array(halfedges, dtype=np.int32),
            # np.array(H_lab, dtype=np.int32),
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
        preferred_edge_length = self._average_hedge_length(vertices, H_twin, H_vertex)
        preferred_cell_area = self._average_cell_area(
            vertices,
            V_hedge,
            H_vertex,
            H_prev,
            H_twin,
        )
        preferred_total_volume = self._total_volume(vertices, faces)
        preferred_total_area = self._total_area(
            vertices,
            V_hedge,
            H_vertex,
            H_prev,
            H_twin,
        )
        return (
            preferred_edge_length,
            preferred_cell_area,
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

    def reinit(
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
    ):
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
        ###########################
        # physical parameters
        self.Kbend = bending_modulus
        self.Ksplay = splay_modulus
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
        self.total_volume = self.preferred_total_volume
        self.total_area = self.preferred_total_area
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

    ###########################################################################
    # mesh navigation functions #
    ############################
    def face(self, f):
        """consistent with initial face assignment"""
        face = []
        h = self.H_prev[self.F_hedge[f]]
        h_start = h
        while True:
            face.append(self.H_vertex[h])
            h = self.H_next[h]
            if h == h_start:
                break
        return np.array(face, dtype=np.int32)

    def halfedge(self, h):
        v1 = self.H_vertex[h]
        v0 = self.H_vertex[self.H_prev[h]]
        hedge = np.array([v0, v1], dtype=np.int32)
        return hedge
        # return self.halfedges[h]

    def twin(self, h):
        return self.H_twin[h]

    def next(self, h):
        return self.H_next[h]

    def prev(self, h):
        return self.H_prev[h]

    def v_of_h(self, h):
        return self.H_vertex[h]

    def f_of_h(self, h):
        return self.H_face[h]

    def h_of_v(self, v):
        return self.V_hedge[v]

    def h_of_f(self, f):
        return self.F_hedge[f]

    ###########################################################################
    # new geometric computations #
    ##########################

    def dihedral_angle_dg(self, h):
        v2 = self.H_vertex[h]
        v3 = self.H_vertex[self.H_next[h]]
        v0 = self.H_vertex[self.H_twin[h]]
        v1 = self.H_vertex[self.H_next[self.H_twin[h]]]

        r0 = self.V_pq[v0, :3]
        r1 = self.V_pq[v1, :3]
        r2 = self.V_pq[v2, :3]
        r3 = self.V_pq[v3, :3]

        Aleft = jitcross(r0, r2) + jitcross(r2, r3) + jitcross(r3, r0)
        Aright = jitcross(r0, r1) + jitcross(r1, r2) + jitcross(r2, r0)

        normAleft = np.sqrt(Aleft[0] ** 2 + Aleft[1] ** 2 + Aleft[2] ** 2)
        normAright = np.sqrt(Aright[0] ** 2 + Aright[1] ** 2 + Aright[2] ** 2)
        Aleft_Aright = (
            Aleft[0] * Aright[0] + Aleft[1] * Aright[1] + Aleft[2] * Aright[2]
        )
        cos_phi = Aleft_Aright / (normAleft * normAright)
        phi = np.arccos(cos_phi)
        return phi

    def mean_curvature_over_edge_dg(self, h):
        v2 = self.H_vertex[h]
        v3 = self.H_vertex[self.H_next[h]]
        v0 = self.H_vertex[self.H_twin[h]]
        v1 = self.H_vertex[self.H_next[self.H_twin[h]]]

        r0 = self.V_pq[v0, :3]
        r1 = self.V_pq[v1, :3]
        r2 = self.V_pq[v2, :3]
        r3 = self.V_pq[v3, :3]

        Aleft = jitcross(r0, r2) + jitcross(r2, r3) + jitcross(r3, r0)
        Aright = jitcross(r0, r1) + jitcross(r1, r2) + jitcross(r2, r0)
        normAleft = np.sqrt(Aleft[0] ** 2 + Aleft[1] ** 2 + Aleft[2] ** 2)
        normAright = np.sqrt(Aright[0] ** 2 + Aright[1] ** 2 + Aright[2] ** 2)
        Aleft_Aright = (
            Aleft[0] * Aright[0] + Aleft[1] * Aright[1] + Aleft[2] * Aright[2]
        )
        cos_phi = Aleft_Aright / (normAleft * normAright)
        phi = np.arccos(cos_phi)
        Eh = r0 - r2
        Lh = np.sqrt(Eh[0] ** 2 + Eh[1] ** 2 + Eh[2] ** 2)
        intH = Lh * phi / 2
        return intH

    def pointwise_mean_curvature_dg(self, v):
        """uses barycell area, H_sphere < 0"""
        vi = v
        pi = self.V_pq[vi, :3]
        intH = 0.0
        # A = self.barcell_area(v)
        A = 0

        hij = self.V_hedge[vi]
        # hir = self.H_next[self.H_twin[hij]]
        hil = self.H_twin[self.H_prev[hij]]
        hil_start = hil
        vj = self.H_vertex[hij]
        vl = self.H_vertex[hil]
        pj = self.V_pq[vj, :3]
        pl = self.V_pq[vl, :3]
        Aijl = jitcross(pi, pj) + jitcross(pj, pl) + jitcross(pl, pi)
        normAijl = np.sqrt(Aijl[0] ** 2 + Aijl[1] ** 2 + Aijl[2] ** 2)
        while True:
            # vr = vj
            # pr = pj
            # vj = vl
            pj = pl
            hil = self.H_twin[self.H_prev[hil]]
            vl = self.H_vertex[hil]
            pl = self.V_pq[vl, :3]
            eij = pj - pi
            Lij = np.sqrt(eij[0] ** 2 + eij[1] ** 2 + eij[2] ** 2)
            Airj = Aijl
            normAirj = normAijl
            Aijl = jitcross(pi, pj) + jitcross(pj, pl) + jitcross(pl, pi)
            normAijl = np.sqrt(Aijl[0] ** 2 + Aijl[1] ** 2 + Aijl[2] ** 2)
            Airj_dot_Aijl = Airj[0] * Aijl[0] + Airj[1] * Aijl[1] + Airj[2] * Aijl[2]
            phi_ij = np.arccos(Airj_dot_Aijl / (normAirj * normAijl))
            intH += Lij * phi_ij / 4
            A += normAirj / 6
            if hil == hil_start:
                break

        H = intH / A

        return H

    def get_mean_curvature_dg(self):
        """uses barycell area, H_sphere < 0"""
        Nv = self.V_pq.shape[0]
        H = np.zeros(Nv)
        for vi in range(Nv):
            # vi = v
            pi = self.V_pq[vi, :3]
            intH = 0.0
            A = 0

            hij = self.V_hedge[vi]
            # hir = self.H_next[self.H_twin[hij]]
            hil = self.H_twin[self.H_prev[hij]]
            hil_start = hil
            vj = self.H_vertex[hij]
            vl = self.H_vertex[hil]
            pj = self.V_pq[vj, :3]
            pl = self.V_pq[vl, :3]
            Aijl = jitcross(pi, pj) + jitcross(pj, pl) + jitcross(pl, pi)
            normAijl = np.sqrt(Aijl[0] ** 2 + Aijl[1] ** 2 + Aijl[2] ** 2)
            while True:
                # vr = vj
                # pr = pj
                # vj = vl
                pj = pl
                hil = self.H_twin[self.H_prev[hil]]
                vl = self.H_vertex[hil]
                pl = self.V_pq[vl, :3]
                eij = pj - pi
                Lij = np.sqrt(eij[0] ** 2 + eij[1] ** 2 + eij[2] ** 2)
                Airj = Aijl
                normAirj = normAijl
                Aijl = jitcross(pi, pj) + jitcross(pj, pl) + jitcross(pl, pi)
                normAijl = np.sqrt(Aijl[0] ** 2 + Aijl[1] ** 2 + Aijl[2] ** 2)
                Airj_dot_Aijl = (
                    Airj[0] * Aijl[0] + Airj[1] * Aijl[1] + Airj[2] * Aijl[2]
                )
                phi_ij = np.arccos(Airj_dot_Aijl / (normAirj * normAijl))
                intH += Lij * phi_ij / 4
                A += normAirj / 6
                if hil == hil_start:
                    break

            H[vi] = intH / A

        return H

    def cotan_laplacian_dg(self, Y):
        """computes the laplacian of Y at each vertex"""
        Nv = self.V_pq.shape[0]
        lapY = np.zeros_like(Y)
        for vi in range(Nv):
            yi = Y[vi]
            pi = self.V_pq[vi, :3]
            h_start = self.V_hedge[vi]
            hij = h_start
            while True:
                vj = self.H_vertex[hij]
                pj = self.V_pq[vj, :3]
                yj = Y[vj]
                hijplus1 = self.H_twin[self.H_prev[hij]]
                vjplus1 = self.H_vertex[hijplus1]
                pjplus1 = self.V_pq[vjplus1, :3]

                ejjplus1 = pjplus1 - pj
                eji = pi - pj
                ejplus1i = pi - pjplus1
                ejplus1j = -ejjplus1
                alpha = np.arccos(
                    jitdot(ejjplus1, eji) / (jitnorm(ejjplus1) * jitnorm(eji))
                )
                beta = np.arccos(
                    jitdot(ejplus1i, ejplus1j) / (jitnorm(ejplus1i) * jitnorm(ejplus1j))
                )
                lapY[vi] += (1 / np.tan(alpha) + 1 / np.tan(beta)) * (yj - yi) / 2
                hij = hijplus1

                if hij == h_start:
                    break

        return lapY

    def Fbend_dg(self):
        """from Tu"""
        Kbend = self.Kbend
        H, K = self.get_curvatures()
        H = self.get_mean_curvature_dg()
        H *= -1
        Nv = H.shape[0]
        lapH = self.cotan_laplacian_dg(H)
        F = np.zeros((Nv, 3))
        # Fn = np.zeros(Nv)
        # Fn = Kbend * (lapH + 2 * H * (H**2 - K))
        Fn = -Kbend * (lapH + 2 * H * (H**2 - K))
        for v in range(Nv):
            n = self.area_weighted_vertex_normal(v)
            A = self.vorcell_area(v)
            Fn[v] *= A
            F[v] = Fn[v] * n
        return F

    ###########################################################################
    # geometric computations #
    ##########################

    def areal_coords(self, p, r0, r1, r2):
        """areal coordinates of p wrt triangle r0-r1-r2"""
        s = np.zeros(3)
        A01 = jitcross(r0, r1)
        A12 = jitcross(r1, r2)
        A20 = jitcross(r2, r0)

        Ap1 = jitcross(p, r1)
        # A12 = jitcross(r1, r2)
        A2p = jitcross(r2, p)

        A0p = jitcross(r0, p)
        Ap2 = jitcross(p, r2)
        # A20 = jitcross(r2, r0)

        # A01 = jitcross(r0, r1)
        A1p = jitcross(r1, p)
        Ap0 = jitcross(p, r0)

        A012 = A01 + A12 + A20
        normsqrA012 = A012[0] ** 2 + A012[1] ** 2 + A012[2] ** 2
        Ap12 = Ap1 + A12 + A2p
        A0p2 = A0p + Ap2 + A20
        A01p = A01 + A1p + Ap0
        s[0] = jitdot(A012, Ap12) / normsqrA012
        s[1] = jitdot(A012, A0p2) / normsqrA012
        s[2] = jitdot(A012, A01p) / normsqrA012
        return s

    def barcell(self, v):
        """returns vertex positions of barycentric cell dual to v
        midpoints of edges and barycenters of faces"""
        V = []
        x0, y0, z0 = self.vertices[v]
        h = self.V_hedge[v]
        h_start = h
        while True:
            v1 = self.H_vertex[h]
            x1, y1, z1 = self.vertices[v1]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            x2, y2, z2 = self.vertices[v2]
            V.append([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
            V.append([(x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3, (z0 + z1 + z2) / 3])

            if h == h_start:
                break
        return np.array(V)

    def vorcell(self, v):
        """returns vertex positions of voroni cell dual to v"""
        V = []
        r0 = self.vertices[v]
        h = self.V_hedge[v]
        h_start = h
        while True:
            v1 = self.H_vertex[h]
            r1 = self.vertices[v1]
            h = self.H_prev[h]
            h = self.H_twin[h]
            v2 = self.H_vertex[h]
            r2 = self.vertices[v2]
            r01 = r1 - r0
            r12 = r2 - r1
            r20 = r0 - r2
            normsqr_r12 = r12[0] ** 2 + r12[1] ** 2 + r12[2] ** 2
            normsqr_r20 = r20[0] ** 2 + r20[1] ** 2 + r20[2] ** 2
            normsqr_r01 = r01[0] ** 2 + r01[1] ** 2 + r01[2] ** 2
            #######################
            c0 = normsqr_r12 * (normsqr_r20 + normsqr_r01 - normsqr_r12)
            c1 = normsqr_r20 * (normsqr_r01 + normsqr_r12 - normsqr_r20)
            c2 = normsqr_r01 * (normsqr_r12 + normsqr_r20 - normsqr_r01)
            c012 = c0 + c1 + c2
            c0 /= c012
            c1 /= c012
            c2 /= c012
            #######################
            # r01_x_r12 = jitcross(r01, r12)
            # normsqr_r01_x_r12 = (
            #     r01_x_r12[0] ** 2 + r01_x_r12[1] ** 2 + r01_x_r12[2] ** 2
            # )
            # r01_dot_r20 = jitdot(r01, r20)
            # r01_dot_r12 = jitdot(r01, r12)
            # r20_dot_r12 = jitdot(r20, r12)
            # c0 = -normsqr_r12 * r01_dot_r20 / (2 * normsqr_r01_x_r12)
            # c1 = -normsqr_r20 * r01_dot_r12 / (2 * normsqr_r01_x_r12)
            # c2 = -normsqr_r01 * r20_dot_r12 / (2 * normsqr_r01_x_r12)
            #######################
            midpoint01 = (r1 + r0) / 2
            circumcenter = c0 * r0 + c1 * r1 + c2 * r2

            V.append([*midpoint01])
            V.append([*circumcenter])

            if h == h_start:
                break
        return np.array(V)

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

    def barcell_area(self, v):
        """area of cell dual to vertex v"""
        r = self.vertex_position(v)
        A = 0.0
        h = self.h_of_v(v)
        h_start = h
        while True:
            v1 = self.v_of_h(h)
            r1 = self.vertex_position(v1)
            h = self.prev(h)
            h = self.twin(h)
            v2 = self.v_of_h(h)
            r2 = self.vertex_position(v2)
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
        Nv = len(self.vertices)
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

    def signed_volume_of_face(self, f):
        v0, v1, v2 = self.face(f)
        p0 = self.vertex_position(v0)
        p1 = self.vertex_position(v1)
        p2 = self.vertex_position(v2)
        vol = triprod(p0, p1, p2) / 6
        return vol

    def volume_of_mesh(self):
        Nf = len(self.faces)
        vol = 0.0
        for f in range(Nf):
            vol += self.signed_volume_of_face(f)
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
        r = self.vertex_position(v)
        h = self.h_of_v(v)
        h_start = h
        defect = 2 * np.pi
        h = h_start
        while True:
            v1 = self.v_of_h(h)
            r1 = self.vertex_position(v1)
            h = self.prev(h)
            h = self.twin(h)
            v2 = self.v_of_h(h)
            r2 = self.vertex_position(v2)
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

    def old_angle_defect(self, v):
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

    def mean_curvature_vector_cot(self, v):
        Atot = 0.0
        r = self.vertex_position(v)
        Hvec = np.zeros(3)

        h_start = self.h_of_v(v)
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
        r = self.vertex_position(v)
        Hvec = np.zeros(3)

        h_start = self.h_of_v(v)
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

    def cotan_laplacian(self, Y):
        """computes the laplacian of Y at each vertex"""
        Nv = self.V_pq.shape[0]
        lapY = np.zeros_like(Y)
        for v in range(Nv):
            Atot = 0.0
            y = Y[v]
            r = self.V_pq[v, :3]
            r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
            h_start = self.V_hedge[v]
            h = h_start
            while True:
                v1 = self.H_vertex[h]
                r1 = self.V_pq[v1, :3]
                y1 = Y[v1]
                h = self.H_twin[self.H_prev[h]]
                v2 = self.H_vertex[h]
                r2 = self.V_pq[v2, :3]
                y2 = Y[v2]

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

                lapY[v] += -(cot_alpha * (y2 - y) / 2 + cot_beta * (y1 - y) / 2)
                Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8

                if h == h_start:
                    break
            lapY[v] /= Atot

        return lapY

    ###########################################################################
    # mesh mutation/regularization functions #
    ###############################
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

    def update_h_of_f(self, f, h_new):
        self.F_hedge[f] = h_new

    def update_h_of_v(self, v, h_new):
        self.V_hedge[v] = h_new

    def update_twin(self, h, h_twin):
        self.H_twin[h] = h_twin
        self.H_twin[h_twin] = h

    def update_next_prev(self, h0, h1):
        """assigns next(h0)=h1 and prev(h1)=h0"""
        self.H_next[h0] = h1
        self.H_prev[h1] = h0

    def update_v_of_h(self, h, v):
        self.H_vertex[h] = v

    def update_f_of_h(self, h, f):
        self.H_face[h] = f

    def update_face(self, f, face):
        v0, v1, v2 = face
        self.faces[f] = np.array([v0, v1, v2], dtype=np.int32)

    def update_halfedge(self, h, halfedge):
        v0, v1 = halfedge
        self.halfedges[h] = np.array([v0, v1], dtype=np.int32)

    def update_vertex_position(self, v, xyz):
        x, y, z = xyz
        self.V_pq[v, :3] = np.array([x, y, z])
        self.vertices[v] = np.array([x, y, z])

    def _edge_flip(self, h):
        r"""
        h/ht can not be on boundary!
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

        ht = self.twin(h)
        h1 = self.next(h)
        h2 = self.prev(h)
        h3 = self.next(ht)
        h4 = self.prev(ht)
        f1 = self.f_of_h(h)
        f2 = self.f_of_h(ht)
        v1 = self.v_of_h(h4)
        v2 = self.v_of_h(h1)
        v3 = self.v_of_h(h2)
        v4 = self.v_of_h(h3)

        self.update_face(f1, np.array([v2, v3, v4]))
        self.update_face(f2, np.array([v4, v1, v2]))
        # and halfedge referenced by new faces
        self.update_h_of_f(f1, h2)
        self.update_h_of_f(f2, h4)

        self.update_halfedge(h, np.array([v4, v2]))
        self.update_halfedge(ht, np.array([v2, v4]))
        # update next/prev halfedge
        self.update_next_prev(h, h2)
        self.update_next_prev(h2, h3)
        self.update_next_prev(h3, h)
        self.update_next_prev(ht, h4)
        self.update_next_prev(h4, h1)
        self.update_next_prev(h1, ht)
        # update face referenced by halfedges
        self.update_f_of_h(h3, f1)

        self.update_f_of_h(h1, f2)

        # update vert referenced by new halfedges
        # and halfedge referenced by verts
        self.update_v_of_h(h, v2)
        self.update_v_of_h(ht, v4)

        self.update_h_of_v(v3, h3)
        self.update_h_of_v(v1, h1)
        self.update_h_of_v(v2, h2)
        self.update_h_of_v(v4, h4)

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

        self.faces[f1] = np.array([v2, v3, v4])
        self.faces[f2] = np.array([v4, v1, v2])
        # and halfedge referenced by new faces
        self.F_hedge[f1] = h2
        self.F_hedge[f2] = h4

        self.halfedges[h] = np.array([v4, v2])
        self.halfedges[ht] = np.array([v2, v4])
        # update next/prev halfedge
        self.H_next[h] = h2
        self.H_prev[h2] = h
        # self.update_next_prev(h2, h3)
        self.H_next[h2] = h3
        self.H_prev[h3] = h2
        # self.update_next_prev(h3, h)
        self.H_next[h3] = h
        self.H_prev[h] = h3
        # self.update_next_prev(ht, h4)
        self.H_next[ht] = h4
        self.H_prev[h4] = ht
        # self.update_next_prev(h4, h1)
        self.H_next[h4] = h1
        self.H_prev[h1] = h4
        # self.update_next_prev(h1, ht)
        self.H_next[h1] = ht
        self.H_prev[ht] = h1
        # update face referenced by halfedges
        # self.update_f_of_h(h3, f1)
        self.H_face[h3] = f1

        # self.update_f_of_h(h1, f2)
        self.H_face[h1] = f2

        # update vert referenced by new halfedges
        # and halfedge referenced by verts
        # self.update_v_of_h(h, v2)
        self.H_vertex[h] = v2
        # self.update_v_of_h(ht, v4)
        self.H_vertex[ht] = v4

        # self.update_h_of_v(v3, h3)
        self.V_hedge[v3] = h3
        # self.update_h_of_v(v1, h1)
        self.V_hedge[v1] = h1
        # self.update_h_of_v(v2, h2)
        self.V_hedge[v2] = h2
        # self.update_h_of_v(v4, h4)
        self.V_hedge[v4] = h4

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

        ht = self.twin(h)
        h1 = self.next(h)
        h3 = self.next(ht)
        v1 = self.v_of_h(h)
        v2 = self.v_of_h(h1)
        v3 = self.v_of_h(ht)
        v4 = self.v_of_h(h3)

        val1 = self.valence(v1)
        val2 = self.valence(v2)
        val3 = self.valence(v3)
        val4 = self.valence(v4)

        flip_it = val1 - val2 + val3 - val4 > 2

        return flip_it

    def is_delaunay(self, h):
        r"""
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

    def shift_vertex_towards_barycenter(self, v, weight):
        """Translates vertex in the direction of the barycenter of
        its neighbors. weight=1 means all the way to the barycenter."""
        r0 = self.vertex_position(v)
        h = self.h_of_v(v)
        h_start = h
        r = np.zeros(3)
        val = 0
        while True:
            vb = self.v_of_h(h)
            r += self.vertex_position(vb)
            val += 1
            h = self.prev(h)
            h = self.twin(h)
            if h == h_start:
                break
        r /= val
        r = weight * r + (1 - weight) * r0
        self.update_vertex_position(v, r)

    def regularize_by_flips(self):
        Nh = len(self.halfedges)
        Nflips = 0
        for h in range(Nh):
            flip_it = self.flip_helps_valence(h)
            if flip_it:
                self.edge_flip(h)
                Nflips += 1
        return Nflips

    def regularize_by_shifts(self, weight):
        Nv = len(self.vertices)
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

    def edge_curve_length(self, h):
        vj = self.H_vertex[h]
        vi = self.H_vertex[self.H_twin[h]]
        ri, qi = self.V_pq[vi, :3], self.V_pq[vi, 3:]
        rj, qj = self.V_pq[vj, :3], self.V_pq[vj, 3:]
        rij = rj - ri
        normrij = jitnorm(rij)
        u = rij / normrij
        q = mul_quaternion(qi, inv_quaternion(qj))
        quq = rotate_by_quaternion(q, u)

        Lij = (normrij / np.sqrt(2)) * np.sqrt(
            1 + u[0] * quq[0] + u[1] * quq[1] + u[2] * quq[2]
        )
        return Lij

    ###########################################################################
    # simulation/data functions #
    ############################
    def mesh_data_min(self):
        """minimum data required to reconstruct mesh"""
        V_pq = self.V_pq
        H_vertex = self.H_vertex
        H_face = self.H_face
        H_next = self.H_next
        H_prev = self.H_prev
        H_twin = self.H_twin
        F_hedge = self.F_hedge
        return (
            V_pq,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            F_hedge,
        )

    def mesh_data(self):
        """(V_pq, V_hedge, halfedges, H_vertex, H_face, H_next, H_prev, H_twin, faces, F_hedge)"""
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

    def param_data(self):
        """to reconstruct mesh"""
        bending_modulus = self.Kbend
        splay_modulus = self.Ksplay
        volume_reg_stiffness = self.Kvolume
        area_reg_stiffness = self.Karea
        length_reg_stiffness = self.Klength
        linear_drag_coeff = self.linear_drag_coeff
        spontaneous_curvature = self.spontaneous_curvature
        preferred_edge_length = self.preferred_edge_length
        preferred_cell_area = self.preferred_cell_area
        preferred_total_volume = self.preferred_total_volume
        total_volume = self.total_volume
        preferred_total_area = self.preferred_total_area
        return np.array(
            [
                bending_modulus,
                splay_modulus,
                volume_reg_stiffness,
                area_reg_stiffness,
                length_reg_stiffness,
                linear_drag_coeff,
                spontaneous_curvature,
                preferred_edge_length,
                preferred_cell_area,
                preferred_total_volume,
                preferred_total_area,
            ]
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

    def visual_data(self):
        V_rgb = self.V_rgb
        V_normal_rgb = self.V_normal_rgb
        V_tangent1_rgb = self.V_tangent1_rgb
        V_tangent2_rgb = self.V_tangent2_rgb
        V_radius = self.V_radius

        H_rgb = self.H_rgb
        # H_radius = self.H_radius
        F_rgb = self.F_rgb
        F_opacity = self.F_opacity
        H_opacity = self.H_opacity
        V_opacity = self.V_opacity
        return (
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
        )

    def get_state_data(self):
        sample_times = self.sample_times
        V_pq_samples = self.V_pq_samples

        # topological/combinatorial
        faces = self.faces
        halfedges = self.halfedges
        V_hedge = self.V_hedge
        H_vertex = self.H_vertex
        H_face = self.H_face
        H_next = self.H_next
        H_prev = self.H_prev
        H_twin = self.H_twin
        H_isboundary = self.H_isboundary
        F_hedge = self.F_hedge
        # V_pq = self.V_pq
        return (
            sample_times,
            V_pq_samples,
            faces,
            halfedges,
            V_hedge,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            H_isboundary,
            F_hedge,
        )

    def get_plot_data(self):
        V_pq = self.V_pq
        faces = self.faces
        V_rgb = self.V_rgb
        V_radius = self.V_radius
        H_rgb = self.H_rgb
        F_rgb = self.F_rgb
        F_opacity = self.F_opacity
        H_opacity = self.H_opacity
        V_opacity = self.V_opacity
        V_normal_rgb = self.V_normal_rgb
        V_frames = self.orthogonal_matrices()
        # plot_data = {"V_pq": V_pq, "faces": faces, "V_rgb": V_rgb}
        return (
            V_pq,
            faces,
            V_rgb,
            V_radius,
            H_rgb,
            F_rgb,
            F_opacity,
            H_opacity,
            V_opacity,
            V_normal_rgb,
            V_frames,
        )
        # return plot_data

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
        xyz = self.vertex_position(v)
        neighbors = self.v_adjacent_to_v(v)
        N = len(neighbors)
        F = np.zeros(3)

        for _v0 in range(0, N):
            v0 = neighbors[_v0]
            xyz0 = self.vertex_position(v0)
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
        r = self.vertex_position(v)
        F = np.zeros(3)

        h_start = self.h_of_v(v)
        h = h_start
        while True:
            v1 = self.v_of_h(h)
            r1 = self.vertex_position(v1)
            h = self.prev(h)
            h = self.twin(h)
            v2 = self.v_of_h(h)
            r2 = self.vertex_position(v2)

            Avec = (jitcross(r, r1) + jitcross(r1, r2) + jitcross(r2, r)) / 2
            A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3

            gradA = jitcross(Avec, r2 - r1) / (2 * A)

            F += -Ka * (A - A0) * gradA / A0

            if h == h_start:
                break
        return F

    def area_reg_force(self, v):
        """E ~ 1/2*Ka*(A-A0)**2"""
        r = self.vertex_position(v)
        F = np.zeros(3)
        area0 = self.preferred_total_area
        area = self.total_area
        Ka = self.Karea

        h_start = self.h_of_v(v)
        h = h_start
        while True:
            v1 = self.v_of_h(h)
            r1 = self.vertex_position(v1)
            h = self.prev(h)
            h = self.twin(h)
            v2 = self.v_of_h(h)
            r2 = self.vertex_position(v2)

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

        h_start = self.h_of_v(v)
        h = h_start
        while True:
            v1 = self.v_of_h(h)
            r1 = self.vertex_position(v1)
            h = self.prev(h)
            h = self.twin(h)
            v2 = self.v_of_h(h)
            r2 = self.vertex_position(v2)

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

    def regularization_step(
        self,
        T,
        dt,
        weight,
        reg_length,
        reg_flips,
        reg_shifts,
        reg_local_area,
        reg_total_area,
        reg_local_volume,
        reg_total_volume,
    ):
        if reg_flips:
            self.regularize_by_flips()
        if reg_shifts:
            self.regularize_by_shifts(weight)
        Nv = len(self.V_pq)
        if reg_total_area:
            self.total_area = self.get_total_area()
        if reg_total_volume:
            self.total_volume = self.volume_of_mesh()

        t = 0
        while t <= T:
            for v in range(Nv):
                F = np.zeros(3)
                F += self.length_reg_force(v)
                F += self.area_reg_force(v)
                F += self.volume_reg_force(v)
                self.V_pq[v, :3] = self.V_pq[v, :3] + dt * F / self.linear_drag_coeff
            t += dt

        # for v in range(Nv):
        #     self.V_pq[v, 3:] = self.get_new_quat_dumb(v)

    def get_new_quat_dumb(self, v):
        qw, qx, qy, qz = self.V_pq[v, 3:]
        Q = np.zeros((3, 3))
        Q[:, 2] = self.area_weighted_vertex_normal(v)

        Q[0, 0] = qw**2 + qx**2 - qy**2 - qz**2
        Q[1, 0] = 2 * qw * qz + 2 * qx * qy
        Q[2, 0] = 2 * qx * qz - 2 * qw * qy
        Q[:, 0] -= jitdot(Q[:, 2], Q[:, 0]) * Q[:, 2]
        Q[:, 0] /= jitnorm(Q[:, 0])
        Q[:, 1] = jitcross(Q[:, 2], Q[:, 0])

        q = matrix_to_quaternion(Q)
        return q

    def get_new_euler_state(self, dt):
        Nv = len(self.V_pq)
        vertices = np.zeros((Nv, 3))
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

    def Fbend2(self):
        """"""
        Kbend = self.Kbend
        H, K = self.get_curvatures()
        Nv = H.shape[0]
        lapH = self.cotan_laplacian2(H)
        F = np.zeros((Nv, 3))
        # Fn = np.zeros(Nv)
        Fn = Kbend * (lapH + 2 * H * (K - H**2))
        for v in range(Nv):
            n = self.area_weighted_vertex_normal(v)
            A = self.vorcell_area(v)
            Fn[v] *= A
            F[v] = Fn[v] * n
        return F

    def Fbend(self):
        """from Tu"""
        Kbend = self.Kbend
        H, K = self.get_curvatures()
        Nv = H.shape[0]
        lapH = self.cotan_laplacian(H)
        F = np.zeros((Nv, 3))
        # Fn = np.zeros(Nv)
        # Fn = Kbend * (lapH + 2 * H * (H**2 - K))
        Fn = -Kbend * (lapH + 2 * H * (H**2 - K))
        for v in range(Nv):
            n = self.area_weighted_vertex_normal(v)
            A = self.vorcell_area(v)
            Fn[v] *= A
            F[v] = Fn[v] * n
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
            h_start = self.h_of_v(v)
            h = h_start
            while True:
                v1 = self.v_of_h(h)
                r1 = self.vertex_position(v1)
                h = self.prev(h)
                h = self.twin(h)
                v2 = self.v_of_h(h)
                r2 = self.vertex_position(v2)

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
            h_start = self.h_of_v(v)
            h = h_start
            while True:
                v1 = self.v_of_h(h)
                r1 = self.vertex_position(v1)
                h = self.prev(h)
                h = self.twin(h)
                v2 = self.v_of_h(h)
                r2 = self.vertex_position(v2)

                # grad_vol = jitcross(r1, r2) / (2 * A)

                F[v] += -Kv * (vol - vol0) * jitcross(r1, r2) / (18 * vol0)

                if h == h_start:
                    break
        return F

    ###########################################################################
    # visualization functions #
    ##########################
    def build_patch(self, v0):
        """assumes triangles"""

        V = [v0]
        F = []
        h_start = self.h_of_v(v0)
        ########################
        # h center-->boundary
        h = h_start
        while True:
            # vertex on boundary
            v = self.v_of_h(h)
            f = self.f_of_h(h)
            V.append(v)
            F.append(f)
            # h to left of boundary
            h = self.next(h)
            # h boundary-->center
            h = self.next(h)
            # h center-->boundary
            h = self.twin(h)
            if h == h_start:
                break

        Nverts = len(V)
        Nfaces = len(F)
        faces = np.zeros((Nfaces, 3), dtype=np.int32)
        vertices = np.zeros((Nverts, 3))
        for _f in range(Nfaces):
            f = F[_f]
            v0, v1, v2 = self.faces[f]
            faces[_f, 0] = V.index(v0)
            faces[_f, 1] = V.index(v1)
            faces[_f, 2] = V.index(v2)
        for v in V:
            vertices[v, :] = self.vertex_position(v)

        return vertices, faces

    def rigid_transform(self, PQ):
        Nverts = len(self.V_pq)
        for i in range(Nverts):
            self.V_pq[i] = mul_se3_quaternion(PQ, self.V_pq[i])
        self.vertices = self.V_pq[:, :3]

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
        scale = 0.8
        hp = self.prev(h)
        hn = self.next(h)
        v0 = self.v_of_h(hp)
        v1 = self.v_of_h(h)
        v2 = self.v_of_h(hn)

        xyz0 = self.vertex_position(v0)
        xyz1 = self.vertex_position(v1)
        xyz2 = self.vertex_position(v2)
        com = (xyz0 + xyz1 + xyz2) / 3

        u = xyz1 - xyz0
        p = xyz0
        ########
        # p += 0.5 * (1 - scale) * u
        p = shift_to_center * com + (1 - shift_to_center) * p
        u *= 1 - shift_to_center

        return p, u

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

    def v_adjacent_to_v(self, v):
        """
        gets vertices adjacent to v in counterclockwise order
        """
        neighbors = []

        h_start = self.h_of_v(v)
        h = h_start
        while True:
            neighbor = self.v_of_h(h)
            neighbors.append(neighbor)
            h = self.prev(h)
            h = self.twin(h)
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

    def vertex_position(self, v):
        """returns copy of vertex position"""
        xyz = np.zeros(3)
        xyz[:] = self.V_pq[v, :3]
        return xyz

    def vertex_positions(self):
        """returns copy of vertex position"""
        V_p = self.V_pq[:, :3].copy()
        return V_p

    def orthogonal_matrices(self):
        V_q = self.V_pq[:, 3:]
        Nv = len(V_q)
        R = np.zeros((Nv, 3, 3))
        for v in range(Nv):
            q = V_q[v]
            R[v] = quaternion_to_matrix(q)
        return R

    def halfedge_vector(self, h):
        """displacement vector of halfedge"""
        h_prev = self.H_prev[h]
        i0 = self.H_vertex[h_prev]
        i1 = self.H_vertex[h]
        v0 = self.V_pq[i0, :3]
        v1 = self.V_pq[i1, :3]
        return v1 - v0

    def hedge_vector(self, h):
        hp = self.prev(h)
        v0 = self.v_of_h(hp)
        v1 = self.v_of_h(h)
        xyz0 = self.vertex_position(v0)
        xyz1 = self.vertex_position(v1)
        u = xyz1 - xyz0
        return u

    def hedge_length(self, h):
        ht = self.twin(h)
        v1 = self.v_of_h(ht)
        v2 = self.v_of_h(h)
        xyz1 = self.vertex_position(v1)
        xyz2 = self.vertex_position(v2)
        u = xyz2 - xyz1
        L = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        return L

    def average_hedge_length(self):
        L = 0.0
        Nh = len(self.halfedges)
        for h in range(Nh):
            L += self.hedge_length(h) / Nh
        return L

    def area_weighted_vertex_normal(self, v):
        """."""
        n = np.zeros(3)

        h_start = self.h_of_v(v)
        h = h_start
        while True:
            f = self.f_of_h(h)
            n += self.face_area_vector(f)

            h = self.prev(h)
            h = self.twin(h)

            if h == h_start:
                break

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    def get_initial_edge_tangents(self):
        Nh = len(self.halfedges)
        H_tangent_components = np.zeros((Nh, 2))
        H_psi = np.zeros((Nh, 6))

        for h in range(Nh):
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


##########################################################
###############################################################
###########################################################
###########################################################
###########################################################
#  HERE HERE HEREHEREHEREHEREHEREHEREHEREHERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE
# HERE         HERE         HERE       HERE     HERE     HERE   HERE

###########################################################################
###########################################################################
# initialization functions #
############################
# these don't call any other class functions and are safe to use before any
# class attributes have been assigned
###########################################################################

# def _face_area_vector(self, f, vertices, H_next, H_vertex, F_hedge):
#     """directed area of face f"""
#     Avec = np.zeros(3)
#     h_start = F_hedge[f]
#     h = h_start
#     while True:
#         h_next = H_next[h]
#         i = H_vertex[h]
#         i_next = H_vertex[h_next]
#         Avec += 0.5 * jitcross(vertices[i], vertices[i_next])
#         h = h_next
#         if h == h_start:
#             break
#
#     return Avec

# def _average_face_area(self, vertices, H_next, H_vertex, F_label, F_hedge):
#     A = 0.0
#     N = len(F_label)
#     for f in F_label:
#         # Avec = self._face_area_vector(f)
#         Avec = np.zeros(3)
#         h_start = F_hedge[f]
#         h = h_start
#         while True:
#             h_next = H_next[h]
#             i = H_vertex[h]
#             i_next = H_vertex[h_next]
#             Avec += 0.5 * jitcross(vertices[i], vertices[i_next])
#             h = h_next
#             if h == h_start:
#                 break
#         A += np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / N
#     return A

###########################################################################
# simulation functions #
############################

###########################################################################
# mesh navigation functions #
############################

###########################################################################
# mesh mutation/regularization functions #
###############################

# def edge_colapse(self, h):

# def should_we_flip_edge(self, h, L):
#     r"""
#     h/ht can not be on boundary!
#       o              o
#      / \            /|\
#     o---o  |---->  o | o
#      \ /            \|/
#       o              o
#
#            v2                           v2
#          /    \                       /  |  \
#         /      \                     /   |   \
#        /h2    h1\                   /h2  |  h1\
#       /    f1    \                 /     |     \
#      /            \               /  f1  |  f2  \
#     /      h       \             /       |       \
#    v3--------------v1  |----->  v3      h|ht     v1
#     \      ht      /             \       |       /
#      \            /               \      |      /
#       \    f2    /                 \     |     /
#        \h3    h4/                   \h3  |  h4/
#         \      /                     \   |   /
#          \    /                       \  |  /
#            v4                           v4
#     """
#     # flip_its = []
#     flip_it = True
#     ht = self.twin(h)
#     h1 = self.next(h)
#     # h2 = self.prev(h)
#     h3 = self.next(ht)
#     # h4 = self.prev(ht)
#     v1 = self.v_of_h(h)
#     v2 = self.v_of_h(h1)
#     v3 = self.v_of_h(ht)
#     v4 = self.v_of_h(h3)
#
#     xyz1 = self.vertex_position(v1)
#     xyz2 = self.vertex_position(v2)
#     xyz3 = self.vertex_position(v3)
#     xyz4 = self.vertex_position(v4)
#     u_before = xyz1 - xyz3
#     u_after = xyz2 - xyz4
#     L_before = np.sqrt(u_before[0] ** 2 + u_before[1] ** 2 + u_before[2] ** 2)
#     L_after = np.sqrt(u_after[0] ** 2 + u_after[1] ** 2 + u_after[2] ** 2)
#     # flip_its.append(abs(L_after - L) < abs(L_before - L))
#     # flip_it = flip_it and (abs(L_after - L) < abs(L_before - L))
#     val1 = self.valence(v1)
#     val2 = self.valence(v2)
#     val3 = self.valence(v3)
#     val4 = self.valence(v4)
#     # flip_its.append(val3 > 2)
#     # flip_its.append(val1 > 2)
#     # flip_it = flip_it and (val3 > 3)
#     # flip_it = flip_it and (val1 > 3)
#     flip_it = flip_it and (val1 > val2)
#     flip_it = flip_it and (val3 > val4)
#     # flip_it = flip_it and (val1 > 3)
#
#     # flip_it = all(flip_its)
#     return flip_it

# def get_bad_hedges(self):
#     bad = []
#     for h in self.H_label:
#         flip_it = self.should_we_flip_edge(h, 1.0)
#
#         if flip_it:
#             bad.append(h)
#
#     # Nbad = len(bad)
#     # bada = np.zeros(Nbad)
#     bada = np.array(bad, dtype=np.int32)
#     return bada

# def flip_bad_edges(self):
#     L0 = self.average_hedge_length()
#     for h in self.H_label:
#         flip_it = self.should_we_flip_edge(h, L0)
#
#         if flip_it:
#             self.edge_flip(h)
#             # self.edge_flip(h)
#             ht = self.twin(h)
#             self.H_rgb[h] = np.array([1.0, 0.0, 0.0])
#             self.H_rgb[ht] = np.array([0.0, 0.0, 1.0])
#             f1 = self.f_of_h(h)
#             f2 = self.f_of_h(ht)
#             self.F_rgb[f1] = np.array([1.0, 0.0, 0.0])
#             self.F_rgb[f2] = np.array([0.0, 0.0, 1.0])

###########################################################################
# forces and time evolution #
#############################

###########################################################################
# visualization functions #
##########################

###########################################################################
# testing functions #
##########################
# def test_next(self):
#     success = True
#     bad = []
#     for h in self.H_label:
#         h0 = self.next(h)
#         h1 = self.next(h0)
#         h2 = self.next(h1)
#         if h - h2 != 0:
#             success = False
#             bad.append(h)
#     return success, bad
#
# def test_prev(self):
#     success = True
#     bad = []
#     for h in self.H_label:
#         h0 = self.prev(h)
#         h1 = self.prev(h0)
#         h2 = self.prev(h1)
#         if h - h2 != 0:
#             success = False
#             bad.append(h)
#
#     return success, bad
#
# def test_twin(self):
#     success = True
#     bad = []
#     for h in self.H_label:
#         h1 = self.twin(h)
#         h2 = self.twin(h1)
#         if h - h2 != 0:
#             success = False
#             bad.append(h)
#     return success, bad

###########################################################################
###########################################################################
# helper functions #
####################################

# def average_face_area(self):
#     A = 0.0
#     N = len(self.F_label)
#     for f in self.F_label:
#         Avec = self.face_area_vector(f)
#         A += np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / N
#     return A

# def area_weighted_vertex_normal(self, v):
#     """."""
#     n = np.zeros(3)
#
#     h_start = self.V_hedge[v]
#     h = h_start
#     while True:
#         f = self.H_face[h]
#         n += self.face_area_vector(f)
#
#         h = self.H_prev[h]
#         h = self.H_twin[h]
#
#         if h == h_start:
#             break
#
#     n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
#     return n

# def get_face_area_vectors(self):
#     F_label = self.F_label
#     Nfaces = len(F_label)
#     F_area_vectors = np.zeros((Nfaces, 3))
#     vertices = self.V_pq[:, :3]
#
#     for _f in range(Nfaces):
#         f = F_label[_f]
#         h = self.F_hedge[f]
#         hn = self.H_next[h]
#         hp = self.H_prev[h]
#
#         v0 = self.H_vertex[hp]
#         v1 = self.H_vertex[h]
#         v2 = self.H_vertex[hn]
#
#         u1 = vertices[v1] - vertices[v0]
#         u2 = vertices[v2] - vertices[v1]
#
#         F_area_vectors[_f] = jitcross(u1, u2)
#     # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
#     return F_area_vectors

# def get_vertex_normal(self, v):
#     F = self.f_adjacent_to_v(v)
#     n = np.zeros(3)
#
#     for f in F:
#         n += self.face_area_vector(f)
#
#     n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
#     return n

# def rcm_pose(self, vertex_list):
#     """computes the SE3-valued Riemannian center of mass of the euclidean transformations associated with vertices in vertex_list"""
#     iters = 4
#     pq = self.V_pq
#     Nsamps = len(vertex_list)
#     G = np.zeros((Nsamps, 7))
#     # G[:] = np.array([pq[i] for i in vertex_list])
#     ##########################################
#     g0 = pq[vertex_list[0]]  # G[0]
#     g0_inv = inv_se3_quaternion(g0)
#     for i in range(Nsamps):
#         G[i] = mul_se3_quaternion(g0_inv, pq[vertex_list[i]])
#     ##########################################
#     mu_g = np.array([G[0, 0], G[0, 1], G[0, 2], G[0, 3], G[0, 4], G[0, 5], G[0, 6]])
#     # mu_g = np.zeros(7)  # sum(G) / Nsamps
#     # for g in G:
#     #     mu_g += g / Nsamps
#     # mu_g[3:] /= np.sqrt(mu_g[3] ** 2 + mu_g[4] ** 2 + mu_g[5] ** 2 + mu_g[6] ** 2)
#
#     for iter in range(iters):
#         mu_g_inv = inv_se3_quaternion(mu_g)
#         Psi = np.zeros(6)
#         for g in G:
#             mu_g_inv_g = mul_se3_quaternion(mu_g_inv, g)
#             Psi += log_se3_quaternion(mu_g_inv_g) / Nsamps
#         mu_g = mul_se3_quaternion(mu_g, exp_se3_quaternion(Psi))
#         # for g in G:
#         #     g_mu_g_inv = mul_se3_quaternion(g, mu_g_inv)
#         #     Psi += log_se3_quaternion(g_mu_g_inv) / Nsamps
#         # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
#     ##########################################
#     mu_g = mul_se3_quaternion(g0, mu_g)
#     ##########################################
#     return mu_g
#
# def acm_pose(self, vertex_list):
#     """computes the affine center of mass of the euclidean transformations associated with vertices in vertex_list"""
#     G = self.V_pq
#     Nsamps = len(vertex_list)
#     g0 = G[vertex_list[0]]
#     mu_g = np.zeros_like(g0)
#     for i in vertex_list:
#         g = G[i]
#         mu_g += g / Nsamps
#     mu_g[3:] /= np.sqrt(mu_g[3] ** 2 + mu_g[4] ** 2 + mu_g[5] ** 2 + mu_g[6] ** 2)
#     return mu_g
#
# def rcm_quaternion(self, vertex_list):
#     """computes the unit quaternion-valued Riemannian center of mass of the euclidean transformations associated with vertices in vertex_list"""
#     iters = 4
#
#     G = self.V_pq[:, 3:]
#     Nsamps = len(vertex_list)
#     g0 = G[vertex_list[0]]
#     mu_g = np.zeros_like(g0)
#     mu_g[:] = g0
#     for iter in range(iters):
#         mu_g_inv = inv_quaternion(mu_g)
#         Psi = np.zeros(3)
#         for i in vertex_list:
#             g = G[i]
#             mu_g_inv_g = mul_quaternion(mu_g_inv, g)
#             Psi += log_unit_quaternion(mu_g_inv_g) / Nsamps
#             # Psi += log_se3_quaternion(mul_se3_quaternion(g, mu_g_inv)) / Nsamps
#         mu_g = mul_quaternion(mu_g, exp_unit_quaternion(Psi))
#         # mu_g = mul_se3_quaternion(exp_se3_quaternion(Psi), mu_g)
#     return mu_g

# def regularize_mesh_rcm(self):
#     pq = np.zeros_like(self.V_pq)
#
#     Nverts = len(pq)
#     for v in range(Nverts):
#         V = self.v_adjacent_to_v(v)
#         # Nsamps = len(V)
#         # G = np.array([PQ[_] for _ in V])
#         pq[v] = self.rcm_pose(V)
#     return pq
#
# def regularize_mesh_acm(self):
#     pq = np.zeros_like(self.V_pq)
#
#     Nverts = len(pq)
#     for v in range(Nverts):
#         V = self.v_adjacent_to_v(v)
#         # Nsamps = len(V)
#         # G = np.array([PQ[_] for _ in V])
#         pq[v] = self.acm_pose(V)
#     return pq
#
# def regularize_mesh_acm_quat(self):
#     pq = np.zeros_like(self.V_pq)
#
#     Nverts = len(pq)
#     for v in range(Nverts):
#         V = self.v_adjacent_to_v(v)
#         # Nsamps = len(V)
#         # G = np.array([PQ[_] for _ in V])
#         pq[v] = self.acm_pose(V)
#         pq[v, 3:] = self.rcm_quaternion(V)
#     return pq

# def face_normal(self, f):
#     h0 = self.F_hedge[f]
#     h1 = self.H_next[h0]
#     h2 = self.H_next[h1]
#
#     v0 = self.H_vertex[h0]
#     v1 = self.H_vertex[h1]
#     v2 = self.H_vertex[h2]
#
#     e1 = self.V_pq[v1, :3] - self.V_pq[v0, :3]
#     e2 = self.V_pq[v2, :3] - self.V_pq[v1, :3]
#
#     e3 = jitcross(e1, e2)
#     e3 /= np.sqrt(e3[0] ** 2 + e3[1] ** 2 + e3[2] ** 2)
#
#     return e3

# def v_of_f(self, f):
#     h_start = self.F_hedge[f]
#     V = []
#
#     h = h_start
#     while True:
#         V.append(self.H_vertex[h])
#         h = self.H_next[h]
#         if h == h_start:
#             break
#
#     return np.array(V, dtype=np.int32)
