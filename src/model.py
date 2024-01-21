from numba import float64, int32, boolean
from numba.experimental import jitclass
import numpy as np
from src.numdiff import (
    jitcross,
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
    index_of_nested,
)

# from mayavi import mlab

Brane_spec = [
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
    ###########################
    # Cosseratt stuff
    ("H_azim", float64[:, :]),
    ###########################
    # plotting stuff
    ("V_rgb", float64[:, :]),
    ("H_rgb", float64[:, :]),
    ("F_rgb", float64[:, :]),
    ("V_normal_rgb", float64[:, :]),
    ("V_tangent1_rgb", float64[:, :]),
    ("V_tangent2_rgb", float64[:, :]),
    ("V_radius", float64[:]),
    ("H_radius", float64[:]),
    ("V_scalar", float64[:]),
    ("H_scalar", float64[:]),
    ("F_scalar", float64[:]),
    ("F_opacity", float64),
    ("H_opacity", float64),
    ("V_opacity", float64),
    # ("name", str),
    # ("H_tangent_components", float64[:, :]),
    # ("H_psi", float64[:, :]),
]


@jitclass(Brane_spec)
class Brane:
    def __init__(self, vertices, faces):
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
        # self.faces, self.F_label = self.check_faces(faces, vertices)
        (
            _vertices,
            self.V_label,
            self.faces,
            self.F_label,
        ) = self.label_vertices_and_faces(vertices, faces)

        (
            self.halfedges,
            self.H_label,
            self.H_isboundary,
        ) = self.label_halfedges(vertices, faces)

        (
            self.V_hedge,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_hedge,
        ) = self.get_combinatorial_mesh_data()

        self.V_pq = self.frame_the_mesh(vertices)
        Nvertices = len(self.V_label)
        Nhedges = len(self.H_label)
        Nfaces = len(self.F_label)

        self.V_scalar = np.zeros(Nvertices)  # self.get_Gaussian_curvature()
        self.H_scalar = np.zeros(Nhedges)
        self.F_scalar = np.zeros(Nfaces)
        # self.H_psi, self.H_tangent_components = self.get_initial_edge_tangents()
        self.set_visuals()

    ###########################################################################
    # initialization functions #
    ############################
    def label_vertices_and_faces(self, vertices_in, faces_in):
        """assigns integers to vertices and faces"""
        vertices = vertices_in.copy()
        faces = faces_in.copy()
        Nvertices = len(vertices)
        Nfaces = len(faces)
        V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)

        return vertices, V_label, faces, F_label

    def label_halfedges(self, vertices, faces):
        """Builds halfedges from vertices and faces, assigns an integer-valued
        label/index to each halfedge, and determines whether the halfedge is
        contained in the boundary of the mesh."""
        halfedges = []
        H_isboundary = []
        H_label = []
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
                H_label.append(h)
                h += 1

        for hedge in halfedges:
            v0, v1 = hedge
            hedge_twin = [v1, v0]
            try:
                halfedges.index(hedge_twin)
            except Exception:
                halfedges.append(hedge_twin)
                H_isboundary.append(True)
                H_label.append(h)
                h += 1

        return (
            np.array(halfedges, dtype=np.int32),
            np.array(H_label, dtype=np.int32),
            np.array(H_isboundary),
        )

    def get_combinatorial_mesh_data(self):
        """."""
        V_label = self.V_label
        H_label = self.H_label
        F_label = self.F_label
        halfedges = self.halfedges

        H_isboundary = self.H_isboundary
        faces = self.faces.copy()
        ####################
        # vertices
        V_hedge = -np.ones_like(V_label)  # outgoing halfedge
        ####################
        # faces
        F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
        ####################
        # halfedges
        H_vertex = -np.ones_like(H_label)  # vertex it points to
        H_face = -np.ones_like(H_label)  # face it belongs to
        # next/previous halfedge inside the face (ordered counter-clockwise)
        H_next = -np.ones_like(H_label)
        H_prev = -np.ones_like(H_label)
        H_twin = -np.ones_like(H_label)  # opposite halfedge
        ####################

        # assign each face a halfedge
        # assign each interior halfedge previous/next halfedge
        # assign each interior halfedge a face
        # assign each halfedge a twin halfedge
        for f in F_label:
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
        for h in H_label:
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
            V_hedge,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            F_hedge,
        )

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
            # framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
            # framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
            # framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
            # framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 3:] = matrix_to_quaternion(R)

            framed_vertices[i, :3] = vertices[i, :]
        return framed_vertices

    def set_visuals(self):
        # face_color = np.array([0.0, 0.2667, 0.1059])
        face_color = np.array([0.0, 0.63335, 0.05295])
        face_alpha = 0.8

        hedge_color = np.array([1.0, 0.498, 0.0])
        hedge_alpha = 1.0
        hedge_radius = 0.0025

        vertex_color = np.array([1.0, 0.498, 0.0])  # np.array([0.7057, 0.0156, 0.1502])
        vertex_alpha = 1.0
        vertex_radius = 0.025

        normal_color = np.array([0.0, 0.0, 0.0])  # (1.0, 0.0, 0.0)
        tangent_color1 = np.array([0.7057, 0.0156, 0.1502])  # (1.0, 0.0, 0.0)
        tangent_color2 = np.array([0.2298, 0.2987, 0.7537])

        Nverts = len(self.V_label)
        Nhedges = len(self.H_label)
        Nfaces = len(self.F_label)

        self.V_rgb = np.zeros((Nverts, 3))
        self.V_normal_rgb = np.zeros((Nverts, 3))
        self.V_tangent1_rgb = np.zeros((Nverts, 3))
        self.V_tangent2_rgb = np.zeros((Nverts, 3))
        self.V_radius = np.zeros(Nverts)
        for _ in range(Nverts):
            self.V_rgb[_] = vertex_color
            self.V_normal_rgb[_] = normal_color
            self.V_tangent1_rgb[_] = tangent_color1
            self.V_tangent2_rgb[_] = tangent_color2
            self.V_radius[_] = vertex_radius
        self.H_rgb = np.zeros((Nhedges, 3))
        self.H_radius = np.zeros(Nhedges)
        for _ in range(Nhedges):
            self.H_rgb[_] = hedge_color
            self.H_radius[_] = hedge_radius
        self.F_rgb = np.zeros((Nfaces, 3))
        for _ in range(Nfaces):
            self.F_rgb[_] = face_color
        # V_scalar = np.array([])
        # H_scalar = np.array([])
        # F_scalar = np.array([])
        self.F_opacity = face_alpha
        self.H_opacity = hedge_alpha
        self.V_opacity = vertex_alpha

        # self.F_opacity = F_opacity

    # def get_cosserat_data(self):
    #     Nhedges = len(self.H_label)
    #     H_azim = np.zeros(Nhedges)
    #     for h in self.H_label:
    #         ht = self.twin(h)

    ###########################################################################
    # mesh navigation functions #
    ############################
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

    ###########################################################################
    # mesh regularization functions #
    ###############################
    def edge_flip(self, h):
        r"""
        h/ht can not be on boundary!
          o              o
         / \            /|\
        o---o  |---->  o | o
         \ /            \|/
          o              o

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
        # if self.h_of_f(f1) == h1:
        self.update_f_of_h(h1, f2)
        # if self.h_of_f(f2) == h3:

        # update vert referenced by new halfedges
        # and halfedge referenced by verts
        self.update_v_of_h(h, v2)
        self.update_v_of_h(ht, v4)
        # if self.h_of_v(v3) == h:
        # self.update_h_of_v(v3, h3)
        # # if self.h_of_v(v1) == ht:
        # self.update_h_of_v(v1, h1)
        self.update_h_of_v(v3, h3)
        self.update_h_of_v(v1, h1)
        self.update_h_of_v(v2, h2)
        self.update_h_of_v(v4, h4)

    def should_we_flip_edge(self, h, L):
        r"""
        h/ht can not be on boundary!
          o              o
         / \            /|\
        o---o  |---->  o | o
         \ /            \|/
          o              o

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
        # flip_its = []
        flip_it = True
        ht = self.twin(h)
        h1 = self.next(h)
        # h2 = self.prev(h)
        h3 = self.next(ht)
        # h4 = self.prev(ht)
        v1 = self.v_of_h(h)
        v2 = self.v_of_h(h1)
        v3 = self.v_of_h(ht)
        v4 = self.v_of_h(h3)

        xyz1 = self.vertex_position(v1)
        xyz2 = self.vertex_position(v2)
        xyz3 = self.vertex_position(v3)
        xyz4 = self.vertex_position(v4)
        u_before = xyz1 - xyz3
        u_after = xyz2 - xyz4
        L_before = np.sqrt(u_before[0] ** 2 + u_before[1] ** 2 + u_before[2] ** 2)
        L_after = np.sqrt(u_after[0] ** 2 + u_after[1] ** 2 + u_after[2] ** 2)
        # flip_its.append(abs(L_after - L) < abs(L_before - L))
        # flip_it = flip_it and (abs(L_after - L) < abs(L_before - L))
        val1 = self.valence(v1)
        val2 = self.valence(v2)
        val3 = self.valence(v3)
        val4 = self.valence(v4)
        # flip_its.append(val3 > 2)
        # flip_its.append(val1 > 2)
        # flip_it = flip_it and (val3 > 3)
        # flip_it = flip_it and (val1 > 3)
        flip_it = flip_it and (val1 > val2)
        flip_it = flip_it and (val3 > val4)
        # flip_it = flip_it and (val1 > 3)

        # flip_it = all(flip_its)
        return flip_it

    def get_bad_hedges(self):
        bad = []
        for h in self.H_label:
            flip_it = self.should_we_flip_edge(h, 1.0)

            if flip_it:
                bad.append(h)

        # Nbad = len(bad)
        # bada = np.zeros(Nbad)
        bada = np.array(bad, dtype=np.int32)
        return bada

    def flip_bad_edges(self):
        Lave = self.avereage_hedge_length()
        for h in self.H_label:
            flip_it = self.should_we_flip_edge(h, Lave)

            if flip_it:
                self.edge_flip(h)
                # self.edge_flip(h)
                ht = self.twin(h)
                self.H_rgb[h] = np.array([1.0, 0.0, 0.0])
                self.H_rgb[ht] = np.array([0.0, 0.0, 1.0])
                f1 = self.f_of_h(h)
                f2 = self.f_of_h(ht)
                self.F_rgb[f1] = np.array([1.0, 0.0, 0.0])
                self.F_rgb[f2] = np.array([0.0, 0.0, 1.0])

    def F_reg_length(self, v, Lpref):
        Ke = 1e-2
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
            F += -Ke * (L - Lpref) * gradL / Lpref

        return F

    def time_step(self, dt):
        v = 2 * dt

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
        # self.vertices = self.V_pq[:, :3]

    def hedge_vectors(self):
        Nhedges = len(self.H_label)
        vecs = np.zeros((Nhedges, 3))
        points = np.zeros((Nhedges, 3))
        for h in self.H_label:
            points[h, :], vecs[h, :] = self.hedge_vector(h)

        return points, vecs

    def hedge_vector(self, h):
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
    # testing functions #
    ##########################
    def test_next(self):
        success = True
        bad = []
        for h in self.H_label:
            h0 = self.next(h)
            h1 = self.next(h0)
            h2 = self.next(h1)
            if h - h2 != 0:
                success = False
                bad.append(h)
        return success, bad

    def test_prev(self):
        success = True
        bad = []
        for h in self.H_label:
            h0 = self.prev(h)
            h1 = self.prev(h0)
            h2 = self.prev(h1)
            if h - h2 != 0:
                success = False
                bad.append(h)

        return success, bad

    def test_twin(self):
        success = True
        bad = []
        for h in self.H_label:
            h1 = self.twin(h)
            h2 = self.twin(h1)
            if h - h2 != 0:
                success = False
                bad.append(h)
        return success, bad

    ###########################################################################
    ###########################################################################
    # helper functions #
    ####################################
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
        h_start = self.h_of_v(v)
        val = 0

        h = h_start
        while True:
            # neighbors.append(self.H_vertex[h])
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
        Q = self.V_pq[:, 3:]
        Nv = len(Q)
        R = np.zeros((Nv, 3, 3))
        for v in range(Nv):
            q = Q[v]
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

    def area_weighted_vertex_normal(self, v):
        """."""
        A = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            f = self.H_face[h]
            A += self.face_area_vector(f)

            h = self.H_prev[h]
            h = self.H_twin[h]

            if h == h_start:
                break

        return A

    def hedge_length(self, h):
        ht = self.twin(h)
        v1 = self.v_of_h(ht)
        v2 = self.v_of_h(h)
        xyz1 = self.vertex_position(v1)
        xyz2 = self.vertex_position(v2)
        u = xyz2 - xyz1
        L = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        return L

    def avereage_hedge_length(self):
        L = 0.0
        N = len(self.H_label)
        for h in self.H_label:
            L += self.hedge_length(h) / N
        return L

    #######################################################

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
        # for v0 in range(Nverts):
        v0 = 0
        K = np.random.rand(Nverts)
        while False:
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

    # def star(self, v):


# @jitclass(Brane_spec)
class testBrane:
    def __init__(self, vertices, faces):
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
        # self.faces, self.F_label = self.check_faces(faces, vertices)
        (
            self.vertices,
            self.V_label,
            self.faces,
            self.F_label,
        ) = self.label_vertices_and_faces(vertices, faces)

        (
            self.halfedges,
            self.H_label,
            self.H_isboundary,
        ) = self.label_halfedges(vertices, faces)

        (
            self.V_hedge,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_hedge,
        ) = self.get_combinatorial_mesh_data()

        self.V_pq = self.frame_the_mesh2(vertices)

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

        # self.V_pq = self.frame_the_mesh(vertices)
        # self.V_scalar = self.get_Gaussian_curvature()
        # self.H_psi, self.H_tangent_components = self.get_initial_edge_tangents()

    def frame_the_mesh2(self, vertices):
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
            # framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
            # framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
            # framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
            # framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 3:] = matrix_to_quaternion(R)

            framed_vertices[i, :3] = vertices[i, :]
        return framed_vertices

    def get_combinatorial_mesh_data(self):
        """og is the one with _"""
        V_label = self.V_label
        H_label = self.H_label
        F_label = self.F_label
        halfedges = self.halfedges

        H_isboundary = self.H_isboundary
        faces = self.faces.copy()
        ####################
        # vertices
        V_hedge = -np.ones_like(V_label)  # outgoing halfedge
        ####################
        # faces
        F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
        ####################
        # halfedges
        H_vertex = -np.ones_like(H_label)  # vertex it points to
        H_face = -np.ones_like(H_label)  # face it belongs to
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
            hedge0 = np.array([face[0], face[1]])
            # h0 = halfedges_list.index(hedge0)
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
                # h = halfedges_list.index(hedge)
                h = index_of_nested(halfedges, hedge)
                # get incident halfedge
                hedge_prev = np.array([vm1, v0])
                # h_prev = halfedges_list.index(hedge_prev)
                h_prev = index_of_nested(halfedges, hedge_prev)
                # assign previous/next halfedge
                H_prev[h] = h_prev
                H_next[h_prev] = h
                # assign face to halfedge
                H_face[h] = f

                hedge_twin = np.array([vp1, v0])
                # h = halfedges_list.index(hedge)
                h_t = index_of_nested(halfedges, hedge_twin)
                H_twin[h] = h_t
                H_twin[h_t] = h

        # assign each halfedge a twin halfedge

        # assign each halfedge a vertex
        # assign each vertex a halfedge
        # assign each boundary halfedge previous/next halfedge
        for h in H_label:
            v0, v1 = halfedges[h]
            # hedge_twin = np.array([v1, v0])
            # h_twin = halfedges_list.index(hedge_twin)
            # h_twin = index_of_nested(halfedges, hedge_twin)
            # H_twin[h] = h_twin
            H_vertex[h] = v1
            if V_hedge[v0] == -1:
                V_hedge[v0] = h

            if H_isboundary[h]:
                # h_t = h_twin
                # h_tp = H_prev[h_t]
                #
                # h_tpt = H_twin[h_tp]
                # h_tptp = H_prev[h_tpt]
                #
                # h_tptpt = H_twin[h_tptp]
                # h_tptptp = H_prev[h_tptpt]
                #
                # h_next = H_twin[h_tptptp]
                # H_next[h] = h_next
                # H_prev[h_next] = h
                # h_next = self.twin(h)
                h_next = H_twin[h]
                while True:
                    # h_next = self.twin(self.prev(h_next))
                    h_next = H_twin[H_prev[h_next]]
                    if H_isboundary[h_next]:
                        break
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

    def label_halfedges(self, vertices, faces):
        """Builds halfedges from vertices and faces, assigns an integer-valued
        label/index to each halfedge, and determines whether the halfedge is
        contained in the boundary of the mesh."""
        halfedges = []
        H_isboundary = []
        H_label = []
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
                H_label.append(h)
                h += 1

        for hedge in halfedges:
            v0, v1 = hedge
            hedge_twin = [v1, v0]
            try:
                halfedges.index(hedge_twin)
            except Exception:
                halfedges.append(hedge_twin)
                H_isboundary.append(True)
                H_label.append(h)
                h += 1

        return (
            np.array(halfedges, dtype=np.int32),
            np.array(H_label, dtype=np.int32),
            np.array(H_isboundary),
        )

    ###########################################################################
    # initialization functions #
    ############################
    def label_vertices_and_faces(self, vertices_in, faces_in):
        """assigns integers to vertices and faces"""
        vertices = vertices_in.copy()
        faces = faces_in.copy()
        Nvertices = len(vertices)
        Nfaces = len(faces)
        V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
        F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)

        return vertices, V_label, faces, F_label

    def build_and_label_halfedges(self, vertices, faces):
        """Builds halfedges from vertices and faces, assigns an integer-valued
        label/index to each halfedge, and determines whether the halfedge is
        contained in the boundary of the mesh."""
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
                # index shift to get next
                _next = (_ + 1) % N_v_of_f
                v0 = face[_]  #
                v1 = face[_next]
                hedge = [v0, v1]
                hedge_twin = [v1, v0]
                # get index of hedge if it already exists, or create
                # index if it doesn't exist yet
                try:
                    h = halfedges.index(hedge)
                except Exception:
                    halfedges.append(hedge)
                    h = halfedges.index(hedge)
                # get index of hedge_twin if it already exists, or create
                # index if it doesn't exist yet
                try:
                    ht = halfedges.index(hedge_twin)
                except Exception:
                    halfedges.append(hedge_twin)
                # if halfedge is contained in a face, add its index to
                # interior_hedge_labels set
                interior_hedge_labels.add(h)

        # if h wasn't added to interior_hedge_labels, then it isn't contained
        # in a face and must be on the boundary
        Nhedges = len(halfedges)
        for h in range(Nhedges):
            H_isboundary.append(h not in interior_hedge_labels)
            H_label.append(h)

        return (
            np.array(halfedges, dtype=np.int32),
            np.array(H_label, dtype=np.int32),
            np.array(H_isboundary),
        )

    def _get_combinatorial_mesh_data(self):
        """"""
        V_label = self.V_label
        H_label = self.H_label
        F_label = self.F_label
        halfedges = self.halfedges
        # make list of halfedges to use .index list method
        # halfedges_list = []
        # for hedge in halfedges:
        #     v0, v1 = hedge
        #     halfedges_list.append([v0, v1])

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
        H_face = -np.ones_like(H_label)  # face it belongs to
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
            hedge0 = np.array([face[0], face[1]])
            # h0 = halfedges_list.index(hedge0)
            h0 = index_of_nested(halfedges, hedge0)
            F_hedge[f] = h0  # assign each face a halfedge
            for _ in range(N_v_of_f):
                # for each vertex in face, get the indices of the
                # previous/next vertex
                _p1 = (
                    _ + 1
                ) % N_v_of_f  # np.mod(_ + 1, N_v_of_f)  # index shift to get next
                _m1 = (
                    _ - 1
                ) % N_v_of_f  # np.mod(_ - 1, N_v_of_f)  # index shift to get prev
                vm1 = face[_m1]
                v0 = face[_]
                vp1 = face[_p1]
                # get outgoing halfedge
                hedge = np.array([v0, vp1])
                # h = halfedges_list.index(hedge)
                h = index_of_nested(halfedges, hedge)
                # get incident halfedge
                hedge_prev = np.array([vm1, v0])
                # h_prev = halfedges_list.index(hedge_prev)
                h_prev = index_of_nested(halfedges, hedge_prev)
                # assign previous/next halfedge
                H_prev[h] = h_prev
                H_next[h_prev] = h
                # assign face to halfedge
                H_face[h] = f

        # assign each halfedge a twin halfedge
        # assign each halfedge a vertex
        # assign each vertex a halfedge
        # assign each boundary halfedge previous/next halfedge
        for h in H_label:
            v0, v1 = halfedges[h]
            hedge_twin = np.array([v1, v0])
            # h_twin = halfedges_list.index(hedge_twin)
            h_twin = index_of_nested(halfedges, hedge_twin)
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
            # framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
            # framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
            # framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
            # framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 3:] = matrix_to_quaternion(R)

            framed_vertices[i, :3] = vertices[i, :]
        return framed_vertices

    def check_faces(self):
        faces = self.faces.copy()
        F_label = self.F_label

        for f in F_label:
            v_of_f = faces[f]
            h = self.F_hedge[f]
            h_twin = self.H_twin[h]
            f_twin = self.H_face[h_twin]

    def remesh(self, vertices, faces):
        (
            self.vertices,
            self.V_label,
            self.faces,
            self.F_label,
        ) = self.label_vertices_and_faces(vertices, faces)

        (
            self.halfedges,
            self.H_label,
            self.H_isboundary,
        ) = self.build_and_label_halfedges(vertices, faces)

        (
            self.V_hedge,
            self.H_vertex,
            self.H_face,
            self.H_next,
            self.H_prev,
            self.H_twin,
            self.F_hedge,
        ) = self.get_combinatorial_mesh_data()
        self.V_pq = self.frame_the_mesh(vertices)

    ###########################################################################
    # helper functions #
    ####################################
    def halfedge_vector(self, h):
        """displacement vector of halfedge"""
        h_prev = self.H_prev[h]
        i0 = self.H_vertex[h_prev]
        i1 = self.H_vertex[h]
        v0 = self.V_pq[i0, :3]
        v1 = self.V_pq[i1, :3]
        return v1 - v0

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

    def area_weighted_vertex_normal(self, v):
        """."""
        A = np.zeros(3)

        h_start = self.V_hedge[v]
        h = h_start
        while True:
            f = self.H_face[h]
            A += self.face_area_vector(f)

            h = self.H_prev[h]
            h = self.H_twin[h]

            if h == h_start:
                break

        return A

    ###########################################################################
    # mesh navigation functions #
    ############################
    def twin(self, h):
        return self.H_twin[h]

    def next(self, h):
        return self.H_next[h]

    def prev(self, h):
        return self.H_prev[h]

    def vert(self, h):
        return self.H_vertex[h]

    def face(self, h):
        return self.H_face[h]

    def h_of_v(self, v):
        return self.V_hedge[v]

    def h_of_f(self, f):
        return self.F_hedge[f]

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

    ###########################################################################
    # mesh regulaization functions #
    ###############################
    def edge_flip(self, h):
        r"""
        h/ht can not be on boundary!
          o              o
         / \            /|\
        o---o  |---->  o | o
         \ /            \|/
          o              o

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

        # update next/prev halfedge
        self.update_next_prev(h, h2)
        self.update_next_prev(h2, h3)
        self.update_next_prev(h3, h)
        self.update_next_prev(ht, h4)
        self.update_next_prev(h4, h1)
        self.update_next_prev(h1, ht)
        # update face referenced by halfedges
        # and halfedge referenced by new faces
        self.update_f_of_h(h3, f1)
        # if self.h_of_f(f1) == h1:
        self.update_h_of_f(f1, h3)
        self.update_f_of_h(h1, f2)
        # if self.h_of_f(f2) == h3:
        self.update_h_of_f(f2, h1)
        # update vert referenced by new halfedges
        # and halfedge referenced by verts
        self.update_v_of_h(h, v2)
        self.update_v_of_h(ht, v4)
        # if self.h_of_v(v3) == h:
        self.update_h_of_v(v3, h3)
        # if self.h_of_v(v1) == ht:
        self.update_h_of_v(v1, h1)

    ###########################################################################

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
                _next = (
                    _ + 1
                ) % N_v_of_f  # np.mod(_ + 1, N_v_of_f)  # index shift to get next
                _prev = (
                    _ - 1
                ) % N_v_of_f  # np.mod(_ - 1, N_v_of_f)  # index shift to get prev
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
        # for v0 in range(Nverts):
        v0 = 0
        K = np.random.rand(Nverts)
        while False:
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
