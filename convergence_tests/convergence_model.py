import numpy as np
from source.ply_utils import HalfEdgeMeshData
from scipy.sparse import coo_matrix


class Brane:
    def __init__(self, V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge):
        self.V = np.copy(V)
        self.V_edge = np.copy(V_edge)
        self.E_vertex = np.copy(E_vertex)
        self.E_face = np.copy(E_face)
        self.E_next = np.copy(E_next)
        self.E_twin = np.copy(E_twin)
        self.F_edge = np.copy(F_edge)

        (
            self.V_rgba,
            self.V_normal_rgba,
            self.V_radius,
            self.E_rgba,
            self.F_rgba,
        ) = self.visual_defaults()

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        he_mesh = HalfEdgeMeshData.from_half_edge_ply(ply_path)
        V = he_mesh.V
        V_edge = he_mesh.V_edge
        E_vertex = he_mesh.E_vertex
        E_face = he_mesh.E_face
        E_next = he_mesh.E_next
        E_twin = he_mesh.E_twin
        F_edge = he_mesh.F_edge
        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        he_mesh = HalfEdgeMeshData.from_vertex_face_ply(ply_path)
        V = he_mesh.V
        V_edge = he_mesh.V_edge
        E_vertex = he_mesh.E_vertex
        E_face = he_mesh.E_face
        E_next = he_mesh.E_next
        E_twin = he_mesh.E_twin
        F_edge = he_mesh.F_edge
        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    ##########################################################################
    # mesh navigation #
    #####################################
    def get_face(self, f):
        face = []
        e = self.E_twin[self.E_next[self.E_twin[self.F_edge[f]]]]
        e_start = e
        while True:
            face.append(self.E_vertex[e])
            e = self.E_next[e]
            if e == e_start:
                break
        return np.array(face, dtype=np.uint32)

    def get_faces(self):
        Nfaces = len(self.F_edge)
        F = np.zeros((Nfaces, 3), dtype=np.int32)
        for f in range(Nfaces):
            e = self.F_edge[f]
            F[f, 0] = self.E_vertex[e]
            e = self.E_next[e]
            F[f, 1] = self.E_vertex[e]
            e = self.E_next[e]
            F[f, 2] = self.E_vertex[e]
        return F

    def get_edge(self, e):
        v1 = self.E_vertex[e]
        v0 = self.E_vertex[self.E_twin[e]]
        edge = np.array([v0, v1], dtype=np.uint32)
        return edge

    def prev(self, e):
        e_next = e
        while True:
            e_prev = e_next
            e_next = self.E_next[e_prev]
            if e_next == e:
                break
        return e_prev

    def next(self, e):

        return self.E_next[e]

    def twin(self, e):
        return self.E_twin[e]

    def e_of_v(self, v):
        return self.V_edge[v]

    def e_of_f(self, f):
        return self.F_edge[f]

    def v_of_e(self, e):
        return self.E_vertex[e]

    def f_of_e(self, e):
        return self.E_face[e]

    def get_V_one_ring_neighbors(self, i):
        """"""
        Vi = []
        e_start = self.V_edge[i]
        eij = e_start
        while True:
            Vi.append(self.E_vertex[eij])
            eij = self.E_twin[self.prev(eij)]
            if eij == e_start:
                break
        return Vi

    def get_E_one_ring_neighbors(self, v):
        """"""
        E_neighbors = []
        e_start = self.next(self.e_of_v(v))
        e = e_start
        while True:
            E_neighbors.append(e)
            e = self.next(self.twin(self.next(e)))
            if e == e_start:
                break
        return E_neighbors

    def get_order_n_plus_one_edges(self, E_inner):
        """
        Compute edges connecting (n+1)th-order neighbors of a vertex from the
        edges connecting nth-order neighbors.

        Args:
        E_inner: Indices of nth-order edges (counter-clockwise!)

        Returns:
            E_inner: Indices of (n+1)th-order edges (counter-clockwise!)
        """
        E_outer = []
        N_inner = len(E_inner)
        for _e in range(N_inner):
            e_inner = E_inner[_e]
            e_next_inner = E_inner[(_e + 1) % N_inner]
            v_break = self.v_of_e(e_next_inner)
            e_outer = self.next(self.twin(self.prev(self.twin(e_inner))))
            while True:
                E_outer.append(e_outer)
                e_outer = self.next(self.twin(self.next(e_outer)))
                v = self.v_of_e(e_outer)
                if v == v_break:
                    break

        return E_outer

    def valence(self, v):
        e_start = self.V_edge[v]
        val = 0
        e = e_start
        while True:
            val += 1
            e = self.twin(self.prev(e))
            if e == e_start:
                break
        return val

    ##########################################################################
    # visualization #
    #####################################
    def visual_defaults(self):
        face_rgba = np.array([0.0, 0.63335, 0.05295, 0.8])
        edge_rgba = np.array([1.0, 0.498, 0.0, 1.0])
        vertex_rgba = np.array([1.0, 0.498, 0.0, 1.0])  # np.array([0.7057, 0.0156, 0.1502])
        vertex_radius = 0.025
        normal_rgba = np.array([0.0, 0.0, 0.0, 1.0])  # (1.0, 0.0, 0.0)

        Nverts = len(self.V)
        Nedges = len(self.E_vertex)
        Nfaces = len(self.F_edge)
        V_rgba = np.zeros((Nverts, 4))
        V_normal_rgba = np.zeros((Nverts, 4))
        V_radius = np.zeros(Nverts)
        for _ in range(Nverts):
            V_rgba[_] = vertex_rgba
            V_normal_rgba[_] = normal_rgba
            V_radius[_] = vertex_radius
        E_rgba = np.zeros((Nedges, 4))
        for _ in range(Nedges):
            E_rgba[_] = edge_rgba
        F_rgba = np.zeros((Nfaces, 4))
        for _ in range(Nfaces):
            F_rgba[_] = face_rgba

        ##########################
        return (
            V_rgba,
            V_normal_rgba,
            V_radius,
            E_rgba,
            F_rgba,
        )

    def shifted_hedge_vectors(self):
        """halfedge vector shifted toward face centroid for visualization"""
        Ne = len(self.E_vertex)
        vecs = np.zeros((Ne, 3))
        points = np.zeros((Ne, 3))
        for e in range(Ne):
            points[e, :], vecs[e, :] = self.shifted_hedge_vector(e)
        return points, vecs

    def shifted_hedge_vector(self, e):
        shift_to_center = 0.15
        v0 = self.E_vertex[self.prev(e)]
        v1 = self.E_vertex[e]
        v2 = self.E_vertex[self.E_next[e]]
        com = (self.V[v0] + self.V[v1] + self.V[v2]) / 3
        p = shift_to_center * com + (1 - shift_to_center) * self.V[v0]
        u = (1 - shift_to_center) * (self.V[v1] - self.V[v0])
        return p, u

    ##########################################################################
    # geometry helper functions #
    #####################################
    def barcell_area(self, v):
        """area of cell dual to vertex v"""
        r = self.V[v]
        A = 0.0
        h = self.V_edge[v]
        h_start = h
        while True:
            v1 = self.E_vertex[h]
            r1 = self.V[v1]
            h = self.prev(h)
            h = self.E_twin[h]
            v2 = self.E_vertex[h]
            r2 = self.V[v2]
            A_face_vec = np.cross(r, r1) / 2 + np.cross(r1, r2) / 2 + np.cross(r2, r) / 2
            A_face = np.sqrt(A_face_vec[0] ** 2 + A_face_vec[1] ** 2 + A_face_vec[2] ** 2)
            A += A_face / 3

            if h == h_start:
                break
        return A

    def meyercell_area(self, v):
        """Meyer's mixed area of cell dual to vertex v"""
        Atot = 0.0
        ri = self.V[v]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        h_start = self.V_edge[v]
        hij = h_start
        while True:
            vj = self.E_vertex[hij]
            rj = self.V[vj]
            hijp1 = self.E_twin[self.prev(hij)]
            vjp1 = self.E_vertex[hijp1]
            rjp1 = self.V[vjp1]

            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
            rjp1_ri = rjp1[0] * ri[0] + rjp1[1] * ri[1] + rjp1[2] * ri[2]

            normDrij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
            # normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
            normDrjjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
            normDrjp1i = np.sqrt(rjp1_rjp1 - 2 * rjp1_ri + ri_ri)
            cos_thetajijp1 = (ri_ri + rj_rjp1 - ri_rj - rjp1_ri) / (normDrij * normDrjp1i)
            cos_thetajp1ji = (rj_rj + rjp1_ri - rj_rjp1 - ri_rj) / (normDrij * normDrjjp1)
            cos_thetaijp1j = (rjp1_rjp1 + ri_rj - rj_rjp1 - rjp1_ri) / (normDrjp1i * normDrjjp1)
            if cos_thetajijp1 < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += np.sqrt(semiP * (semiP - normDrij) * (semiP - normDrjjp1) * (semiP - normDrjp1i)) / 2
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 4
            elif cos_thetajp1ji < 0 or cos_thetaijp1j < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += np.sqrt(semiP * (semiP - normDrij) * (semiP - normDrjjp1) * (semiP - normDrjp1i)) / 4
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 8
            else:
                cot_thetaijp1j = cos_thetaijp1j / np.sqrt(1 - cos_thetaijp1j**2)
                cot_thetajp1ji = cos_thetajp1ji / np.sqrt(1 - cos_thetajp1ji**2)
                Atot += normDrij**2 * cot_thetaijp1j / 8 + normDrjp1i**2 * cot_thetajp1ji / 8

            hij = hijp1
            if hij == h_start:
                break

        return Atot

    def vorcell_area(self, v):
        """area of cell dual to vertex v"""
        Atot = 0.0
        r = self.V[v]
        r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
        h_start = self.V_edge[v]
        h = h_start
        while True:
            v1 = self.E_vertex[h]
            r1 = self.V[v1]
            h = self.E_twin[self.prev(h)]
            v2 = self.E_vertex[h]
            r2 = self.V[v2]

            r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
            r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
            r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
            r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
            r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]

            normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
            normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
            normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
            cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
            cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)

            cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
            cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
            Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8

            if h == h_start:
                break

        return Atot

    def face_area_vector(self, f):
        """directed area of face f"""
        A = np.zeros(3)

        h_start = self.F_edge[f]
        h = h_start
        while True:
            h_next = self.E_next[h]
            i = self.E_vertex[h]
            i_next = self.E_vertex[h_next]
            A += 0.5 * np.cross(self.V[i], self.V[i_next])
            h = h_next
            if h == h_start:
                break

        return A

    def area_weighted_vertex_normal(self, v):
        """."""
        n = np.zeros(3)

        h_start = self.V_edge[v]
        h = h_start
        while True:
            f = self.E_face[h]
            n += self.face_area_vector(f)
            h = self.E_twin[self.prev(h)]

            if h == h_start:
                break

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    def other_weighted_vertex_normal(self, v):
        """Weights for Computing Vertex Normals from Facet Normals Max99"""
        n = np.zeros(3)
        ri = self.V[v]
        h_start = self.V_edge[v]
        hj = h_start
        while True:
            hjp1 = self.E_twin[self.prev(hj)]
            vj = self.E_vertex[hj]
            vjp1 = self.E_vertex[hjp1]
            Drj = self.V[vj] - ri
            Drjp1 = self.V[vjp1] - ri
            Drj_dot_Drj = Drj[0] ** 2 + Drj[1] ** 2 + Drj[2] ** 2
            Drjp1_dot_Drjp1 = Drjp1[0] ** 2 + Drjp1[1] ** 2 + Drjp1[2] ** 2

            n += np.cross(Drj, Drjp1) / (Drj_dot_Drj * Drjp1_dot_Drjp1)
            hj = hjp1

            if hj == h_start:
                break

        n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        return n

    ##########################################################################
    # Heat Laplacian #
    #####################################
    # def get_neighbors_within_tol(self, v):
    #     neighbors = []
    #     return []

    def get_heat_laplacian_weights(self, cutoff_distance=1e-2):
        data = []
        row_indices = []
        col_indices = []
        Nv = len(self.V)
        for x in range(Nv):
            Mx = -1
            for y in range(Nv):
                len_xy = np.linalg.norm(self.V[y] - self.V[x])
                if len_xy <= cutoff_distance:
                    if Mx == -1:
                        Mx = self.meyercell_area(x)
                    My = self.meyercell_area(y)
                    Wxy = My * np.exp(-(len_xy**2) / (4 * Mx)) / (4 * np.pi * Mx**2)
                    data.append(Wxy)
                    row_indices.append(x)
                    col_indices.append(y)

        return data, row_indices, col_indices

    ##########################################################################
    # Cotan Laplacian #
    #####################################
    def get_cot_laplacian_weights(self):
        """L_ij(y_j-y_i)"""
        data = []
        row_indices = []
        col_indices = []
        Nv = len(self.V)
        for i in range(Nv):
            data_i = []
            Atot = 0.0
            ri = self.V[i]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            e_start = self.e_of_v(i)
            eij = e_start
            while True:
                eijm1 = self.next(self.twin(eij))
                eijp1 = self.twin(self.prev(eij))
                jm1 = self.v_of_e(eijm1)
                j = self.v_of_e(eij)
                jp1 = self.v_of_e(eijp1)

                rjm1 = self.V[jm1]
                rj = self.V[j]
                rjp1 = self.V[jp1]

                rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
                rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
                ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
                rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

                len_ijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
                len_jjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
                len_ijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
                len_jjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
                len_ij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

                cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (len_ijm1 * len_jjm1)

                cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (len_ijp1 * len_jjp1)

                cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
                cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

                Atot += len_ij**2 * (cot_thetam + cot_thetap) / 8
                data_i.append((cot_thetam + cot_thetap) / 2)
                row_indices.append(i)
                col_indices.append(j)
                eij = eijp1

                if eij == e_start:
                    break
            data_arr_i = np.array(data_i) / Atot
            # data_i = data_arr_i.tolist()
            data.extend(data_arr_i)

        return data, row_indices, col_indices

    def get_cot_laplacian_csr(self):
        """L_ij(y_j-y_i)"""
        data = []
        row_indices = []
        col_indices = []
        Nv = len(self.V)
        for i in range(Nv):
            data_i = []
            Atot = 0.0
            ri = self.V[i]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            e_start = self.e_of_v(i)
            eij = e_start
            while True:
                eijm1 = self.next(self.twin(eij))
                eijp1 = self.twin(self.prev(eij))
                jm1 = self.v_of_e(eijm1)
                j = self.v_of_e(eij)
                jp1 = self.v_of_e(eijp1)

                rjm1 = self.V[jm1]
                rj = self.V[j]
                rjp1 = self.V[jp1]

                rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
                rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
                rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
                ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
                ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
                rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
                ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
                rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

                len_ijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
                len_jjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
                len_ijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
                len_jjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
                len_ij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

                cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (len_ijm1 * len_jjm1)

                cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (len_ijp1 * len_jjp1)

                cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
                cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

                Atot += len_ij**2 * (cot_thetam + cot_thetap) / 8
                data_i.append((cot_thetam + cot_thetap) / 2)
                row_indices.append(i)
                col_indices.append(j)
                eij = eijp1

                if eij == e_start:
                    break
            data_arr_i = np.array(data_i) / Atot
            # data_i = data_arr_i.tolist()
            data.extend(data_arr_i)

        # Convert the lists into numpy arrays
        data = np.array(data)
        row_indices = np.array(row_indices)
        col_indices = np.array(col_indices)

        # Create the sparse matrix
        L_coo = coo_matrix((data, (row_indices, col_indices)))

        # Convert to CSR or CSC format for efficient arithmetic and matrix operations
        L_csr = L_coo.tocsr()  # or sparse_matrix.tocsc()

        return L_csr

    def cotan_laplacian(self, Y):
        """computes the laplacian of Y at each vertex"""
        Nv = len(self.V)
        lapY = np.zeros_like(Y)
        for vi in range(Nv):
            Atot = 0.0
            ri = self.V[vi]
            yi = Y[vi]
            ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
            h_start = self.V_edge[vi]
            hij = h_start
            while True:
                hijm1 = self.E_next[self.E_twin[hij]]
                hijp1 = self.E_twin[self.prev(hij)]
                vjm1 = self.E_vertex[hijm1]
                vj = self.E_vertex[hij]
                vjp1 = self.E_vertex[hijp1]

                yj = Y[vj]

                rjm1 = self.V[vjm1]
                rj = self.V[vj]
                rjp1 = self.V[vjp1]

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

    # ##########################################################################
    # # Cotan Laplacian #
    # #####################################
    # def Fbend_mixed(self):
    #     Nv = len(self.V)
    #     F = np.zeros((Nv, 3))
    #     Kb = self.bending_modulus
    #     K = self.get_gaussian_curvature_meyer()
    #     H = np.zeros_like(K)
    #     H0 = self.spontaneous_curvature
    #     Hn = self.slow_smoothed_laplacian(self.V[:]) / 2
    #     n = np.zeros((Nv, 3))
    #     A = np.zeros(Nv)
    #     for i in range(Nv):
    #         n[i] = self.other_weighted_vertex_normal(i)
    #         A[i] = self.meyercell_area(i)
    #         H[i] = n[i, 0] * Hn[i, 0] + n[i, 1] * Hn[i, 1] + n[i, 2] * Hn[i, 2]
    #     lapH = self.slow_smoothed_laplacian(H)
    #
    #     Fn = -2 * Kb * (lapH + 2 * (H - H0) * (H**2 + H0 * H - K))
    #
    #     for i in range(Nv):
    #         F[i] = Fn[i] * n[i] * A[i]
    #     return F
    #
    # ###########################################################################
    # # Curvature #
    # ##########################
    #
    # def get_meyer_masses(self):
    #     Nv = len(self.vertices)
    #     M = np.zeros(Nv)
    #     for v in range(Nv):
    #         M[v] = self.meyercell_area(v)
    #     return M
    #
    # def heat_laplacian_weight(self, x, y, h):
    #     X = self.V[x]
    #     Y = self.V[y]
    #     dxy = np.linalg.norm(Y - X)
    #     Wxy = np.exp(-(dxy**2) / (4 * h)) / (4 * np.pi * h**2)
    #     return Wxy
    #
    # def get_meyer_weighted_heat_laplacian(self, tol=1e-9):
    #
    #     index_list = []
    #     matrix_element_list = []
    #     Nv = len(self.V)
    #     for x in range(Nv):
    #         Mx = self.meyercell_area(x)
    #         for y in range(Nv):
    #             My = self.meyercell_area(y)
    #             Wxy = self.heat_laplacian_weight(x, y, Mx)
    #             Lxy = Wxy * My
    #             if Lxy > tol:
    #                 index_list.append([x, y])
    #                 matrix_element_list.append(Lxy)
    #
    #     return np.array(matrix_element_list), np.array(index_list, dtype=np.int32)
    #
    # def get_heat_laplacian(self, h, tol=1e-9):
    #
    #     index_list = []
    #     matrix_element_list = []
    #     Nv = len(self.V)
    #     for x in range(Nv):
    #         for y in range(Nv):
    #             My = self.meyercell_area(y)
    #             Wxy = self.heat_laplacian_weight(x, y, h)
    #             Lxy = Wxy * My
    #             if Lxy > tol:
    #                 index_list.append([x, y])
    #                 matrix_element_list.append(Lxy)
    #
    #     return np.array(matrix_element_list), np.array(index_list, dtype=np.int32)
    #
    # ###########################################################################
    # # cotan stuff #
    # ##########################
    # def meyercell_area(self, v):
    #     """Meyer's mixed area of cell dual to vertex v"""
    #     Atot = 0.0
    #     ri = self.V[v]
    #     ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
    #     h_start = self.V_edge[v]
    #     hij = h_start
    #     while True:
    #         vj = self.E_vertex[hij]
    #         rj = self.V[vj]
    #         hijp1 = self.E_twin[self.prev(hij)]
    #         vjp1 = self.E_vertex[hijp1]
    #         rjp1 = self.V[vjp1]
    #
    #         rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
    #         rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
    #         ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
    #         rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
    #         rjp1_ri = rjp1[0] * ri[0] + rjp1[1] * ri[1] + rjp1[2] * ri[2]
    #
    #         normDrij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
    #         # normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #         normDrjjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
    #         # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #         normDrjp1i = np.sqrt(rjp1_rjp1 - 2 * rjp1_ri + ri_ri)
    #         cos_thetajijp1 = (ri_ri + rj_rjp1 - ri_rj - rjp1_ri) / (normDrij * normDrjp1i)
    #         cos_thetajp1ji = (rj_rj + rjp1_ri - rj_rjp1 - ri_rj) / (normDrij * normDrjjp1)
    #         cos_thetaijp1j = (rjp1_rjp1 + ri_rj - rj_rjp1 - rjp1_ri) / (normDrjp1i * normDrjjp1)
    #         if cos_thetajijp1 < 0:
    #             semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
    #             Atot += np.sqrt(semiP * (semiP - normDrij) * (semiP - normDrjjp1) * (semiP - normDrjp1i)) / 2
    #             # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 4
    #         elif cos_thetajp1ji < 0 or cos_thetaijp1j < 0:
    #             semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
    #             Atot += np.sqrt(semiP * (semiP - normDrij) * (semiP - normDrjjp1) * (semiP - normDrjp1i)) / 4
    #             # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 8
    #         else:
    #             cot_thetaijp1j = cos_thetaijp1j / np.sqrt(1 - cos_thetaijp1j**2)
    #             cot_thetajp1ji = cos_thetajp1ji / np.sqrt(1 - cos_thetajp1ji**2)
    #             Atot += normDrij**2 * cot_thetaijp1j / 8 + normDrjp1i**2 * cot_thetajp1ji / 8
    #
    #         hij = hijp1
    #         if hij == h_start:
    #             break
    #
    #     return Atot
    #
    # def vorcell_area(self, v):
    #     """area of cell dual to vertex v"""
    #     Atot = 0.0
    #     r = self.V[v]
    #     r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v1 = self.E_vertex[h]
    #         r1 = self.V[v1]
    #         h = self.E_twin[self.prev(h)]
    #         v2 = self.E_vertex[h]
    #         r2 = self.V[v2]
    #
    #         r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #         r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #         r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #         r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #         r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #
    #         normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #         normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #         normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #         cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #         cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #
    #         cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #         cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #         Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #
    #         if h == h_start:
    #             break
    #
    #     return Atot
    #
    # def cotan_laplacian(self, Y):
    #     """computes the laplacian of Y at each vertex"""
    #     Nv = self.V.shape[0]
    #     lapY = np.zeros_like(Y)
    #     for vi in range(Nv):
    #         Atot = 0.0
    #         ri = self.V[vi]
    #         yi = Y[vi]
    #         ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
    #         h_start = self.V_edge[vi]
    #         hij = h_start
    #         while True:
    #             hijm1 = self.E_next[self.E_twin[hij]]
    #             hijp1 = self.E_twin[self.prev(hij)]
    #             vjm1 = self.E_vertex[hijm1]
    #             vj = self.E_vertex[hij]
    #             vjp1 = self.E_vertex[hijp1]
    #
    #             yj = Y[vj]
    #
    #             rjm1 = self.V[vjm1]
    #             rj = self.V[vj]
    #             rjp1 = self.V[vjp1]
    #
    #             rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
    #             rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
    #             rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
    #             ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
    #             ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
    #             rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
    #             ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
    #             rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
    #
    #             Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
    #             Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
    #             Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
    #             Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
    #             Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
    #
    #             cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)
    #
    #             cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)
    #
    #             cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
    #             cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)
    #
    #             Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
    #             lapY[vi] += (cot_thetam + cot_thetap) * (yj - yi) / 2
    #
    #             hij = hijp1
    #
    #             if hij == h_start:
    #                 break
    #         lapY[vi] /= Atot
    #
    #     return lapY
    #
    # def smoothed_laplacian(self, Y):
    #     """computes the laplacian of Y at each vertex"""
    #
    #     Nv = len(self.V)
    #     lapY = np.zeros_like(Y)
    #     for vi in range(Nv):
    #         ri = self.V[vi]
    #         ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
    #         Ai = self.meyercell_area(vi)
    #         h_start = self.V_edge[vi]
    #         hij = h_start
    #         while True:
    #             vj = self.E_vertex[hij]
    #             hijm1 = self.E_next[self.E_twin[hij]]
    #             hijp1 = self.E_twin[self.prev(hij)]
    #             vjm1 = self.E_vertex[hijm1]
    #             vjp1 = self.E_vertex[hijp1]
    #             rjm1 = self.V[vjm1]
    #             rj = self.V[vj]
    #             rjp1 = self.V[vjp1]
    #             rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
    #             ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
    #             vecAijjp1 = (np.cross(ri, rj) + np.cross(rj, rjp1) + np.cross(rjp1, ri)) / 2
    #             vecAijm1j = (np.cross(ri, rjm1) + np.cross(rjm1, rj) + np.cross(rj, ri)) / 2
    #             Aijjp1 = np.sqrt(vecAijjp1[0] ** 2 + vecAijjp1[1] ** 2 + vecAijjp1[2] ** 2)
    #             Aijm1j = np.sqrt(vecAijm1j[0] ** 2 + vecAijm1j[1] ** 2 + vecAijm1j[2] ** 2)
    #             lapY[vi] += (
    #                 (Aijjp1 + Aijm1j)
    #                 * np.exp(-(ri_ri - 2 * ri_rj + rj_rj) / (4 * Ai))
    #                 * (Y[vj] - Y[vi])
    #                 / (12 * np.pi * Ai**2)
    #             )
    #             hij = hijp1
    #             if hij == h_start:
    #                 break
    #     return lapY
    #
    # def slow_smoothed_laplacian(self, Y):
    #     """computes the laplacian of Y at each vertex"""
    #
    #     Nv = len(self.V)
    #     Nf = len(self.faces)
    #     lapY = np.zeros_like(Y)
    #     for vi in range(Nv):
    #         ri = self.V[vi]
    #         ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
    #         Ai = self.meyercell_area(vi)
    #         for f in range(Nf):
    #             vj, vk, vl = self.faces[f]
    #             rj = self.V[vj]
    #             rk = self.V[vk]
    #             rl = self.V[vl]
    #             vecAf = (np.cross(rj, rk) + np.cross(rk, rl) + np.cross(rl, rj)) / 2
    #             Af = np.sqrt(vecAf[0] ** 2 + vecAf[1] ** 2 + vecAf[2] ** 2)
    #             for vm in self.faces[f]:
    #                 rm = self.V[vm]
    #                 rm_rm = rm[0] ** 2 + rm[1] ** 2 + rm[2] ** 2
    #                 ri_rm = ri[0] * rm[0] + ri[1] * rm[1] + ri[2] * rm[2]
    #                 Drim_sqr = ri_ri - 2 * ri_rm + rm_rm
    #                 lapY[vi] += (Af / (12 * np.pi * Ai**2)) * np.exp(-Drim_sqr / (4 * Ai)) * (Y[vm] - Y[vi])
    #
    #     return lapY
    #
    # def mean_curvature_vector_cot(self, v):
    #     Atot = 0.0
    #     r = self.V[v]
    #     Hvec = np.zeros(3)
    #
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v0 = self.E_vertex[h]
    #         r0 = self.V[v0]
    #         vm1 = self.E_vertex[self.E_next[self.E_twin[h]]]
    #         rm1 = self.V[vm1]
    #         h = self.E_twin[self.prev(h)]
    #         vp1 = self.E_vertex[h]
    #         rp1 = self.V[vp1]
    #
    #         ua1 = r0 - rm1
    #         ua2 = r - rm1
    #         alpha = np.arccos(jitdot(ua1, ua2) / (np.crossnp.linalg.norm(ua1) * np.crossnp.linalg.norm(ua2)))
    #
    #         ub1 = r - rp1
    #         ub2 = r0 - rp1
    #         beta = np.arccos(jitdot(ub1, ub2) / (np.crossnp.linalg.norm(ub1) * np.crossnp.linalg.norm(ub2)))
    #
    #         Hvec += (1 / np.tan(alpha) + 1 / np.tan(beta)) * (r0 - r) / 2
    #         Atot += (1 / np.tan(alpha) + 1 / np.tan(beta)) * jitdot(r0 - r, r0 - r) / 8
    #
    #         if h == h_start:
    #             break
    #     return Hvec / (2 * Atot)
    #
    # def mean_curvature_cot(self, v):
    #     Atot = 0.0
    #     r = self.V[v]
    #     Hvec = np.zeros(3)
    #
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v0 = self.E_vertex[h]
    #         r0 = self.V[v0]
    #         vm1 = self.E_vertex[self.E_next[self.E_twin[h]]]
    #         rm1 = self.V[vm1]
    #         h = self.E_twin[self.prev(h)]
    #         vp1 = self.E_vertex[h]
    #         rp1 = self.V[vp1]
    #
    #         ua1 = r0 - rm1
    #         ua2 = r - rm1
    #         alpha = np.arccos(jitdot(ua1, ua2) / (np.crossnp.linalg.norm(ua1) * np.crossnp.linalg.norm(ua2)))
    #
    #         ub1 = r - rp1
    #         ub2 = r0 - rp1
    #         beta = np.arccos(jitdot(ub1, ub2) / (np.crossnp.linalg.norm(ub1) * np.crossnp.linalg.norm(ub2)))
    #
    #         Hvec += (1 / np.tan(alpha) + 1 / np.tan(beta)) * (r0 - r) / 2
    #         Atot += (1 / np.tan(alpha) + 1 / np.tan(beta)) * jitdot(r0 - r, r0 - r) / 8
    #
    #         if h == h_start:
    #             break
    #
    #     Hvec /= 2 * Atot
    #     n = self.area_weighted_vertex_normal(v)
    #     H = jitdot(n, Hvec)
    #     return H
    #
    # def mean_curvature(self, v):
    #     Atot = 0.0
    #     r = self.V[v]
    #     r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #     Hvec = np.zeros(3)
    #     n = np.zeros(3)
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v1 = self.E_vertex[h]
    #         r1 = self.V[v1]
    #         h = self.E_twin[self.prev(h)]
    #         v2 = self.E_vertex[h]
    #         r2 = self.V[v2]
    #
    #         r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #         r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #         r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #         r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #         r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #         # u1_u1 = r1_r1 - 2 * r_r1 + r_r
    #         # u2_u2 = r2_r2 - 2 * r1_r2 + r1_r1
    #         # u3_u3 = r_r - 2 * r2_r + r2_r2
    #
    #         # u1 = r1 - r
    #         # u2 = r2 - r1
    #         # u3 = r - r2
    #
    #         normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #         normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #         normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #         cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #         cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #         # sin_alpha = np.sqrt(1-cos_alpha**2)
    #         # sin_beta = np.sqrt(1-cos_beta**2)
    #         cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #         cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #         # alpha = np.arccos(jitdot(u2, u1) / (normu2 * normu1))
    #         # beta = np.arccos(jitdot(u3, u2) / (normu3 * normu2))
    #
    #         Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
    #         Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #         n += np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)
    #         # Atot += (1 / np.tan(alpha) + 1 / np.tan(beta)) * jitdot(r - r0, r - r0) / 8
    #
    #         if h == h_start:
    #             break
    #
    #     Hvec /= 2 * Atot
    #     n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    #     # n = self.area_weighted_vertex_normal(v)
    #     H = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
    #     return H
    #
    # def get_mean_curvature(self):
    #     """ """
    #     Nv = self.V.shape[0]
    #     H = np.zeros(Nv)
    #
    #     for v in range(Nv):
    #         Atot = 0.0
    #         r = self.V[v]
    #         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #         Hvec = np.zeros(3)
    #         n = np.zeros(3)
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.E_twin[self.prev(h)]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #
    #             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #             normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #             cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #             cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #             cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #             cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #
    #             Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
    #             Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #             n += np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)
    #
    #             if h == h_start:
    #                 break
    #
    #         Hvec /= 2 * Atot
    #         n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    #         H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
    #     return H
    #
    # def get_mean_curvature_vector(self):
    #     """ """
    #     Nv = self.V.shape[0]
    #     Hvec = np.zeros((Nv, 3))
    #
    #     for v in range(Nv):
    #         Atot = 0.0
    #         r = self.V[v]
    #         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #         # Hvec = np.zeros(3)
    #         # n = np.zeros(3)
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.E_twin[self.prev(h)]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #
    #             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #             normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #             cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #             cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #             cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #             cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #
    #             Hvec[v] += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
    #             Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #             # n += np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)
    #
    #             if h == h_start:
    #                 break
    #
    #         Hvec[v] /= 2 * Atot
    #         # n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    #         # H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
    #     return Hvec
    #
    # def get_gaussian_curvature(self):
    #     """gaussian curvature using voroni cell areas"""
    #     Nv = self.V.shape[0]
    #     K = np.zeros(Nv)
    #
    #     for v in range(Nv):
    #         Atot = 0.0
    #         r = self.V[v]
    #         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #         defect = 2 * np.pi
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.E_twin[self.prev(h)]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #
    #             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #             normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #             cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #             cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #             cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
    #             cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #             cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #             gamma = np.arccos(cos_gamma)
    #             defect -= gamma
    #             Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #
    #             if h == h_start:
    #                 break
    #
    #         K[v] = defect / Atot
    #     return K
    #
    # def get_curvatures(self):
    #     """ """
    #     Nv = self.V.shape[0]
    #     H = np.zeros(Nv)
    #     K = np.zeros(Nv)
    #
    #     for v in range(Nv):
    #         Atot = 0.0
    #         r = self.V[v]
    #         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #         Hvec = np.zeros(3)
    #         n = np.zeros(3)
    #         defect = 2 * np.pi
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.E_twin[self.prev(h)]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #
    #             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #             normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #             cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #             cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #             cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
    #             cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #             cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #
    #             defect -= np.arccos(cos_gamma)
    #             Hvec += cot_alpha * (r2 - r) / 2 + cot_beta * (r1 - r) / 2
    #             Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #             n += np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)
    #
    #             if h == h_start:
    #                 break
    #
    #         Hvec /= 2 * Atot
    #         n /= np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    #         H[v] = n[0] * Hvec[0] + n[1] * Hvec[1] + n[2] * Hvec[2]
    #         K[v] = defect / Atot
    #     return H, K
    #
    # def get_gaussian_curvature_meyer(self):
    #     """gaussian curvature using voroni cell areas"""
    #     Nv = self.V.shape[0]
    #     K = np.zeros(Nv)
    #
    #     for v in range(Nv):
    #         Atot = self.meyercell_area(v)
    #         r = self.V[v]
    #         r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #         defect = 2 * np.pi
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.E_twin[self.prev(h)]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             r1_r1 = r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2
    #             r2_r2 = r2[0] ** 2 + r2[1] ** 2 + r2[2] ** 2
    #             r_r1 = r[0] * r1[0] + r[1] * r1[1] + r[2] * r1[2]
    #             r1_r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
    #             r2_r = r2[0] * r[0] + r2[1] * r[1] + r2[2] * r[2]
    #
    #             normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # np.crossnp.linalg.norm(u1)
    #             # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # np.crossnp.linalg.norm(u2)
    #             normu3 = np.sqrt(r_r - 2 * r2_r + r2_r2)  # np.crossnp.linalg.norm(u3)
    #             # cos_alpha = (r1_r1 + r2_r - r_r1 - r1_r2) / (normu1 * normu2)
    #             # cos_beta = (r2_r2 + r_r1 - r1_r2 - r2_r) / (normu2 * normu3)
    #             cos_gamma = (r_r + r1_r2 - r_r1 - r2_r) / (normu1 * normu3)
    #             # cot_alpha = cos_alpha / np.sqrt(1 - cos_alpha**2)
    #             # cot_beta = cos_beta / np.sqrt(1 - cos_beta**2)
    #             gamma = np.arccos(cos_gamma)
    #             defect -= gamma
    #             # Atot += normu3**2 * cot_alpha / 8 + normu1**2 * cot_beta / 8
    #
    #             if h == h_start:
    #                 break
    #
    #         K[v] = defect / Atot
    #     return K
    #
    # ###########################################################################
    # # geometric computations #
    # ##########################
    #
    # def get_total_area(self):
    #     vertices = self.V[:]
    #     Nv = self.V.shape[0]
    #     V_edge = self.V_edge
    #     E_vertex = self.E_vertex
    #     H_prev = self.H_prev
    #     E_twin = self.E_twin
    #     A = 0.0
    #     for v in range(Nv):
    #         r = vertices[v]
    #         Av = 0.0
    #         h = V_edge[v]
    #         h_start = h
    #         while True:
    #             v1 = E_vertex[h]
    #             r1 = vertices[v1]
    #             h = H_prev[h]
    #             h = E_twin[h]
    #             v2 = E_vertex[h]
    #             r2 = vertices[v2]
    #             a = np.cross(r, r1) / 2 + np.cross(r1, r2) / 2 + np.cross(r2, r) / 2
    #             Av += np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) / 3
    #
    #             if h == h_start:
    #                 break
    #
    #         A += Av
    #     return A
    #
    # def volume_of_mesh(self):
    #     """computes total mesh volume as the sum of signed volume of faces"""
    #     Nf = len(self.faces)
    #     vol = 0.0
    #     for f in range(Nf):
    #         v0, v1, v2 = self.faces[f]
    #         p0 = self.V[v0]
    #         p1 = self.V[v1]
    #         p2 = self.V[v2]
    #         vol += triprod(p0, p1, p2) / 6
    #     return vol
    #
    # def angle_defect(self, v):
    #     """
    #     2*pi - sum_f (angle_f)
    #     """
    #     r = self.V[v]
    #     h = self.V_edge[v]
    #     h_start = h
    #     defect = 2 * np.pi
    #     h = h_start
    #     while True:
    #         v1 = self.E_vertex[h]
    #         r1 = self.V[v1]
    #         h = self.prev(h)
    #         h = self.E_twin[h]
    #         v2 = self.E_vertex[h]
    #         r2 = self.V[v2]
    #         e1 = r1 - r
    #         e2 = r2 - r
    #         norm_e1 = np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
    #         norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
    #         cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (norm_e1 * norm_e2)
    #         defect -= np.arccos(cos_angle)
    #         if h == h_start:
    #             break
    #
    #     return defect
    #
    # def get_angle_defects(self):
    #     """
    #     2*pi - sum_f (angle_f)
    #     """
    #     Nverts = len(self.V)
    #     defects = np.zeros(Nverts)
    #     for v0 in range(Nverts):
    #         h_start = self.V_edge[v0]
    #         defects[v0] = 2 * np.pi
    #
    #         h = h_start
    #         v = self.E_vertex[h]
    #         e2 = self.V[v] - self.V[v0]
    #         norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
    #         h = self.E_next[self.E_twin[h]]
    #
    #         while True:
    #             e1 = e2
    #             norm_e1 = norm_e2
    #             v = self.E_vertex[h]  # 2nd vert
    #             e2 = self.V[v] - self.V[v0]
    #             norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
    #             cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (norm_e1 * norm_e2)
    #             defects[v0] -= np.arccos(cos_angle)
    #
    #             h = self.E_next[self.E_twin[h]]
    #             if h == h_start:
    #                 break
    #
    #     return defects
    #
    # def get_gaussian_curvature2(self):
    #     """
    #     2*pi - sum_f (angle_f)
    #     """
    #     Nv = len(self.V)
    #     # defects = np.zeros(Nverts)
    #     K = np.zeros(Nv)
    #     # Nv = len(self.V)
    #     # for v0 in range(Nv):
    #     # v0 = 0
    #     # K = np.random.rand(Nverts)
    #     # while True:
    #     #     # p0 = self.V[v]
    #     #     h_start = self.V_edge[v0]
    #     #     defect = 2 * np.pi
    #     #     area = 0.0
    #     #
    #     #     h = h_start
    #     #     v = self.E_vertex[h]
    #     #     e2 = self.V[v] - self.V[v0]
    #     #     norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
    #     #     h = self.E_next[self.E_twin[h]]
    #     #
    #     #     while True:
    #     #         e1 = e2
    #     #         norm_e1 = norm_e2
    #     #         v = self.E_vertex[h]  # 2nd vert
    #     #         e2 = self.V[v] - self.V[v0]
    #     #         norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
    #     #         # e1_cross_e2 = np.cross(e1, e2)
    #     #         # norm_e1_cross_e2 = np.sqrt(
    #     #         #     e1_cross_e2[0] ** 2 + e1_cross_e2[1] ** 2 + e1_cross_e2[2] ** 2
    #     #         # )
    #     #         # sin_angle = norm_e1_cross_e2 / (norm_e1 * norm_e2)
    #     #         cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
    #     #             norm_e1 * norm_e2
    #     #         )
    #     #         angle = np.arccos(cos_angle)
    #     #
    #     #         defect -= angle
    #     #         area += 0.5 * norm_e1 * norm_e2 * np.sin(angle) / 3
    #     #
    #     #         h = self.E_next[self.E_twin[h]]
    #     #         if h == h_start:
    #     #             break
    #     #     K[v0] = defect / area
    #     for v in range(Nv):
    #         K[v] = self.gaussian_curvature(v)
    #
    #     return K
    #
    # def gaussian_curvature(self, v):
    #     K = self.angle_defect(v)
    #     K /= self.barcell_area(v)
    #     return K
    #
    # ###########################################################################
    # # mesh mutation/regularization functions #
    # ###############################
    # def orientation_check(self, h):
    #     """checks if faces adjacent to an edge have consistent normals"""
    #
    #     fl = self.E_face[h]
    #     fr = self.E_face[self.E_twin[h]]
    #     il, jl, kl = self.faces[fl]
    #     ir, jr, kr = self.faces[fr]
    #     ril, rjl, rkl = self.V[il], self.V[jl], self.V[kl]
    #     rir, rjr, rkr = self.V[ir], self.V[jr], self.V[kr]
    #     Al = np.cross(ril, rjl) + np.cross(rjl, rkl) + np.cross(rkl, ril)
    #     Ar = np.cross(rir, rjr) + np.cross(rjr, rkr) + np.cross(rkr, rir)
    #     Al_Ar = Al[0] * Ar[0] + Al[1] * Ar[1] + Al[2] * Ar[2]
    #     is_good = Al_Ar > 0
    #     return is_good
    #
    # def volume_length_quality_metric(self, f):
    #     v0, v1, v2 = self.faces[f]
    #     r0 = self.V[v0]
    #     r1 = self.V[v1]
    #     r2 = self.V[v2]
    #     Av = (np.cross(r0, r1) + np.cross(r1, r2) + np.cross(r2, r0)) / 2
    #     A = np.crossnp.linalg.norm(Av)
    #     e01 = r1 - r0
    #     e12 = r2 - r1
    #     e20 = r0 - r2
    #
    #     e_rms = (
    #         np.sqrt(
    #             e01[0] ** 2
    #             + e01[1] ** 2
    #             + e01[2] ** 2
    #             + e12[0] ** 2
    #             + e12[1] ** 2
    #             + e12[2] ** 2
    #             + e20[0] ** 2
    #             + e20[1] ** 2
    #             + e20[2] ** 2
    #         )
    #         / 3
    #     )
    #
    #     a = 4 * np.sqrt(3) * A / (3 * e_rms**2)
    #     return a
    #
    # def edge_flip(self, h):
    #     r"""
    #     h/ht can not be on boundary!
    #     keeps fa
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
    #     ht = self.E_twin[h]
    #     h1 = self.E_next[h]
    #     h2 = self.prev(h)
    #     h3 = self.E_next[ht]
    #     h4 = self.H_prev[ht]
    #     f1 = self.E_face[h]
    #     f2 = self.E_face[ht]
    #     v1 = self.E_vertex[h4]
    #     v2 = self.E_vertex[h1]
    #     v3 = self.E_vertex[h2]
    #     v4 = self.E_vertex[h3]
    #     self.faces[f1] = np.array([v2, v3, v4], dtype=np.int32)
    #     self.faces[f2] = np.array([v4, v1, v2], dtype=np.int32)
    #     self.F_edge[f1] = h2
    #     self.F_edge[f2] = h4
    #     self.halfedges[h] = np.array([v4, v2], dtype=np.int32)
    #     self.halfedges[ht] = np.array([v2, v4], dtype=np.int32)
    #     self.E_next[h] = h2
    #     self.H_prev[h2] = h
    #     self.E_next[h2] = h3
    #     self.H_prev[h3] = h2
    #     self.E_next[h3] = h
    #     self.prev(h) = h3  #
    #     self.E_next[ht] = h4
    #     self.H_prev[h4] = ht
    #     self.E_next[h4] = h1
    #     self.H_prev[h1] = h4
    #     self.E_next[h1] = ht
    #     self.H_prev[ht] = h1
    #     self.E_face[h3] = f1
    #     self.E_face[h1] = f2
    #     self.E_vertex[h] = v2
    #     self.E_vertex[ht] = v4
    #     self.V_edge[v3] = h3
    #     self.V_edge[v1] = h1
    #     self.V_edge[v2] = h2
    #     self.V_edge[v4] = h4
    #
    # def monte_wants_flip(self, h):
    #     Ka = self.Karea
    #     A0 = self.preferred_face_area
    #     l0 = self.preferred_edge_length
    #     ht = self.E_twin[h]
    #     h1 = self.E_next[h]
    #     h2 = self.prev(h)
    #     h3 = self.E_next[ht]
    #     h4 = self.H_prev[ht]
    #
    #     v1 = self.E_vertex[h4]
    #     v2 = self.E_vertex[h1]
    #     v3 = self.E_vertex[h2]
    #     v4 = self.E_vertex[h3]
    #     r1 = self.V[v1]
    #     r2 = self.V[v2]
    #     r3 = self.V[v3]
    #     r4 = self.V[v4]
    #     vecA123 = (np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)) / 2
    #     vecA134 = (np.cross(r1, r3) + np.cross(r3, r4) + np.cross(r4, r1)) / 2
    #     vecA234 = (np.cross(r2, r3) + np.cross(r3, r4) + np.cross(r4, r2)) / 2
    #     vecA124 = (np.cross(r1, r2) + np.cross(r2, r4) + np.cross(r4, r1)) / 2
    #     A123 = np.sqrt(vecA123[0] ** 2 + vecA123[1] ** 2 + vecA123[2] ** 2)
    #     A134 = np.sqrt(vecA134[0] ** 2 + vecA134[1] ** 2 + vecA134[2] ** 2)
    #     A234 = np.sqrt(vecA234[0] ** 2 + vecA234[1] ** 2 + vecA234[2] ** 2)
    #     A124 = np.sqrt(vecA124[0] ** 2 + vecA124[1] ** 2 + vecA124[2] ** 2)
    #
    #     Upre = (Ka / (2 * A0)) * ((A123 - A0) ** 2 + (A134 - A0) ** 2)
    #     Upos = (Ka / (2 * A0)) * ((A234 - A0) ** 2 + (A124 - A0) ** 2)
    #
    #     Dr13 = self.V[v1] - self.V[v3]
    #     Dr24 = self.V[v2] - self.V[v4]
    #     L13 = np.sqrt(Dr13[0] ** 2 + Dr13[1] ** 2 + Dr13[2] ** 2)
    #     L24 = np.sqrt(Dr24[0] ** 2 + Dr24[1] ** 2 + Dr24[2] ** 2)
    #     Upre += self.Utether(L13, l0)
    #     Upos += self.Utether(L24, l0)
    #     flip_it = Upos < Upre
    #     return flip_it
    #
    # def do_the_monte_flips(self):
    #     Nflips = 0
    #     Nh = len(self.halfedges)
    #     for h in range(Nh):
    #         flip_it = self.is_flippable(h) and self.monte_wants_flip(h)
    #         if flip_it:
    #             self.edge_flip(h)
    #             Nflips += 1
    #     return Nflips
    #
    # def flip_helps_valence(self, h):
    #     r"""
    #     Returns True if flipping edge 'h' decreases variance
    #     of the valence of the four vertices illustrated below.
    #     \sum_{i=1}^4 valence(i)^2/4 - (\sum_{i=1}^4 valence(i))^2/4
    #     This favors vertices with valence=6 and is equivalent
    #     returning True if the flip decreases the energy
    #     \sum_{i in all vertices} (valence(i)-6)^2.
    #       v2             v2
    #      / \            /|\
    #    v3---v1  |---> v3 | v1
    #      \ /            \|/
    #       v4             v4
    #     """
    #
    #     ht = self.E_twin[h]
    #     h1 = self.E_next[h]
    #     h3 = self.E_next[ht]
    #     v1 = self.E_vertex[h]
    #     v2 = self.E_vertex[h1]
    #     v3 = self.E_vertex[ht]
    #     v4 = self.E_vertex[h3]
    #
    #     val1 = self.valence(v1)
    #     val2 = self.valence(v2)
    #     val3 = self.valence(v3)
    #     val4 = self.valence(v4)
    #
    #     flip_it = val1 - val2 + val3 - val4 > 2
    #
    #     return flip_it
    #
    # def is_delaunay(self, h):
    #     r"""
    #     checks if edge is locally delaunay
    #       v2
    #       /|\
    #     v3 | v1
    #       \|/
    #        v4
    #     """
    #     vi = self.E_vertex[self.E_next[self.E_twin[h]]]
    #     vj = self.E_vertex[h]
    #     vk = self.E_vertex[self.E_next[h]]
    #     vl = self.E_vertex[self.prev(h)]
    #
    #     pij = self.V[vj] - self.V[vi]
    #     pil = self.V[vl] - self.V[vi]
    #     pkj = self.V[vj] - self.V[vk]
    #     pkl = self.V[vl] - self.V[vk]
    #
    #     pij_pil = pij[0] * pil[0] + pij[1] * pil[1] + pij[2] * pil[2]
    #     pkl_pkj = pkl[0] * pkj[0] + pkl[1] * pkj[1] + pkl[2] * pkj[2]
    #     normpij = np.sqrt(pij[0] ** 2 + pij[1] ** 2 + pij[2] ** 2)
    #     normpil = np.sqrt(pil[0] ** 2 + pil[1] ** 2 + pil[2] ** 2)
    #     normpkj = np.sqrt(pkj[0] ** 2 + pkj[1] ** 2 + pkj[2] ** 2)
    #     normpkl = np.sqrt(pkl[0] ** 2 + pkl[1] ** 2 + pkl[2] ** 2)
    #
    #     alphai = np.arccos(pij_pil / (normpij * normpil))
    #     alphak = np.arccos(pkl_pkj / (normpkl * normpkj))
    #
    #     return alphai + alphak <= np.pi
    #
    # def is_flippable_old(self, h):
    #     r"""
    #       vj
    #       /|\
    #     vk | vi
    #       \|/
    #        vl
    #     """
    #     hlj = h
    #     hjk = self.E_next[hlj]
    #     hkl = self.E_next[hjk]
    #
    #     hjl = self.E_twin[hlj]
    #     hli = self.E_next[hjl]
    #     hij = self.E_next[hli]
    #
    #     hkj = self.E_twin[hjk]
    #     fkj = self.E_face[hkj]
    #     hji = self.E_twin[hij]
    #     fji = self.E_face[hji]
    #
    #     hlk = self.E_twin[hkl]
    #     flk = self.E_face[hlk]
    #     hil = self.E_twin[hli]
    #     fil = self.E_face[hil]
    #
    #     flippable = (fkj != fji) and (flk != fil)
    #
    #     return flippable
    #
    # def is_flippable(self, h):
    #     """
    #     edge flip hlj-->hki is allowed unless vi and vk are already neighbors
    #       vj
    #       /|\
    #     vk | vi
    #       \|/
    #        vl
    #     """
    #     hlj = h
    #     hjk = self.E_next[hlj]
    #     hjl = self.E_twin[hlj]
    #     hli = self.E_next[hjl]
    #     flippable = True
    #
    #     vj = self.E_vertex[hlj]
    #     vk = self.E_vertex[hjk]
    #
    #     him = self.E_twin[hli]
    #     while True:
    #         him = self.E_twin[self.H_prev[him]]
    #         vm = self.E_vertex[him]
    #         if vm == vk:
    #             flippable = False
    #             break
    #         if vm == vj:
    #             break
    #
    #     return flippable
    #
    # def shift_vertex_towards_barycenter(self, v, weight):
    #     """Translates vertex in the direction of the barycenter of
    #     its neighbors. weight=1 means all the way to the barycenter."""
    #     r0 = self.V[v].copy()
    #     h = self.V_edge[v]
    #     h_start = h
    #     r = np.zeros(3)
    #     val = 0
    #     while True:
    #         vb = self.E_vertex[h]
    #         r += self.V[vb]
    #         val += 1
    #         h = self.prev(h)
    #         h = self.E_twin[h]
    #         if h == h_start:
    #             break
    #     r /= val
    #     self.V[v] = weight * r + (1 - weight) * r0
    #
    # def regularize_by_flips(self):
    #     Nh = len(self.halfedges)
    #     Nflips = 0
    #     for h in range(Nh):
    #         flip_it = self.flip_helps_valence(h) & self.is_flippable(h)
    #         if flip_it:
    #             self.edge_flip(h)
    #             Nflips += 1
    #     return Nflips
    #
    # def regularize_by_shifts(self, weight):
    #     Nv = self.V.shape[0]
    #     for v in range(Nv):
    #         self.shift_vertex_towards_barycenter(v, weight)
    #
    # def smooth_samples(self, samples_in, weight, iters):
    #     """shifts samples towards the avereage value of their neighbors"""
    #     samples = samples_in.copy()
    #     Nv = self.V.shape[0]
    #     for iter in range(iters):
    #         for v in range(Nv):
    #             h = self.V_edge[v]
    #             h_start = h
    #             samp = np.zeros_like(samples[v])
    #             val = 0
    #             while True:
    #                 vb = self.E_vertex[h]
    #                 samp += samples[vb]
    #                 val += 1
    #                 h = self.E_twin[self.prev(h)]
    #                 if h == h_start:
    #                     break
    #             samp /= val
    #             samples[v] = weight * samp + (1 - weight) * samples[v]
    #     return samples
    #
    # def delaunay_regularize_by_flips(self):
    #     Nh = len(self.halfedges)
    #     Nflips = 0
    #     for h in range(Nh):
    #         is_del = self.is_delaunay(h)
    #         if not is_del:
    #             flippable = self.is_flippable(h)
    #             if flippable:
    #                 self.edge_flip(h)
    #                 Nflips += 1
    #     return Nflips
    #
    # def gaussian_smooth_samples(self, samples_in, iters, a):
    #     """shifts samples towards the avereage value of their neighbors"""
    #     samples = samples_in.copy()
    #     Nv = self.V.shape[0]
    #     for iter in range(iters):
    #         # samples0 = samples_in.copy()
    #         for v in range(Nv):
    #             r0 = self.V[v]
    #             r0_r0 = r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2
    #             h = self.V_edge[v]
    #             h_start = h
    #             samp = samples[v]
    #             W = 1
    #             while True:
    #                 vb = self.E_vertex[h]
    #                 r = self.V[vb]
    #                 r_r0 = r[0] * r0[0] + r[1] * r0[1] + r[2] * r0[2]
    #                 r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2
    #                 w = np.exp(-(r_r - 2 * r_r0 + r0_r0) / a**2)
    #                 samp += w * samples[vb]
    #                 W += w
    #                 h = self.E_twin[self.prev(h)]
    #                 if h == h_start:
    #                     break
    #
    #             samples[v] = samp / W
    #     return samples
    #
    # ###########################################################################
    # # forces and time evolution #
    # #############################
    #
    # def Flj(self, v):
    #     r0 = self.V[v]
    #     F = np.zeros(3)
    #     L0 = self.preferred_edge_length
    #     eps = self.Klength
    #     A = 2 ** (-1 / 6) * L0
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v = self.E_vertex[h]
    #         R = self.V[v] - r0
    #         normR = np.crossnp.linalg.norm(R)
    #         A_normR = A / normR
    #         Dg = (-24 * eps / A) * (2 * A_normR**13 - A_normR**7)
    #         F += -Dg * R / normR
    #         h = self.E_twin[self.prev(h)]
    #
    #         if h == h_start:
    #             break
    #     return F
    #
    # def Ulj(self, R):
    #     L0 = self.preferred_edge_length
    #     eps = self.Klength
    #     A = 2 ** (-1 / 6) * L0
    #
    #     normR = np.crossnp.linalg.norm(R)
    #     A_normR = A / normR
    #     g = 4 * eps * (A_normR**12 - A_normR**6)
    #
    #     return g
    #
    # def length_reg_force(self, v):
    #     """E ~ 1/2*Ke*(L-L0)**2/L0"""
    #     Ke = self.Klength
    #     L0 = self.preferred_edge_length
    #     xyz = self.V[v]
    #     neighbors = self.v_adjacent_to_v(v)
    #     N = len(neighbors)
    #     F = np.zeros(3)
    #
    #     for _v0 in range(0, N):
    #         v0 = neighbors[_v0]
    #         xyz0 = self.V[v0]
    #         r = xyz - xyz0
    #         L = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    #         gradL = r / L
    #         # if L >1e-12:
    #         #     gradL = r / L
    #         # else:
    #         #     gradL = r
    #         F += -Ke * (L - L0) * gradL / L0
    #
    #     return F
    #
    # def LOCAL_area_reg_force(self, v):
    #     """E ~ 1/2*Ka*(A-A0)**2"""
    #     A0 = self.preferred_cell_area
    #     Ka = self.Karea
    #     r = self.V[v]
    #     F = np.zeros(3)
    #
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v1 = self.E_vertex[h]
    #         r1 = self.V[v1]
    #         h = self.prev(h)
    #         h = self.E_twin[h]
    #         v2 = self.E_vertex[h]
    #         r2 = self.V[v2]
    #
    #         Avec = (np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)) / 2
    #         A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3
    #
    #         gradA = np.cross(Avec, r2 - r1) / (2 * A)
    #
    #         F += -Ka * (A - A0) * gradA / A0
    #
    #         if h == h_start:
    #             break
    #     return F
    #
    # def area_reg_force(self, v):
    #     """E ~ 1/2*Ka*(A-A0)**2"""
    #     r = self.V[v]
    #     F = np.zeros(3)
    #     area0 = self.preferred_total_area
    #     area = self.total_area
    #     Ka = self.Karea
    #
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v1 = self.E_vertex[h]
    #         r1 = self.V[v1]
    #         h = self.prev(h)
    #         h = self.E_twin[h]
    #         v2 = self.E_vertex[h]
    #         r2 = self.V[v2]
    #
    #         Avec = (np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)) / 2
    #         A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3
    #
    #         gradA = np.cross(Avec, r2 - r1) / (2 * A)
    #
    #         F += -Ka * (area - area0) * gradA / area0
    #
    #         if h == h_start:
    #             break
    #     return F
    #
    # def volume_reg_force(self, v):
    #     """."""
    #     vol0 = self.preferred_total_volume
    #     vol = self.total_volume
    #     Kv = self.Kvolume
    #
    #     F = np.zeros(3)
    #
    #     h_start = self.V_edge[v]
    #     h = h_start
    #     while True:
    #         v1 = self.E_vertex[h]
    #         r1 = self.V[v1]
    #         h = self.prev(h)
    #         h = self.E_twin[h]
    #         v2 = self.E_vertex[h]
    #         r2 = self.V[v2]
    #
    #         # grad_vol = np.cross(r1, r2) / (2 * A)
    #
    #         F += -Kv * (vol - vol0) * np.cross(r1, r2) / (18 * vol0)
    #
    #         if h == h_start:
    #             break
    #     return F
    #
    # def forward_euler_reg_step(self, dt):
    #     Nv = len(self.V)
    #     linear_drag_coeff = self.linear_drag_coeff
    #     self.total_volume = self.volume_of_mesh()
    #     self.total_area = self.get_total_area()
    #
    #     for v in range(Nv):
    #         F = np.zeros(3)
    #         # F += self.length_reg_force(v)
    #         F += self.area_reg_force(v)
    #         F += self.volume_reg_force(v)
    #         F += self.Flj(v)
    #         self.V[v] = self.V[v] + dt * F / linear_drag_coeff
    #
    #     # for v in range(Nv):
    #     #     self.V[v, 3:] = self.get_new_quat_dumb(v)
    #
    # def get_new_euler_state(self, dt):
    #     # Nv = len(self.V)
    #     # vertices = np.zeros((Nv, 3))
    #     linear_drag_coeff = self.linear_drag_coeff
    #     self.total_volume = self.volume_of_mesh()
    #     self.total_area = self.get_total_area()
    #     Fb = self.Fbend()
    #     Fl = self.Flength()
    #     Fa = self.Farea()
    #     Fv = self.Fvolume()
    #     F = Fb + Fl + Fa + Fv
    #     vertices = self.V[:] + dt * F / linear_drag_coeff
    #
    #     # for v in range(Nv):
    #     #     self.V[v, 3:] = self.get_new_quat_dumb(v)
    #     success = True
    #     return vertices, success
    #
    # def Fbend(self, Nsmooth=0):
    #     """from Tu"""
    #     a = self.preferred_edge_length
    #     Kbend = self.bending_modulus
    #     H, K = self.get_angle_weighted_arc_curvatures()
    #     for _ in range(Nsmooth):
    #         H = self.gaussian_smooth_samples(H, 1, a)
    #         K = self.gaussian_smooth_samples(K, 1, a)
    #     Nv = H.shape[0]
    #     lapH = self.cotan_laplacian(H)
    #     F = np.zeros((Nv, 3))
    #
    #     Fn = -2 * Kbend * (lapH + 2 * H * (H**2 - K))
    #
    #     for v in range(Nv):
    #         n = self.other_weighted_vertex_normal(v)
    #         Av = self.vorcell_area(v)
    #         F[v] = Fn[v] * n * Av
    #     return F
    #
    # def Flength(self):
    #     Nv = len(self.V)
    #     F = np.zeros((Nv, 3))
    #     Ke = self.Klength
    #     L0 = self.preferred_edge_length
    #     for v in range(Nv):
    #         r = self.V[v]
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v0 = self.E_vertex[h]
    #             r0 = self.V[v0]
    #             u = r - r0
    #             L = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    #             gradL = u / L
    #             F[v] += -Ke * (L - L0) * gradL / L0
    #             h = self.E_twin[self.prev(h)]
    #             if h == h_start:
    #                 break
    #     return F
    #
    # def Farea(self):
    #     """local cell area refulation"""
    #     Nv = len(self.V)
    #     F = np.zeros((Nv, 3))
    #     # area0 = self.preferred_total_area
    #     # area = self.total_area
    #     A0 = self.preferred_cell_area
    #     Ka = self.Karea
    #     for v in range(Nv):
    #         r = self.V[v]
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.prev(h)
    #             h = self.E_twin[h]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             Avec = (np.cross(r, r1) + np.cross(r1, r2) + np.cross(r2, r)) / 2
    #             A = np.sqrt(Avec[0] ** 2 + Avec[1] ** 2 + Avec[2] ** 2) / 3
    #
    #             gradA = np.cross(Avec, r2 - r1) / (2 * A)
    #
    #             # F[v] += -Ka * (area - area0) * gradA / area0
    #             F[v] += -Ka * (A - A0) * gradA / A0
    #
    #             if h == h_start:
    #                 break
    #     return F
    #
    # def Fvolume(self):
    #     Nv = len(self.V)
    #     F = np.zeros((Nv, 3))
    #     vol0 = self.preferred_total_volume
    #     vol = self.volume_of_mesh()
    #     Kv = self.Kvolume
    #     for v in range(Nv):
    #         h_start = self.V_edge[v]
    #         h = h_start
    #         while True:
    #             v1 = self.E_vertex[h]
    #             r1 = self.V[v1]
    #             h = self.prev(h)
    #             h = self.E_twin[h]
    #             v2 = self.E_vertex[h]
    #             r2 = self.V[v2]
    #
    #             # grad_vol = np.cross(r1, r2) / (2 * A)
    #
    #             F[v] += -Kv * (vol - vol0) * np.cross(r1, r2) / (18 * vol0)
    #
    #             if h == h_start:
    #                 break
    #     return F
    #
    # ###########################################################################
