from functools import lru_cache
from src.python.ply_tools import VertTri2HalfEdgeConverter
from src.python.half_edge_mesh import HalfEdgeMesh
import numpy as np
from scipy.sparse import csr_matrix
from time import time
import pickle


class HalfEdgePatch:
    """
    A submanifold of a HalfEdgeMesh topologically equivalent to a disk.
    """

    def __init__(
        self,
        supermesh,
        V,
        H,
        F,
        h_comp_B=None,
        V_bdry=None,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        if h_comp_B is None:
            self.h_comp_B = self.find_h_comp_B()
        if V_bdry is None:
            self.V_bdry = set(self.generate_V_cw_B())

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        if isinstance(value, set):
            self._V = value
        elif hasattr(value, "__iter__"):
            self._V = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        if isinstance(value, set):
            self._H = value
        elif hasattr(value, "__iter__"):
            self._H = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        if isinstance(value, set):
            self._F = value
        elif hasattr(value, "__iter__"):
            self._F = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    @property
    def V_bdry(self):
        return self._V_bdry

    @V_bdry.setter
    def V_bdry(self, value):
        if isinstance(value, set):
            self._V_bdry = value
        elif hasattr(value, "__iter__"):
            self._V_bdry = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    ##############################################
    @classmethod
    def from_seed_vertex(cls, v_seed, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If v_seed is not in a boundary of supermesh, the patch will be a disk centered at v_seed.

        Parameters:
            v_seed (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = supermesh.closure(*supermesh.star_of_vertex(v_seed))
        self = cls(supermesh, V, H, F)
        # self.h_comp_B = self.find_h_comp_B()

        return self

    def complement_boundary_contains_h(self, h):
        """check if half-edge h is in the boundary of the mesh"""
        return h in self.H and self.supermesh.f_left_h(h) not in self.F

    def interior_boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return (
            h in self.H
            and self.supermesh.f_left_h(self.supermesh.h_twin_h(h)) not in self.F
        )

    def boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        if h in self.H:
            if self.supermesh.f_left_h(h) not in self.F:
                return True
            if self.supermesh.f_left_h(self.supermesh.h_twin_h(h)) not in self.F:
                return True
        return False

    # def boundary_contains_v(self, v):
    #     """check if vertex v is on the boundary of the mesh"""
    #     return self.boundary_contains_h(self.supermesh.h_out_v(v))

    ##############################################
    def xyz_coord_v(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self.supermesh.xyz_coord_v(v)

    def h_out_v(self, v):
        """
        get index of an non-boundary outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        if v not in self.V:
            raise ValueError("Vertex not in patch.")
        for h in self.supermesh.generate_H_out_v_clockwise(v):
            if h not in self.H:
                continue
            elif self.supermesh.f_left_h(h) in self.F:
                return h

    def v_origin_h(self, h):
        """get index of the vertex at the origin of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        return self.supermesh.v_origin_h(h)

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face/boundary cycle
        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        elif self.supermesh.f_left_h(h) in self.F:
            return self.supermesh.h_next_h(h)
        n = self.supermesh.h_next_h(h)
        while n not in self.H:
            n = self.supermesh.h_out_cw_from_h(n)
        return n

    def h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        return self.supermesh.h_twin_h(h)

    def f_left_h(self, h):
        """get index of the face to the left of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: face index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        elif self.supermesh.f_left_h(h) in self.F:
            return self.supermesh.f_left_h(h)
        else:
            return -1

    def h_bound_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        if f not in self.F:
            raise ValueError("Face not in patch.")
        return self.supermesh.h_bound_f(f)

    def generate_H_next_h(self, h_start):
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_H_cw_B(self):
        for bdry, h in self.h_comp_B.items():
            for h in self.generate_H_next_h(h):
                yield h

    def generate_V_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.v_origin_h(h)

    def generate_F_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.f_left_h(self.h_twin_h(h))

    def find_h_comp_B(self, F_need2check=None):
        h_comp_B = dict()
        bdry_count = 0
        H_in_cw_boundary = set()
        # boundary_is_right_of_H = set()
        if F_need2check is None:
            F_need2check = self.F.copy()  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            h = self.h_bound_f(f)
            for h in self.generate_H_next_h(h):
                if self.interior_boundary_contains_h(h):
                    H_in_cw_boundary.add(self.h_twin_h(h))
        while H_in_cw_boundary:
            bdry_count += 1
            h = H_in_cw_boundary.pop()
            bdry = -bdry_count
            h_comp_B[bdry] = h
            for h in self.generate_H_next_h(h):
                H_in_cw_boundary.discard(h)
        return h_comp_B

    def _expand_boundary(self):
        """
        **slow but actually works***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V_bdry_old = set(self.generate_V_cw_B())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V.update(V)
        self.H.update(H)
        self.F.update(F)
        self.h_comp_B = self.find_h_comp_B(F_need2check=F)
        return V_bdry_new

    def expand_boundary(self):
        """
        ***a little bit faster***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V, H, F = set(), set(), set()
        V_bdry_old = set(self.generate_V_cw_B())
        for h_start in self.generate_H_cw_B():
            if self.supermesh.complement_boundary_contains_h(h_start):
                continue
            for h in self.supermesh.generate_H_in_cw_from_h(h_start):
                f = self.supermesh.f_left_h(h)
                if f in self.F:
                    break
                n = self.supermesh.h_next_h(h)
                v = self.supermesh.v_origin_h(h)
                V.add(v)
                H.update([h, n])
                if f >= 0:
                    nn = self.supermesh.h_next_h(n)
                    H.update([nn, self.supermesh.h_twin_h(nn)])
                    F.add(f)

        # V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V.update(V)
        self.H.update(H)
        self.F.update(F)
        self.h_comp_B = self.find_h_comp_B(F_need2check=F)
        return V_bdry_new

    def expand_boundary_doesnt_work_yet(self):
        """
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V, H, F = set(), set(), set()
        V_bdry_old = set(self.generate_V_cw_B())
        for h_start in self.generate_H_cw_B():
            if self.supermesh.complement_boundary_contains_h(h_start):
                continue
            for h in self.supermesh.generate_H_in_cw_from_h(h_start):
                f = self.supermesh.f_left_h(h)
                if f in self.F:
                    break
                n = self.supermesh.h_next_h(h)
                v = self.supermesh.v_origin_h(h)
                V.add(v)
                H.update([h, n])
                if f >= 0:
                    nn = self.supermesh.h_next_h(n)
                    H.update([nn, self.supermesh.h_twin_h(nn)])
                    F.add(f)

        # V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V = V
        self.H = H
        self.F = F.copy()
        self.h_comp_B = self.find_h_comp_B(F_need2check=F)
        return V_bdry_new

    ##############################################
    def to_half_edge_mesh(self):
        V = sorted(self.V)
        F = sorted(self.F)
        xyz_coord_V = [self.xyz_coord_v(v) for v in V]
        F = [
            [
                V.index(self.v_origin_h(h))
                for h in self.generate_H_next_h(self.h_bound_f(f))
            ]
            for f in F
        ]
        return HalfEdgeMesh.from_vert_face_list(xyz_coord_V, F)

    @property
    def data_lists(self):
        """ """
        V = sorted(self.V)
        H = sorted(self.H)
        F = sorted(self.F)
        # [x if x<.5 else 33 for x in X]
        xyz_coord_V = [self.xyz_coord_v(v) for v in V]
        h_out_V = [H.index(self.h_out_v(v)) for v in V]
        v_origin_H = [V.index(self.v_origin_h(h)) for h in H]
        h_next_H = [H.index(self.h_next_h(h)) for h in H]
        h_twin_H = [H.index(self.h_twin_h(h)) for h in H]
        f_left_H = [
            self.f_left_h(h) if self.f_left_h(h) < 0 else F.index(self.f_left_h(h))
            for h in H
        ]
        h_bound_F = [H.index(self.h_bound_f(f)) for f in F]

        return (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

    ##############################################
    ##############################################
    # to be deprecated


class LaplaceOperatorBase:
    def __init__(self, mesh):
        self.mesh = mesh

    def compute_weight(self, vi, vj):
        pass

    def compute_matrix(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class CotanLaplaceOperator:
    def __init__(self, mesh):
        self.mesh = mesh
        self.csr_data = []
        self.csr_indices = []
        self.csr_indptr = []
        self.V = self.mesh.V
        self.Vdict = {vi: i for i, vi in enumerate(self.V)}

        # self.csr_data, self.csr_indices, self.csr_indptr = self.compute_csr_matrix()
        # self.matrix = csr_matrix(
        #     (self.csr_data, self.csr_indices, self.csr_indptr),
        #     shape=(len(self.V), len(self.V)),
        # )

    @lru_cache(maxsize=None)
    def cot_theta_h_opposite(self, hij):
        if self.mesh.complement_boundary_contains_h(hij):
            return 0.0
        vi = self.mesh.v_origin_h(hij)
        ri = self.mesh.xyz_coord_v(vi)
        vj = self.mesh.v_head_h(hij)
        rj = self.mesh.xyz_coord_v(vj)
        hijp1 = self.mesh.h_out_ccw_from_h(hij)
        vjp1 = self.mesh.v_head_h(hijp1)
        rjp1 = self.mesh.xyz_coord_v(vjp1)

        ui = ri - rjp1
        uj = rj - rjp1
        cos_theta = np.dot(ui, uj) / (np.linalg.norm(ui) * np.linalg.norm(uj))
        cot_theta = cos_theta / np.sqrt(1 - cos_theta**2)
        return cot_theta

    @lru_cache(maxsize=None)
    def dual_cell_area(self, vi):
        return self.mesh.meyercell_area(vi)

    def cache_clear(self):
        self.cot_theta_h_opposite.cache_clear()
        self.dual_cell_area.cache_clear()

    def compute_csr_matrix(self):
        self.T_compute = time()
        csr_data = []
        csr_indices = []
        csr_indptr = []
        nonzero_count = 0
        csr_indptr.append(nonzero_count)
        for i, vi in enumerate(self.V):
            M_i = self.dual_cell_area(vi)
            indices_i = [i]
            data_i = [0.0]
            nonzero_count += 1
            for hij in self.mesh.generate_H_out_v_clockwise(vi):
                hji = self.mesh.h_twin_h(hij)
                vj = self.mesh.v_head_h(hij)
                j = self.Vdict[vj]
                indices_i.append(j)
                data_ij = (
                    self.cot_theta_h_opposite(hij) + self.cot_theta_h_opposite(hji)
                ) / (2 * M_i)
                data_i[0] -= data_ij
                data_i.append(data_ij)
                nonzero_count += 1
            csr_indices.extend(indices_i)
            csr_data.extend(data_i)
            csr_indptr.append(nonzero_count)
        self.T_compute = time() - self.T_compute

        return csr_data, csr_indices, csr_indptr

    def run_unit_sphere_mean_curvature_normal_tests(self):
        self.csr_data, self.csr_indices, self.csr_indptr = self.compute_csr_matrix()
        self.matrix = csr_matrix(
            (self.csr_data, self.csr_indices, self.csr_indptr),
            shape=(len(self.V), len(self.V)),
        )

        self.weights_matrix = self.compute_weights_matrix()
        self.matrix = self.compute_matrix()
        self.Y = self.mesh.xyz_array
        self.lapY = self.apply(self.Y)
        self.lapY_actual = -2 * self.Y
        self.lapY_error = np.linalg.norm(self.lapY - self.lapY_actual, axis=1)
        self.lapY_error_max = np.linalg.norm(self.lapY_error, np.inf)
        self.lapY_error_ave = np.mean(self.lapY_error)
        self.H = 0.5 * np.linalg.norm(self.lapY, axis=1)
        self.H_max = np.linalg.norm(self.H, np.inf)
        self.H_ave = np.mean(self.H)
        self.sparsity = self.matrix.nnz / self.matrix.shape[0] ** 2
        self.weights_sparsity = (
            self.weights_matrix.nnz / self.weights_matrix.shape[0] ** 2
        )


class HeatLaplaceOperator:
    def __init__(self, mesh):
        self.mesh = mesh
        self.csr_data = []
        self.csr_indices = []
        self.csr_indptr = []
        self.V = self.mesh.V
        self.Vdict = {vi: i for i, vi in enumerate(self.V)}
        self.csr_data, self.csr_indices, self.csr_indptr = self.compute_csr_matrix()
        self.matrix = csr_matrix(
            (self.csr_data, self.csr_indices, self.csr_indptr),
            shape=(len(self.V), len(self.V)),
        )

    @lru_cache(maxsize=None)
    def dual_cell_area(self, vi):
        return self.mesh.meyercell_area(vi)

    @lru_cache(maxsize=None)
    def interaction_vv(self, v0, v1):
        r0, r1 = self.mesh.xyz_coord_v(v0), self.mesh.xyz_coord_v(v1)
        A0, A1 = self.dual_cell_area(v0), self.dual_cell_area(v1)
        W01 = (
            A1 * np.exp(-np.linalg.norm(r1 - r0) ** 2 / (4 * A0)) / (4 * np.pi * A0**2)
        )
        return W01

    @lru_cache(maxsize=None)
    def interaction_vv_symmetricish(self, v0, v1):
        r0, r1 = self.mesh.xyz_coord_v(v0), self.mesh.xyz_coord_v(v1)
        A0, A1 = self.dual_cell_area(v0), self.dual_cell_area(v1)
        W01 = (
            A1
            * np.exp(-np.linalg.norm(r1 - r0) ** 2 / (2 * (A0 + A1)))
            / (np.pi * (A0 + A1) ** 2)
        )
        # 1/Mi=sum_j Aj/(Ai+Aj)**2
        return W01

    def cache_clear(self):
        self.cot_theta_h_opposite.cache_clear()
        self.dual_cell_area.cache_clear()

    def nearest_neighbors_interaction(self, v0):
        data = [0.0]
        indices = [self.Vdict[v0]]
        for h in self.mesh.generate_H_out_v_clockwise(v0):
            v1 = self.mesh.v_head_h(h)
            W01 = self.interaction_vv(v0, v1)
            data.append(W01)
            data[0] -= W01
            indices.append(self.Vdict[v1])
        return data, indices

    def next_ring_interactions(self, v0, data, indices):
        return 1

    def compute_csr_matrix(self):
        t = time()
        csr_data = []
        csr_indices = []
        csr_indptr = []
        nonzero_count = 0
        csr_indptr.append(nonzero_count)
        for i, vi in enumerate(self.V):
            M_i = self.dual_cell_area(vi)
            indices_i = [i]
            data_i = [0.0]
            nonzero_count += 1
            for hij in self.mesh.generate_H_out_v_clockwise(vi):
                hji = self.mesh.h_twin_h(hij)
                vj = self.mesh.v_head_h(hij)
                j = self.Vdict[vj]
                indices_i.append(j)
                data_ij = (
                    self.cot_theta_h_opposite(hij) + self.cot_theta_h_opposite(hji)
                ) / (2 * M_i)
                data_i[0] -= data_ij
                data_i.append(data_ij)
                nonzero_count += 1
            csr_indices.extend(indices_i)
            csr_data.extend(data_i)
            csr_indptr.append(nonzero_count)

        return csr_data, csr_indices, csr_indptr


class CotanLaplacian(LaplaceOperatorBase):
    def __init__(
        self, mesh, rtol=1e-6, atol=1e-6, compute=False, data_path=None, run_tests=False
    ):
        super().__init__(mesh)
        self.data_path = data_path
        self.rtol = rtol
        self.atol = atol
        self.num_vertices = mesh.num_vertices
        self.Vindices = sorted(self.mesh.xyz_coord_V.keys())
        self.Vdict = {v: i for i, v in enumerate(self.Vindices)}
        if compute:
            self.weights_matrix = self.compute_weights_matrix()
            self.matrix = self.compute_matrix()
        if run_tests:
            self.run_unit_sphere_mean_curvature_normal_tests()

    def save(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
        if self.data_path is None:
            raise ValueError("No path to save data.")
        # with open(self.data_path, "wb") as f:
        #     pickle.dump(self.__dict__, f)
        with open(self.data_path + ".pickle", "wb") as f:
            pickle.dump(self, f)

    @lru_cache(maxsize=None)
    def barcell_area(self, v):
        return self.mesh.barcell_area(v)

    @lru_cache(maxsize=None)
    def meyercell_area(self, v):
        return self.mesh.meyercell_area(v)

    # @lru_cache(maxsize=None)
    # def compute_weight(self, vi, vj):
    #     Ai = self.meyercell_area(vi)
    #     ri = self.mesh.xyz_coord_v(vi)
    #     ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2

    #     return wij

    def compute_weights_row(self, vi):
        """computes the laplacian of Y at each vertex"""
        data, col_indices = [], []
        i = self.Vdict[vi]
        Atot = 0.0
        ri = self.mesh.xyz_coord_v(vi)
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        for hij in self.mesh.generate_H_out_v_clockwise(vi):
            vj = self.mesh.v_head_h(hij)
            j = self.Vdict[vj]
            col_indices.append(j)
            hijm1 = self.mesh.h_out_cw_from_h(hij)
            hijp1 = self.mesh.h_out_ccw_from_h(hij)
            vjm1 = self.mesh.v_head_h(hijm1)
            vjp1 = self.mesh.v_head_h(hijp1)

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
            data.append((cot_thetam + cot_thetap) / 2)

        for k in range(len(data)):
            data[k] /= Atot
        return data, col_indices

    def compute_weights_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_weights_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        # self.wrow_indices = row_indices
        # self.wcol_indices = col_indices
        # self.wdata = data
        self.T_compute_weights_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def compute_row(self, vi):
        i = self.Vdict[vi]
        data, col_indices = self.compute_weights_row(vi)
        # data[0] -= sum(data)

        # return data, col_indices
        return [-sum(data), *data], [i, *col_indices]

    def compute_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        self.T_compute_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def apply2array(self, Y):
        return self.matrix.dot(Y)

    def apply2dict(self, Y):
        return self.matrix.dot([Y[v] for v in self.Vindices])

    def apply(self, Y):
        t = time()
        if isinstance(Y, dict):
            lapY = self.apply2dict(Y)
        elif isinstance(Y, np.ndarray):
            lapY = self.apply2array(Y)
        else:
            raise ValueError("Argument must be dict or numpy.ndarray.")
        self.T_apply = time() - t
        return lapY

    def run_unit_sphere_mean_curvature_normal_tests(self):
        self.weights_matrix = self.compute_weights_matrix()
        self.matrix = self.compute_matrix()
        self.Y = self.mesh.xyz_array
        self.lapY = self.apply(self.Y)
        self.lapY_actual = -2 * self.Y
        self.lapY_error = np.linalg.norm(self.lapY - self.lapY_actual, axis=1)
        self.lapY_error_max = np.linalg.norm(self.lapY_error, np.inf)
        self.lapY_error_ave = np.mean(self.lapY_error)
        self.H = 0.5 * np.linalg.norm(self.lapY, axis=1)
        self.H_max = np.linalg.norm(self.H, np.inf)
        self.H_ave = np.mean(self.H)
        self.sparsity = self.matrix.nnz / self.matrix.shape[0] ** 2
        self.weights_sparsity = (
            self.weights_matrix.nnz / self.weights_matrix.shape[0] ** 2
        )


class HeatLaplacian(LaplaceOperatorBase):
    def __init__(
        self, mesh, rtol=1e-6, atol=1e-6, compute=False, data_path=None, run_tests=False
    ):
        super().__init__(mesh)
        self.data_path = data_path
        self.rtol = rtol
        self.atol = atol
        self.num_vertices = mesh.num_vertices
        self.Vindices = sorted(self.mesh.xyz_coord_V.keys())
        self.Vdict = {v: i for i, v in enumerate(self.Vindices)}
        if compute:
            self.weights_matrix = self.compute_weights_matrix()
            self.matrix = self.compute_matrix()
        if run_tests:
            self.run_unit_sphere_mean_curvature_normal_tests()

    def save(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
        if self.data_path is None:
            raise ValueError("No path to save data.")
        # with open(self.data_path, "wb") as f:
        #     pickle.dump(self.__dict__, f)
        with open(self.data_path + ".pickle", "wb") as f:
            pickle.dump(self, f)

    @lru_cache(maxsize=None)
    def barcell_area(self, v):
        return self.mesh.barcell_area(v)

    @lru_cache(maxsize=None)
    def meyercell_area(self, v):
        return self.mesh.meyercell_area(v)

    @lru_cache(maxsize=None)
    def compute_weight(self, vi, vj):
        # vi,vj = self.Vindices[i], self.Vindices[j]
        ri, rj = self.mesh.xyz_coord_v(vi), self.mesh.xyz_coord_v(vj)
        Ai, Aj = self.barcell_area(vi), self.barcell_area(vj)
        # Ai, Aj = self.meyercell_area(vi), self.meyercell_area(vj)
        wij = (
            Aj * np.exp(-np.linalg.norm(rj - ri) ** 2 / (4 * Ai)) / (4 * np.pi * Ai**2)
        )
        return wij

    def compute_weights_row(self, vi):
        p = HalfEdgePatch.from_seed_vertex(vi, self.mesh)
        V = p.V - {vi}
        i = self.Vdict[vi]
        wii = self.compute_weight(vi, vi)
        data = [wii]
        col_indices = [i]
        for vj in V:
            j = self.Vdict[vj]
            wij = self.compute_weight(vi, vj)
            data.append(wij)
            col_indices.append(j)

        norm_wi = np.linalg.norm(data)

        while True:
            V = p.expand_boundary()
            new_data = []
            new_col_indices = []
            for vj in V:
                j = self.Vdict[vj]
                wij = self.compute_weight(vi, vj)
                new_data.append(wij)
                new_col_indices.append(j)

            norm_dwi = np.linalg.norm(new_data)
            data.extend(new_data)
            col_indices.extend(new_col_indices)

            if norm_dwi < self.rtol * norm_wi + self.atol:
                break
            if len(data) >= self.num_vertices:
                break
            norm_wi = np.linalg.norm(data)

        return data, col_indices

    def compute_weights_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_weights_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        # self.wrow_indices = row_indices
        # self.wcol_indices = col_indices
        # self.wdata = data
        self.T_compute_weights_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def compute_row(self, vi):
        p = HalfEdgePatch.from_seed_vertex(vi, self.mesh)
        V = p.V - {vi}
        i = self.Vdict[vi]
        Lii = 0.0
        data = [Lii]
        col_indices = [i]
        for vj in V:
            j = self.Vdict[vj]
            wij = self.compute_weight(vi, vj)
            data.append(wij)
            col_indices.append(j)
            data[0] -= wij

        norm_Li = np.linalg.norm(data)

        while True:
            V = p.expand_boundary()
            new_data = [0.0]
            new_col_indices = [i]
            for vj in V:
                j = self.Vdict[vj]
                wij = self.compute_weight(vi, vj)
                new_data.append(wij)
                new_col_indices.append(j)
                new_data[0] -= wij
            norm_dLi = np.linalg.norm(new_data)
            data[0] += new_data[0]
            data.extend(new_data[1:])
            col_indices.extend(new_col_indices[1:])

            if norm_dLi < self.rtol * norm_Li + self.atol:
                break
            if len(data) >= self.num_vertices:
                break
            norm_Li = np.linalg.norm(data)

        return data, col_indices

    def compute_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        self.T_compute_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def apply2array(self, Y):
        return self.matrix.dot(Y)

    def apply2dict(self, Y):
        return self.matrix.dot([Y[v] for v in self.Vindices])

    def apply(self, Y):
        t = time()
        if isinstance(Y, dict):
            lapY = self.apply2dict(Y)
        elif isinstance(Y, np.ndarray):
            lapY = self.apply2array(Y)
        else:
            raise ValueError("Argument must be dict or numpy.ndarray.")
        self.T_apply = time() - t
        return lapY

    def run_unit_sphere_mean_curvature_normal_tests(self):
        self.weights_matrix = self.compute_weights_matrix()
        self.matrix = self.compute_matrix()
        self.Y = self.mesh.xyz_array
        self.lapY = self.apply(self.Y)
        self.lapY_actual = -2 * self.Y
        self.lapY_error = np.linalg.norm(self.lapY - self.lapY_actual, axis=1)
        self.lapY_error_max = np.linalg.norm(self.lapY_error, np.inf)
        self.lapY_error_ave = np.mean(self.lapY_error)
        self.H = 0.5 * np.linalg.norm(self.lapY, axis=1)
        self.H_max = np.linalg.norm(self.H, np.inf)
        self.H_ave = np.mean(self.H)
        self.sparsity = self.matrix.nnz / self.matrix.shape[0] ** 2
        self.weights_sparsity = (
            self.weights_matrix.nnz / self.weights_matrix.shape[0] ** 2
        )


class MeyerHeatLaplacian(LaplaceOperatorBase):
    def __init__(
        self, mesh, rtol=1e-6, atol=1e-6, compute=False, data_path=None, run_tests=False
    ):
        super().__init__(mesh)
        self.data_path = data_path
        self.rtol = rtol
        self.atol = atol
        self.num_vertices = mesh.num_vertices
        self.Vindices = sorted(self.mesh.xyz_coord_V.keys())
        self.Vdict = {v: i for i, v in enumerate(self.Vindices)}
        if compute:
            self.weights_matrix = self.compute_weights_matrix()
            self.matrix = self.compute_matrix()
        if run_tests:
            self.run_unit_sphere_mean_curvature_normal_tests()

    def save(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
        if self.data_path is None:
            raise ValueError("No path to save data.")
        with open(self.data_path + ".pickle", "wb") as f:
            pickle.dump(self, f)

    @lru_cache(maxsize=None)
    def barcell_area(self, v):
        return self.mesh.barcell_area(v)

    @lru_cache(maxsize=None)
    def meyercell_area(self, v):
        return self.mesh.meyercell_area(v)

    @lru_cache(maxsize=None)
    def compute_weight(self, vi, vj):
        ri, rj = self.mesh.xyz_coord_v(vi), self.mesh.xyz_coord_v(vj)
        Ai, Aj = self.meyercell_area(vi), self.meyercell_area(vj)
        wij = (
            Aj * np.exp(-np.linalg.norm(rj - ri) ** 2 / (4 * Ai)) / (4 * np.pi * Ai**2)
        )
        return wij

    def compute_weights_row(self, vi):
        p = HalfEdgePatch.from_seed_vertex(vi, self.mesh)
        V = p.V - {vi}
        i = self.Vdict[vi]
        wii = self.compute_weight(vi, vi)
        data = [wii]
        col_indices = [i]
        for vj in V:
            j = self.Vdict[vj]
            wij = self.compute_weight(vi, vj)
            data.append(wij)
            col_indices.append(j)

        norm_wi = np.linalg.norm(data)

        while True:
            V = p.expand_boundary()
            new_data = []
            new_col_indices = []
            for vj in V:
                j = self.Vdict[vj]
                wij = self.compute_weight(vi, vj)
                new_data.append(wij)
                new_col_indices.append(j)

            norm_dwi = np.linalg.norm(new_data)
            data.extend(new_data)
            col_indices.extend(new_col_indices)

            if norm_dwi < self.rtol * norm_wi + self.atol:
                break
            if len(data) >= self.num_vertices:
                break
            norm_wi = np.linalg.norm(data)

        return data, col_indices

    def compute_weights_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_weights_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        # self.wrow_indices = row_indices
        # self.wcol_indices = col_indices
        # self.wdata = data
        self.T_compute_weights_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def compute_row(self, vi):
        p = HalfEdgePatch.from_seed_vertex(vi, self.mesh)
        V = p.V - {vi}
        i = self.Vdict[vi]
        Lii = 0.0
        data = [Lii]
        col_indices = [i]
        for vj in V:
            j = self.Vdict[vj]
            wij = self.compute_weight(vi, vj)
            data.append(wij)
            col_indices.append(j)
            data[0] -= wij

        norm_Li = np.linalg.norm(data)

        while True:
            V = p.expand_boundary()
            new_data = [0.0]
            new_col_indices = [i]
            for vj in V:
                j = self.Vdict[vj]
                wij = self.compute_weight(vi, vj)
                new_data.append(wij)
                new_col_indices.append(j)
                new_data[0] -= wij
            norm_dLi = np.linalg.norm(new_data)
            data[0] += new_data[0]
            data.extend(new_data[1:])
            col_indices.extend(new_col_indices[1:])

            if norm_dLi < self.rtol * norm_Li + self.atol:
                break
            if len(data) >= self.num_vertices:
                break
            norm_Li = np.linalg.norm(data)

        return data, col_indices

    def compute_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        self.T_compute_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def apply2array(self, Y):
        return self.matrix.dot(Y)

    def apply2dict(self, Y):
        return self.matrix.dot([Y[v] for v in self.Vindices])

    def apply(self, Y):
        t = time()
        if isinstance(Y, dict):
            lapY = self.apply2dict(Y)
        elif isinstance(Y, np.ndarray):
            lapY = self.apply2array(Y)
        else:
            raise ValueError("Argument must be dict or numpy.ndarray.")
        self.T_apply = time() - t
        return lapY

    def run_unit_sphere_mean_curvature_normal_tests(self):
        self.weights_matrix = self.compute_weights_matrix()
        self.matrix = self.compute_matrix()
        self.Y = self.mesh.xyz_array
        self.lapY = self.apply(self.Y)
        self.lapY_actual = -2 * self.Y
        self.lapY_error = np.linalg.norm(self.lapY - self.lapY_actual, axis=1)
        self.lapY_error_max = np.linalg.norm(self.lapY_error, np.inf)
        self.lapY_error_ave = np.mean(self.lapY_error)
        self.H = 0.5 * np.linalg.norm(self.lapY, axis=1)
        self.H_max = np.linalg.norm(self.H, np.inf)
        self.H_ave = np.mean(self.H)
        self.sparsity = self.matrix.nnz / self.matrix.shape[0] ** 2
        self.weights_sparsity = (
            self.weights_matrix.nnz / self.weights_matrix.shape[0] ** 2
        )


class FixedTimelikeParamHeatLaplacian(LaplaceOperatorBase):
    def __init__(
        self, mesh, rtol=1e-6, atol=1e-6, compute=False, data_path=None, run_tests=False
    ):
        super().__init__(mesh)
        self.Ai = self.mesh.total_area_of_faces() / self.mesh.num_vertices
        self.data_path = data_path
        self.rtol = rtol
        self.atol = atol
        self.num_vertices = mesh.num_vertices
        self.Vindices = sorted(self.mesh.xyz_coord_V.keys())
        self.Vdict = {v: i for i, v in enumerate(self.Vindices)}
        if compute:
            self.weights_matrix = self.compute_weights_matrix()
            self.matrix = self.compute_matrix()
        if run_tests:
            self.run_unit_sphere_mean_curvature_normal_tests()

    def save(self, data_path=None):
        if data_path is not None:
            self.data_path = data_path
        if self.data_path is None:
            raise ValueError("No path to save data.")
        with open(self.data_path + ".pickle", "wb") as f:
            pickle.dump(self, f)

    @lru_cache(maxsize=None)
    def barcell_area(self, v):
        return self.mesh.barcell_area(v)

    @lru_cache(maxsize=None)
    def meyercell_area(self, v):
        return self.mesh.meyercell_area(v)

    @lru_cache(maxsize=None)
    def compute_weight(self, vi, vj):
        ri, rj = self.mesh.xyz_coord_v(vi), self.mesh.xyz_coord_v(vj)
        Ai, Aj = self.Ai, self.barcell_area(vj)
        wij = (
            Aj * np.exp(-np.linalg.norm(rj - ri) ** 2 / (4 * Ai)) / (4 * np.pi * Ai**2)
        )
        return wij

    def compute_weights_row(self, vi):
        p = HalfEdgePatch.from_seed_vertex(vi, self.mesh)
        V = p.V - {vi}
        i = self.Vdict[vi]
        wii = self.compute_weight(vi, vi)
        data = [wii]
        col_indices = [i]
        for vj in V:
            j = self.Vdict[vj]
            wij = self.compute_weight(vi, vj)
            data.append(wij)
            col_indices.append(j)

        norm_wi = np.linalg.norm(data)

        while True:
            V = p.expand_boundary()
            new_data = []
            new_col_indices = []
            for vj in V:
                j = self.Vdict[vj]
                wij = self.compute_weight(vi, vj)
                new_data.append(wij)
                new_col_indices.append(j)

            norm_dwi = np.linalg.norm(new_data)
            data.extend(new_data)
            col_indices.extend(new_col_indices)

            if norm_dwi < self.rtol * norm_wi + self.atol:
                break
            if len(data) >= self.num_vertices:
                break
            norm_wi = np.linalg.norm(data)

        return data, col_indices

    def compute_weights_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_weights_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        # self.wrow_indices = row_indices
        # self.wcol_indices = col_indices
        # self.wdata = data
        self.T_compute_weights_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def compute_row(self, vi):
        p = HalfEdgePatch.from_seed_vertex(vi, self.mesh)
        V = p.V - {vi}
        i = self.Vdict[vi]
        Lii = 0.0
        data = [Lii]
        col_indices = [i]
        for vj in V:
            j = self.Vdict[vj]
            wij = self.compute_weight(vi, vj)
            data.append(wij)
            col_indices.append(j)
            data[0] -= wij

        norm_Li = np.linalg.norm(data)

        while True:
            V = p.expand_boundary()
            new_data = [0.0]
            new_col_indices = [i]
            for vj in V:
                j = self.Vdict[vj]
                wij = self.compute_weight(vi, vj)
                new_data.append(wij)
                new_col_indices.append(j)
                new_data[0] -= wij
            norm_dLi = np.linalg.norm(new_data)
            data[0] += new_data[0]
            data.extend(new_data[1:])
            col_indices.extend(new_col_indices[1:])

            if norm_dLi < self.rtol * norm_Li + self.atol:
                break
            if len(data) >= self.num_vertices:
                break
            norm_Li = np.linalg.norm(data)

        return data, col_indices

    def compute_matrix(self):
        """Construct the sparse Laplacian matrix using cached areas."""
        t = time()
        row_indices = []
        col_indices = []
        data = []

        for vi, i in self.Vdict.items():
            new_data, new_col_indices = self.compute_row(vi)
            data.extend(new_data)
            col_indices.extend(new_col_indices)
            row_indices.extend([i] * len(new_data))
        self.T_compute_matrix = time() - t
        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.mesh.num_vertices, self.mesh.num_vertices),
        )

    def apply2array(self, Y):
        return self.matrix.dot(Y)

    def apply2dict(self, Y):
        return self.matrix.dot([Y[v] for v in self.Vindices])

    def apply(self, Y):
        t = time()
        if isinstance(Y, dict):
            lapY = self.apply2dict(Y)
        elif isinstance(Y, np.ndarray):
            lapY = self.apply2array(Y)
        else:
            raise ValueError("Argument must be dict or numpy.ndarray.")
        self.T_apply = time() - t
        return lapY

    def run_unit_sphere_mean_curvature_normal_tests(self):
        self.weights_matrix = self.compute_weights_matrix()
        self.matrix = self.compute_matrix()
        self.Y = self.mesh.xyz_array
        self.lapY = self.apply(self.Y)
        self.lapY_actual = -2 * self.Y
        self.lapY_error = np.linalg.norm(self.lapY - self.lapY_actual, axis=1)
        self.lapY_error_max = np.linalg.norm(self.lapY_error, np.inf)
        self.lapY_error_ave = np.mean(self.lapY_error)
        self.H = 0.5 * np.linalg.norm(self.lapY, axis=1)
        self.H_max = np.linalg.norm(self.H, np.inf)
        self.H_ave = np.mean(self.H)
        self.sparsity = self.matrix.nnz / self.matrix.shape[0] ** 2
        self.weights_sparsity = (
            self.weights_matrix.nnz / self.weights_matrix.shape[0] ** 2
        )


class PatchBoundary:
    def __init__(self, supermesh):
        self.supermesh = supermesh
        self.h_next_H = dict()
        self.h_twin_H = dict()
        self.v_origin_H = dict()
        self.front_contains_h = dict()

    @classmethod
    def from_seed_vertex(cls, v_seed, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If v_seed is not in a boundary of supermesh, the patch will be a disk centered at v_seed.

        Parameters:
            v_seed (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        b = cls(supermesh)
        p = HalfEdgePatch.from_seed_vertex(v_seed, supermesh)
        H_complement_B = p.generate_H_cw_B()
        h_prev_H = dict()
        for h in H_complement_B:
            n = p.h_next_h(h)
            t = p.h_twin_h(h)
            b.h_next_H[h] = n
            b.h_twin_H[h] = t
            b.h_twin_H[t] = h

            h_prev_H[t] = p.h_twin_h(n)
        for key, val in h_prev_h.items():
            b.h_next_h[val] = key

        return b
