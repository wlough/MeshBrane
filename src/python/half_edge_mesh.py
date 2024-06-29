from functools import lru_cache
from src.python.ply_tools import VertTri2HalfEdgeConverter
import numpy as np
from scipy.sparse import csr_matrix
from time import time
import pickle


class HalfEdgeMeshBase:
    """
    h_bound_F-->h_adjacent_F
    Dict-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.arrays of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as dicts of vertex/half-edge/face labels. Each data dict has a name of the form "a_description_B", where "a" denotes the type of object associated with the dict elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "B" denotes the type of object associated with the dict indices ("V" for vertex, "H" for half-edge, or "F" for face), and "description" is a description of information represented by the dict. For example, "_v_origin_H" is a dict of vertices at the origin of each half-edge. The i-th element of data dict "a_description_B" can be accessed using the "a_description_b(i)" method.

    Properties
    ----------
    xyz_coord_V : dict of numpy.array
        _xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : dict of int
        _h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : dict of int
        _v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : dict of int
        _h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : dict of int
        _h_twin_H[j] = half-edge antiparalel to half-edge j
    f_left_H : dict of int
        _f_left_H[j] = face to the left of half-edge j
    h_bound_F : dict of int
        _h_bound_F[k] = some half-edge on the ccw boudary of face k
    h_cw_B : dict of int
        _h_cw_B[n] = half-edge in boundary n of the mesh

    Initialization
    ---------------
    The HalfEdgeMesh class can be initialized in several ways:
    - Directly from half-edge mesh data dicts:
        HalfEdgeMesh(xyz_coord_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_bound_F)
    - From a dict of vertex positions and a dict of face vertices:
        HalfEdgeMesh.from_vert_face_dict(xyz_coord_V, vvv_of_F)
    - From a ply file (binary/ascii) containing vertex/face data:
        HalfEdgeMesh.from_vertex_face_ply(ply_path)
        * See HalfEdgeMeshBuilder for more details about ply format.
    - From a ply file (binary/ascii) containing half-edge mesh data:
        HalfEdgeMesh.from_half_edge_ply(ply_path)
        * See HalfEdgeMeshBuilder for more details about ply format.
    - From C-Glass binary data:
        HalfEdgeMesh.from_cglass_binary(data)
        * To be implemented...
    """

    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_cw_B=None,
    ):
        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.h_bound_F = h_bound_F
        self.f_left_H = f_left_H
        if h_cw_B is None:
            self.h_cw_B = self.find_h_cw_B()
        else:
            self.h_cw_B = h_cw_B

    @classmethod
    def from_vert_face_list(cls, xyz_coord_V, vvv_of_F):
        """
        Initialize a half-edge mesh from vertex/face data.

        Parameters:
        ----------
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i
        vvv_of_F : list of lists of integers
            vvv_of_F[j] = [v0, v1, v2] = vertices in face j.

        Returns:
        -------
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with the given vertices and faces.
        """
        return cls(
            *VertTri2HalfEdgeConverter.from_source_samples(
                xyz_coord_V, vvv_of_F
            ).target_samples
        )

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        return cls(*VertTri2HalfEdgeConverter.from_source_ply(ply_path).target_samples)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(*VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples)

    @classmethod
    def from_half_edge_ply_no_bdry_twin(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data using the h_twin = -1 convention for boundary edges.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(*VertTri2HalfEdgeConverter.from_target_ply(ply_path).source_samples)

    #######################################################
    @property
    def V(self):
        return sorted(self._xyz_coord_V.keys())

    @property
    def H(self):
        return sorted(self._v_origin_H.keys())

    @property
    def F(self):
        return sorted(self._h_bound_F.keys())

    @property
    def h_cw_B(self):
        return self._h_cw_B

    @h_cw_B.setter
    def h_cw_B(self, value):
        if isinstance(value, dict):
            self._h_cw_B = value
        elif hasattr(value, "__iter__"):
            self._h_cw_B = dict(enumerate(value))
        else:
            raise TypeError("h_cw_B must be a dictionary or an iterable.")

    @property
    def xyz_coord_V(self):
        return self._xyz_coord_V

    @property
    def xyz_array(self):
        return np.array([self.xyz_coord_v(v) for v in sorted(self.xyz_coord_V.keys())])

    @xyz_coord_V.setter
    def xyz_coord_V(self, value):
        if isinstance(value, dict):
            self._xyz_coord_V = value
        elif hasattr(value, "__iter__"):
            self._xyz_coord_V = dict(enumerate(value))
        else:
            raise TypeError("xyz_coord_V must be a dictionary or an iterable.")

    @property
    def h_out_V(self):
        return self._h_out_V

    @h_out_V.setter
    def h_out_V(self, value):
        if isinstance(value, dict):
            self._h_out_V = value
        elif hasattr(value, "__iter__"):
            self._h_out_V = dict(enumerate(value))
        else:
            raise TypeError("h_out_V must be a dictionary or an iterable.")

    @property
    def v_origin_H(self):
        return self._v_origin_H

    @v_origin_H.setter
    def v_origin_H(self, value):
        if isinstance(value, dict):
            self._v_origin_H = value
        elif hasattr(value, "__iter__"):
            self._v_origin_H = dict(enumerate(value))
        else:
            raise TypeError("v_origin_H must be a dictionary or an iterable.")

    @property
    def h_next_H(self):
        return self._h_next_H

    @h_next_H.setter
    def h_next_H(self, value):
        if isinstance(value, dict):
            self._h_next_H = value
        elif hasattr(value, "__iter__"):
            self._h_next_H = dict(enumerate(value))
        else:
            raise TypeError("h_next_H must be a dictionary or an iterable.")

    @property
    def h_twin_H(self):
        return self._h_twin_H

    @h_twin_H.setter
    def h_twin_H(self, value):
        if isinstance(value, dict):
            self._h_twin_H = value
        elif hasattr(value, "__iter__"):
            self._h_twin_H = dict(enumerate(value))
        else:
            raise TypeError("h_twin_H must be a dictionary or an iterable.")

    @property
    def f_left_H(self):
        return self._f_left_H

    @f_left_H.setter
    def f_left_H(self, value):
        if isinstance(value, dict):
            self._f_left_H = value
        elif hasattr(value, "__iter__"):
            self._f_left_H = dict(enumerate(value))
        else:
            raise TypeError("f_left_H must be a dictionary or an iterable.")

    @property
    def h_bound_F(self):
        return self._h_bound_F

    @h_bound_F.setter
    def h_bound_F(self, value):
        if isinstance(value, dict):
            self._h_bound_F = value
        elif hasattr(value, "__iter__"):
            self._h_bound_F = dict(enumerate(value))
        else:
            raise TypeError("h_bound_F must be a dictionary or an iterable.")

    @property
    def num_vertices(self):
        return len(self._xyz_coord_V)

    @property
    def num_edges(self):
        return len(self._v_origin_H) // 2

    @property
    def num_faces(self):
        return len(self._h_bound_F)

    @property
    def num_boundaries(self):
        return len(self._h_cw_B)

    @property
    def genus(self):
        return 1 - (self.euler_characteristic + self.num_boundaries) // 2

    @property
    def euler_characteristic(self):
        return self.num_vertices - self.num_edges + self.num_faces

    @property
    def data_lists(self):
        """
        get lists of vertex positions and connectivity data and required to reconstruct mesh or write to ply file. Vertex/half-edge/face indices are sorted in ascending order and relabeled so that the first index is 0, the second index is 1, etc...
        """
        V = sorted(self._xyz_coord_V.keys())
        H = sorted(self._v_origin_H.keys())
        F = sorted(self._h_bound_F.keys())

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

    @property
    def data_dicts(self):
        """
        get dicts of vertex positions and connectivity data and required to reconstruct mesh or write to ply file
        """
        return (
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
        )

    #######################################################
    def xyz_coord_v(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self._xyz_coord_V[v]

    def h_out_v(self, v):
        """
        get index of an outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        return self._h_out_V[v]

    def h_bound_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        return self._h_bound_F[f]

    def v_origin_h(self, h):
        """get index of the vertex at the origin of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self._v_origin_H[h]

    def f_left_h(self, h):
        """get index of the face to the left of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: face index
        """
        return self._f_left_H[h]

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face cycle

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self._h_next_H[h]

    def h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self._h_twin_H[h]

    ######################################################
    # basic helpers
    def v_head_h(self, h):
        """get index of the vertex at the head of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self.v_origin_h(self.h_twin_h(h))

    def h_out_cw_from_h(self, h):
        return self.h_next_h(self.h_twin_h(h))

    def h_out_ccw_from_h(self, h):

        return self.h_twin_h(self.h_prev_h(h))

    def h_in_cw_from_h(self, h):
        return self.h_twin_h(self.h_next_h(h))

    def h_prev_h(self, h):
        """works forhalf-edges of non-triangle faces and boundaries"""
        h_next = self.h_next_h(h)

        while h_next != h:
            h_prev = h_next
            h_next = self.h_next_h(h_prev)
        return h_prev

    def h_prev_h_tri(self, h):
        """only works for triangle faces"""
        return self.h_twin_h(self.h_next_h(self.h_next_h(h)))

    def complement_boundary_contains_h(self, h):
        """check if half-edge h is in the boundary of the mesh"""
        return self.f_left_h(h) < 0

    def interior_boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.f_left_h(self.h_twin_h(h)) < 0

    def boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.f_left_h(h) < 0 or self.f_left_h(self.h_twin_h(h)) < 0

    def boundary_contains_v(self, v):
        """check if vertex v is on the boundary of the mesh"""
        return self.boundary_contains_h(self.h_out_v(v))

    ######################################################
    # generators
    def generate_H_out_v_clockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in clockwise order until one of the following conditions is met:
        1) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def generate_H_in_cw_from_h(self, h):
        """ """
        h_start = h
        while True:
            yield h
            h = self.h_in_cw_from_h(h)
            if h == h_start:
                break

    def generate_H_bound_f(self, f):
        """Generate half-edges on the boundary of face f"""
        h = self.h_bound_f(f)
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_H_next_h(self, h):
        """Generate half-edges in the face/boundary cycle of half-edge h"""
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    ######################################################
    # The star St(s) of a k-simplex s consists of: s and all (n>k)-simplices that contain s.
    # The closure Cl(s) of a k-simplex s consists of: s and all (n<k)-simplices that are proper faces of s.
    # The link Lk(s)=Cl(St(s))-St(Cl(s)).
    def star_of_vertex(self, v):
        """Star of a vertex is the set of all simplices that contain the vertex."""
        V = {v}
        H = set()
        F = set()
        for h in self.generate_H_out_v_clockwise(v):
            ht = self.h_twin_h(h)
            H.update([h, ht])
            if not self.complement_boundary_contains_h(h):
                F.add(self.f_left_h(h))

        return V, H, F

    def star_of_edge(self, h):
        """Star of an edge is the set of all simplices that contain the edge."""
        V = set()
        H = {h, self.h_twin_h(h)}
        F = set()
        for hi in H:
            if not self.complement_boundary_contains_h(hi):
                F.add(self.f_left_h(hi))

        return V, H, F

    def star(self, V_in, H_in, F_in):
        """The star of a single simplex is the set of all simplices that have the simplex as a face."""
        F = F_in.copy()
        H = H_in.copy()
        V = V_in.copy()

        for h in H_in:
            ht = self.h_twin_h(h)
            H.add(ht)
            if not self.complement_boundary_contains_h(h):
                F.add(self.f_left_h(h))
            if not self.complement_boundary_contains_h(ht):
                F.add(self.f_left_h(ht))
        for v in V_in:
            for h in self.generate_H_out_v_clockwise(v):
                H.add(h)
                H.add(self.h_twin_h(h))
                if not self.complement_boundary_contains_h(h):
                    F.add(self.f_left_h(h))
        return V, H, F

    def closure(self, V_in, H_in, F_in):
        """The closure of a single simplex is the set of all simplices that contain the simplex as a subset of their vertices."""
        F = F_in.copy()
        H = H_in.copy()
        V = V_in.copy()

        for f in F_in:
            for h in self.generate_H_bound_f(f):
                H.add(h)
                H.add(self.h_twin_h(h))
                V.add(self.v_origin_h(h))
                V.add(self.v_origin_h(self.h_twin_h(h)))
        for h in H_in:
            V.add(self.v_origin_h(h))
            V.add(self.v_origin_h(self.h_twin_h(h)))
        return V, H, F

    def link(self, V, H, F):
        """"""
        StCl_V, StCl_H, StCl_F = self.star(*self.closure(V, H, F))
        ClSt_V, ClSt_H, ClSt_F = self.closure(*self.star(V, H, F))
        return ClSt_V - StCl_V, ClSt_H - StCl_H, ClSt_F - StCl_F

    def find_h_cw_B(self):
        """Find all boundary edges in the mesh"""
        h_cw_B = dict()
        complement_boundary_contains_H = set()
        F_need2check = set(self.h_bound_F)  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            for h in self.generate_H_bound_f(f):
                if self.interior_boundary_contains_h(h):
                    complement_boundary_contains_H.add(self.h_twin_h(h))
        while complement_boundary_contains_H:
            h = complement_boundary_contains_H.pop()
            bdry = self.f_left_h(h)
            h_cw_B[bdry] = h
            for h in self.generate_H_next_h(h):
                complement_boundary_contains_H.discard(h)
        return h_cw_B

    ######################################################
    def xyz_com_f(self, f):
        h0 = self.h_bound_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        r0 = self.xyz_coord_v(self.v_origin_h(h0))
        r1 = self.xyz_coord_v(self.v_origin_h(h1))
        r2 = self.xyz_coord_v(self.v_origin_h(h2))
        return (r0 + r1 + r2) / 3

    def vec_area_f(self, f):
        h0 = self.h_bound_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        r0 = self.xyz_coord_v(self.v_origin_h(h0))
        r1 = self.xyz_coord_v(self.v_origin_h(h1))
        r2 = self.xyz_coord_v(self.v_origin_h(h2))
        vec_area = 0.5 * (np.cross(r0, r1) + np.cross(r1, r2) + np.cross(r2, r0))
        return vec_area

    def area_f(self, f):
        return np.linalg.norm(self.vec_area_f(f))

    def total_area_of_faces(self):
        Atot = 0.0
        for f in self.h_bound_F.keys():
            Atot += self.area_f(f)

        return Atot

    def barcell_area(self, v):
        """area of the barycentric cell dual to vertex v"""
        r = self.xyz_coord_v(v)
        A = 0.0
        for h in self.generate_H_out_v_clockwise(v):
            if self.complement_boundary_contains_h(h):
                continue
            # r1 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(h)))
            # r2 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(self.h_next_h(h))))
            # A_face_vec = (
            #     np.cross(r, r1) / 2 + np.cross(r1, r2) / 2 + np.cross(r2, r) / 2
            # )
            # A_face = np.sqrt(
            #     A_face_vec[0] ** 2 + A_face_vec[1] ** 2 + A_face_vec[2] ** 2
            # )

            A += self.area_f(self.f_left_h(h)) / 3

        return A

    def total_area_of_dual_barcells(self):
        Atot = 0.0
        for v in self.xyz_coord_V.keys():
            Atot += self.barcell_area(v)

        return Atot

    def vorcell_area(self, v):
        """area of cell dual to vertex v"""
        Atot = 0.0
        r = self.xyz_coord_v(v)
        r_r = r[0] ** 2 + r[1] ** 2 + r[2] ** 2

        for h in self.generate_H_out_v_clockwise(v):
            if self.complement_boundary_contains_h(h):
                continue
            h1 = self.h_next_h(h)
            h2 = self.h_next_h(h1)
            v1 = self.v_origin_h(h1)
            r1 = self.xyz_coord_v(v1)
            v2 = self.v_origin_h(h2)
            r2 = self.xyz_coord_v(v2)

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

        return Atot

    def total_area_of_dual_vorcells(self):
        Atot = 0.0
        for v in self.xyz_coord_V.keys():
            Atot += self.vorcell_area(v)

        return Atot

    def meyercell_area(self, v):
        """Meyer's mixed area of cell dual to vertex v"""
        Atot = 0.0
        ri = self.xyz_coord_v(v)
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        # h_start = self.V_hedge[v]
        # hij = h_start
        # while True:
        for hij in self.generate_H_out_v_clockwise(v):
            if self.complement_boundary_contains_h(hij):
                continue
            hjjp1 = self.h_next_h(hij)
            hjp1i = self.h_next_h(hjjp1)
            vj = self.v_origin_h(hjjp1)
            rj = self.xyz_coord_v(vj)
            vjp1 = self.v_origin_h(hjp1i)
            rjp1 = self.xyz_coord_v(vjp1)

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

        return Atot

    def total_area_of_dual_meyercells(self):
        Atot = 0.0
        for v in self.xyz_coord_V.keys():
            Atot += self.meyercell_area(v)

        return Atot

    ######################################################
    # to be deprecated


######################################
######################################
class HalfEdgeMesh(HalfEdgeMeshBase):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

    ######################################################
    # mesh mutation
    def is_delaunay(self, h):
        r"""
        checks if edge is locally delaunay
             v2
             /|\
           v3 | v1
             \|/
             v4
        
             v2
             /|\
           v3 | v1
             \|/
             vj
        """
        vi = self.v_origin_h(self.h_next_h(self.h_twin_h(h)))
        vj = self.v_origin_h(h)
        vk = self.v_origin_h(self.h_next_h(h))
        vl = self.v_origin_h(self.h_prev_h(h))

        pij = self.xyz_coord_v(vj) - self.xyz_coord_v(vi)
        pil = self.xyz_coord_v(vl) - self.xyz_coord_v(vi)
        pkj = self.xyz_coord_v(vj) - self.xyz_coord_v(vk)
        pkl = self.xyz_coord_v(vl) - self.xyz_coord_v(vk)

        pij_pil = pij[0] * pil[0] + pij[1] * pil[1] + pij[2] * pil[2]
        pkl_pkj = pkl[0] * pkj[0] + pkl[1] * pkj[1] + pkl[2] * pkj[2]
        normpij = np.sqrt(pij[0] ** 2 + pij[1] ** 2 + pij[2] ** 2)
        normpil = np.sqrt(pil[0] ** 2 + pil[1] ** 2 + pil[2] ** 2)
        normpkj = np.sqrt(pkj[0] ** 2 + pkj[1] ** 2 + pkj[2] ** 2)
        normpkl = np.sqrt(pkl[0] ** 2 + pkl[1] ** 2 + pkl[2] ** 2)

        alphai = np.arccos(pij_pil / (normpij * normpil))
        alphak = np.arccos(pkl_pkj / (normpkl * normpkj))

        return alphai + alphak <= np.pi

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
        hjk = self.h_next_h(hlj)
        hjl = self.h_twin_h(hlj)
        hli = self.h_next_h(hjl)
        flippable = True

        vj = self.v_origin_h(hlj)
        vk = self.v_origin_h(hjk)

        him = self.h_twin_h(hli)
        while True:
            him = self.h_twin_h(self.h_prev_h(him))
            vm = self.v_origin_h(him)
            if vm == vk:
                flippable = False
                break
            if vm == vj:
                break

        return flippable

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
        n_h = self.h_next_h(h)  # ->h2
        p_h = self.h_prev_h(h)  # ->h3
        t_h = self.h_twin_h(h)  #
        nt_h = self.h_next_h(t_h)  # ->h4
        pt_h = self.h_prev_h(t_h)  # ->h1

        f_h = self.f_left_h(h)  # ->f1
        ft_h = self.f_left_h(t_h)  # ->f2

        o_h = self.v_origin_h(h)  # ->v4
        ot_h = self.v_origin_h(t_h)  # ->v2
        op_h = self.v_origin_h(p_h)
        opt_h = self.v_origin_h(pt_h)

        # update h_out for v1,v3

        # update next for h2,h,ht,h4

        # update h_bound for f1,f2
        if self.h_bound_f(f_h) == n_h:
            self.update_face(f_h, h_bound=h)
        if self.h_bound_f(ft_h) == nt_h:
            self.update_face(ft_h, h_bound=t_h)

        ht = self.h_twin_h(h)
        h1 = self.h_next_h(h)
        h2 = self.h_prev_h(h)
        h3 = self.h_next_h(ht)
        h4 = self.h_prev_h(ht)

        f1 = self.f_left_h(h)
        f2 = self.f_left_h(ht)

        # v1 = self.v_origin_h(h4)
        # v2 = self.v_origin_h(h1)
        # v3 = self.v_origin_h(h2)
        # v4 = self.v_origin_h(h3)
        v1 = self.v_origin_h(ht)
        v2 = self.v_origin_h(h2)
        v3 = self.v_origin_h(h)
        v4 = self.v_origin_h(h4)

        # self.faces[f1] = np.array([v2, v3, v4], dtype=np.int32)
        # self.faces[f2] = np.array([v4, v1, v2], dtype=np.int32)
        self.update_face(f1, h_bound=h2)
        self.update_face(f2, h_bound=h4)

        # self.halfedges[h] = np.array([v4, v2], dtype=np.int32)
        # self.halfedges[ht] = np.array([v2, v4], dtype=np.int32)

        # self.H_next[h] = h2
        # self.H_vertex[h] = v2
        self.update_hedge(h, h_next=h2, v_origin=v4)

        # self.H_next[ht] = h4
        # self.H_vertex[ht] = v4
        self.update_hedge(ht, h_next=h4, v_origin=v2)

        # self.H_next[h1] = ht
        # self.H_face[h1] = f2
        self.update_hedge(h1, h_next=ht, f_left=f2)

        # self.H_next[h2] = h3
        self.update_hedge(h2, h_next=h3)

        # self.H_face[h3] = f1
        # self.H_next[h3] = h
        self.update_hedge(h3, f_left=f1, h_next=h)

        # self.H_next[h4] = h1
        self.update_hedge(h4, h_next=h1)

        # self.V_hedge[v3] = h3
        # self.V_hedge[v1] = h1
        # self.V_hedge[v2] = h2
        # self.V_hedge[v4] = h4
        self.update_vertex(v3, h_out=h3)
        self.update_vertex(v1, h_out=h1)
        self.update_vertex(v2, h_out=h2)
        self.update_vertex(v4, h_out=h4)

    @property
    def v_new(self):
        return max(self._xyz_coord_V.keys()) + 1

    @property
    def h_new(self):
        return max(self._v_origin_H.keys()) + 1

    @property
    def f_new(self):
        return max(self._h_bound_F.keys()) + 1

    def add_vertex(self, xyz=None, h_out=None):
        v_new = self.v_new
        self._xyz_coord_V[v_new] = xyz
        self._h_out_V[v_new] = h_out
        return v_new

    def update_vertex(self, v, xyz=None, h_out=None):
        if xyz is not None:
            self._xyz_coord_V[v] = xyz
        if h_out is not None:
            self._h_out_V[v] = h_out

    def add_face(self, h_bound=None):
        f_new = self.f_new
        self._h_bound_F[f_new] = h_bound
        return f_new

    def update_face(self, f, h_bound=None):
        if h_bound is not None:
            self._h_bound_F[f] = h_bound

    def add_hedge(self, h_next=None, h_twin=None, v_origin=None, f_left=None):
        h_new = self.h_new
        self._v_origin_H[h_new] = v_origin
        self._h_next_H[h_new] = h_next
        self._h_twin_H[h_new] = h_twin
        self._f_left_H[h_new] = f_left
        return h_new

    def update_hedge(self, h, h_next=None, h_twin=None, v_origin=None, f_left=None):
        if h_next is not None:
            self._h_next_H[h] = h_next
        if h_twin is not None:
            self._h_twin_H[h] = h_twin
        if v_origin is not None:
            self._v_origin_H[h] = v_origin
        if f_left is not None:
            self._f_left_H[h] = f_left

    def split_edge(self, h):
        # choose hij = h or twin(h) so that its face is a thing
        if not self.complement_boundary_contains_h(h):
            hij = h
        else:
            hij = self.h_twin_h(h)
        hji = self.h_twin_h(hij)
        vi = self.v_origin_h(hij)
        vj = self.v_origin_h(hji)
        fijk = self.f_left_h(hij)
        hjk = self.h_next_h(hij)
        hki = self.h_next_h(hjk)
        vk = self.v_origin_h(hki)

        ##############################
        # add new stuff
        vm = self.add_vertex()
        him = hij  # re-use hij label
        hmj = self.add_hedge()
        hmk = self.add_hedge()
        hkm = self.add_hedge()
        fimk = fijk  # re-use fijk label
        fmjk = self.add_face()
        # maybe bdry
        hmi = self.add_hedge()
        hjm = hji  # re-use hji label

        self.update_vertex(
            vm, xyz=0.5 * (self.xyz_coord_v(vi) + self.xyz_coord_v(vj)), h_out=hmj
        )

        self.update_hedge(him, h_next=hmk, h_twin=hmi, v_origin=vi, f_left=fimk)
        self.update_hedge(hmj, h_next=hjk, h_twin=hjm, v_origin=vm, f_left=fmjk)
        self.update_hedge(hmk, h_next=hki, h_twin=hkm, v_origin=vm, f_left=fimk)
        self.update_hedge(hkm, h_next=hmj, h_twin=hmk, v_origin=vk, f_left=fmjk)

        self.update_face(fimk, h_bound=him)
        self.update_face(fmjk, h_bound=hmj)

        ##############################
        if not self.complement_boundary_contains_h(hji):
            filj = self.f_left_h(hji)
            hil = self.h_next_h(hji)
            hlj = self.h_next_h(hil)
            vl = self.v_origin_h(hlj)
            # new stuff
            hlm = self.add_hedge()
            hml = self.add_hedge()
            film = self.add_face()
            fmlj = filj  # re-use filj label
            ############################

            self.update_hedge(hmi, h_next=hil, h_twin=him, v_origin=vm, f_left=film)
            self.update_hedge(hjm, h_next=hml, h_twin=hmj, v_origin=vj, f_left=fmlj)
            self.update_hedge(hlm, h_next=hmi, h_twin=hlm, v_origin=vl, f_left=film)
            self.update_hedge(hml, h_next=hlj, h_twin=hlm, v_origin=vm, f_left=fmlj)

            self.update_face(film, h_bound=hmi)
            self.update_face(fmlj, h_bound=hjm)

        else:
            f_bdry = self.f_left_h(hji)
            hmi_next = self.h_next_h(hji)
            hjm_next = hmi
            hjm_prev = self.h_prev_h(hji)
            self.update_hedge(
                hmi, h_next=hmi_next, h_twin=him, v_origin=vm, f_left=f_bdry
            )
            self.update_hedge(
                hjm, h_next=hjm_next, h_twin=hmj, v_origin=vj, f_left=f_bdry
            )

            self.update_hedge(hjm_prev, h_next=hjm)

    ######################################################
    ######################################################
    # to be deprecated


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
        h_cw_B=None,
        V_bdry=None,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        if h_cw_B is None:
            self.h_cw_B = self.find_h_cw_B()
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
        # self.h_cw_B = self.find_h_cw_B()

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

    def boundary_contains_v(self, v):
        """check if vertex v is on the boundary of the mesh"""
        return self.boundary_contains_h(self.supermesh.h_out_v(v))

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
        for bdry, h in self.h_cw_B.items():
            for h in self.generate_H_next_h(h):
                yield h

    def generate_V_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.v_origin_h(h)

    def generate_F_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.f_left_h(self.h_twin_h(h))

    def find_h_cw_B(self, F_need2check=None):
        h_cw_B = dict()
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
            h_cw_B[bdry] = h
            for h in self.generate_H_next_h(h):
                H_in_cw_boundary.discard(h)
        return h_cw_B

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
        self.h_cw_B = self.find_h_cw_B(F_need2check=F)
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
        self.h_cw_B = self.find_h_cw_B(F_need2check=F)
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
        self.h_cw_B = self.find_h_cw_B(F_need2check=F)
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
        self.csr_data, self.csr_indices, self.csr_indptr = self.compute_csr_matrix()
        self.matrix = csr_matrix(
            (self.csr_data, self.csr_indices, self.csr_indptr),
            shape=(len(self.V), len(self.V)),
        )

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
