from src.python.ply_tools import VertTri2HalfEdgeConverter
import numpy as np


class HalfEdgeMeshOperator:
    def __init__(self, mesh):
        self.mesh = mesh

    def _generate_twin_pairs_around_v_clockwise(self, v):
        e = self.mesh.h_out_v(v)
        e_start = e
        while True:
            e_twin = self.mesh.h_twin_h(e)
            yield (e, e_twin)
            e = self.mesh.h_next_h(e_twin)
            if e == e_start:
                break

    def _Cl(self, V, E, F):
        for f in F:
            e0 = self.mesh.h_left_F(f)
            e1 = self.mesh.h_next_h(e0)
            e2 = self.mesh.h_next_h(e1)
            E.update({e0, e1, e2})
        E_twins = set()
        for e in E:
            E_twins.add(self.mesh.h_twin_h(e))
        E.update(E_twins)
        for e in E:
            v0 = self.mesh.v_origin_h(e)
            e_twin = self.mesh.h_twin_h(e)
            v1 = self.mesh.v_origin_h(e_twin)
            V.update({v0, v1})
        return V, E, F

    def _St_of_v(self, v):
        V = {v}
        E = set()
        F = set()
        E_Et_v = self.generate_twin_pairs_around_v_clockwise(v)
        for e_et in E_Et_v:
            # V.add(self.v_origin_h(e_et[1]))
            E.update(e_et)
            F.add(self.mesh.f_left_h(e_et[0]))
        return V, E, F

    def _St_of_e(self, e):
        """face of a simplex is any subset of its vertices. the star of a simplex is defined as the set of all simplices that have the simplex as a face. the set of all simplices that contain its vertices as a subset of their vertices"""
        et = self.mesh.h_twin_h(e)
        V = set()
        E = {e, et}
        F = {self.mesh.f_left_h(e), self.mesh.f_left_h(et)}
        return V, E, F

    def _St(self, V_in, E_in, F_in):
        F = F_in
        E = set()
        V = set()
        for e in E_in:
            pass


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
    def h_out_cw_from_h(self, h):
        return self.h_next_h(self.h_twin_h(h))

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

    def _h_adjacent_to_boundary(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.f_left_h(self.h_twin_h(h)) < 0

    def boundary_is_left_of_h(self, h):
        """check if half-edge h is in the boundary of the mesh"""
        return self.f_left_h(h) < 0

    def boundary_is_right_of_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.f_left_h(self.h_twin_h(h)) < 0

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
    # geometric computations
    def barcell_area(self, v):
        """area of the barycentric cell dual to vertex v"""
        r = self.xyz_coord_v(v)
        A = 0.0
        for h in self.generate_H_out_v_clockwise(v):
            if self.boundary_is_left_of_h(h):
                continue
            r1 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(h)))
            r2 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(self.h_next_h(h))))
            A_face_vec = (
                np.cross(r, r1) / 2 + np.cross(r1, r2) / 2 + np.cross(r2, r) / 2
            )
            A_face = np.sqrt(
                A_face_vec[0] ** 2 + A_face_vec[1] ** 2 + A_face_vec[2] ** 2
            )
            A += A_face / 3

        return A

    def find_h_cw_B(self):
        """Find all boundary edges in the mesh"""
        h_cw_B = dict()
        boundary_is_left_of_H = set()
        # boundary_is_right_of_H = set()
        F_need2check = set(self.h_bound_F)  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            for h in self.generate_H_bound_f(f):
                if self.boundary_is_right_of_h(h):
                    boundary_is_left_of_H.add(self.h_twin_h(h))
        cw_boundary_H_cycles = []
        ccw_boundary_H_cycles = []
        while boundary_is_left_of_H:
            h = boundary_is_left_of_H.pop()
            bdry = self.f_left_h(h)
            h_cw_B[bdry] = h
            for h in self.generate_H_next_h(h):
                boundary_is_left_of_H.discard(h)
        return h_cw_B

    ######################################################
    # star of a k-simplex s consists of:
    # 1) s
    # 2) all (n>k)-simplices that contain s

    def star_of_vertex(self, v):
        """Star of a vertex is the set of all simplices that contain the vertex."""
        V = {v}
        H = set()
        F = set()
        for h in self.generate_H_out_v_clockwise(v):
            ht = self.h_twin_h(h)
            H.update([h, ht])
            if not self.boundary_is_left_of_h(h):
                F.add(self.f_left_h(h))

        return V, H, F

    def star_of_edge(self, h):
        """Star of an edge is the set of all simplices that contain the edge."""
        V = set()
        H = {h, self.h_twin_h(h)}
        F = set()
        for hi in H:
            if not self.boundary_is_left_of_h(hi):
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
            if not self.boundary_is_left_of_h(h):
                F.add(self.f_left_h(h))
            if not self.boundary_is_left_of_h(ht):
                F.add(self.f_left_h(ht))
        for v in V_in:
            for h in self.generate_H_out_v_clockwise(v):
                H.add(h)
                H.add(self.h_twin_h(h))
                if not self.boundary_is_left_of_h(h):
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

    ######################################################
    ######################################################
    # old generators to be deprecated

    def _generate_H_out_v_clockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            if self.h_adjacent_to_boundary(h):
                break
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def _generate_H_out_v_counterclockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            if self.h_adjacent_to_boundary(self.h_prev_h(h)):
                break
            h = self.h_twin_h(self.h_prev_h(h))
            if h == h_start:
                break

    def _generate_H_out_cw_from_h(self, h_start):
        """
        Starting with h_start, generate outgoing half-edges from origin of h_start in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            if self.h_adjacent_to_boundary(h):
                break
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def _generate_H_out_ccw_from_h(self, h_start):
        """
        Starting with h_start, generate outgoing half-edges from origin of h_start in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            if self.h_adjacent_to_boundary(self.h_prev_h(h)):
                break
            h = self.h_twin_h(self.h_prev_h(h))
            if h == h_start:
                break

    def _generate_H_in_cw_from_h(self, h_start):
        """
        Starting with h_start, generate incoming half-edges toward origin(twin(h_start)) in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if self.h_adjacent_to_boundary(h):
                break
            h = self.h_twin_h(h)
            if h == h_start:
                break

    def _generate_H_in_ccw_from_h(self, h_start):
        """
        Starting with h_start, generate incoming half-edges toward origin(twin(h_start)) in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            if self.h_adjacent_to_boundary(h):
                break
            h = self.h_prev_h(self.h_twin_h(h))
            if h == h_start:
                break

    def _H_out_v_counterclockwise_safe(self, v):
        """Returns a list of all outgoing half-edges from vertex v in counterclockwise order. If the vertex is on a boundary, the list starts with a the outgoing boundary half-edge."""
        ccw = self.generate_H_out_v_counterclockwise(v)
        need_cw = False
        Hccw = []
        for h in ccw:
            Hccw.append(h)
            if self.h_adjacent_to_boundary(self.h_prev_h(h)):
                cw = True
                break
        if need_cw:
            cw = self.generate_H_out_v_clockwise(v)
            Hcw = []
            for h in cw:
                Hcw.append(h)
            return [*Hcw[::-1], *Hccw[1:]]
        else:
            return Hccw

    ######################################################


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

    def find_boundary_cycles(self):
        """Find all boundary edges in the mesh"""
        boundary_is_left_of_H = set()
        # boundary_is_right_of_H = set()
        F_need2check = set(self.h_bound_F)  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            for h in self.generate_H_bound_f(f):
                if self.boundary_is_right_of_h(h):
                    boundary_is_left_of_H.add(self.h_twin_h(h))
        cw_boundary_H_cycles = []
        ccw_boundary_H_cycles = []
        while boundary_is_left_of_H:
            h = boundary_is_left_of_H.pop()
            cw_cycle = list(self.generate_H_next_h(h))
            cw_boundary_H_cycles.append(cw_cycle)
            ccw_cycle = []
            for h in cw_cycle:
                ccw_cycle.append(self.h_twin_h(h))
                boundary_is_left_of_H.discard(h)
            ccw_boundary_H_cycles.append(ccw_cycle[::-1])
        return cw_boundary_H_cycles, ccw_boundary_H_cycles

    ######################################################
    def _generate_twin_pairs_around_v_clockwise(self, v):
        e = self.h_out_v(v)
        e_start = e
        while True:
            e_twin = self.h_twin_h(e)
            yield (e, e_twin)
            e = self.h_next_h(e_twin)
            if e == e_start:
                break

    def _Cl(self, V, E, F):
        for f in F:
            e0 = self.h_left_F(f)
            e1 = self.h_next_h(e0)
            e2 = self.h_next_h(e1)
            E.update({e0, e1, e2})
        E_twins = set()
        for e in E:
            E_twins.add(self.h_twin_h(e))
        E.update(E_twins)
        for e in E:
            v0 = self.v_origin_h(e)
            e_twin = self.h_twin_h(e)
            v1 = self.v_origin_h(e_twin)
            V.update({v0, v1})
        return V, E, F

    def _St_of_v(self, v):
        V = {v}
        E = set()
        F = set()
        E_Et_v = self.generate_twin_pairs_around_v_clockwise(v)
        for e_et in E_Et_v:
            # V.add(self.v_origin_h(e_et[1]))
            E.update(e_et)
            F.add(self.f_left_h(e_et[0]))
        return V, E, F

    def _St_of_e(self, e):
        """face of a simplex is any subset of its vertices. the star of a simplex is defined as the set of all simplices that have the simplex as a face. the set of all simplices that contain its vertices as a subset of their vertices"""
        et = self.h_twin_h(e)
        V = set()
        E = {e, et}
        F = {self.f_left_h(e), self.f_left_h(et)}
        return V, E, F

    def _St(self, V_in, E_in, F_in):
        F = F_in
        E = set()
        V = set()
        for e in E_in:
            V_e, E_e, F_e = self.St_of_e(e)
            # V.update(V_e)
            E.update(E_e)
            F.update(F_e)
        for v in V_in:
            V_v, E_v, F_v = self.St_of_v(v)
            V.update(V_v)
            E.update(E_v)
            F.update(F_v)
        return V, E, F

    def _Lk(self, V, E, F):
        clV, clE, clF = self.Cl(V, E, F)
        stclV, stclE, stclF = self.St(clV, clE, clF)
        stV, stE, stF = self.St(V, E, F)
        clstV, clstE, clstF = self.Cl(stV, stE, stF)
        return clstV - stclV, clstE - stclE, clstF - stclF

    def _one_ring_vhf_sets_with_bdry(self, v_center):
        # gets the vertex/edge/face labels one-ring mesh of an interior vertex
        boundary_edges = set()
        vertices = set()
        edges = set()
        faces = set()
        vertices.add(v_center)
        e = self.h_out_v(v_center)
        e_start = e
        while True:
            ########################################
            # e starts as an outgoing half-edge of v_center
            f = self.f_left_h(e)
            edges.add(e)
            faces.add(f)
            ########################################
            # next edge is on the boundary
            e = self.h_next_h(e)
            v = self.v_origin_h(e)
            vertices.add(v)
            edges.add(e)
            boundary_edges.add(e)
            ########################################
            # next edge is in to v_center
            e = self.h_next_h(e)
            edges.add(e)
            ########################################
            # twin edge is out of v_center
            e = self.h_twin_h(e)
            if e == e_start:
                break
        return {
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "boundary_edges": boundary_edges,
        }

    def _expand_boundary_safe(self, vertices, edges, faces, boundary_edges):
        maybe_boundary_edges = set()

        while boundary_edges:
            e = boundary_edges.pop()
            # get new face and its edges
            e0 = self.h_twin_h(e)
            e1 = self.h_next_h(e0)
            e2 = self.h_next_h(e1)
            f = self.f_left_h(e0)
            faces.add(f)
            edges.update({e0, e1, e2})
            # add possible new boundary edges
            maybe_boundary_edges.update({e1, e2})
            # possibly a new vertex so add to vertices
            v = self.v_origin_h(e)
            vertices.add(v)
        # remove edges that are not on the boundary
        boundary_edges = {
            e for e in maybe_boundary_edges if self.h_twin_h(e) not in edges
        }

        return {
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "boundary_edges": boundary_edges,
        }

    def _expand_frontier(self, vertices, edges, faces, frontier_edges):
        new_frontier_edges = set()

        while frontier_edges:
            e0 = boundary_edges.pop()
            # get new face and its edges
            # e0 = self.h_twin_h(e)
            e1 = self.h_next_h(e0)
            e2 = self.h_next_h(e1)
            f = self.f_left_h(e0)
            faces.add(f)
            edges.update({e0, e1, e2})
            # add twins of possible new boundary edges to frontier
            new_frontier_edges.update({e1, e2})
            # possibly a new vertex so add to vertices
            v = self.v_origin_h(e)
            vertices.add(v)
        # remove edges that are not on the boundary
        boundary_edges = {
            e for e in maybe_boundary_edges if self.h_twin_h(e) not in edges
        }

        return {
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "boundary_edges": boundary_edges,
        }

    ######################################################

    def _laplacian_interact(self, vi, vj):
        xi = self.xyz_coord_v(vi)
        xj = self.xyz_coord_v(vj)
        Ai = self.barcell_area(vi)
        Aj = self.barcell_area(vj)
        dist_xi_xj = np.linalg.norm(xj - xi)
        Lij = Aj * np.exp(-(dist_xi_xj**2) / (4 * Ai)) / (4 * np.pi * Ai**2)
        return Lij

    def _laplacian_propogate_to_tol(
        self, vi, Y, tol_rel=1e-6, tol_abs=1e-6, max_iter=100
    ):
        Yi = Y[vi]
        LYi = 0
        _rel = 1
        _abs = 1
        labels = self.one_ring_vhf_sets_with_bdry(vi)
        for hj in labels["boundary_edges"]:
            vj = self.v_origin_h(hj)
            Yj = Y[vj]
            LYi += self.laplacian_interact(vi, vj) * (Yj - Yi)
        # while _rel > tol_rel and _abs > tol_abs:
        while True:
            labels = self.expand_boundary_safe(**labels)
            LYibdry = 0.0
            for hj in labels["boundary_edges"]:
                vj = self.v_origin_h(hj)
                Yj = Y[vj]
                LYibdry += self.laplacian_interact(vi, vj) * (Yj - Yi)
            LYi += LYibdry
            _rel = abs(LYibdry / LYi)
            _abs = abs(LYibdry)

        return LYi, labels

    ######################################################


class HalfEdgePatch:
    """
    A submanifold of a HalfEdgeMesh topologically equivalent to a disk.
    """

    def __init__(
        self,
        V,
        H,
        F,
        H_boundary_generators,
        supermesh,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        self.H_boundary_generators = H_boundary_generators

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, _V):
        if not isinstance(_V, set):
            if hasattr(_V, "__iter__"):
                self._V = set(_V)
            else:
                raise ValueError("Argument must be set or iterable.")
        self._V = _V

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, _H):
        if not isinstance(_H, set):
            if hasattr(_H, "__iter__"):
                self._H = set(_H)
            else:
                raise ValueError("Argument must be set or iterable.")
        self._H = _H

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, _F):
        if not isinstance(_F, set):
            if hasattr(_F, "__iter__"):
                self._F = set(_F)
            else:
                raise ValueError("Argument must be set or iterable.")
        self._F = _F

    @classmethod
    def from_seed_vertex(cls, v, supermesh):
        V, H, F = supermesh.closure(*supermesh.star_of_vertex(v))
        self = cls(V, H, F, [], supermesh)
        self.H_boundary_generators = self.find_boundary_generators(
            feasible_H_bdry_generators=H.copy()
        )

        return self

    ##########################################################3

    def find_boundary_generators(self, feasible_H_bdry_generators=set()):
        H_boundary_generators = []
        if len(feasible_H_bdry_generators) == 0:
            feasible_H_bdry_generators.update(self.H)
        while feasible_H_bdry_generators:
            h = feasible_H_bdry_generators.pop()
            if self.h_is_on_boundary(h):
                H_boundary_generators.append(h)
                H_next_h = self.generate_H_next_h(h)
                for n in H_next_h:
                    feasible_H_bdry_generators.discard(n)

        return H_boundary_generators

    @classmethod
    def _from_seed_vertex(cls, v, supermesh):
        V = {v}
        H = set()
        F = set()
        H_boundary_cycle = []
        H_out_v_counterclockwise = supermesh.H_out_v_counterclockwise_safe(v)
        for h in H_out_v_counterclockwise:
            # h is v-->bdry
            n = supermesh.h_next_h(h)  # n is bdry-->bdry
            v1 = supermesh.v_origin_h(n)
            nn = supermesh.h_next_h(n)  # nn is bdry-->v
            v2 = supermesh.v_origin_h(nn)
            f = supermesh.f_left_h(h)
            V.update([v1, v2])
            H.update([h, n, nn])
            F.add(f)  # new face
            if supermesh.h_adjacent_to_boundary(
                h
            ):  # if h is on boundary of supermesh it's also on boundary of patch
                H_boundary_cycle.append(h)
            H_boundary_cycle.append(n)  # always on patch boundary

        if supermesh.h_adjacent_to_boundary(
            nn
        ):  # if nn is on boundary of supermesh it's also on boundary of patch
            H_boundary_cycle.append(nn)

        return cls(V, H, F, H_boundary_cycle, supermesh)

    def xyz_coord_v(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self.supermesh.xyz_coord_v(v)

    def h_is_on_boundary(self, h):
        """check if half-edge h is on the boundary of the patch"""
        if h in self.H:
            return self.supermesh.f_left_h(h) not in self.F
        return False

    def h_is_in_interior(self, h):
        """check if half-edge h is on the boundary of the patch"""
        if h in self.H:
            return self.supermesh.f_left_h(h) in self.F
        return False

    def h_out_v(self, v):
        """
        get index of an non-boundary outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        # H_out_v = self.supermesh.generate_H_out_v_clockwise(v)
        for h in self.supermesh.generate_H_out_v_clockwise(v):
            if h in self.H and not self.h_is_on_boundary(h):
                return h

    def h_bound_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        return self.supermesh.h_bound_f(f)

    def v_origin_h(self, h):
        """get index of the vertex at the origin of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self.supermesh.v_origin_h(h)

    def f_left_h(self, h):
        """get index of the face to the left of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: face index
        """
        return self.supermesh.f_left_h(h)

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face cycle

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if self.h_is_in_interior(h):
            return self.supermesh.h_next_h(h)
        n = self.supermesh.h_next_h(h)
        while n not in self.H:
            n = self.supermesh.h_out_cw_from_h(n)
        return n

    def generate_H_next_h(self, h_start):
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def boundary_H_cycles(self):
        bdry_cycles = []
        for h_start in self.H_boundary_generators:
            bdry_cycles.append([])
            for h in self.generate_H_next_h(h_start):
                bdry_cycles[-1].append(h)
        return bdry_cycles

    ##############################################
    def h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        t = self.supermesh.h_twin_h(h)
        if t in self.H:
            return t
        else:
            return -1

    def h_adjacent_to_boundary(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        t = self.supermesh.h_twin_h(h)
        return t == -1 or t not in self.H

    def _find_boundary_cycle(self, feasible_H_bdry_set=set()):
        H_boundary_cycle = []
        if len(feasible_H_bdry_set) == 0:  # find a boundary half-edge
            feasible_H_bdry_set.update(self.H)
        for h_start in feasible_H_bdry_set:
            if self.h_adjacent_to_boundary(h_start):
                break
        h = h_start
        while self.h_adjacent_to_boundary(h):
            H_boundary_cycle.append(h)
            H_in_cw = self.supermesh.generate_H_in_cw_from_h(h)
            for h in H_in_cw:  # cycle through half-edges pointing the boundary vertex
                h = self.h_next_h(h)
                if self.h_adjacent_to_boundary(h):
                    break
            if h == h_start:
                break
        return H_boundary_cycle

    def to_half_edge_mesh(self):
        V = sorted(self.V)
        H = sorted(self.H)
        F = sorted(self.F)
        # [x if x<.5 else 33 for x in X]
        xyz_coord_V = {i: self.xyz_coord_v(v) for i, v in enumerate(V)}
        h_out_V = {i: H.index(self.h_out_v(v)) for i, v in enumerate(V)}
        v_origin_H = {i: V.index(self.v_origin_h(h)) for i, h in enumerate(H)}
        h_next_H = {i: H.index(self.h_next_h(h)) for i, h in enumerate(H)}
        h_twin_H = {
            i: -1 if self.h_adjacent_to_boundary(h) else H.index(self.h_twin_h(h))
            for i, h in enumerate(H)
        }
        f_left_H = {i: F.index(self.f_left_h(h)) for i, h in enumerate(H)}
        h_bound_F = {i: H.index(self.h_bound_f(f)) for i, f in enumerate(F)}

        return HalfEdgeMesh(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

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
        h_twin_H = [
            -1 if self.h_adjacent_to_boundary(h) else H.index(self.h_twin_h(h))
            for h in H
        ]
        f_left_H = [F.index(self.f_left_h(h)) for h in H]
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

    def expand_boundary(self):
        """ """
        feasible_H_bdry_set = set()
        N_old_bdry = len(self.H_boundary_cycle)
        for i in range(N_old_bdry):
            h_start = self.H_boundary_cycle[i]
            h_stop = self.H_boundary_cycle[
                (i + 1) % N_old_bdry
            ]  # the next h in old boundary cycle
            H_in_ccw = self.supermesh.generate_H_in_ccw_from_h(h_start)
            h = next(H_in_ccw)  # already have h_start
            # cycle through remaining half-edges pointing toward the old boundary vertex
            for h in H_in_ccw:
                self.V.add(self.v_origin_h(h))
                self.F.add(self.f_left_h(h))
                n = self.h_next_h(h)
                nn = self.h_next_h(n)
                # self.H.update([h, n, nn])
                feasible_H_bdry_set.update([h, n, nn])
                # if we reach the old boundary stop
                if self.supermesh.h_twin_h(h) == h_stop:
                    break
            # if we reached the boundary of supermesh work cw from next h on old boundary...
            if self.supermesh.h_adjacent_to_boundary(h):
                # v = self.v_origin_h(h_stop)
                H_out_cw = self.supermesh.generate_H_out_cw_from_h(h_stop)
                h = next(H_out_cw)
                for h in H_out_cw:
                    self.V.add(self.v_origin_h(h))
                    self.F.add(self.f_left_h(h))
                    n = self.h_next_h(h)
                    nn = self.h_next_h(n)
                    # self.H.update([h, n, nn])
                    feasible_H_bdry_set.update([h, n, nn])
        self.H.update(feasible_H_bdry_set)
        self.H_boundary_cycle = self.find_boundary_cycle(
            feasible_H_bdry_set=feasible_H_bdry_set
        )
