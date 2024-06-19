from plyfile import PlyData, PlyElement
from src.python.ply_tools import VertTri2HalfEdgeConverter
import numpy as np
from src.python.combinatorics import arg_right_action
from src.python.key_manager import KeyManager


class HalfEdgeMeshBase:
    """
    Dict-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.arrays of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as dicts of vertex/half-edge/face labels. Each data dict has a name of the form "a_description_B", where "a" denotes the type of object associated with the dict elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "B" denotes the type of object associated with the dict indices ("V" for vertex, "H" for half-edge, or "F" for face), and "description" is a description of information represented by the dict. For example, "_v_origin_H" is a dict of vertices at the origin of each half-edge. The i-th element of data dict "a_description_B" can be accessed using the "a_description_b(i)" method.

    Attributes
    ----------
    _xyz_coord_V : dict of numpy.array
        _xyz_coord_V[i] = xyz coordinates of vertex i
    _h_out_V : dict of int
        _h_out_V[i] = some outgoing half-edge incident on vertex i
    _v_origin_H : dict of int
        _v_origin_H[j] = vertex at the origin of half-edge j
    _h_next_H : dict of int
        _h_next_H[j] next half-edge after half-edge j in the face cycle
    _h_twin_H : dict of int
        _h_twin_H[j] = half-edge antiparalel to half-edge j
        _h_twin_H[j] = -1 if half-edge j is on a boundary of the mesh
    _f_left_H : dict of int
        _f_left_H[j] = face to the left of half-edge j
    _h_bound_F : dict of int
        _h_bound_F[k] = some half-edge on the boudary of face k

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
    ):
        self._xyz_coord_V = xyz_coord_V
        self._h_out_V = h_out_V
        self._v_origin_H = v_origin_H
        self._h_next_H = h_next_H
        self._h_twin_H = h_twin_H
        self._h_bound_F = h_bound_F
        self._f_left_H = f_left_H

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
        v2h = VertTri2HalfEdgeConverter.from_source_samples(xyz_coord_V, vvv_of_F)
        data_dicts = [
            {key: val for key, val in enumerate(data)} for data in v2h.target_samples
        ]
        return cls(*data_dicts)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        v2h = VertTri2HalfEdgeConverter.from_source_ply(ply_path)
        data_dicts = [
            {key: val for key, val in enumerate(data)} for data in v2h.target_samples
        ]
        return cls(*data_dicts)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        v2h = VertTri2HalfEdgeConverter.from_target_ply(ply_path)
        data_dicts = [
            {key: val for key, val in enumerate(data)} for data in v2h.target_samples
        ]
        return cls(*data_dicts)

    #######################################################
    @property
    def xyz_coord_V(self):
        return self._xyz_coord_V

    @property
    def h_out_V(self):
        return self._h_out_V

    @property
    def v_origin_H(self):
        return self._v_origin_H

    @property
    def h_next_H(self):
        return self._h_next_H

    @property
    def h_twin_H(self):
        return self._h_twin_H

    @property
    def f_left_H(self):
        return self._f_left_H

    @property
    def h_bound_F(self):
        return self._h_bound_F

    @property
    def num_vertices(self):
        return len(self._xyz_coord_V)

    @property
    def num_half_edges(self):
        return len(self._v_origin_H)

    @property
    def num_faces(self):
        return len(self._h_bound_F)

    @property
    def data_lists(self):
        """get lists of data required for basic getters.vertex positions and vertex/half-edge/face indices mesh connectivity data and required for basic getters

        Returns:
            _type_: _description_
        """
        Vkeys = self._xyz_coord_V.keys()
        min_v = min(Vkeys)
        max_v = max(Vkeys)
        V = sorted(Vkeys)
        Hkeys = self._v_origin_H.keys()
        min_h = min(Hkeys)
        max_h = max(Hkeys)
        H = sorted(Hkeys)
        Fkeys = self._h_bound_F.keys()
        min_f = min(Fkeys)
        max_f = max(Fkeys)
        F = sorted(Fkeys)

        xyz_coord_V = [self.xyz_coord_v(v) for v in V]
        h_out_V = [H.index(self.h_out_v(v)) for v in V]
        v_origin_H = [V.index(self.v_origin_h(h)) for h in H]
        h_next_H = [H.index(self.h_next_h(h)) for h in H]
        h_twin_H = [
            -1 if self.h_on_boundary(h) else H.index(self.h_twin_h(h)) for h in H
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
    def h_prev_h(self, h):
        """works for non-triangle meshes"""
        h_next = self.h_next_h(h)
        while h_next != h:
            h_prev = h_next
            h_next = self.h_next_h(h_prev)
        return h_prev

    def h_prev_h_tri(self, h):
        """only works for triangle meshes"""
        return self.h_twin_h(self.h_next_h(self.h_next_h(h)))

    def h_on_boundary(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.h_twin_h(h) == -1

    ######################################################
    # generators
    def generate_H_out_v_clockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            if self.h_on_boundary(h):
                break
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def generate_H_out_v_counterclockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            if self.h_on_boundary(self.h_prev_h(h)):
                break
            h = self.h_twin_h(self.h_prev_h(h))
            if h == h_start:
                break

    def generate_H_out_cw_from_h(self, h_start):
        """
        Starting with h_start, generate outgoing half-edges from origin of h_start in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            if self.h_on_boundary(h):
                break
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def generate_H_out_ccw_from_h(self, h_start):
        """
        Starting with h_start, generate outgoing half-edges from origin of h_start in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            if self.h_on_boundary(self.h_prev_h(h)):
                break
            h = self.h_twin_h(self.h_prev_h(h))
            if h == h_start:
                break

    def generate_H_in_cw_from_h(self, h_start):
        """
        Starting with h_start, generate incoming half-edges toward origin(twin(h_start)) in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if self.h_on_boundary(h):
                break
            h = self.h_twin_h(h)
            if h == h_start:
                break

    def generate_H_in_ccw_from_h(self, h_start):
        """
        Starting with h_start, generate incoming half-edges toward origin(twin(h_start)) in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = h_start
        while True:
            yield h
            if self.h_on_boundary(h):
                break
            h = self.h_prev_h(self.h_twin_h(h))
            if h == h_start:
                break

    def H_out_v_counterclockwise_safe(self, v):
        """Returns a list of all outgoing half-edges from vertex v in counterclockwise order. If the vertex is on a boundary, the list starts with a the outgoing boundary half-edge."""
        ccw = self.generate_H_out_v_counterclockwise(v)
        need_cw = False
        Hccw = []
        for h in ccw:
            Hccw.append(h)
            if self.h_on_boundary(self.h_prev_h(h)):
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


class HalfEdgeMeshList:
    """
    List-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.arrays of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as lists of vertex/half-edge/face labels. Each data list has a name of the form "a_description_B", where "a" denotes the type of object associated with the list elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "B" denotes the type of object associated with the list indices ("V" for vertex, "H" for half-edge, or "F" for face), and "description" is a description of information represented by the list. For example, "_v_origin_H" is a list of vertices at the origin of each half-edge. The i-th element of data list "a_description_B" can be accessed using the "a_description_b(i)" method.

    Attributes
    ----------
    _xyz_coord_V : list of numpy.array
        _xyz_coord_V[i] = xyz coordinates of vertex i
    _h_out_V : list of int
        _h_out_V[i] = some outgoing half-edge incident on vertex i
    _v_origin_H : list of int
        _v_origin_H[j] = vertex at the origin of half-edge j
    _h_next_H : list of int
        _h_next_H[j] next half-edge after half-edge j in the face cycle
    _h_twin_H : list of int
        _h_twin_H[j] = half-edge antiparalel to half-edge j
        _h_twin_H[j] = -1 if half-edge j is on a boundary of the mesh
    _f_left_H : list of int
        _f_left_H[j] = face to the left of half-edge j
    _h_bound_F : list of int
        _h_bound_F[k] = some half-edge on the boudary of face k

    Initialization
    ---------------
    The HalfEdgeMesh class can be initialized in several ways:
    - Directly from half-edge mesh data lists:
        HalfEdgeMesh(xyz_coord_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_bound_F)
    - From a list of vertex positions and a list of face vertices:
        HalfEdgeMesh.from_vert_face_list(xyz_coord_V, vvv_of_F)
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
    ):
        self._xyz_coord_V = xyz_coord_V
        self._h_out_V = h_out_V
        self._v_origin_H = v_origin_H
        self._h_next_H = h_next_H
        self._h_twin_H = h_twin_H
        self._h_bound_F = h_bound_F
        self._f_left_H = f_left_H

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
        v2h = VertTri2HalfEdgeConverter.from_source_samples(xyz_coord_V, vvv_of_F)
        return cls(*v2h.target_samples)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        v2h = VertTri2HalfEdgeConverter.from_source_ply(ply_path)
        return cls(*v2h.target_samples)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        v2h = VertTri2HalfEdgeConverter.from_target_ply(ply_path)
        return cls(*v2h.target_samples)

    #######################################################
    @property
    def xyz_coord_V(self):
        return self._xyz_coord_V

    @property
    def h_out_V(self):
        return self._h_out_V

    @property
    def v_origin_H(self):
        return self._v_origin_H

    @property
    def h_next_H(self):
        return self._h_next_H

    @property
    def h_twin_H(self):
        return self._h_twin_H

    @property
    def f_left_H(self):
        return self._f_left_H

    @property
    def h_bound_F(self):
        return self._h_bound_F

    @property
    def num_vertices(self):
        return len(self._xyz_coord_V)

    @property
    def num_half_edges(self):
        return len(self._v_origin_H)

    @property
    def num_faces(self):
        return len(self._h_bound_F)

    @property
    def data_lists(self):
        """get lists of data required for basic getters.vertex positions and vertex/half-edge/face indices mesh connectivity data and required for basic getters

        Returns:
            _type_: _description_
        """
        return (
            self._xyz_coord_V,
            self._h_out_V,
            self._v_origin_H,
            self._h_next_H,
            self._h_twin_H,
            self._f_left_H,
            self._h_bound_F,
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
    def h_prev_h(self, h):
        """works for non-triangle meshes"""
        h_next = self.h_next_h(h)
        while h_next != h:
            h_prev = h_next
            h_next = self.h_next_h(h_prev)
        return h_prev

    def h_prev_h_tri(self, h):
        """only works for triangle meshes"""
        return self.h_twin_h(self.h_next_h(self.h_next_h(h)))

    def h_on_boundary(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.h_twin_h(h) == -1

    ######################################################
    # generators
    def generate_H_out_v_clockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in clockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            if self.h_on_boundary(h):
                break
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def generate_H_out_v_counterclockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in counterclockwise order until one of the following conditions is met:
        1) a boundary is reached
        2) the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            if self.h_on_boundary(self.h_prev_h(h)):
                break
            h = self.h_twin_h(self.h_prev_h(h))
            if h == h_start:
                break

    def H_out_v_counterclockwise_safe(self, v):
        """Returns a list of all outgoing half-edges from vertex v in counterclockwise order. If the vertex is on a boundary, the list starts with a the outgoing boundary half-edge."""
        ccw = self.generate_H_out_v_counterclockwise(v)
        need_cw = False
        Hccw = []
        for h in ccw:
            Hccw.append(h)
            if self.h_on_boundary(self.h_prev_h(h)):
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

    ######################################################
    # safe methods

    def poly_sector_data_sublists(self, v):
        """builds sets of vertices, edges, and faces around vertex v"""
        V = [v]
        h_out_V = [self.h_out_v(v)]
        H = []
        v_origin_H = []
        h_next_H,
        h_twin_H = []
        f_left_H = []
        h_bound_F = []
        F = []

        h_start = self.h_out_v(v)
        h = h_start
        while True:
            # h is v-->bdry
            H.append(h)
            o = v  # already have v
            n = self.h_next_h(h)
            t = self.h_twin_h(h)
            f = self.f_left_h(h)
            v_origin_H.append(o)
            h_next_H.append(n)
            h_twin_H.append(t)
            f_left_H.append(f)
            F.append(f)  # new face
            # next h is bdry<--bdry
            h = n
            H.append(h)
            o = self.v_origin_h(h)  # new vertex
            V.append(o)
            n = self.h_next_h(h)
            t = -1  # twin is not in the sector
            f = self.f_left_h(h)  # already have f
            v_origin_H.append(o)
            h_next_H.append(n)
            h_twin_H.append(t)
            f_left_H.append(f)
            # next next h is v<--bdry
            h = n
            H.append(h)
            o = self.v_origin_h(h)  # new vertex
            V.append(o)
            n = self.h_next_h(h)
            t = self.h_twin_h(h)  # twin may not be -1
            f = self.f_left_h(h)  # already have f
            v_origin_H.append(o)
            h_next_H.append(n)
            h_twin_H.append(t)
            f_left_H.append(f)
            #
            if self.h_on_boundary(h):
                break
            h = ht
            if h == h_start:
                break

        return V, H, F

    def poly_sector_VHF(self, v):
        """builds sets of vertices, edges, and faces around vertex v"""
        V = [v]
        H = []
        F = []
        for h in self.H_out_v_counterclockwise_safe(v):
            # h is v-->bdry
            n = self.h_next_h(h)
            v1 = self.v_origin_h(n)
            nn = self.h_next_h(n)
            v2 = self.v_origin_h(nn)
            f = self.f_left_h(h)
            V.extend([v1, v2])
            H.extend([h, n, nn])
            F.append(f)  # new face
        return V, H, F

    ######################################################

    def generate_twin_pairs_around_v_clockwise(self, v):
        e = self.h_out_v(v)
        e_start = e
        while True:
            e_twin = self.h_twin_h(e)
            yield (e, e_twin)
            e = self.h_next_h(e_twin)
            if e == e_start:
                break

    ######################################################

    def barcell_area(self, v):
        """area of cell dual to vertex v"""
        r = self.xyz_coord_v(v)
        A = 0.0
        # h = self.h_out_v(v)
        # h_start = h
        for h in self.generate_H_out_v_clockwise(v):
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

    def laplacian_interact(self, vi, vj):
        xi = self.xyz_coord_v(vi)
        xj = self.xyz_coord_v(vj)
        Ai = self.barcell_area(vi)
        Aj = self.barcell_area(vj)
        dist_xi_xj = np.linalg.norm(xj - xi)
        Lij = Aj * np.exp(-(dist_xi_xj**2) / (4 * Ai)) / (4 * np.pi * Ai**2)
        return Lij

    def laplacian_propogate_to_tol(
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
    def Cl(self, V, E, F):
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

    def St_of_v(self, v):
        V = {v}
        E = set()
        F = set()
        E_Et_v = self.generate_twin_pairs_around_v_clockwise(v)
        for e_et in E_Et_v:
            # V.add(self.v_origin_h(e_et[1]))
            E.update(e_et)
            F.add(self.f_left_h(e_et[0]))
        return V, E, F

    def St_of_e(self, e):
        """face of a simplex is any subset of its vertices. the star of a simplex is defined as the set of all simplices that have the simplex as a face. the set of all simplices that contain its vertices as a subset of their vertices"""
        et = self.h_twin_h(e)
        V = set()
        E = {e, et}
        F = {self.f_left_h(e), self.f_left_h(et)}
        return V, E, F

    def St(self, V_in, E_in, F_in):
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

    def Lk(self, V, E, F):
        clV, clE, clF = self.Cl(V, E, F)
        stclV, stclE, stclF = self.St(clV, clE, clF)
        stV, stE, stF = self.St(V, E, F)
        clstV, clstE, clstF = self.Cl(stV, stE, stF)
        return clstV - stclV, clstE - stclE, clstF - stclF

    def one_ring_vhf_sets_with_bdry(self, v_center):
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

    def expand_boundary_safe(self, vertices, edges, faces, boundary_edges):
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

    def expand_frontier(self, vertices, edges, faces, frontier_edges):
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


######################################
######################################
class HalfEdgePatch:
    """
    A submanifold of a HalfEdgeMesh topologically equivalent to a disk.
    """

    def __init__(
        self,
        V,
        H,
        F,
        H_boundary_cycle,
        supermesh,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        self.H_boundary_cycle = H_boundary_cycle

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
        get index of an outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        H_out_v = self.supermesh.generate_H_out_v_counterclockwise(v)
        for h in H_out_v:
            if h in self.H:
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
        return self.supermesh.h_next_h(h)

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

    def h_on_boundary(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        t = self.supermesh.h_twin_h(h)
        return t == -1 or t not in self.H

    def find_boundary_cycle(self, feasible_H_bdry_set=set()):
        H_boundary_cycle = []
        if len(feasible_H_bdry_set) == 0:  # find a boundary half-edge
            feasible_H_bdry_set.update(self.H)
        for h_start in feasible_H_bdry_set:
            if self.h_on_boundary(h_start):
                break
        h = h_start
        while self.h_on_boundary(h):
            H_boundary_cycle.append(h)
            H_in_cw = self.supermesh.generate_H_in_cw_from_h(h)
            for h in H_in_cw:  # cycle through half-edges pointing the boundary vertex
                h = self.h_next_h(h)
                if self.h_on_boundary(h):
                    break
            if h == h_start:
                break
        return H_boundary_cycle

    @classmethod
    def from_seed_vertex(cls, v, supermesh):
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
            if supermesh.h_on_boundary(
                h
            ):  # if h is on boundary of supermesh it's also on boundary of patch
                H_boundary_cycle.append(h)
            H_boundary_cycle.append(n)  # always on patch boundary

        if supermesh.h_on_boundary(
            nn
        ):  # if nn is on boundary of supermesh it's also on boundary of patch
            H_boundary_cycle.append(nn)

        return cls(V, H, F, H_boundary_cycle, supermesh)

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
            i: -1 if self.h_on_boundary(h) else H.index(self.h_twin_h(h))
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
            -1 if self.h_on_boundary(h) else H.index(self.h_twin_h(h)) for h in H
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
            if self.supermesh.h_on_boundary(h):
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


######################################
######################################


class HalfEdgeSubMesh(HalfEdgeMesh):
    """
    A sub-mesh of a HalfEdgeMesh.
    """

    def __init__(
        self,
        V,
        H,
        F,
        supermesh,
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
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F

    @classmethod
    def from_seed_vertex(cls, v, supermesh):
        V = [v]
        H = []
        F = []
        super_h_out_V = [supermesh.h_out_v(v)]
        for h in supermesh.H_out_v_counterclockwise_safe(v):
            # h is v-->bdry
            n = supermesh.h_next_h(h)
            v1 = supermesh.v_origin_h(n)
            nn = supermesh.h_next_h(n)
            v2 = supermesh.v_origin_h(nn)
            f = supermesh.f_left_h(h)
            V.extend([v1, v2])
            H.extend([h, n, nn])
            super_h_out_V.extend([supermesh.h_out_v(v1), supermesh.h_out_v(v2)])
            F.append(f)  # new face

        xyz_coord_V = [supermesh.xyz_coord_v(v) for v in V]
        h_out_V = [H.index(h) for h in super_h_out_V]
        v_origin_H = []  # [V.index(supermesh.v_origin_H(h)) for h in H]
        h_next_H = []  # [H.index(supermesh.h_next_H(h)) for h in H]
        h_twin_H = []  # [H.index(supermesh.h_twin_H(h)) for h in H]
        f_left_H = []  # [F.index(supermesh.f_left_H(h)) for h in H]
        for h in H:
            v_origin_H.append(V.index(supermesh.v_origin_h(h)))
            h_next_H.append(H.index(supermesh.h_next_h(h)))
            ht = supermesh.h_twin_h(h)
            if ht in H:
                h_twin_H.append(H.index(ht))
            else:
                h_twin_H.append(-1)
            f_left_H.append(F.index(supermesh.f_left_h(h)))

        h_bound_F = [H.index(supermesh.h_bound_f(f)) for f in F]
        return cls(
            V,
            H,
            F,
            supermesh,
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

    @classmethod
    def from_VHF_lists(cls, V, H, F, supermesh):
        """fix capitalization"""
        xyz_coord_V = np.zeros((len(V), 3))
        h_out_V = [H.index(supermesh.h_out_v(v)) for v in V]
        v_origin_H = []  # [V.index(supermesh.v_origin_H(h)) for h in H]
        h_next_H = []  # [H.index(supermesh.h_next_H(h)) for h in H]
        h_twin_H = []  # [H.index(supermesh.h_twin_H(h)) for h in H]
        f_left_H = []  # [F.index(supermesh.f_left_H(h)) for h in H]
        for h in H:
            v_origin_H.append(V.index(supermesh.v_origin_H(h)))
            h_next_H.append(H.index(supermesh.h_next_H(h)))
            ht = supermesh.h_twin_H(h)
            if ht in H:
                h_twin_H.append(H.index(ht))
            else:
                h_twin_H.append(-1)
            f_left_H.append(F.index(supermesh.f_left_H(h)))

        h_bound_F = [H.index(supermesh.h_bound_F(f)) for f in F]
        return cls(
            V,
            H,
            F,
            supermesh,
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

    @classmethod
    def from_supermesh_sublists(
        cls,
        V,
        H,
        F,
        supermesh,
        super_xyz_coord_V,
        super_h_out_V,
        super_v_origin_H,
        super_h_next_H,
        super_h_twin_H,
        super_f_left_H,
        super_h_bound_F,
    ):
        xyz_coord_V = np.zeros((len(V), 3))
        h_out_V = [H.index(h) for h in super_h_out_V]
        v_origin_H = [
            V.index(v) for v in super_v_origin_H
        ]  # [V.index(supermesh.v_origin_H(h)) for h in H]
        h_next_H = [
            H.index(h) for h in super_h_next_H
        ]  # [H.index(supermesh.h_next_H(h)) for h in H]
        h_twin_H = []
        # [H.index(h) for h in super_h_twin_H]  # [H.index(supermesh.h_twin_H(h)) for h in H]
        for h in super_h_twin_H:
            if h == -1:
                h_twin_H.append(-1)
            else:
                h_twin_H.append(H.index(h))
        f_left_H = [F.index(f) for f in super_f_left_H]
        h_bound_F = [H.index(h) for h in super_h_bound_F]
        return cls(
            V,
            H,
            F,
            supermesh,
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )
