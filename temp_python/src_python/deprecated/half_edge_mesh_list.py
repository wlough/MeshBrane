from temp_python.src_python.ply_tools import (
    VertTri2HalfEdgeConverter_no_bdry_twin as VertTri2HalfEdgeConverter,
)


class HalfEdgeMeshList:
    """
    Deprecated... See MeshBrane/src/python/half_edge_mesh.py

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
