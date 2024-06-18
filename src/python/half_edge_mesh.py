from plyfile import PlyData, PlyElement
from src.python.ply_tools import VertTri2HalfEdgeConverter

# import numpy as np
# import glob
# import os


class HalfEdgeMesh:
    """
    List-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.arrays of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as lists of vertex/half-edge/face labels. Each data list has a name of the form "_a_description_B", where "a" denotes the type of object associated with the list elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "B" denotes the type of object associated with the list indices ("V" for vertex, "H" for half-edge, or "F" for face), and "description" is a description of information represented by the list. For example, "_v_origin_H" is a list of vertices at the origin of each half-edge. Each data list has a corresponding getter "a_description_b()" that returns elements of the list.

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
        HalfEdgeMesh(xyz_coordinates_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_bound_F)
    - From a list of vertex positions and a list of face vertices:
        HalfEdgeMesh.from_vert_face_list(xyz_coordinates_V, vvv_of_F)
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
        xyz_coordinates_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    ):
        self._xyz_coord_V = xyz_coordinates_V
        self._h_out_V = h_out_V
        self._v_origin_H = v_origin_H
        self._h_next_H = h_next_H
        self._h_twin_H = h_twin_H
        self._h_bound_F = h_bound_F
        self._f_left_H = f_left_H

    @classmethod
    def from_vert_face_list(cls, xyz_coordinates_V, vvv_of_F):
        """
        Initialize a half-edge mesh from vertex/face data.

        Parameters:
        ----------
        xyz_coordinates_V : list of numpy.array
            xyz_coordinates_V[i] = xyz coordinates of vertex i
        vvv_of_F : list of lists of integers
            vvv_of_F[j] = [v0, v1, v2] = vertices in face j.

        Returns:
        -------
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with the given vertices and faces.
        """
        v2h = VertTri2HalfEdgeConverter.from_source_samples(xyz_coordinates_V, vvv_of_F)
        return cls(*v2h.target_samples)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        v2h = VertTri2HalfEdgeConverter.from_ply_file(ply_path)
        return cls(*v2h.target_samples)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        mb = HalfEdgeMeshBuilder.from_half_edge_ply(ply_path)
        return cls(
            mb.V, mb.V_edge, mb.E_vertex, mb.E_face, mb.E_next, mb.E_twin, mb.F_edge
        )

    #######################################################
    # getters
    def data_lists(self):
        """get lists of data required for basic getters.vertex positions and vertex/half-edge/face indices mesh connectivity data and required for basic getters

        Returns:
            _type_: _description_
        """
        return (
            self._xyz_coord_V,
            self._h_out_V,
            self._v_origin_H,
            self._f_left_H,
            self._h_next_H,
            self._h_twin_H,
            self._h_bound_F,
        )

    def xyz_coord_V(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self._xyz_coord_V[v]

    def h_out_V(self, v):
        """
        get index of an outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        return self._h_out_V[v]

    def h_bound_F(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        return self._h_bound_F[f]

    def v_origin_H(self, e):
        """get index of the vertex at the origin of half-edge e

        Args:
            e (int): half-edge index

        Returns:
            int: vertex index
        """
        return self._v_origin_H[e]

    def f_left_H(self, e):
        """get index of the face to the left of half-edge e

        Args:
            e (int): half-edge index

        Returns:
            int: face index
        """
        return self._f_left_H[e]

    def h_next_H(self, e):
        """get index of the next half-edge in the face cycle

        Args:
            e (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self._h_next_H[e]

    def h_twin_H(self, e):
        """get index of the half-edge anti-parallel to half-edge e

        Args:
            e (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self._h_twin_H[e]

    ######################################################
    # basic helpers
    def h_prev_H_safe(self, e):
        """works for non-triangle meshes"""
        e_next = self.h_next_H(e)
        while e_next != e:
            e_prev = e_next
            e_next = self.h_next_H(e_prev)
        return e_prev

    def h_prev_H(self, e):
        """only works for triangle meshes"""
        return self.h_twin_H(self.h_next_H(self.h_next_H(e)))

    def v_opposite_H(self, h):
        """vertex opposite to half-edge h in face"""
        return self.v_origin_H(self.h_next_H(h))

    def f_inc_V(self, v):
        """face incident on vertex v"""
        return self.f_left_H(self.h_out_V(v))

    ######################################################
    # generators
    def generate_H_out_v_clockwise(self, v):
        e = self.h_out_V(v)
        e_start = e
        while True:
            yield e
            e = self.h_next_H(self.h_twin_H(e))
            if e == e_start:
                break

    def generate_H_out_v_anticlockwise(self, v):
        e = self.h_out_V(v)
        e_start = e
        while True:
            yield e
            e = self.h_twin_H(self.h_pre_H(e))
            if e == e_start:
                break

    def generate_twin_pairs_around_v_clockwise(self, v):
        e = self.h_out_V(v)
        e_start = e
        while True:
            e_twin = self.h_twin_H(e)
            yield (e, e_twin)
            e = self.h_next_H(e_twin)
            if e == e_start:
                break

    def generate_V_around_f_anticlockwise(self, f):
        e = self.h_bound_F(f)
        e_start = e
        while True:
            yield self.v_origin_H(e)
            e = self.h_next_H(e)
            if e == e_start:
                break

    def generate_V_around_f_anticlockwise(self, f):
        e = self.h_bound_F(f)
        e_start = e
        while True:
            yield self.v_origin_H(e)
            e = self.h_next_H(e)
            if e == e_start:
                break

    ######################################################

    ######################################################
    def Cl(self, V, E, F):
        for f in F:
            e0 = self.h_left_F(f)
            e1 = self.h_next_H(e0)
            e2 = self.h_next_H(e1)
            E.update({e0, e1, e2})
        E_twins = set()
        for e in E:
            E_twins.add(self.h_twin_H(e))
        E.update(E_twins)
        for e in E:
            v0 = self.v_origin_H(e)
            e_twin = self.h_twin_H(e)
            v1 = self.v_origin_H(e_twin)
            V.update({v0, v1})
        return V, E, F

    def St_of_v(self, v):
        V = {v}
        E = set()
        F = set()
        E_Et_v = self.generate_twin_pairs_around_v_clockwise(v)
        for e_et in E_Et_v:
            # V.add(self.v_origin_H(e_et[1]))
            E.update(e_et)
            F.add(self.f_left_H(e_et[0]))
        return V, E, F

    def St_of_e(self, e):
        """face of a simplex is any subset of its vertices. the star of a simplex is defined as the set of all simplices that have the simplex as a face. the set of all simplices that contain its vertices as a subset of their vertices"""
        et = self.h_twin_H(e)
        V = set()
        E = {e, et}
        F = {self.f_left_H(e), self.f_left_H(et)}
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
        e = self.h_out_V(v_center)
        e_start = e
        while True:
            ########################################
            # e starts as an outgoing half-edge of v_center
            f = self.f_left_H(e)
            edges.add(e)
            faces.add(f)
            ########################################
            # next edge is on the boundary
            e = self.h_next_H(e)
            v = self.v_origin_H(e)
            vertices.add(v)
            edges.add(e)
            boundary_edges.add(e)
            ########################################
            # next edge is in to v_center
            e = self.h_next_H(e)
            edges.add(e)
            ########################################
            # twin edge is out of v_center
            e = self.h_twin_H(e)
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
            e0 = self.h_twin_H(e)
            e1 = self.h_next_H(e0)
            e2 = self.h_next_H(e1)
            f = self.f_left_H(e0)
            faces.add(f)
            edges.update({e0, e1, e2})
            # add possible new boundary edges
            maybe_boundary_edges.update({e1, e2})
            # possibly a new vertex so add to vertices
            v = self.v_origin_H(e)
            vertices.add(v)
        # remove edges that are not on the boundary
        boundary_edges = {
            e for e in maybe_boundary_edges if self.h_twin_H(e) not in edges
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
            # e0 = self.h_twin_H(e)
            e1 = self.h_next_H(e0)
            e2 = self.h_next_H(e1)
            f = self.f_left_H(e0)
            faces.add(f)
            edges.update({e0, e1, e2})
            # add twins of possible new boundary edges to frontier
            new_frontier_edges.update({e1, e2})
            # possibly a new vertex so add to vertices
            v = self.v_origin_H(e)
            vertices.add(v)
        # remove edges that are not on the boundary
        boundary_edges = {
            e for e in maybe_boundary_edges if self.h_twin_H(e) not in edges
        }

        return {
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "boundary_edges": boundary_edges,
        }


class HalfEdgeMeshBuilder:
    """Constructs data to initialize a half-edge mesh from a vertex-face list

    parameters
    ----------
    V: list
        numpy arrays containing xyz coordinates of each vertex
    V_edge: list
        half-edge indices of a half-edge incident on each vertex
    E_vertex: list
        vertex indices for the origin of each half-edge
    E_face: list
        face indices of face to the left of each half-edge
    E_next: list
        half-edge indices for next half-edge
    E_twin: list
        half-edge indices for the twin of each half-edge
    F_edge: list
        half-edge indices of a half-edge on the boudary of each face
    """

    def __init__(self, V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge):
        self.V = [np.copy(xyz) for xyz in V]
        self.V_edge = V_edge.copy()
        self.E_vertex = E_vertex.copy()
        self.E_face = E_face.copy()
        self.E_next = E_next.copy()
        self.E_twin = E_twin.copy()
        self.F_edge = F_edge.copy()

    @classmethod
    def get_index_of_twin(self, E, e):
        Nedges = len(E)
        v0 = E[e][0]
        v1 = E[e][1]
        for e_twin in range(Nedges):
            if E[e_twin][0] == v1 and E[e_twin][1] == v0:
                return e_twin

        return -1

    @classmethod
    def from_vert_face_list(cls, V, F):
        Nfaces = len(F)
        Nvertices = len(V)
        Nedges = 3 * Nfaces

        E = Nedges * [[0, 0]]

        V_edge = Nvertices * [-1]
        E_vertex = Nedges * [0]
        E_face = Nedges * [0]
        E_next = Nedges * [0]
        E_twin = Nedges * [-2]
        F_edge = Nfaces * [0]

        for f in range(Nfaces):
            F_edge[f] = 3 * f
            for i in range(3):
                e = 3 * f + i
                e_next = 3 * f + (i + 1) % 3
                v0 = F[f][i]
                v1 = F[f][(i + 1) % 3]
                E[e] = [v0, v1]
                E_vertex[e] = v0
                E_face[e] = f
                E_next[e] = e_next
                if V_edge[v0] == -1:
                    V_edge[v0] = e

        for e in range(Nedges):
            if E_twin[e] == -2:
                e_twin = cls.get_index_of_twin(E, e)
                E_twin[e] = e_twin
                if e_twin != -1:
                    E_twin[e_twin] = e

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]
            )
        ]
        F = [verts.tolist() for verts in plydata["face"]["vertex_indices"]]
        return cls.from_vert_face_list(V, F)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = [
            np.array([x, y, z])
            for x, y, z in zip(
                plydata["he_vertex"]["x"],
                plydata["he_vertex"]["y"],
                plydata["he_vertex"]["z"],
            )
        ]
        V_edge = plydata["he_vertex"]["e"].tolist()
        F_edge = plydata["he_face"]["e"].tolist()
        E_vertex = plydata["he_edge"]["v"].tolist()
        E_face = plydata["he_edge"]["f"].tolist()
        E_next = plydata["he_edge"]["n"].tolist()
        E_twin = plydata["he_edge"]["t"].tolist()

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    def get_faces(self):
        return [
            [
                self.E_vertex[e],
                self.E_vertex[self.E_next[e]],
                self.E_vertex[self.E_next[self.E_next[e]]],
            ]
            for e in self.F_edge
        ]

    def to_half_edge_ply(self, ply_path, use_binary=False):
        V_data = np.array(
            [
                (vertex[0], vertex[1], vertex[2], e)
                for vertex, e in zip(self.V, self.V_edge)
            ],
            dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("e", "uint32")],
        )
        F_data = np.array(self.F_edge, dtype=[("e", "uint32")])
        E_data = np.array(
            [
                (v, f, n, t)
                for v, f, n, t in zip(
                    self.E_vertex, self.E_face, self.E_next, self.E_twin
                )
            ],
            dtype=[("v", "uint32"), ("f", "uint32"), ("n", "uint32"), ("t", "i4")],
        )
        V_element = PlyElement.describe(V_data, "he_vertex")
        E_element = PlyElement.describe(E_data, "he_edge")
        F_element = PlyElement.describe(F_data, "he_face")
        PlyData([V_element, E_element, F_element], text=not use_binary).write(ply_path)

    def to_vertex_face_ply(self, ply_path, use_binary=False):
        F = self.get_faces()
        V_data = np.array(
            [tuple(v) for v in self.V], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")]
        )
        F_data = np.empty(len(F), dtype=[("vertex_indices", "i4", (3,))])
        F_data["vertex_indices"] = F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        PlyData([vertex_element, face_element], text=not use_binary).write(ply_path)
