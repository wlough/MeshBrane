from plyfile import PlyData, PlyElement
from source.ply_utils import HalfEdgeMeshData, TriMeshData
import numpy as np


class HalfEdgeMesh:
    # """E_twin[e]=-1 if half-edge e is on the boundary (i.e. it has no twin half-edge)"""
    """List based half-edge mesh data structure.

    parameters
    ----------
    V: list
        numpy arrays containing xyz coordinates of each vertex
    V_edge: list
        half-edge indices of a half-edge incident on each vertex
    E_vertex: list
        vertex indices for the origin of each half-edge
    E_face: list
        face indices to the left of each half-edge
    E_next: list
        half-edge indices for next half-edge
    E_twin: list
        half-edge indices for the twin of each half-edge
    F_edge: list
        half-edge indices of a half-edge on the boudary of each face

    mesh navigation
    ---------------

    notes
    -----
    -E_twin[e]=-1 if half-edge e is on a boundary (it has no twin half-edge)
    """

    def __init__(self, V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge):
        self.V = V.copy()
        self.V_edge = V_edge.copy()
        self.E_vertex = E_vertex.copy()
        self.E_face = E_face.copy()
        self.E_next = E_next.copy()
        self.E_twin = E_twin.copy()
        self.F_edge = F_edge.copy()

    #######################################################
    def xyz_of_v(self, v):
        return self.V[v]

    def e_of_v(self, v):
        return self.V_edge[v]

    def e_of_f(self, f):
        return self.F_edge[f]

    def origin(self, e):
        return self.E_vertex[e]

    def left(self, e):
        return self.E_face[e]

    def next(self, e):
        return self.E_next[e]

    def twin(self, e):
        return self.E_twin[e]

    ######################################################
    def prev(self, e):
        e_next = self.next(e)
        while e_next != e:
            e_prev = e_next
            e_next = self.next(e_prev)
        return e_prev

    def next_edge_out_of_vertex(self, e):
        return self.twin(self.prev(e))

    def get_order_one_edge_neighbors(self, v):
        E_order_one = []
        e_start = self.next(self.e_of_v(v))
        e_neighbor = e_start
        while True:
            E_order_one.append(e_neighbor)
            e_neighbor = self.next(self.twin(self.next(e_neighbor)))
            if e_neighbor == e_start:
                return E_order_one

    def get_order_n_plus_one_edge_neighbors(self, E_inner):
        N_inner = len(E_inner)
        E_outer = []
        for _e_inner in range(N_inner):
            e_inner = E_inner[_e_inner]
            e_inner_next = E_inner[(_e_inner + 1) % N_inner]
            e_inner2outer = self.twin(self.next(self.next(self.twin(e_inner))))
            e_inner2outer_stop = self.next(self.twin(e_inner_next))
            while e_inner2outer != e_inner2outer_stop:
                e_outer = self.next(e_inner2outer)
                E_outer.append(e_outer)
                e_inner2outer = self.twin(self.next(self.next(e_inner2outer)))
        return E_outer

        #
        # N_inner = len(E_inner)
        # for _e in range(N_inner):
        #     e_inner = E_inner[_e]
        #     e_next_inner = E_inner[(_e + 1) % N_inner]
        #     v_break = self.v_of_e(e_next_inner)
        #     e_outer = self.next(self.twin(self.prev(self.twin(e_inner))))
        #     while True:
        #         E_outer.append(e_outer)
        #         e_outer = self.next(self.twin(self.next(e_outer)))
        #         v = self.v_of_e(e_outer)
        #         if v == v_break:
        #             break
        #
        # return E_outer
