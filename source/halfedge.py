from source.ply_utils import HalfEdgeMeshData
import numpy as np


class HalfEdgeMesh:
    def __init__(self, halfEdgeMeshData):
        """E_twin[e]=-1 if half-edge e is on the boundary (i.e. it has no twin half-edge)"""
        self.V = np.copy(halfEdgeMeshData.V)
        self.V_edge = np.copy(halfEdgeMeshData.V_edge)
        self.E_vertex = np.copy(halfEdgeMeshData.E_vertex)
        self.E_face = np.copy(halfEdgeMeshData.E_face)
        self.E_next = np.copy(halfEdgeMeshData.E_next)
        self.E_twin = np.copy(halfEdgeMeshData.E_twin)
        self.F_edge = np.copy(halfEdgeMeshData.F_edge)

    def next(self, e):
        return self.E_next[e]

    def prev(self, e):
        e_next = e
        while True:
            e_prev = e_next
            e_next = self.E_next[e_prev]
            if e_next == e:
                break
        return e_prev

    def twin(self, e):
        return self.E_twin[e]

    def v_of_e(self, e):
        return self.E_vertex[e]

    def f_of_e(self, e):
        return self.E_face[e]

    def e_of_v(self, v):
        return self.V_edge[v]

    def e_of_f(self, f):
        return self.F_edge[f]

    def get_E_outer(self, E_inner):
        E_outer = []
        e_inner_start = E_inner[0]
        e_inner = e_inner_start
        while True:
            v_inner = self.v_of_e(e_inner)
            e_outer = self.next(self.twin(self.prev(self.twin(e_inner))))
            E_outer.append(e_outer)
            while True:
                e_outer = self.next(self.twin(self.next(e_outer)))
                v = self.v_of_e(self.next(e_outer))  # assumes tri faces
                if v == v_inner:
                    E_outer.append(e_outer)
                else:
                    e_inner = self.twin(self.next(e_outer))
                    break

            if e_inner == e_inner_start:
                break

        return E_outer


#


#


#
