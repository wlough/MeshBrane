from plyfile import PlyData, PlyElement
import numpy as np
import glob
import os


def generate_half_edge_mesh_ply(vf_ply_dir, he_ply_dir, use_binary=True):
    """Generates a half edge mesh ply in he_ply_dir for each vertex-face list
    ply in vf_ply_dir"""
    vf_ply_files = glob.glob(vf_ply_dir + "/*.ply")
    print("Generating half edge mesh data for:")
    for vf_ply in vf_ply_files:
        print(os.path.basename(vf_ply))
    print("-----------------------------------")
    Nply = len(vf_ply_files)
    for _, vf_ply in enumerate(vf_ply_files):
        n_ply = _ + 1
        ply_name = os.path.basename(vf_ply)
        he_ply = f"{he_ply_dir}/{ply_name}"
        print(f"{ply_name} ({n_ply}/{Nply})")

        mesh = HalfEdgeMeshData.from_vertex_face_ply(vf_ply)
        mesh.to_ply(he_ply, use_binary=use_binary)
    print("-----------------------------------")
    print("done")


class TriMeshData:
    def __init__(self, V, F):
        self.V = V
        self.F = F

    @classmethod
    def from_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
        F = np.vstack(plydata["face"]["vertex_indices"])
        if not isinstance(V[0], np.float64):
            V = V.astype(np.float64)
        if not isinstance(F[0], np.uint32):
            F = F.astype(np.uint32)
        return cls(V, F)

    def to_ply(self, ply_path, use_binary=False):
        V_data = np.array([tuple(v) for v in self.V], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
        F_data = np.empty(len(self.F), dtype=[("vertex_indices", "i4", (3,))])
        F_data["vertex_indices"] = self.F
        vertex_element = PlyElement.describe(V_data, "vertex")
        face_element = PlyElement.describe(F_data, "face")
        PlyData([vertex_element, face_element], text=not use_binary).write(ply_path)


class HalfEdgeMeshData:
    """E_twin[e]=-1 if half-edge e is on the boundary (i.e. it has no twin half-edge)"""

    def __init__(self, V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge):
        self.V = np.copy(V)
        self.V_edge = np.copy(V_edge)
        self.E_vertex = np.copy(E_vertex)
        self.E_face = np.copy(E_face)
        self.E_next = np.copy(E_next)
        self.E_twin = np.copy(E_twin)
        self.F_edge = np.copy(F_edge)

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

        V_edge = -1 * np.ones(Nvertices, dtype=np.uint32)

        E = np.zeros((Nedges, 2), dtype=np.uint32)
        E_vertex = np.zeros(Nedges, dtype=np.uint32)
        E_face = np.zeros(Nedges, dtype=np.uint32)
        E_next = np.zeros(Nedges, dtype=np.uint32)
        E_twin = -2 * np.ones(Nedges, dtype=np.int32)  # -2

        F_edge = np.zeros(Nfaces, dtype=np.uint32)

        for f in range(Nfaces):
            F_edge[f] = 3 * f
            for i in range(3):
                e = 3 * f + i
                e_next = 3 * f + (i + 1) % 3
                v0 = F[f][i]
                v1 = F[f][(i + 1) % 3]
                E[e] = [v0, v1]
                E_vertex[e] = v1
                E_face[e] = f
                E_next[e] = e_next
                # F_edge[f] = e
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
    def from_tri_mesh_data(cls, vf_data):
        V, F = vf_data.V, vf_data.F
        return cls.from_vert_face_list(V, F)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
        F = np.vstack(plydata["face"]["vertex_indices"])
        if not isinstance(V[0], np.float64):
            V = V.astype(np.float64)
        if not isinstance(F[0], np.uint32):
            F = F.astype(np.uint32)
        return cls.from_vert_face_list(V, F)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
        V_edge = plydata["vertex"]["e"]
        F_edge = plydata["face"]["e"]
        E_vertex = plydata["edge"]["v"]
        E_face = plydata["edge"]["f"]
        E_next = plydata["edge"]["n"]
        E_twin = plydata["edge"]["t"]  # np.vstack(plydata["edge"]["t"])

        if not isinstance(V[0], np.float64):
            V = V.astype(np.float64)
        if not isinstance(V_edge[0], np.uint32):
            V_edge = V_edge.astype(np.uint32)

        if not isinstance(F_edge[0], np.uint32):
            F_edge = F_edge.astype(np.uint32)

        if not isinstance(E_vertex[0], np.uint32):
            E_vertex = E_vertex.astype(np.uint32)
        if not isinstance(E_face[0], np.uint32):
            E_face = E_face.astype(np.uint32)
        if not isinstance(E_next[0], np.uint32):
            E_next = E_next.astype(np.uint32)
        if not isinstance(E_twin[0], np.int32):
            E_twin = E_twin.astype(np.int32)

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    def to_ply(self, ply_path, use_binary=False):
        V_data = np.array(
            [(v[0], v[1], v[2], e) for v, e in zip(self.V, self.V_edge)],
            dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("e", "uint32")],
        )
        F_data = np.array(self.F_edge, dtype=[("e", "uint32")])
        E_data = np.array(
            [(v, f, n, t) for v, f, n, t in zip(self.E_vertex, self.E_face, self.E_next, self.E_twin)],
            dtype=[("v", "uint32"), ("f", "uint32"), ("n", "uint32"), ("t", "i4")],
        )
        V_element = PlyElement.describe(V_data, "vertex")
        E_element = PlyElement.describe(E_data, "edge")
        F_element = PlyElement.describe(F_data, "face")
        PlyData([V_element, E_element, F_element], text=not use_binary).write(ply_path)

    def to_vertex_face_ply(self, ply_path, use_binary=False):
        V = self.V
        Nfaces = len(self.F_edge)
        F = np.zeros((Nfaces, 3), dtype=np.uint32)
        # Nedges = len(self.E_vertex)
        for f in range(Nfaces):
            e = self.F_edge[f]
            F[f, 0] = self.E_vertex[e]
            e = self.E_next[e]
            F[f, 1] = self.E_vertex[e]
            e = self.E_next[e]
            F[f, 2] = self.E_vertex[e]

        vf_mesh = TriMeshData(V, F)
        vf_mesh.to_ply(ply_path, use_binary=use_binary)


class HalfEdgeMeshBuilder:
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
        face indices of face to the left of each half-edge
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

        V_edge = -1 * np.ones(Nvertices, dtype=np.uint32)

        E = np.zeros((Nedges, 2), dtype=np.uint32)
        E_vertex = np.zeros(Nedges, dtype=np.uint32)
        E_face = np.zeros(Nedges, dtype=np.uint32)
        E_next = np.zeros(Nedges, dtype=np.uint32)
        E_twin = -2 * np.ones(Nedges, dtype=np.int32)  # -2

        F_edge = np.zeros(Nfaces, dtype=np.uint32)

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
        V = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
        F = np.vstack(plydata["face"]["vertex_indices"])
        if not isinstance(V[0], np.float64):
            V = V.astype(np.float64)
        if not isinstance(F[0], np.uint32):
            F = F.astype(np.uint32)
        return cls.from_vert_face_list(V, F)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        V = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
        V_edge = plydata["vertex"]["e"]
        F_edge = plydata["face"]["e"]
        E_vertex = plydata["edge"]["v"]
        E_face = plydata["edge"]["f"]
        E_next = plydata["edge"]["n"]
        E_twin = plydata["edge"]["t"]  # np.vstack(plydata["edge"]["t"])

        if not isinstance(V[0], np.float64):
            V = V.astype(np.float64)
        if not isinstance(V_edge[0], np.uint32):
            V_edge = V_edge.astype(np.uint32)

        if not isinstance(F_edge[0], np.uint32):
            F_edge = F_edge.astype(np.uint32)

        if not isinstance(E_vertex[0], np.uint32):
            E_vertex = E_vertex.astype(np.uint32)
        if not isinstance(E_face[0], np.uint32):
            E_face = E_face.astype(np.uint32)
        if not isinstance(E_next[0], np.uint32):
            E_next = E_next.astype(np.uint32)
        if not isinstance(E_twin[0], np.int32):
            E_twin = E_twin.astype(np.int32)

        return cls(V, V_edge, E_vertex, E_face, E_next, E_twin, F_edge)

    def to_ply(self, ply_path, use_binary=False):
        V_data = np.array(
            [(v[0], v[1], v[2], e) for v, e in zip(self.V, self.V_edge)],
            dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("e", "uint32")],
        )
        F_data = np.array(self.F_edge, dtype=[("e", "uint32")])
        E_data = np.array(
            [(o, f, n, t) for o, f, n, t in zip(self.E_vertex, self.E_face, self.E_next, self.E_twin)],
            dtype=[("v", "uint32"), ("f", "uint32"), ("n", "uint32"), ("t", "i4")],
        )
        V_element = PlyElement.describe(V_data, "vertex")
        E_element = PlyElement.describe(E_data, "edge")
        F_element = PlyElement.describe(F_data, "face")
        PlyData([V_element, E_element, F_element], text=not use_binary).write(ply_path)

    def to_vertex_face_ply(self, ply_path, use_binary=False):
        V = self.V
        Nfaces = len(self.F_edge)
        F = np.zeros((Nfaces, 3), dtype=np.uint32)
        # Nedges = len(self.E_vertex)
        for f in range(Nfaces):
            e = self.F_edge[f]
            F[f, 0] = self.E_vertex[e]
            e = self.E_next[e]
            F[f, 1] = self.E_vertex[e]
            e = self.E_next[e]
            F[f, 2] = self.E_vertex[e]

        vf_mesh = TriMeshData(V, F)
        vf_mesh.to_ply(ply_path, use_binary=use_binary)
