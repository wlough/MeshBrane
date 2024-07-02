from src.python.half_edge_mesh import HalfEdgeMesh, HalfEdgePatch, CotanLaplaceOperator, CotanLaplacian
from src.python.mesh_viewer import MeshViewer
import numpy as np

image_dir = "./output/debug"
source_ply = "./data/ply/binary/unit_sphere_00162.ply"
# source_ply = "./data/ply/binary/annulus.ply"
# source_ply = "./data/ply/binary/dumbbell_ultracoarse.ply"
# source_ply = "./data/ply/binary/torus.ply"
# source_ply = "./data/ply/binary/neovius.ply"
# source_ply = "./data/ply/ascii/pyramid3.ply"

viewer_kwargs = {
    "image_dir": image_dir,
    "show_vertices": True,
    # "v_radius": 0.03,
}

m = HalfEdgeMesh.from_half_edge_ply(source_ply)
# m = HalfEdgeMesh.from_vertex_face_ply(source_ply)
val = [m.valence_v(v) for v in m.V]
# for h in m.H:
#     if m.h_is_flippable(h) and not m.h_is_locally_delaunay(h):
#         print(f"flipping {h=}")
#         m.flip_edge(h)
# m.num_boundaries
# m.split_edge(3)
mv = MeshViewer(*m.data_lists, **viewer_kwargs)

mv.plot()
# %%
mv = MeshViewer(*m.data_lists, **viewer_kwargs)
# A = np.array([m.meyercell_area(v) for v in m.V])

mv.plot()
L0 = CotanLaplaceOperator(m)
L1 = CotanLaplacian(m, compute=True)
dir(L1.matrix)
L0.matrix.data
L1.matrix.data
(L0.matrix - L1.matrix).toarray()
np.linalg.norm((L0.matrix - L1.matrix).toarray().ravel())
# %%


def divide_faces(self):
    F = []
    V = [np.array(xyz) for xyz in self.V]
    v_midpt_vv = dict()
    for tri in self.F:
        v0, v1, v2 = tri
        v01 = v_midpt_vv.get((v0, v1))
        v12 = v_midpt_vv.get((v1, v2))
        v20 = v_midpt_vv.get((v2, v0))
        if v01 is None:
            v01 = len(V)
            xyz01 = (V[v0] + V[v1]) / 2
            xyz01 *= self.r / np.linalg.norm(xyz01)
            V.append(xyz01)
            v_midpt_vv[(v0, v1)] = v01
            v_midpt_vv[(v1, v0)] = v01
        if v12 is None:
            v12 = len(V)
            xyz12 = (V[v1] + V[v2]) / 2
            xyz12 *= self.r / np.linalg.norm(xyz12)
            V.append(xyz12)
            v_midpt_vv[(v1, v2)] = v12
            v_midpt_vv[(v2, v1)] = v12
        if v20 is None:
            v20 = len(V)
            xyz20 = (V[v2] + V[v0]) / 2
            xyz20 *= self.r / np.linalg.norm(xyz20)
            V.append(xyz20)
            v_midpt_vv[(v2, v0)] = v20
            v_midpt_vv[(v0, v2)] = v20
        F.append([v0, v01, v20])
        F.append([v01, v1, v12])
        F.append([v20, v12, v2])
        F.append([v01, v12, v20])
