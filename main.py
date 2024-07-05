from src.python.ply_tools import VertTri2HalfEdgeConverter, SphereBuilder, DoughnutBuilder
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_ops import CotanLaplaceOperator
from src.python.rigid_body import exp_so3

ply = "./data/ply/ascii/torus.ply"
m = HalfEdgeMesh.from_vertex_face_ply(ply)
# sb = SphereBuilder()
#
# sb._name = "test_sphere"
# sb.write_plys()
# sb.divide_faces()
import numpy as np


# V,F=VF_torus(6)
db = DoughnutBuilder()
for iter in range(2):
    db.refine()
db.convert_to_half_edge()
db.write_plys()
# %%
db.num_vertices(-1)
max_level = len(db.pow) - 1
M = [HalfEdgeMesh.from_vert_face_list(*db.VF(level=level)) for level in range(len(db.pow))]
[m.num_vertices for m in M]
# %%
level = 2
m = HalfEdgeMesh.from_vert_face_list(*db.VF(level=level))
mv = MeshViewer(*m.data_lists)
mv.plot()
