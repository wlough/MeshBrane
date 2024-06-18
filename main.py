from src.python.half_edge_mesh import HalfEdgeMesh, HalfEdgeSubMesh
from src.python.mesh_viewer import MeshViewer
import numpy as np

# source_path = "./data/ply/ascii/hex_sector.ply"
# m = HalfEdgeMesh.from_vertex_face_ply(source_path)
source_path = "./data/ply/binary/dumbbell.ply"
m = HalfEdgeMesh.from_half_edge_ply(source_path)
sm = HalfEdgeSubMesh.from_seed_vertex(0, m)
mv = MeshViewer(*m.data_lists, show_vertices=True)
smv = MeshViewer(*sm.data_lists, show_vertices=False)
#
# V,H,F=from_seed_vertex(m, 2, m)
#
#
# mv.set_subset_F_rgba(rgba=mv.colors["blue"], indices=F)
# mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=H)
# mv.set_subset_V_rgba(rgba=mv.colors["blue"], indices=V)
# mv.plot()
smv.plot()
