import numpy as np
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_base_mesh import HalfEdgeMeshBase

# from src.python.half_edge_base_utils import make_half_edge_base_numba_utils
# make_half_edge_base_numba_utils()
# from src.python.half_edge_base_utils import (
#     find_h_right_B,
#     vf_samples_to_he_samples,
#     he_samples_to_vf_samples,
# )
# from src.python.half_edge_base_ply_tools import (
#     VertexTriMeshSchema,
#     HalfEdgeMeshSchema,
#     MeshConverter,
#     VertTri2HalfEdgeMeshConverter,
# )


vf_ply = "./data/half_edge_base/ply/torus_003072_vf.ply"
he_ply = "./data/half_edge_base/ply/torus_003072_he.ply"
he_ply = "./data/half_edge_base/ply/dumbbell_he.ply"

# %%
m = HalfEdgeMesh.from_half_edge_ply(he_ply)
mb = HalfEdgeMeshBase.from_half_edge_ply(he_ply)
mv = MeshViewer(*mb.data_arrays)
mv.plot()
