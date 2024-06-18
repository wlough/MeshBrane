from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer

# source_path = "./data/ply/binary/sphere.ply"
# m = HalfEdgeMesh.from_vertex_face_ply(source_path)
source_paths = [
    f"./data/ply/binary/sphere{_}.ply" for _ in ["_ultracoarse", "_coarse", "", "_fine", "_ultrafine"]
]
m = [HalfEdgeMesh.from_half_edge_ply(source_path) for source_path in source_paths[:-1]]
mv = [MeshViewer(*_.data_lists) for _ in m]
