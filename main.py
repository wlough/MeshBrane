from src.python.half_edge_mesh import *
from src.python.ply_tools import VertTri2HalfEdgeConverter
from plyfile import PlyData

source_ply = "./data/ply/ascii/dumbbell.ply"
target_ply = "./data/ply/binary/dumbell.ply"
v2h = VertTri2HalfEdgeConverter.from_ply_file(source_ply)
v2h.write_target_ply(target_path=target_ply, use_ascii=True)

pd = PlyData.read(source_ply)
pd.text = False
