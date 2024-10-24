from temp_python.src_python.mesh.half_edge import HalfEdgeMesh
from temp_python.src_python.mesh.ply_tools import MeshConverter
from temp_python.src_python.mesh.viewer import MeshViewer  # , MultiMeshViewer
import numpy as np

# %%
# test imports
ply_in = "./data/example_ply/vutukuri_vesicle_005120_he.ply"
ply_in = "./data/example_ply/dumbbell_coarse_he.ply"
ply_out = "./output/cglass.dumbbell_coarse_he.ply"
scale_factor = 40
m = HalfEdgeMesh.load(ply_path=ply_in)
m.xyz_coord_V *= scale_factor
mc = MeshConverter.from_he_samples(*m.he_samples)
mv = MeshViewer(m)
mv.plot()


# %%
def scale_ply(ply_in, ply_out, scale_factor):
    m = HalfEdgeMesh.load(ply_path=ply_in)
    m.xyz_coord_V *= scale_factor
    mc = MeshConverter.from_he_samples(*m.he_samples)
    mc.write_he_ply(ply_out)
    # mv = MeshViewer(m)
    # mv.plot()
    #


surfs = [
    "dumbbell_coarse_he",
    "dumbbell_he",
    "dumbbell_fine_he",
    "oblate_001280_he",
    "oblate_005120_he",
    "oblate_081920_he",
    "unit_sphere_001280_he",
    "unit_sphere_005120_he",
    "unit_sphere_081920_he",
    "unit_torus_3_1_raw_001536_he",
    "unit_torus_3_1_raw_006144_he",
    "unit_torus_3_1_raw_098304_he",
]

ply_ins = [f"./data/half_edge_base/ply/{_}.ply" for _ in surfs]
ply_outs = [f"./output/cglass.{_}.ply" for _ in surfs]

for ply_in, ply_out in zip(ply_ins, ply_outs):
    print(ply_in)
    scale_ply(ply_in, ply_out, scale_factor)
    print(ply_out)
