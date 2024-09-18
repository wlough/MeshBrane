from src.python.half_edge_base_brane import Brane

from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim, SpbForce, ParamManager
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np


parameters_path = "./data/stretch_sim_unit_sphere.yaml"

sim = StretchSim.from_parameters_file(parameters_path, overwrite_output_dir=True)
# sim.run()

# parameters_path = "./data/stretch_sim_unit_sphere_uniform.yaml"
#
# sim = StretchSim.from_parameters_file(parameters_path, overwrite=True)
# sim.run()
# sim.mesh_viewer.movie()
# %%
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_patch import HalfEdgePatch

# image_dir = "./output/aaaaa_stretch_sim/temp_images"
# image_dir = "./output/temp_images"
# image_dir = "./output/stretch_test/temp_images"
image_dir = "./output/torus_mesh_gen2/temp_images_001536"
ply = "./data/half_edge_base/ply/unit_torus_3_1_raw_001536_he.ply"
m = Brane.from_he_ply(ply)
mv = MeshViewer(m, image_dir=image_dir)
seed_vertex = 13
p = HalfEdgePatch.from_seed_vertex(seed_vertex, m)
p2 = HalfEdgePatch.from_vertex_set(set(list(p.V.copy())), m)
