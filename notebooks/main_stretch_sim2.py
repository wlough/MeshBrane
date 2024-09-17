from src.python.half_edge_base_brane import Brane

from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim, SpbForce, ParamManager
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np


# parameters_path = "./data/stretch_sim_unit_sphere.yaml"
#
# sim = StretchSim.from_parameters_file(parameters_path, overwrite=True)
# sim.run()

parameters_path = "./data/stretch_sim_unit_sphere_uniform.yaml"

sim = StretchSim.from_parameters_file(parameters_path, overwrite=True)
sim.run()
