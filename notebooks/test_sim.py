# from src.python.half_edge_base_brane import Brane
# from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim  # , Spindle, SPB, Envelope, ParamManager

# from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from src.python.pretty_pictures import RGBA_DICT

yaml_path = "./data/parameters.yaml"

sim = StretchSim.from_parameters_file(yaml_path, overwrite_output_dir=False)
# sim.run()



m = sim.envelope
mv = sim.mesh_viewer
mv.plot()
# %%
# Af = m.area_F()
vorcell_area_V = m.vorcell_area_V()
barcell_area_V = m.barcell_area_V()
%timeit m.Farea_harmonic()
%timeit m.Fvolume_harmonic()
%timeit m.Fbend_analytic()
%timeit m.area_F()
L1, A1 = m._get_cotan_laplacian_lil()
L2, A2 = m.get_cotan_laplacian_lil()
L3, A3 = m.get_cotan_laplacian_lilsafe()
%timeit m._get_cotan_laplacian_lil()
%timeit m.get_cotan_laplacian_lil()
%timeit m.get_cotan_laplacian_lilsafe()
# get_cotan_laplacian_csr = m.get_cotan_laplacian_csr()
np.linalg.norm((L1 - L2).toarray().ravel(), np.inf)
np.linalg.norm((L2 - L3).toarray().ravel(), np.inf)
np.linalg.norm((L3 - L1).toarray().ravel(), np.inf)

np.linalg.norm((A1 - A2), np.inf)
np.linalg.norm((A2 - A3), np.inf)
np.linalg.norm((A3 - A1), np.inf)

np.linalg.norm((A1 - vorcell_area_V), np.inf)
np.linalg.norm((A1 - barcell_area_V), np.inf)

np.linalg.norm((A2 - vorcell_area_V), np.inf)
np.linalg.norm((A2 - barcell_area_V), np.inf)

np.linalg.norm((A3 - vorcell_area_V), np.inf)
np.linalg.norm((A3 - barcell_area_V), np.inf)

np.linalg.norm((barcell_area_V - vorcell_area_V), np.inf)
