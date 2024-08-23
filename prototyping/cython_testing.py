import numpy as np
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_base_utils import vf_samples_to_he_samples as nvf_samples_to_he_samples
from src.python.half_edge_base_utils import find_h_comp_B as nfind_h_comp_B

from src.python.half_edge_base_jit_utils import find_h_comp_B,vf_samples_to_he_samples

# from src.python.half_edge_base_cython_utils import

# %%
# m = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/sphere_000642_he.ply")
m = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/neovius.ply")
# %%
V, F = m.xyz_array.tolist(), m.V_of_F.tolist()
Varr, Farr = m.xyz_array, np.int32(m.V_of_F)

%timeit vf_samples_to_he_samples(Varr, Farr)
%timeit nvf_samples_to_he_samples(Varr, Farr)


# %%
# from libcpp.vector cimport vector
from cython.cimports.gpp import vector
from cython cimport int, float
from typing import List, Tuple

from cython.cimports import numpy as cnp
from cython.cimports.libgpp.vector import math
math.vector
