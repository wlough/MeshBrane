from src.python.half_edge_base_brane import Brane

from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim, SpbForce, ParamManager
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np
from scipy.sparse import lil_matrix


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
# ply = "./data/half_edge_base/ply/unit_torus_3_1_raw_001536_he.ply"
ply = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
m = Brane.from_he_ply(ply)
mv = MeshViewer(m, image_dir=image_dir)
seed_vertex = 13
p = HalfEdgePatch.from_seed_vertex(seed_vertex, m)
# %%
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

# ply = "./data/half_edge_base/ply/unit_sphere_001280_he.ply"
# ply = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
# ply = "./data/half_edge_base/ply/unit_sphere_020480_he.ply"
ply = "./data/half_edge_base/ply/neovius_he.ply"
# plys = [
#     "./data/half_edge_base/ply/unit_sphere_001280_he.ply",
#     "./data/half_edge_base/ply/unit_sphere_005120_he.ply",
#     "./data/half_edge_base/ply/unit_sphere_020480_he.ply",
# ]
m = Brane.from_he_ply(ply, **Brane.default_params())

mv = MeshViewer(m)
# M = [Brane.from_he_ply(ply, **Brane.default_params()) for ply in plys]

# m.num_vertices
# m = Brane.default_sphere()
# L, A = m.get_cotan_laplacian_lil()
L, A = m.get_cotan_laplacian_lil()
Lsafe, Asafe = m.get_cotan_laplacian_lilsafe()
_L, _A = m._get_cotan_laplacian_lil()
# F = m.Fbend_analytic()

# %%
diag = A
S = L
scale = 1e-1

Q = m.xyz_coord_V
D = lil_matrix(np.diag(1/diag))
Lap = (D@S).toarray()
vecs = Lap@Q

I = np.array([i for i in range(m.num_vertices) if not m.boundary_contains_v(i)])
vf_kwargs = {"points": Q, "vectors": scale*vecs}
# vf_kwargs = {"points": Q[I], "vectors": scale*vecs[I]}
mv.clear_vector_field_data()
mv.add_vector_field(**vf_kwargs)
mv.plot()
# # %%
for m in M:
    print(f"{m.num_vertices}")
    %timeit m.get_cotan_laplacian_lil()
    %timeit m._get_cotan_laplacian_lil()
from src.python.pretty_pictures import plot_log_log_fit
t = np.array([55.4e-3, 220e-3, 884e-3])
_t = np.array([255e-3, 1.02, 4.08])
Nv = np.array([m.num_vertices for m in M])
Nv[1:]/Nv[:-1]
plot_log_log_fit(Nv,t)
