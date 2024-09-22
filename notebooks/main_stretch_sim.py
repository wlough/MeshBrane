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
from src.python.half_edge_base_viewer import MeshViewer, MultiMeshViewer
from src.python.linear_algebra import (
    exp_so3_quaternion,
    quaternion_to_matrix,
    quaternion_to_matrix_numba,
    rigid_transform,
)
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

# ply = "./data/half_edge_base/ply/unit_sphere_001280_he.ply"
# ply = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
# ply = "./data/half_edge_base/ply/unit_sphere_020480_he.ply"
ply = "./data/half_edge_base/ply/dumbbell_he.ply"
# ply = "./data/half_edge_base/ply/unit_torus_3_1_raw_001536_he.ply"
# plys = [
#     "./data/half_edge_base/ply/unit_sphere_001280_he.ply",
#     "./data/half_edge_base/ply/unit_sphere_005120_he.ply",
#     "./data/half_edge_base/ply/unit_sphere_020480_he.ply",
# ]
# m0 = Brane.from_he_ply(ply, **Brane.default_params())
# m1 = Brane.from_he_ply(ply, **Brane.default_params())
meshes = [Brane.from_he_ply(ply, **Brane.default_params()) for _ in range(4)]
_params = {
    "radius_vertex": 0.0125,
    "rgba_vertex": (0.7057, 0.0156, 0.1502, 0.75),
    "rgba_half_edge": (1.0, 0.498, 0.0, 1.0),
    "rgba_face": (0.0, 0.63335, 0.05295, 0.65),
    "rgba_boundary_half_edge": (0.0, 0.4471, 0.6980, 1.0),
    "rgba_edge": (1.0, 0.498, 0.0, 1.0),
    # mlab data that depends on mesh size
    "radius_V": None,
    "rgba_V": None,
    "rgba_H": None,
    "rgba_F": None,
    # Additional vector fields to plot
    "vector_field_data": None,
    # mlab data that does NOT depend on mesh size
    "show_wireframe_surface": True,
    "show_face_colored_surface": True,
    "show_vertex_colored_surface": False,
    "show_vertices": False,
    "show_half_edges": False,
    "show_vector_fields": True,
}

# mv = MeshViewer(
#     m[0],
#     show_plot_axes=True,
#     view={"azimuth": 0, "elevation": 0.0, "distance": 2.75, "focalpoint": [0, 0, 0]},
# )

mmv = MultiMeshViewer(
    meshes,
    show_plot_axes=True,
    view={"azimuth": 0, "elevation": 0.0, "distance": 2.75, "focalpoint": [0, 0, 0]},
)

# translation = 0 * np.array([[1, 0, 0]])
# angle_vec = 0*np.pi / 4 * np.array([0, 1, 0])
# m0.rigid_transform(translation, angle_vec)
# b = 0
s = set()
s.update(range(3))
# np.random.rand(33, 4)@rgba0
rgba0 = mmv.colors["purple"]
rgba1 = mmv.colors["red"]
# mmv.mesh_viewers[0].update_rgba_F_incident_b(rgba0, b)
# mmv.mesh_viewers[1].update_rgba_F_incident_b(rgba1, b)
# mmv.mesh_viewers[0].update_rgba_F_incident_B(rgba0)
# mmv.mesh_viewers[1].update_rgba_F_incident_B(rgba1)
# mmv.update_mesh_params(
#     0, rgba_boundary_face=rgba0, show_half_edges=True, show_wireframe_surface=False
# )
# mmv.update_mesh_params(
#     1, rgba_boundary_face=rgba1, show_half_edges=True, show_wireframe_surface=False
# )
# mmv.update_mesh_params(1, rgba_boundary_half_edge=rgba1, show_half_edges=True, show_wireframe_surface=False)
pad = 0.1
axis = 0
mmv.spread_meshes(pad, axis)
mmv.plot()
# %%
set(range(3))
# - 0.0
# - 0.0
# - 0.0

# M = [Brane.from_he_ply(ply, **Brane.default_params()) for ply in plys]
np.linalg.norm(q, axis=-1, keepdims=True)
# m.num_vertices
# m = Brane.default_sphere()
# L, A = m.get_cotan_laplacian_lil()
L, A = m.get_cotan_laplacian_lil()
Lsafe, Asafe = m.get_cotan_laplacian_lilsafe()
_L, _A = m._get_cotan_laplacian_lil()
# F = m.Fbend_analytic()

# %%
I = np.array([i for i in range(m.num_vertices) if not m.boundary_contains_v(i)])
Q = m.xyz_coord_V

diag = A
S = L
scale = 1e-1
D = lil_matrix(np.diag(1 / diag))
Lap = (D @ S).toarray()
vecs = Lap @ Q
vf_kwargs = {"points": Q[I], "vectors": scale * vecs[I]}

diag = Asafe
S = Lsafe
scale = 1e-1
D = lil_matrix(np.diag(1 / diag))
Lap = (D @ S).toarray()
vecs = Lap @ Q
vf_kwargssafe = {"points": Q[I], "vectors": scale * vecs[I]}

diag = _A
S = _L
scale = 1e-1
D = lil_matrix(np.diag(1 / diag))
Lap = (D @ S).toarray()
vecs = Lap @ Q
_vf_kwargs = {"points": Q[I], "vectors": scale * vecs[I]}

# vf_kwargs = {"points": Q, "vectors": scale*vecs}
kwargs = {"points": Q[I], "vectors": _vf_kwargs["vectors"] - vf_kwargssafe["vectors"]}

mv.clear_vector_field_data()
mv.add_vector_field(**kwargs)
mv.plot()
# # %%
# for m in M:
#     print(f"{m.num_vertices}")
#     %timeit m.get_cotan_laplacian_lil()
#     %timeit m._get_cotan_laplacian_lil()
from src.python.pretty_pictures import plot_log_log_fit

t = np.array([55.4e-3, 220e-3, 884e-3])
_t = np.array([255e-3, 1.02, 4.08])
Nv = np.array([m.num_vertices for m in M])
Nv[1:] / Nv[:-1]
plot_log_log_fit(Nv, t)
