from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
import numpy as np
import os

source_paths = [
    f"./data/ply/binary/sphere{_}.ply" for _ in ["_ultracoarse", "_coarse", "", "_fine", "_ultrafine"]
]
m = [HalfEdgeMesh.from_half_edge_ply(source_path) for source_path in source_paths[:-1]]
mv = [MeshViewer(*_.data_lists) for _ in m]

mb = m[2]
Y = [xyz[1] for xyz in mb.data_lists[0]]
LYi, labels = mb.laplacian_propogate(3,Y)
# %%
# ply_path = "./ply_files_vf/dumbbell.ply"
ply_path = "./data/ply/binary/torus.ply"
m = HalfEdgeMesh.from_half_edge_ply(ply_path)
viewer_kwargs = {
    "image_dir": "./output/convergence_test/temp_images",
    "view": {
        "azimuth": 0,
        "elevation": 55,
        "distance": 4,
        "focalpoint": (0, 0, 0),
    },
    "show_vertices": True,
}
mv = MeshViewer(*m.data_lists, **viewer_kwargs)
# %%
Vs, Es, Fs = {13, 6}, set(), set()
# Vs, Es, Fs = m.Lk(Vs, Es, Fs)
Vs, Es, Fs = m.St(Vs, Es, Fs)
# %%
V, E, F = list(Vs), list(Es), list(Fs)
mv.set_F_rgba(f_rgba=mv.colors["green20"])
mv.set_E_rgba(e_rgba=mv.colors["orange20"])
mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=E)
# mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=F)
mv.show_plot()
# %%
mv.colors["transparent"]
black = np.array([0.0, 0.0, 0.0, 1.0])
red = np.array([1.0, 0.0, 0.0, 1.0])
green = np.array([0.0, 1.0, 0.0, 1.0])
blue = np.array([0.0, 0.0, 1.0, 1.0])
orange50 = np.array([1.0, 0.498, 0.0, 0.5])
orange25 = np.array([1.0, 0.498, 0.0, 0.25])
orange10 = np.array([1.0, 0.498, 0.0, 0.1])
transp = np.array([0, 0, 0, 0])
v0 = 916
# v0 = int(len(m.V)*np.random.rand())
labels = m.one_ring_vhf_sets_with_bdry(v0)

E = list(labels["boundary_edges"])
F = list(labels["faces"])
for iter in range(60):
    print(f"iter={iter}")
    mv.set_F_rgba(f_rgba=mv.colors["green20"])
    mv.set_E_rgba(e_rgba=mv.colors["orange20"])
    mv.set_V_rgba(v_rgba=mv.colors["red10"])

    mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=E)
    mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=F)
    mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
    mv.save_plot()
    labels = m.expand_boundary_safe(**labels)
    V = list(labels["vertices"])
    E = list(labels["boundary_edges"])
    F = list(labels["faces"])


# %%
v0 = int(len(m._xyz_coordinates_v) * np.random.rand())
E1 = m.get_order_one_edge_neighbors(v0)
E2 = m.get_order_n_plus_one_edge_neighbors(E1)
E3 = m.get_order_n_plus_one_edge_neighbors(E2)
E4 = m.get_order_n_plus_one_edge_neighbors(E3)
E5 = m.get_order_n_plus_one_edge_neighbors(E4)
# E6 = m.get_order_n_plus_one_edge_neighbors(E5)
# E7 = m.get_order_n_plus_one_edge_neighbors(E6)
# E8 = m.get_order_n_plus_one_edge_neighbors(E7)
# %%
mv.set_V_rgba(v_rgba=mv.colors["green"])
mv.set_E_rgba(e_rgba=mv.colors["orange20"])
mv.set_F_rgba(f_rgba=mv.colors["green20"])

mv.set_subset_E_rgba(rgba=blue, indices=E1)
mv.set_subset_E_rgba(rgba=black, indices=E2)
mv.set_subset_E_rgba(rgba=red, indices=E3)
mv.set_subset_E_rgba(rgba=blue, indices=E4)
mv.set_subset_E_rgba(rgba=black, indices=E5)
# m.set_E_rgba(red, E6)
# m.set_E_rgba(blue, E7)
# m.set_E_rgba(black, E8)
mv.E_rgba
mv.show_plot()
