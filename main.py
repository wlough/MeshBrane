from src.python.half_edge_mesh import HalfEdgeMesh, HalfEdgePatch
from src.python.mesh_viewer import MeshViewer
import numpy as np

source_path = "./data/ply/binary/neovius.ply"
viewer_kwargs = {
    "image_dir": "./output/convergence_test/temp_images",
    # "view": {
    #     "azimuth": 0,
    #     "elevation": 55,
    #     "distance": 6,
    #     "focalpoint": (0, 0, 0),
    # },
    "show_vertices": True,
    "v_radius": 0.03,
}
# m = HalfEdgeMesh.from_vertex_face_ply(source_path)
m = HalfEdgeMesh.from_half_edge_ply(source_path)
mv = MeshViewer(*m.data_lists, **viewer_kwargs)

p = HalfEdgePatch.from_seed_vertex(3, m)
for iter in range(60):
    print(f"iter={iter}")

    # LkV, LkH, LkF = m.link(V, H, F)
    mv.set_F_rgba(f_rgba=mv.colors["green10"])
    mv.set_E_rgba(e_rgba=mv.colors["orange10"])
    mv.set_V_rgba(v_rgba=mv.colors["transparent"])

    V = list(p.expand_boundary())
    H = list(p.generate_H_cw_B())
    F = list(p.generate_F_cw_B())

    mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=H)
    mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=list(F))
    mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
    mv.save_plot()


def movie(
    image_dir,
    image_type="png",
    image_prefix="frame",
    index_length=4,
    movie_name="movie",
    movie_dir=None,
    movie_type="mp4",
):
    import os
    import subprocess

    image_name = f"{image_prefix}_%0{index_length}d.{image_type}"
    image_path = os.path.join(image_dir, image_name)
    if movie_dir is None:
        movie_dir = image_dir
    movie_path = os.path.join(image_dir, f"{movie_name}.{movie_type}")
    run_command = [
        "ffmpeg",
        # overwrite output file without asking
        "-y",
        # frame rate (Hz)
        "-r",
        "20",
        # frame size (width x height)
        "-s",
        "1080x720",
        # input files
        "-i",
        image_path,
        # video codec
        "-vcodec",
        "libx264",
        # video quality, lower means better
        "-crf",
        "25",
        # pixel format
        "-pix_fmt",
        "yuv420p",
        # output file
        movie_path,
    ]

    # Start the process
    process = subprocess.Popen(
        run_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=image_dir,
    )
    for line in iter(process.stdout.readline, b""):
        print(line.decode(), end="")
    # relative_path = os.path.relpath(target_dir, start_dir)
    # os.path.join,os.path.basename,os.makedirs,os.system,os.path.exists,os.makedirs,shutil.rmtree


movie(
    "./output/convergence_test/temp_images",
    image_type="png",
    image_prefix="frame",
    index_length=4,
    movie_name="movie",
    movie_dir=None,
    movie_type="mp4",
)
# %%
# cw, ccw = m.find_boundary_cycles()
mv.set_F_rgba(f_rgba=mv.colors["green20"])
mv.set_E_rgba(e_rgba=mv.colors["orange20"])
mv.set_V_rgba(v_rgba=mv.colors["red10"])
H = []
for b, h in m.h_cw_B.items():
    H.extend(m.generate_H_next_h(h))
mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=H)
# for cycle in cw:
#     mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=cycle)
# for cycle in ccw:
#     mv.set_subset_E_rgba(rgba=mv.colors["red"], indices=cycle)
mv.plot()

# %%
V, H, F = set(), {3}, set()
V, H, F = m.star(V, H, F)

# %%
Hl, Hr = m.find_boundary_cycles()
mv.plot()
# %%
V, H, F = {3}, set(), set()
V, H, F = m.link(V, H, F)
# V, H, F = (list(_) for _ in m.link(V, H, F))
# V, H, F = m.star(V, H, F)
# V, H, F = m.closure(V, H, F)
p = HalfEdgePatch.from_seed_vertex(3, m)

list(p.generate_V_cw_B())
# V, F, H = list(p.V), list(p.F), list(p.H)
mv.set_F_rgba(f_rgba=mv.colors["green20"])
mv.set_E_rgba(e_rgba=mv.colors["orange20"])
mv.set_V_rgba(v_rgba=mv.colors["red10"])

mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=list(H))
mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=list(F))
mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=list(V))
mv.plot()
# %%

# V, H, F = m.link(V, H, F)
# %%
# p = HalfEdgePatch.from_seed_vertex(3, m)
# V, F, H = list(p.V), list(p.F), list(p.H)


for iter in range(60):
    print(f"iter={iter}")
    V, F, H = list(p.V), list(p.F), list(p.H)
    Hbdry = p.H_boundary_cycle
    mv.set_F_rgba(f_rgba=mv.colors["green20"])
    mv.set_E_rgba(e_rgba=mv.colors["orange20"])
    mv.set_V_rgba(v_rgba=mv.colors["red10"])

    mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=Hbdry)
    mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=F)
    mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
    mv.save_plot()

    p.expand_boundary()
    # need2visit = p.H.copy()
    # Ne = 0
    # need2visit.discard(13)
    # while need2visit:
    #     h = need2visit.pop()
    #     ht = p.supermesh.h_twin_h(h)
    #     need2visit.discard(ht)
    #     Ne += 1
    # Nf = len(p.F)
    # Nv = len(p.V)
    # chi = Nf - Ne + Nv
    # print(f"chi={chi}")


#
#
######################################################################
######################################################################
######################################################################
######################################################################

# %%
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
import numpy as np
import os

source_paths = [
    f"./data/ply/binary/sphere{_}.ply" for _ in ["_ultracoarse", "_coarse", "", "_fine", "_ultrafine"]
]
m = [HalfEdgeMesh.from_half_edge_ply(source_path) for source_path in source_paths[:-1]]
mv = [MeshViewer(*_.data_lists) for _ in m]

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
