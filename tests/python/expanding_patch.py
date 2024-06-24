from src.python.half_edge_mesh import HalfEdgeMesh, HalfEdgePatch
from src.python.mesh_viewer import MeshViewer
import os


def test(source_ply, image_dir, v_seed=3, iters=60):
    # source_ply = "./data/ply/binary/neovius.ply"
    # image_dir = "./output/expanding_patch_test/neovius"
    if os.path.exists(image_dir):
        os.system(f"rm -r {image_dir}")
    os.system(f"mkdir -p {image_dir}")
    viewer_kwargs = {
        "image_dir": image_dir,  # "./output/convergence_test/temp_images",
        # "view": {
        #     "azimuth": 0,
        #     "elevation": 55,
        #     "distance": 6,
        #     "focalpoint": (0, 0, 0),
        # },
        "show_vertices": True,
        "v_radius": 0.03,
    }

    m = HalfEdgeMesh.from_half_edge_ply(source_ply)
    mv = MeshViewer(*m.data_lists, **viewer_kwargs)
    p = HalfEdgePatch.from_seed_vertex(v_seed, m)

    V = [v_seed]
    H, F = [], []
    mv.set_F_rgba(f_rgba=mv.colors["green10"])
    mv.set_E_rgba(e_rgba=mv.colors["orange10"])
    mv.set_V_rgba(v_rgba=mv.colors["transparent"])
    mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
    mv.save_plot()
    Vnew = list(p.V_bdry)

    for iter in range(iters):
        print(f"iter={iter}")

        V = Vnew
        H = list(p.generate_H_cw_B())
        F = list(p.generate_F_cw_B())

        mv.set_F_rgba(f_rgba=mv.colors["green10"])
        mv.set_E_rgba(e_rgba=mv.colors["orange10"])
        mv.set_V_rgba(v_rgba=mv.colors["transparent"])

        mv.set_subset_E_rgba(rgba=mv.colors["blue"], indices=H)
        mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=list(F))
        mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V)
        mv.save_plot()
        Vnew = list(p.expand_boundary())

    mv.movie()


# %%
source_ply = "./data/ply/binary/dumbbell.ply"
image_dir = "./output/expanding_patch_test/dumbbell"
test(source_ply, image_dir, v_seed=3, iters=60)

source_ply = "./data/ply/binary/torus.ply"
image_dir = "./output/expanding_patch_test/torus"
test(source_ply, image_dir, v_seed=3, iters=60)

source_ply = "./data/ply/binary/neovius.ply"
image_dir = "./output/expanding_patch_test/neovius"
test(source_ply, image_dir, v_seed=3, iters=60)
