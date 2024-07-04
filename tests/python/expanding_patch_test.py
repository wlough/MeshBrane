from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.half_edge_patch import HalfEdgePatch
from src.python.mesh_viewer import MeshViewer
import os

# %%


def expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60):
    """
    Makes a movie of surface patch expanding from seed vertex
    """
    if os.path.exists(image_dir):
        os.system(f"rm -r {image_dir}")
    os.system(f"mkdir -p {image_dir}")
    viewer_kwargs = {
        "image_dir": image_dir,
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
    # V_new = list(p.V_bdry)
    #
    # for iter in range(iters):
    #     print(f"iter={iter}")
    #
    #     H_bdry = list(p.generate_H_cw_B())
    #     F_bdry = list(p.generate_F_cw_B())
    #     H = list(p.H)
    #     F = list(p.F)
    #
    #     mv.set_F_rgba(f_rgba=mv.colors["green10"])
    #     mv.set_E_rgba(e_rgba=mv.colors["orange10"])
    #     mv.set_V_rgba(v_rgba=mv.colors["transparent"])
    #
    #     mv.set_subset_E_rgba(rgba=mv.colors["orange50"], indices=H)
    #     mv.set_subset_F_rgba(rgba=mv.colors["green50"], indices=F)
    #
    #     mv.set_subset_E_rgba(rgba=mv.colors["red"], indices=H_bdry)
    #     mv.set_subset_F_rgba(rgba=mv.colors["blue"], indices=F_bdry)
    #     mv.set_subset_V_rgba(rgba=mv.colors["red50"], indices=V_new)
    #     mv.save_plot()
    #     V_new = list(p.expand_boundary())

    mv.movie()


# 2 min 40 sec
# %%
source_ply = "./data/ply/binary/dumbbell.ply"
image_dir = "./output/expanding_patch_test/dumbbell"
expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60)

source_ply = "./data/ply/binary/torus.ply"
image_dir = "./output/expanding_patch_test/torus"
expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60)

source_ply = "./data/ply/binary/neovius.ply"
image_dir = "./output/expanding_patch_test/neovius"
expanding_patch_movie(source_ply, image_dir, v_seed=3, iters=60)

# %%


def expanding_patch_vertex_count(source_ply, v_seed=3):
    """
    check new verts from HalfEdgePatch.expand_boundary() for repeats
    """

    m = HalfEdgeMesh.from_half_edge_ply(source_ply)
    Vcounts = {v: 0 for v in m.xyz_coord_V.keys()}
    p = HalfEdgePatch.from_seed_vertex(v_seed, m)

    Vnew = p.V
    for v in Vnew:
        Vcounts[v] += 1

    while Vnew:
        Vnew = p.expand_boundary()
        for v in Vnew:
            Vcounts[v] += 1

    V = [val for key, val in Vcounts.items()]
    count_max = max(V)
    count_min = min(V)
    print(f"{count_max=}, {count_min=}")


source_ply = "./data/ply/binary/torus.ply"
source_ply = "./data/ply/binary/neovius.ply"
source_ply = "./data/ply/binary/dumbbell.ply"
expanding_patch_vertex_count(source_ply, v_seed=3)
