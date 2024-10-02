# from src.python.half_edge_base_brane import Brane
# from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim  # , Spindle, SPB, Envelope, ParamManager

from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from src.python.pretty_pictures import RGBA_DICT

yaml_path = "./data/parameters.yaml"

sim = StretchSim.from_parameters_file(yaml_path, overwrite_output_dir=True)
# sim.run()
# sim.update(patch=True, force=True, pretty=True)
# sim.plot(save=False, show=True, title="")
# sim.evolve_for_DT(1e-2, 1e-3)
# %%

m = sim.envelope
mv = sim.mesh_viewer
mv.plot()
# %%


# %%
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_patch import HalfEdgePatch
from src.python.half_edge_mesh import HalfEdgeMeshBase, HalfEdgeBoundary
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
ply_path = "./data/half_edge_base/ply/neovius_he.ply"

m = HalfEdgeMeshBase.load(ply_path=ply_path)

p = HalfEdgePatch.from_seed_to_radius(313, m, 0.3)

mv = MeshViewer(m, show_half_edges=True, show_wireframe_surface=False)
mv.update_rgba_H((0, 0, 0, 0))

bm = HalfEdgeBoundary.from_mesh(m)
bm_rgba_H = (0, 0, 0, 1)
bm_rgba_F = bm_rgba_H[:-1] + (0.5,)
bm_arrH = bm.arrH
bm_arrF_interior = np.array(sorted(bm.get_F_interior()))
mv.update_rgba_H(bm_rgba_H, bm_arrH)
mv.update_rgba_F(bm_rgba_F, bm_arrF_interior)

# bm2 = HalfEdgeBoundary.from_faces(m, set(range(m.num_faces)))
# bm2_rgba_H = (1, 1, 1, 1)
# bm2_rgba_F = bm2_rgba_H[:-1] + (0.5,)
# bm2_arrH = bm2.arrH
# bm2_arrF_interior = np.array(sorted(bm2.get_interior_F()))
# mv.update_rgba_H(bm2_rgba_H, bm2_arrH)
# mv.update_rgba_F(bm2_rgba_F, bm2_arrF_interior)

bp = HalfEdgeBoundary.from_faces(m, p.F)
bp_rgba_H = (1, 0, 0, 1)
bp_rgba_F = bp_rgba_H[:-1] + (0.5,)
bp_arrH = bp.arrH
bp_arrF_interior = np.array(sorted(bp.get_F_interior()))
mv.update_rgba_H(bp_rgba_H, bp_arrH)
mv.update_rgba_F(bp_rgba_F, bp_arrF_interior)

# mv.plot()


def apply_fun_iter(self, fun, num_iters=1):
    m = self.M
    self.plot(save=True, show=False, title=f"iter_{0}")
    for i in range(num_iters):
        print(f"Applying fun to mesh {i+1} of {num_iters}")
        fun(m)
        self.plot(save=True, show=False, title=f"iter_{i+1}")
    self.movie()


# %%


def movie_generate_cumulative_F_interior():
    from src.python.half_edge_base_viewer import MeshViewer
    from src.python.half_edge_mesh import HalfEdgeMeshBase, HalfEdgeBoundary
    from src.python.pretty_pictures import RGBA_DICT

    rgba_F_surface = RGBA_DICT["green50"]
    rgba_H_surface = RGBA_DICT["orange80"]

    rgba_H_boundary = RGBA_DICT["blue"]
    rgba_F_interior = RGBA_DICT["purple70"]
    rgba_F_frontier = RGBA_DICT["purple30"]
    mv_kwargs = {
        "show_half_edges": True,
        "show_wireframe_surface": False,
        "show_face_colored_surface": True,
        "show_vertex_colored_surface": False,
        "rgba_half_edge": rgba_H_surface,
        "rgba_face": rgba_F_surface,
    }
    ply_path = "./data/half_edge_base/ply/neovius_he.ply"

    m = HalfEdgeMeshBase.load(ply_path=ply_path)
    b = HalfEdgeBoundary.from_mesh(m)
    mv = MeshViewer(m, **mv_kwargs)

    mv.update_rgba_H(rgba_H_boundary, b.arrH)
    title = f"{mv.image_prefix}_{mv.image_count:0{mv.image_index_length}d}.{mv.image_format}"
    mv.plot(save=True, show=False, title=title)
    for F_interior, F_frontier in b.generate_cumulative_F_interior():
        # print(f"creating frame {i} of {num_iters}...                          ", end="\r")
        arrF_interior = np.array(list(F_interior))
        arrF_frontier = np.array(list(F_frontier))
        mv.update_rgba_F(rgba_F_interior, arrF_interior)
        mv.update_rgba_F(rgba_F_frontier, arrF_frontier)
        title = f"{mv.image_prefix}_{mv.image_count:0{mv.image_index_length}d}.{mv.image_format}"
        print(f"\r{' ' * 50}\n{' ' * 50}", end="")
        # Print the progress messages
        print(f"\rCreating {title}...", end="")
        print(
            f"\nnum_interior={len(F_interior)}, num_frontier={len(F_frontier)}", end=""
        )
        mv.plot(save=True, show=False, title=title)
    mv.movie()


movie_generate_cumulative_F_interior()
# %%
from src.python.atlas_test_sim import AtlasTestSim


s = TestSim(clean_output_dir=True, dF_draw=100, make_movie_frames=True)
s.run()
# np.array(list(s.F_frontier))
