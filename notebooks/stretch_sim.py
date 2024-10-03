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
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_patch import HalfEdgePatch
from src.python.half_edge_mesh import HalfEdgeMeshBase, HalfEdgeBoundary, SubMesh
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
ply_path = "./data/half_edge_base/ply/neovius_he.ply"

m = HalfEdgeMeshBase.load(ply_path=ply_path)
sm = SubMesh.from_seed_vertex(13, m)
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

# %%
from src.python.atlas_test_sim import AtlasTestSim


s = TestSim(clean_output_dir=True, dF_draw=100, make_movie_frames=True)
s.run()
# np.array(list(s.F_frontier))
set([1, 2, 3]) - set([2, 6, 7])
