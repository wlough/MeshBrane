from src.python.half_edge_mesh import (
    HalfEdgeMeshBase,
    HalfEdgeComplex,
    HalfEdgeCurve,
    HalfEdgeLoop,
    Boundary2D,
)


from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
ply_path = "./data/half_edge_base/ply/neovius_he.ply"

m = HalfEdgeComplex.load(ply_path=ply_path)
p = HalfEdgePatch.from_seed_to_radius(313, m, 0.3)
b = Boundary2D.from_supermesh(m)
F = b.get_interior_faces()
len(F)
m.num_faces
# p.h_right_B
# p.get_VH_bdry()
# %%
# sm = SubMesh.from_seed_vertex(13, m)


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
