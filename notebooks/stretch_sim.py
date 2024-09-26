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
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np
from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
m = Brane.load(ply_path=ply_path)
v = 1322
p1 = HalfEdgePatch.from_seed_to_radius(v, m, 0.3)
p2 = HalfEdgePatch.from_seed_to_radius(v, m, 0.2)
p0 = HalfEdgePatch.from_seed_to_radius(v, m, 0.1)
F = p2.F - p0.F
H = p2.H - p0.H
V = p2.V - p0.V
V0, H0, F0 = m.closure(V.copy(), H.copy(), F.copy())
V1, H1, F1 = m.closure1(V.copy(), H.copy(), F.copy())
V2, H2, F2 = m.closure2(V.copy(), H.copy(), F.copy())
[
    V1.symmetric_difference(V0),
    V2.symmetric_difference(V1),
    V0.symmetric_difference(V2),
    H1.symmetric_difference(H0),
    H2.symmetric_difference(H1),
    H0.symmetric_difference(H2),
    F1.symmetric_difference(F0),
    F2.symmetric_difference(F1),
    F0.symmetric_difference(F2),
]

# %timeit m.closure(V.copy(), H.copy(), F.copy())
# %timeit m.closure1(V.copy(), H.copy(), F.copy())
# %timeit m.closure2(V.copy(), H.copy(), F.copy())


# %%
from src.python.half_edge_base_brane import Brane

# from src.python.half_edge_base_viewer import MeshViewer
# from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
m = Brane.load(ply_path=ply_path)


# class MaterialPoint:
#     def __init__(self, coord_xyz=None, h_out=None):
#         self.xyz = xyz


class HalfEdgeVertex:
    def __init__(self, xyz_coord=None, h_out=None):
        self.xyz_coord = xyz_coord
        self.h_out = h_out

    def __hash__(self):
        return hash(
            (
                tuple(self.xyz_coord),
                self.h_out,
            )
        )


class HalfEdge:
    def __init__(self, v_origin=None, h_twin=None, h_next=None, f_left=None):
        self.v_origin = v_origin
        self.h_twin = h_twin
        self.h_next = h_next
        self.f_left = f_left

    def __hash__(self):
        return hash(
            (
                self.v_origin,
                self.h_next,
                self.h_twin,
                self.f_left,
            )
        )


class HalfEdgeFace:
    def __init__(self, h_bound=None):
        self.h_bound = h_bound

    def __hash__(self):
        return hash(
            (
                tuple(self.xyz_coord),
                self.h_out,
                self.v_origin,
                self.h_next,
                self.h_twin,
                self.f_left,
                self.h_bound,
                self.h_right,
            )
        )


# M = gen(V,H,F)
H = set(np.random.randint(0, 555, 5, dtype="int32"))
arrH = np.array(list(H), dtype="int32")
# _f_left_H = m.f_left_h(_H)
H.update(set(m.h_next_h(arrH)) | set(m.h_next_h(m.h_next_h(arrH))))
arrH = np.array(list(H), dtype="int32")

v_origin_H = m.v_origin_h(arrH)
f_left_H = m.f_left_h(arrH)

V = set(v_origin_H)
F = set(f_left_H)
arrV = np.array(list(V), dtype="int32")
arrF = np.array(list(F), dtype="int32")
xyz_coord_V = m.xyz_coord_v(arrV)
h_out_V = m.h_out_v(arrV)
h_bound_F = m.h_bound_f(arrF)

h_next_H = m.h_next_h(arrH)
h_twin_H = m.h_twin_h(arrH)
f_left_H = m.f_left_h(arrH)

H_interior = (
    set(h_bound_F) | set(m.h_next_h(h_bound_F)) | set(m.h_next_h(m.h_next_h(h_bound_F)))
)
H_boundary = H - H_interior
vertices = np.array([HalfEdgeVertex(xyz_coord=_) for _ in xyz_coord_V], dtype=object)
halfedges = np.array([HalfEdge() for _ in arrH], dtype=object)
faces = np.array([HalfEdgeFace() for _ in arrF], dtype=object)
# %%
vertices = [HalfEdgeVertex() for _ in V]
halfedges = [HalfEdge() for _ in H]
faces = [HalfEdgeFace() for _ in V]
# h =
# v = HalfEdgeVertex(m, )
