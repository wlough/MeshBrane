import sys

# sys.path.append("../src/python")
sys.path.append("./")
# import numpy as np
# from numba import njit, float64, int32, int64, boolean, prange
# from numba.typed import Dict
# from src.python.half_edge_mesh import HalfEdgeMesh
# from src.python.mesh_viewer import MeshViewer
from src.python.sphere_builder import SphereFactory
import pickle
from time import time


# m = HalfEdgeMesh.from_half_edge_ply("./data/ply/binary/annulus.ply")
# m.num_boundaries
# m.num_edges
# m.num_faces*
# m.num_vertices
# len(m.V_of_H)
# m.euler_characteristic
# import dill
def refine_icososphere_and_save(
    num_refine, output_dir="./output/sphere_builder", save_name="refinement"
):
    # num_refine = 10
    # data_paths = [f"./output/{save_name}_{n:06d}.pickle" for n in range(num_refine)]
    sf = SphereFactory()
    for n in range(1, num_refine + 1):
        print("------------------")
        print(f"refinement {n=}")
        t = time()
        sf.refine()
        t = time() - t
        print(f"refine time {t=}")
        data_path = f"./output/{save_name}_{n:06d}.pickle"
        t = time()
        with open(data_path, "wb") as f:
            pickle.dump(sf, f)
        t = time() - t
        print(f"pickle time {t=}")
        print("------------------")


# V, F = sf.VF()
# m = HalfEdgeMesh.from_vert_face_list(V, F)
# V = sorted(m._xyz_coord_V.keys())
# H = sorted(m._v_origin_H.keys())
# F = sorted(m._h_bound_F.keys())
# xyz_coord_V = [m.xyz_coord_v(v) for v in V]
# # h_out_V = [H.index(m.h_out_v(v)) for v in V]
# m._h_out_V
# v_origin_H = [V.index(m.v_origin_h(h)) for h in H]
# h_next_H = [H.index(m.h_next_h(h)) for h in H]
# h_twin_H = [H.index(m.h_twin_h(h)) for h in H]
# f_left_H = [m.f_left_h(h) if m.f_left_h(h) < 0 else F.index(m.f_left_h(h)) for h in H]
# h_bound_F = [H.index(m.h_bound_f(f)) for f in F]
# m.data_dicts
# m.data_lists
# (
#     V,
#     h_out_V,
#     v_origin_H,
#     h_next_H,
#     h_twin_H,
#     f_left_H,
#     h_bound_F,
# ) = sf.HE()
# # m = HalfEdgeMesh(*sf.HE())
# mv = MeshViewer(*sf.HE())
# mv.plot()

# %%

import numpy as np
from src.python.utilities import log_log_fit
from src.python.pretty_pictures import plot_log_log_fit
import matplotlib.pyplot as plt

T = np.array(
    [
        0.00022459030151367188,
        0.00023674964904785156,
        0.0017919540405273438,
        0.021833181381225586,
        0.32131028175354004,
        5.501546859741211,
        92.80365824699402,
        1780.0954852104187,
        32649.431557655334,
    ]
)
h = np.int32(T // 3600)
m = np.int32((T - 3600 * h) // 60)
s = np.int32(T - 3600 * h - 60 * m)
N = np.array([10 * 4**k + 2 for k in range(1, 10)])
fit = plot_log_log_fit(N[4:], T[4:], rcparams=dict())
