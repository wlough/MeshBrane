import sys

sys.path.append("./")
import pickle
from src.python.jit_brane import HalfEdgeMeshBuilder
from src.python.convergence_tests import ConvergenceTestData
import numpy as np
import os
from src.python.pretty_pictures import plot_log_log_fit

# from src.python.mesh_viewer import MeshViewer

_TEST_DIR_ = "./output/convergence_tests"
_NUM_VERTS_ = [
    # 12,
    # 42,
    162,
    642,
    2562,
    10242,
    40962,
    # 163842,
    # 655362,
    # 2621442,
]
_NUM_FACES_ = [
    # 20,
    # 80,
    320,
    1280,
    5120,
    20480,
    81920,
    # 327680,
    # 1310720,
    # 5242880,
]
try:
    _M
except NameError:
    _M = HalfEdgeMeshBuilder.load_test_spheres()


def make_output_dir(output_dir=_TEST_DIR_, overwrite=False):
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError(
            f"{output_dir} already exists. Choose a different output_dir, or Set overwrite=True"
        )
    os.system(f"mkdir -p {output_dir}")


# %%
def test_sphere_mcvec_belkin(heat_order=1, s_num=1):
    M = _M[2:7]
    # S = np.array([10**-p for p in range(1, 6)])
    S = np.array([np.mean(m.area_F()) for m in M])
    # S = np.array([0.025, 0.01, 0.0025])

    s = S[s_num]
    name = f"sphere_mcvec_belkin_{heat_order:03d}_{s_num:03d}"
    data_path = _TEST_DIR_ + "/" + name + ".pkl"
    params = {"s": s}
    independent_var = np.array([m.num_faces for m in M])
    he_keys = [
        "xyz_coord_V",
        "h_out_V",
        "v_origin_H",
        "h_next_H",
        "h_twin_H",
        "f_left_H",
        "h_bound_F",
        "h_comp_B",
    ]
    half_edge_mesh_arrays = [{k: v for k, v in zip(he_keys, m.data_arrays)} for m in M]
    samples_numerical = []
    samples_actual = []
    Nsamps = len(M)
    for n in range(Nsamps):
        m = M[n]
        Q = m.xyz_array
        lapQ = m.order_p_belkin_laplacian(Q, s, heat_order)
        H = -np.linalg.norm(lapQ, axis=-1) / 2
        samples_numerical.append(H)
        samples_actual.append(-np.ones_like(H))
    test_kwargs = {
        "name": name,
        "samples_numerical": samples_numerical,
        "samples_actual": samples_actual,
        "independent_var": independent_var,
        "params": params,
        # "fun_actual": lambda xyz: -1.0,
        "half_edge_mesh_arrays": half_edge_mesh_arrays,
        "data_path": data_path,
    }
    T = ConvergenceTestData(**test_kwargs)
    T.save()
    return T


def run_tests():
    M = _M[2:7]
    # M = _M[:4]
    # S = np.array([np.mean(m.area_F()) for m in M])
    S = np.array([0.03853078, 0.0097707, 0.00245144, 0.00061341, 0.00015339])
    Snum = [_ for _, m in enumerate(S)]
    Hnum = [1, 2, 3]
    Tdict = dict()
    for heat_order in Hnum:
        print("--------------------")
        print(f"{heat_order=}")
        for s_num in Snum:
            print(f"- {s_num=}")
            s = S[s_num]
            name = f"sphere_mcvec_belkin_{heat_order:03d}_{s_num:03d}"
            data_path = _TEST_DIR_ + "/" + name + ".pkl"
            params = {"s": s}
            independent_var = np.array([m.num_faces for m in M])
            he_keys = [
                "xyz_coord_V",
                "h_out_V",
                "v_origin_H",
                "h_next_H",
                "h_twin_H",
                "f_left_H",
                "h_bound_F",
                "h_comp_B",
            ]
            half_edge_mesh_arrays = [
                {k: v for k, v in zip(he_keys, m.data_arrays)} for m in M
            ]
            samples_numerical = []
            samples_actual = []
            Nsamps = len(M)
            for n in range(Nsamps):
                m = M[n]
                num_faces = m.num_faces
                print(f"-- {num_faces=}")
                Q = m.xyz_array
                lapQ = m.order_p_belkin_laplacian(Q, s, heat_order)
                H = -np.linalg.norm(lapQ, axis=-1) / 2
                samples_numerical.append(H)
                samples_actual.append(-np.ones_like(H))
            test_kwargs = {
                "name": name,
                "samples_numerical": samples_numerical,
                "samples_actual": samples_actual,
                "independent_var": independent_var,
                "params": params,
                # "fun_actual": lambda xyz: -1.0,
                "half_edge_mesh_arrays": half_edge_mesh_arrays,
                "data_path": data_path,
            }
            T = ConvergenceTestData(**test_kwargs)
            T.save()
            Tdict[(heat_order, s_num)] = T
    print("Done.")
    return Tdict


Td = run_tests()
#
# T10 = test_sphere_mcvec_belkin(heat_order=1, s_num=0)
# T11 = test_sphere_mcvec_belkin(heat_order=1, s_num=1)
# T12 = test_sphere_mcvec_belkin(heat_order=1, s_num=2)
# T13 = test_sphere_mcvec_belkin(heat_order=1, s_num=3)
#
# T20 = test_sphere_mcvec_belkin(heat_order=2, s_num=0)
# T21 = test_sphere_mcvec_belkin(heat_order=2, s_num=1)
# T22 = test_sphere_mcvec_belkin(heat_order=2, s_num=2)
# T23 = test_sphere_mcvec_belkin(heat_order=2, s_num=3)
#
# T30 = test_sphere_mcvec_belkin(heat_order=3, s_num=0)
# T31 = test_sphere_mcvec_belkin(heat_order=3, s_num=1)
# T32 = test_sphere_mcvec_belkin(heat_order=3, s_num=2)
# T33 = test_sphere_mcvec_belkin(heat_order=3, s_num=3)

# %%
# heat_order = 2
# s_num = 1
# name = f"sphere_mcvec_belkin_{heat_order:03d}_{s_num:03d}"
# data_path = _TEST_DIR_ + "/" + name + ".pkl"
# T = ConvergenceTestData.load(data_path)
# X = T.independent_var[2:]
# Y = T.normalized_L2_error[2:]
# plot_log_log_fit(
#     X,
#     Y,
#     Xlabel="X",
#     Ylabel="Y",
#     title="log-log fit",
#     show=True,
# )
