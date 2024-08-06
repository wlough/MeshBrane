import sys
import os

# sys.path.append("../src/python")
sys.path.append("./")

import pickle
from src.python.half_edge_test import HalfEdgeTestSphere as TestSurf

# [12, 42, 162, 642, 2562, 10242, 40962, 163842, 655362, 2621442]
# see scratch2.py
# check run_analyticdiffextrap_laplacian_mcvec_fixed_heat_param_test
# %%
_TEST_DIR_ = "./output/sphere_tests"
_NUM_VERTS_ = [
    # 162,
    642,
    2562,
    10242,
    40962,
    163842,
]  # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
_SURF_NAMES_ = [f"sphere_{N:06d}_he" for N in _NUM_VERTS_]
_SURF_PARAMS_ = [1.0]


def run_mcvec_tests(
    run_name="mcvec0",
    test_dir=_TEST_DIR_,
    surf_names=_SURF_NAMES_,
    surface_params=_SURF_PARAMS_,
    overwrite=False,
):
    # fixed_heat_param_vals = [0.0025, 0.01, 0.025]
    fixed_heat_param_vals = [10**-p for p in range(1, 6)]
    output_dir = f"{test_dir}/{run_name}"
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError("Ahhhhhh")
    os.system(f"mkdir -p {output_dir}")
    M = []
    for _ in range(len(surf_names)):
        surf = surf_names[_]
        print("\n" + surf + "\n" + len(surf) * "-")
        data_path = f"{output_dir}/{surf}"
        ply = f"./data/ply/binary/{surf}.ply"
        m = TestSurf.from_half_edge_ply(ply, *surface_params)
        print("run_belkin_laplacian_mcvec_fixed_heat_param_test")
        m.run_belkin_laplacian_mcvec_fixed_heat_param_test(fixed_heat_param_vals)
        # print("run_belkin_laplacian_mcvec_fixed_heat_param_test")
        # m.run_analyticdiffextrap_laplacian_mcvec_fixed_heat_param_test(fixed_heat_param_vals)

        print("run_belkin_laplacian_mcvec_average_face_area_test")
        m.run_belkin_laplacian_mcvec_average_face_area_test()
        print("run_cotan_laplacian_mcvec_test")
        m.run_cotan_laplacian_mcvec_test()
        print("run_guckenberger_laplacian_mcvec_test")
        m.run_guckenberger_laplacian_mcvec_test()
        m.save(data_path)
        M.append(m)
    print("Done")


def load_test_surfs(
    run_name="mcvec0",
    test_dir=_TEST_DIR_,
    surf_names=_SURF_NAMES_,
):
    output_dir = f"{test_dir}/{run_name}"
    M = []
    for surf in surf_names:
        data_path = f"{output_dir}/{surf}"
        with open(data_path + ".pickle", "rb") as f:
            M.append(pickle.load(f))
    return M


# %%
# noise_scales = [0.0]
# for n, noise_scale in enumerate(noise_scales):
# run_mcvec_tests(run_name="mcvec", overwrite=True)

# results = get_mcvec_test_results(run_name="mcvec0")
