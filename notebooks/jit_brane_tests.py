# import sys
# sys.path.append("./")
#################################
# Belkin tests
# %%
from src.python.convergence_tests import SphereBelkinTest
from src.python.pretty_pictures import plot_log_log_fit

# import numpy as np

ST = SphereBelkinTest(
    test_dir="./output/sphere_belkin_tests",
    npz_dir="./output/half_edge_arrays",
    num_vertices=[
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
    ],
    load_half_edge_meshes=False,
    test_names=[
        "mean_curvature",
        "unit_normal",
        "lap_x",
        "lap_x_squared",
        "lap_exp_x_y",
    ],
)

# Td0 = ST.run_tests()

Td = ST.load_test_results()
for n_findiff in ST.tfindiff_order[:2]:
    for n_tstep in ST.time_step_num[:3]:
        s = ST.time_step[n_tstep]
        T = Td["mean_curvature"][(n_findiff, n_tstep)]
        X = T.independent_var[-3:]
        Y = T.normalized_L2_error[-3:]

        plot_log_log_fit(
            X,
            Y,
            Xlabel="X",
            Ylabel="Y",
            title=f"{n_findiff=}, {s=}\n {X=}",
            show=True,
        )

#################################
# Guckenberger tests
# %%
from src.python.convergence_tests import SphereMcvecGuckenbergerTest
from src.python.pretty_pictures import plot_log_log_fit
import numpy as np

STG = SphereMcvecGuckenbergerTest(
    test_dir="./output/sphere_mcvec_guckenberger_tests",
    npz_dir="./output/half_edge_arrays",
    num_vertices=[
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
    ],
    load_half_edge_meshes=False,
)
# TGd0 = STG.run_tests()
Td = ST.load_test_results()
for key in ST.test_keys:
    T = Td[key]
    n_findiff = key
    X = T.independent_var
    Y = T.normalized_L2_error
    plot_log_log_fit(
        X,
        Y,
        Xlabel="X",
        Ylabel="Y",
        title=f"{n_findiff=}",
        show=True,
    )

#################################
# OLD Belkin tests
# %%
from src.python.convergence_tests import SphereMcvecBelkinTest
from src.python.pretty_pictures import plot_log_log_fit
import numpy as np


ST = SphereMcvecBelkinTest(
    test_dir="./output/sphere_mcvec_belkin_tests",
    npz_dir="./output/half_edge_arrays",
    num_vertices=[
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
    ],
    load_half_edge_meshes=False,
)
# Td0 = ST.run_tests()
Td = ST.load_test_results()

for n_findiff in ST.tfindiff_order[:1]:
    for n_tstep in ST.time_step_num[-2:-1]:
        s = ST.time_step[n_tstep]
        T = Td[(n_findiff, n_tstep)]
        X = T.independent_var[:-2]
        Y = T.normalized_L2_error[:-2]

        plot_log_log_fit(
            X,
            Y,
            Xlabel="X",
            Ylabel="Y",
            title=f"{n_findiff=}, {s=}\n{X=}",
            show=True,
        )


#################################
# Cotan tests
# %%
from src.python.convergence_tests import SphereCotanTest
from src.python.pretty_pictures import plot_log_log_fit

import numpy as np

ST = SphereCotanTest(
    test_dir="./output/sphere_cotan_tests",
    npz_dir="./output/half_edge_arrays",
    num_vertices=[
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
    ],
    load_half_edge_meshes=True,
    test_names=[
        "mean_curvature",
        "unit_normal",
        "lap_x",
        "lap_x_squared",
        "lap_exp_x_y",
    ],
)

# Td0 = ST.run_tests(overwrite=True)


Td = ST.load_test_results()
T = Td["lap_x"][0]
X = T.independent_var
# Y = T.normalized_L2_error
Y = T.compute_absolut_Lp_error(np.inf)
dir(T)
plot_log_log_fit(
    X,
    Y,
    Xlabel="X",
    Ylabel="Y",
    title=f"cotan\n {X=}",
    show=True,
)
