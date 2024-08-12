from src.python.mesh_viewer import MeshViewer
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.python.utilities import round_to, log_log_fit
from src.python.half_edge_test import (
    get_plt_combos,
    scalars_to_rgba,
    to_scinotation_tex,
)
from src.python.half_edge_test import HalfEdgeTestSphere as TestSurf

# [12, 42, 162, 642, 2562, 10242, 40962, 163842, 655362, 2621442]
# see scratch2.py
# check run_analyticdiffextrap_laplacian_mcvec_fixed_heat_param_test
# %%
_TEST_DIR_ = "./output/jit_sphere_tests"
_NUM_VERTS_ = [
    # 162,
    642,
    2562,
    10242,
    # 40962,
    # 163842,
]  # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
_SURF_NAMES_ = [f"unit_sphere_{N:07d}" for N in _NUM_VERTS_]
_SURF_PARAMS_ = [1.0]
_HE_DATA_PATHS_ = [
    f"./data/half_edge_arrays/unit_sphere_{N:07d}.npz" for N in _NUM_VERTS_
]


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
        # ply = f"./data/ply/binary/{surf}.ply"
        path_npz = f"./data/half_edge_arrays/{surf}.npz"
        m = TestSurf.from_data_arrays(path_npz, *surface_params)
        # m = TestSurf.from_half_edge_ply(ply, *surface_params)
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


def get_mcvec_test_results(run_name="mcvec0"):
    """
    For tests with multiple param values, the first index corresponds to the
    parameter value and the second corresponds to the number of vertices.
    """
    test_results = dict()
    M = load_test_surfs(run_name=run_name)
    test_results["M"] = M
    # test_results["noise_scale"] = M[0].cotan_laplacian_mcvec_results["noise_scale"]
    test_results["num_vertices"] = np.array([m.num_vertices for m in M])
    test_results["mcvec_actual"] = [m.mcvec_actual for m in M]
    #########
    # Cotan #
    #########
    test_results["cotan_mcvec"] = [m.cotan_laplacian_mcvec_results["mcvec"] for m in M]
    test_results["cotan_L2error"] = np.array(
        [m.cotan_laplacian_mcvec_results["L2error"] for m in M]
    )
    test_results["cotan_Linftyerror"] = np.array(
        [m.cotan_laplacian_mcvec_results["Linftyerror"] for m in M]
    )
    #########################################################
    # Guckenberger with local heat_param=Meyer's mixed area #
    #########################################################
    test_results["guckenberger_mcvec"] = [
        m.guckenberger_laplacian_mcvec_results["mcvec"] for m in M
    ]
    test_results["guckenberger_L2error"] = np.array(
        [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M]
    )
    test_results["guckenberger_Linftyerror"] = np.array(
        [m.guckenberger_laplacian_mcvec_results["Linftyerror"] for m in M]
    )
    #######################################################
    # Belkin with fixed values for heat_param=[s0,s1,...] #
    #######################################################
    num_heat_param = len(
        M[0].belkin_laplacian_mcvec_fixed_heat_param_results["heat_param"]
    )
    test_results["belkin_fixed_mcvec"] = [
        [m.belkin_laplacian_mcvec_fixed_heat_param_results["mcvec"][n_param] for m in M]
        for n_param in range(num_heat_param)
    ]
    test_results["belkin_fixed_heat_param"] = np.array(
        [
            M[0].belkin_laplacian_mcvec_fixed_heat_param_results["heat_param"][n_param]
            for n_param in range(num_heat_param)
        ]
    )
    test_results["belkin_fixed_L2error"] = np.array(
        [
            [
                m.belkin_laplacian_mcvec_fixed_heat_param_results["L2error"][n_param]
                for m in M
            ]
            for n_param in range(num_heat_param)
        ]
    )
    test_results["belkin_fixed_Linftyerror"] = np.array(
        [
            [
                m.belkin_laplacian_mcvec_fixed_heat_param_results["Linftyerror"][
                    n_param
                ]
                for m in M
            ]
            for n_param in range(num_heat_param)
        ]
    )
    ################################################
    # Belkin with heat_param=[sqrt(Af), Af, Af**2] #
    # where Af=average face area                   #
    ################################################
    num_heat_param = len(
        M[0].belkin_laplacian_mcvec_average_face_area_results["heat_param"]
    )
    test_results["belkin_afe_mcvec"] = [
        [
            m.belkin_laplacian_mcvec_average_face_area_results["mcvec"][n_param]
            for m in M
        ]
        for n_param in range(num_heat_param)
    ]
    test_results["belkin_afe_heat_param"] = np.array(
        [
            [
                m.belkin_laplacian_mcvec_average_face_area_results["heat_param"][
                    n_param
                ]
                for m in M
            ]
            for n_param in range(num_heat_param)
        ]
    )
    test_results["belkin_afe_L2error"] = np.array(
        [
            [
                m.belkin_laplacian_mcvec_average_face_area_results["L2error"][n_param]
                for m in M
            ]
            for n_param in range(num_heat_param)
        ]
    )
    test_results["belkin_afe_Linftyerror"] = np.array(
        [
            [
                m.belkin_laplacian_mcvec_average_face_area_results["Linftyerror"][
                    n_param
                ]
                for m in M
            ]
            for n_param in range(num_heat_param)
        ]
    )

    return test_results


def single_plot_L2_fit(
    results,
    belkin_fixed=True,
    belkin_afe=True,
    cotan=True,
    guckenberger=True,
    Xlabel="",
    Ylabel="",
    suptitle="",
    # num_v = np.array(range(len(_NUM_VERTS_)))
):
    num_vertices = results["num_vertices"]
    cotan_error = results["cotan_L2error"]
    guckenberger_error = results["guckenberger_L2error"]
    belkin_fixed_heat_param = results["belkin_fixed_heat_param"]
    belkin_fixed_error = results["belkin_fixed_L2error"]
    belkin_afe_heat_param = results["belkin_afe_heat_param"]
    belkin_afe_error = results["belkin_afe_L2error"]

    N_fit = 0
    if belkin_fixed:
        N_fit += len(belkin_fixed_error)
    if belkin_afe:
        N_fit += len(belkin_afe_error)
    if cotan:
        N_fit += len(cotan_error)
    if guckenberger:
        N_fit += len(guckenberger_error)
    color_marker_linestyle = get_plt_combos(N_fit)

    textsize = 14
    plt.rcParams.update(
        {
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            "xtick.labeltop": False,
            "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "text.usetex": False,
            "figure.titlesize": 18,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            "lines.linewidth": 3,
            "lines.markersize": 14,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.loc": "best",
        }
    )
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(r"$" + f"{suptitle}" + r"$")
    ax = fig.add_subplot(1, 1, 1)

    ymin = np.inf
    ymax = -np.inf
    errmin = np.inf
    errmax = -np.inf
    n_fit = 0

    if belkin_fixed:
        error = belkin_fixed_error
        parameter = belkin_fixed_heat_param
        param_tex = to_scinotation_tex(parameter, decimals=1, mode="plain")
        for _ in range(len(error)):
            param = parameter[_]
            err = error[_]
            color, marker, linestyle = color_marker_linestyle[_]
            fit_dict = log_log_fit(num_vertices, err)
            m = fit_dict["m"]
            y_fit = fit_dict["F"]
            y = fit_dict["logY"]
            x = fit_dict["logX"]
            fit_label = (
                r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
            )
            sample_label = r"$s_B=" + param_tex[_] + r"$"
            ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
            ax.plot(x, y, marker, color=color, label=sample_label)

            ymin = np.min([ymin, *y])
            ymax = np.max([ymax, *y])
            errmin = np.min([errmin, *err])
            errmax = np.max([errmax, *err])
            n_fit += 1

    if belkin_afe:
        error = belkin_afe_error
        parameter = belkin_afe_heat_param
        sample_pow = [r"a^{1/2}", r"a", r"a^2"]

        sample_labels = []
        for n_pow in range(3):
            a_pow = sample_pow[n_pow]
            valsp = parameter[n_pow]
            sample_label = r" $s_{afe}=" + a_pow + r"=("
            sample_tex = to_scinotation_tex(valsp, decimals=1, mode="plain")
            for tex in sample_tex:
                # sample_label += f"{round_to(param, n=3)}, "
                sample_label += tex + r", "
            sample_label = sample_label[:-2]
            sample_label += r")$"
            sample_labels.append(sample_label)

        for _ in range(len(error)):
            # param = parameter[_]
            err = error[_]
            color, marker, linestyle = color_marker_linestyle[_]
            fit_dict = log_log_fit(num_vertices, err)
            m = fit_dict["m"]
            y_fit = fit_dict["F"]
            y = fit_dict["logY"]
            x = fit_dict["logX"]
            fit_label = (
                r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
            )
            sample_label = sample_labels[_]
            ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
            ax.plot(x, y, marker, color=color, label=sample_label)

            ymin = np.min([ymin, *y])
            ymax = np.max([ymax, *y])
            errmin = np.min([errmin, *err])
            errmax = np.max([errmax, *err])
            n_fit += 1

    if guckenberger:
        err = guckenberger_error

        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]
        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
        sample_label = r"$s_G=a_x$"
        ax.plot(x, y, marker, color=color, label=sample_label)

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])
        n_fit += 1

    if cotan:
        err = cotan_error
        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])

        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        sample_label = "cotan"
        ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
        ax.plot(x, y, marker, color=color, label=sample_label)
        n_fit += 1

    ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_vertices))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")
    yminmax = np.linspace(ymin, ymax, 4)
    errminmax = np.exp(yminmax)
    ax.set_yticks(yminmax)
    ax.set_yticklabels(to_scinotation_tex(errminmax, decimals=2))

    ax.legend()
    plt.tight_layout()
    plt.show()


def single_plot_Linfty_fit(
    results,
    belkin_fixed=True,
    belkin_afe=True,
    cotan=True,
    guckenberger=True,
    Xlabel="",
    Ylabel="",
    suptitle="",
    # num_v = np.array(range(len(_NUM_VERTS_)))
):
    num_vertices = results["num_vertices"]
    cotan_error = results["cotan_Linftyerror"]
    guckenberger_error = results["guckenberger_Linftyerror"]
    belkin_fixed_heat_param = results["belkin_fixed_heat_param"]
    belkin_fixed_error = results["belkin_fixed_Linftyerror"]
    belkin_afe_heat_param = results["belkin_afe_heat_param"]
    belkin_afe_error = results["belkin_afe_Linftyerror"]

    N_fit = 0
    if belkin_fixed:
        N_fit += len(belkin_fixed_error)
    if belkin_afe:
        N_fit += len(belkin_afe_error)
    if cotan:
        N_fit += len(cotan_error)
    if guckenberger:
        N_fit += len(guckenberger_error)
    color_marker_linestyle = get_plt_combos(N_fit)

    textsize = 14
    plt.rcParams.update(
        {
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            "xtick.labeltop": False,
            "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "text.usetex": False,
            "figure.titlesize": 18,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            "lines.linewidth": 3,
            "lines.markersize": 14,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.loc": "best",
        }
    )
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(r"$" + f"{suptitle}" + r"$")
    ax = fig.add_subplot(1, 1, 1)

    ymin = np.inf
    ymax = -np.inf
    errmin = np.inf
    errmax = -np.inf
    n_fit = 0

    if belkin_fixed:
        error = belkin_fixed_error
        parameter = belkin_fixed_heat_param
        param_tex = to_scinotation_tex(parameter, decimals=1, mode="plain")
        for _ in range(len(error)):
            # param = parameter[_]
            err = error[_]
            color, marker, linestyle = color_marker_linestyle[_]
            fit_dict = log_log_fit(num_vertices, err)
            m = fit_dict["m"]
            y_fit = fit_dict["F"]
            y = fit_dict["logY"]
            x = fit_dict["logX"]
            fit_label = (
                r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
            )
            sample_label = r"$s_B=" + param_tex[_] + r"$"
            ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
            ax.plot(x, y, marker, color=color, label=sample_label)

            ymin = np.min([ymin, *y])
            ymax = np.max([ymax, *y])
            errmin = np.min([errmin, *err])
            errmax = np.max([errmax, *err])
            n_fit += 1

    if belkin_afe:
        error = belkin_afe_error
        parameter = belkin_afe_heat_param
        sample_pow = [r"a^{1/2}", r"a", r"a^2"]

        sample_labels = []
        for n_pow in range(3):
            a_pow = sample_pow[n_pow]
            valsp = parameter[n_pow]
            sample_label = r" $s_{afe}=" + a_pow + r"=("
            sample_tex = to_scinotation_tex(valsp, decimals=1, mode="plain")
            for tex in sample_tex:
                # sample_label += f"{round_to(param, n=3)}, "
                sample_label += tex + r", "
            sample_label = sample_label[:-2]
            sample_label += r")$"
            sample_labels.append(sample_label)

        for _ in range(len(error)):
            # param = parameter[_]
            err = error[_]
            color, marker, linestyle = color_marker_linestyle[_]
            fit_dict = log_log_fit(num_vertices, err)
            m = fit_dict["m"]
            y_fit = fit_dict["F"]
            y = fit_dict["logY"]
            x = fit_dict["logX"]
            fit_label = (
                r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
            )
            sample_label = sample_labels[_]
            ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
            ax.plot(x, y, marker, color=color, label=sample_label)

            ymin = np.min([ymin, *y])
            ymax = np.max([ymax, *y])
            errmin = np.min([errmin, *err])
            errmax = np.max([errmax, *err])
            n_fit += 1

    if guckenberger:
        err = guckenberger_error

        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]
        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
        sample_label = r"$s_G=a_x$"
        ax.plot(x, y, marker, color=color, label=sample_label)

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])
        n_fit += 1

    if cotan:
        err = cotan_error
        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])

        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        sample_label = "cotan"
        ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
        ax.plot(x, y, marker, color=color, label=sample_label)
        n_fit += 1

    ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_vertices))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")
    yminmax = np.linspace(ymin, ymax, 4)
    errminmax = np.exp(yminmax)
    ax.set_yticks(yminmax)
    ax.set_yticklabels(to_scinotation_tex(errminmax, decimals=2))

    ax.legend()
    plt.tight_layout()
    plt.show()


def surf_error3d(
    results,
    lap_method="cotan",
    num_v=0,
    num_fixed_s=0,
    num_afe_s=0,
    scale_vec=2.0,
    alpha=0.4,
):
    m = results["M"][num_v]
    mcvec_actual = results["mcvec_actual"][num_v]
    # num_vertices = results["num_vertices"][num_v]
    # cotan_error = results["cotan_L2error"][num_v]
    cotan_mcvec = results["cotan_mcvec"][num_v]

    # guckenberger_error = results["guckenberger_L2error"][num_v]
    guckenberger_mcvec = results["guckenberger_mcvec"][num_v]

    # belkin_fixed_heat_param = results["belkin_fixed_heat_param"][num_fixed_s]
    # belkin_fixed_error = results["belkin_fixed_L2error"][num_fixed_s][num_v]
    belkin_fixed_mcvec = results["belkin_fixed_mcvec"][num_fixed_s][num_v]

    # belkin_afe_heat_param = results["belkin_afe_heat_param"][num_afe_s][num_v]
    # belkin_afe_error = results["belkin_afe_L2error"][num_afe_s][num_v]
    belkin_afe_mcvec = results["belkin_afe_mcvec"][num_afe_s][num_v]

    if lap_method == "cotan":
        mcvec = cotan_mcvec
    if lap_method == "belkin_fixed":
        mcvec = belkin_fixed_mcvec
    if lap_method == "belkin_afe":
        mcvec = belkin_afe_mcvec
    if lap_method == "guckenberger":
        mcvec = guckenberger_mcvec

    local_error = np.linalg.norm(mcvec - mcvec_actual, axis=-1)
    V_rgba = scalars_to_rgba(local_error)
    # V_rgba = scalars_to_rgba([-m.valence_v(v) for v in m.Vkeys])
    V_rgba[:, -1] = alpha
    E_rgba = np.zeros((len(m._v_origin_H), 4))

    vfdat = [m.xyz_array, scale_vec * (mcvec - mcvec_actual)]
    mv_kwargs = {
        "vector_field_data": [vfdat],
        "V_rgba": V_rgba,
        "color_by_V_rgba": True,
        "E_rgba": E_rgba,
    }
    mv = MeshViewer(*m.data_lists, **mv_kwargs)
    mv.plot()


def surf_mcvec3d(
    results,
    lap_method="cotan",
    num_v=0,
    num_fixed_s=0,
    num_afe_s=0,
    scale_vec=1.0,
    alpha=0.4,
):
    m = results["M"][num_v]
    mcvec_actual = results["mcvec_actual"][num_v]
    # num_vertices = results["num_vertices"][num_v]
    # cotan_error = results["cotan_L2error"][num_v]
    cotan_mcvec = results["cotan_mcvec"][num_v]

    # guckenberger_error = results["guckenberger_L2error"][num_v]
    guckenberger_mcvec = results["guckenberger_mcvec"][num_v]

    # belkin_fixed_heat_param = results["belkin_fixed_heat_param"][num_fixed_s]
    # belkin_fixed_error = results["belkin_fixed_L2error"][num_fixed_s][num_v]
    belkin_fixed_mcvec = results["belkin_fixed_mcvec"][num_fixed_s][num_v]

    # belkin_afe_heat_param = results["belkin_afe_heat_param"][num_afe_s][num_v]
    # belkin_afe_error = results["belkin_afe_L2error"][num_afe_s][num_v]
    belkin_afe_mcvec = results["belkin_afe_mcvec"][num_afe_s][num_v]

    if lap_method == "cotan":
        mcvec = cotan_mcvec
    if lap_method == "belkin_fixed":
        mcvec = belkin_fixed_mcvec
    if lap_method == "belkin_afe":
        mcvec = belkin_afe_mcvec
    if lap_method == "guckenberger":
        mcvec = guckenberger_mcvec

    local_error = np.linalg.norm(mcvec - mcvec_actual, axis=-1)
    V_rgba = scalars_to_rgba(local_error)
    # V_rgba = scalars_to_rgba([-m.valence_v(v) for v in m.Vkeys])
    V_rgba[:, -1] = alpha
    E_rgba = np.zeros((len(m._v_origin_H), 4))

    vfdat = [m.xyz_array, scale_vec * mcvec]
    mv_kwargs = {
        "vector_field_data": [vfdat],
        "V_rgba": V_rgba,
        "color_by_V_rgba": True,
        "E_rgba": E_rgba,
    }
    mv = MeshViewer(*m.data_lists, **mv_kwargs)
    mv.plot()


# %%
# noise_scales = [0.0]
# for n, noise_scale in enumerate(noise_scales):
# run_mcvec_tests(run_name="mcvec0", overwrite=True)

results = get_mcvec_test_results(run_name="mcvec")

# mv = MeshViewer(*m.data_lists)
# mv.plot()
# %%

# %%
surf_error3d(
    results,
    lap_method="belkin_fixed",
    num_v=0,
    num_fixed_s=0,
    num_afe_s=0,
    scale_vec=-1.0,
    alpha=0.4,
)
# %%
surf_mcvec3d(
    results,
    lap_method="belkin_fixed",
    num_v=1,
    num_fixed_s=0,
    num_afe_s=0,
    scale_vec=-0.12,
    alpha=0.4,
)
# %%
plt_kwargs = {
    "Xlabel": r"N",
    "Ylabel": r"\varepsilon",
    "suptitle": r"{\bf{r}}(x),\quad"
    + r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{2}}{||\Delta {\bf{r}}||_{2}}",
}
single_plot_L2_fit(
    results,
    belkin_fixed=False,
    belkin_afe=True,
    cotan=True,
    guckenberger=False,
    **plt_kwargs,
)

# %%
plt_kwargs = {
    "Xlabel": r"N",
    "Ylabel": r"\varepsilon",
    "suptitle": r"{\bf{r}}(x),\quad"
    + r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{\infty}}{||\Delta {\bf{r}}||_{\infty}}",
}
single_plot_Linfty_fit(
    results,
    belkin_fixed=True,
    belkin_afe=False,
    cotan=True,
    guckenberger=False,
    **plt_kwargs,
)


# %%
def make_noisy_sphere_data(
    output_dir="./output/noisy_sphere_tests", overwrite=False, loc=0, scale=0.01
):
    # heat_param_vals = [0.0025, 0.01, 0.025]
    u = 10
    heat_param_vals = [u**-p for p in range(1, 6)]
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    # output_dir = "./output/noisy_sphere_tests"
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError("Ahhhhhh")
    os.system(f"mkdir -p {output_dir}")
    M = []
    for _ in range(len(surfs)):
        surf = surfs[_]
        print(f"running tests for {surf}")
        data_path = f"{output_dir}/{surf}"
        ply = f"./data/ply/binary/{surf}.ply"
        m = HalfEdgeTestSphere.from_half_edge_ply(ply, 1.0)

        m.run_noisy_belkin_laplacian_mcvec_fixed_param_test(
            heat_param_vals, loc=loc, scale=scale
        )
        m.run_noisy_belkin_laplacian_mcvec_average_face_area_test(loc=loc, scale=scale)
        m.run_noisy_cotan_laplacian_mcvec_test(loc=loc, scale=scale)
        m.run_noisy_guckenberger_laplacian_mcvec_test(loc=loc, scale=scale)
        m.save(data_path)
        M.append(m)
    return M


def load_spheres(output_dir="./output/sphere_tests3"):
    Nverts = [12, 42, 162, 642, 2562, 10242]
    # Nverts = [12, 42, 162, 642]
    surfs = [f"unit_sphere_{N:05d}" for N in Nverts]
    # output_dir = "./output/sphere_tests3"
    M = []
    for surf in surfs:
        data_path = f"{output_dir}/{surf}"
        with open(data_path + ".pickle", "rb") as f:
            M.append(pickle.load(f))
    return M


def get_test_data(with_M=False, output_dir="./output/sphere_tests3"):
    M = load_spheres(output_dir=output_dir)
    # belkin_laplacian_mcvec_fixed_param_results
    # cotan_laplacian_mcvec_results
    timelike_param = np.array(
        [m.belkin_laplacian_mcvec_fixed_param_results["s"] for m in M][0]
    )
    mcvec_actual = [m.mcvec_actual for m in M]

    mcvec_cotan = [m.cotan_laplacian_mcvec_results["mcvec"] for m in M]
    mcvec_cotan_L2error = np.array(
        [m.cotan_laplacian_mcvec_results["L2error"] for m in M]
    )
    mcvec_cotan_Linftyerror = np.array(
        [m.cotan_laplacian_mcvec_results["Linftyerror"] for m in M]
    )

    # _mcvec_belkin = [m.mcvec_belkin for m in M]
    # _mcvec_belkin_L2error = [m.mcvec_belkin_L2error for m in M]
    # _mcvec_belkin_Linftyerror = [m.mcvec_belkin_Linftyerror for m in M]
    #
    # num_M = len(M)
    num_timelike = len(timelike_param)
    mcvec_belkin = [
        [m.belkin_laplacian_mcvec_fixed_param_results["mcvec"][_] for m in M]
        for _ in range(num_timelike)
    ]
    mcvec_belkin_L2error = np.array(
        [
            [m.belkin_laplacian_mcvec_fixed_param_results["L2error"][_] for m in M]
            for _ in range(num_timelike)
        ]
    )
    mcvec_belkin_Linftyerror = np.array(
        [
            [m.belkin_laplacian_mcvec_fixed_param_results["Linftyerror"][_] for m in M]
            for _ in range(num_timelike)
        ]
    )

    mcvec_belkin_afe = [
        m.belkin_laplacian_mcvec_average_face_area_results["mcvec"] for m in M
    ]
    mcvec_belkin_afe_L2error = np.array(
        [m.belkin_laplacian_mcvec_average_face_area_results["L2error"] for m in M]
    )
    mcvec_belkin_afe_Linftyerror = np.array(
        [m.belkin_laplacian_mcvec_average_face_area_results["Linftyerror"] for m in M]
    )

    # error_belkin_afe = run_noisy_belkin_laplacian_mcvec_average_face_area_test

    num_vertices = np.array([m.num_vertices for m in M])
    if with_M:
        return {
            "timelike_param": timelike_param,
            "mcvec_actual": mcvec_actual,
            "mcvec_cotan": mcvec_cotan,
            "mcvec_cotan_L2error": mcvec_cotan_L2error,
            "mcvec_cotan_Linftyerror": mcvec_cotan_Linftyerror,
            "mcvec_belkin": mcvec_belkin,
            "mcvec_belkin_L2error": mcvec_belkin_L2error,
            "mcvec_belkin_Linftyerror": mcvec_belkin_Linftyerror,
            "mcvec_belkin_afe": mcvec_belkin_afe,
            "mcvec_belkin_afe_L2error": mcvec_belkin_afe_L2error,
            "mcvec_belkin_afe_Linftyerror": mcvec_belkin_afe_Linftyerror,
            "N_vertices": num_vertices,
            "M": M,
        }
    else:
        return {
            "timelike_param": timelike_param,
            "mcvec_actual": mcvec_actual,
            "mcvec_cotan": mcvec_cotan,
            "mcvec_cotan_L2error": mcvec_cotan_L2error,
            "mcvec_cotan_Linftyerror": mcvec_cotan_Linftyerror,
            "mcvec_belkin": mcvec_belkin,
            "mcvec_belkin_L2error": mcvec_belkin_L2error,
            "mcvec_belkin_Linftyerror": mcvec_belkin_Linftyerror,
            "N_vertices": num_vertices,
        }


def fig_single(
    # num_vertices,
    # error_belkin,
    # error_cot,
    # error_guckenberger,
    # parameter_belkin,
    # parameter_guckenberger,
    # test_data,
    # suptitle="",
    # Xlabel="",
    # Ylabel="",
    # legendlabel="",
    **kwargs,
):
    N_fit = 0
    N_fit += len(error_belkin)  # belkin
    N_fit += 1  # cot
    N_fit += 1  # guckenberger
    # N_fit += 1
    color_marker_linestyle = get_plt_combos(len(N_fit))

    textsize = 14
    plt.rcParams.update(
        {
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            "xtick.labeltop": False,
            "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "text.usetex": False,
            "figure.titlesize": 18,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            "lines.linewidth": 3,
            "lines.markersize": 14,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.loc": "best",
        }
    )
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(r"$" + f"{suptitle}" + r"$")
    ax = fig.add_subplot(1, 1, 1)

    ymin = np.inf
    ymax = -np.inf
    errmin = np.inf
    errmax = -np.inf
    n_fit = 0

    if True:
        error = error_belkin
        parameter = parameter_belkin
        for _ in range(len(error)):
            param = parameter[_]
            err = error[_]
            color, marker, linestyle = color_marker_linestyle[_]
            fit_dict = log_log_fit(num_vertices, err)
            m = fit_dict["m"]
            y_fit = fit_dict["F"]
            y = fit_dict["logY"]
            x = fit_dict["logX"]
            fit_label = (
                r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
            )
            ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
            ax.plot(x, y, marker, color=color, label=r"$s_B=" + f"{param}" + r"$")

            ymin = np.min([ymin, *y])
            ymax = np.max([ymax, *y])
            errmin = np.min([errmin, *err])
            errmax = np.max([errmax, *err])
            n_fit += 1

    if True:
        err = error_guckenberger
        parameter = parameter_guckenberger
        # for _ in range(n_fit+len(error)):

        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]
        fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
        ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
        sample_label = r" $s_G=("
        for param in parameter:
            sample_label += f"{round_to(param, n=3)}, "
        sample_label = sample_label[:-2]
        sample_label += r")$"
        ax.plot(x, y, marker, color=color, label=sample_label)

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])
        n_fit += 1

    ax.set_xlabel(r"$\text{Vertex count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_vertices))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")

    if True:
        err = error_cot
        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_vertices, err)
        m = fit_dict["m"]
        y_fit = fit_dict["F"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]

        ymin = np.min([ymin, *y])
        ymax = np.max([ymax, *y])
        errmin = np.min([errmin, *err])
        errmax = np.max([errmax, *err])

    # errminmax = np.linspace(errmin, errmax, 4)
    yminmax = np.linspace(ymin, ymax, 4)
    errminmax = np.exp(yminmax)
    ax.set_yticks(yminmax)
    ax.set_yticklabels(to_scinotation_tex(errminmax, decimals=2))
    # ax.set_yticks([ymin, ymax])
    # ax.set_yticklabels(to_scinotation_tex([errmin, errmax]))

    fit_label = r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"

    ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
    ax.plot(x, y, marker, color=color, label="cotan")
    n_fit += 1

    ax.legend()
    plt.tight_layout()
    plt.show()


# M = make_sphere_data(overwrite=True)
# test_data = get_test_data(with_M=True)
output_dir = "./output/noisy_sphere_tests"
# M = make_noisy_sphere_data(output_dir=output_dir, overwrite=True, loc=0, scale=0.01)
# output_dir = "./output/noisy_sphere_tests"
# output_dir = "./output/noisy_tests"
test_data = get_test_data(with_M=True, output_dir=output_dir)

timelike_param = test_data["timelike_param"]
mcvec_actual = test_data["mcvec_actual"]
mcvec_cotan = test_data["mcvec_cotan"]
mcvec_cotan_L2error = test_data["mcvec_cotan_L2error"]
mcvec_cotan_Linftyerror = test_data["mcvec_cotan_Linftyerror"]
mcvec_belkin = test_data["mcvec_belkin"]
mcvec_belkin_L2error = test_data["mcvec_belkin_L2error"]
mcvec_belkin_Linftyerror = test_data["mcvec_belkin_Linftyerror"]
mcvec_belkin_afe = test_data["mcvec_belkin_afe"]
mcvec_belkin_afe_L2error = test_data["mcvec_belkin_afe_L2error"]
mcvec_belkin_afe_Linftyerror = test_data["mcvec_belkin_afe_Linftyerror"]
N_vertices = test_data["N_vertices"]
M = test_data["M"]
# %%
###############################################################
# Normalized L2 error for spherical mesh.
###############################################################

fun = r"{\bf{r}}(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{2}}{||\Delta {\bf{r}}||_{2}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
n0 = 3

num_vertices = N_vertices[n0:]
error_belkin = mcvec_belkin_L2error[:, n0:]
error_belkin_afe = run_noisy_belkin_laplacian_mcvec_average_face_area_test
error_cot = np.array([*mcvec_cotan_L2error[n0:]])
error_guckenberger = [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M[n0:]]

parameter_belkin = timelike_param
parameter_guckenberger = [
    m.guckenberger_laplacian_mcvec_results["L2error"] for m in M[n0:]
]

# error_heat = np.array([*error_belkin, error_guckenberger])

# error_belkin[-1:] = [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M[n0:]]
# error_belkin = [*_error_belkin, [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M]]
plot_kwargs = {
    "num_vertices": num_vertices,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "error_guckenberger": error_guckenberger,
    "parameter_belkin": parameter_belkin,
    "parameter_guckenberger": parameter_guckenberger,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
fig_single(**plot_kwargs)

# %%
###############################################################
# Normalized Linf error for spherical mesh.
###############################################################
parameter = timelike_param
num_vertices = N_vertices[3:]

fun = r"{\bf{r}}(x)"
err_tex = r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{\infty}}{||\Delta {\bf{r}}||_{\infty}}"
Xlabel = r"N"
Ylabel = r"\varepsilon"
suptitle = f"{fun}" + r",\quad" + err_tex
error_cot = mcvec_cotan_Linftyerror[3:]
error_belkin = mcvec_belkin_Linftyerror[:, 3:]
plot_kwargs = {
    "num_vertices": num_vertices,
    "parameter": parameter,
    "error_belkin": error_belkin,
    "error_cot": error_cot,
    "suptitle": suptitle,
    "Xlabel": Xlabel,
    "Ylabel": Ylabel,
}
fig_single(**plot_kwargs)


#########################################################
# %%
