# from src.python.mesh_viewer import MeshViewer
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.python.utilities.misc_utils import round_to, log_log_fit
from src.python.half_edge_test import (
    get_plt_combos,
    scalars_to_rgba,
    to_scinotation_tex,
)
from src.python.convergence_tests import (
    SphereBelkinTest,
    SphereMcvecGuckenbergerTest,
    SphereCotanTest,
)


def load_tests(
    load_half_edge_meshes=False,
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
):
    #################################
    # Belkin tests
    #
    belkinST = SphereBelkinTest(
        test_dir="./output/sphere_belkin_tests",
        npz_dir="./output/half_edge_arrays",
        num_vertices=num_vertices,
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

    belkinTdict = belkinST.load_test_results()
    #################################
    # Guckenberger tests
    #
    guckenbergerST = SphereMcvecGuckenbergerTest(
        test_dir="./output/sphere_mcvec_guckenberger_tests",
        npz_dir="./output/half_edge_arrays",
        num_vertices=num_vertices,
        load_half_edge_meshes=False,
    )

    guckenbergerTdict = guckenbergerST.load_test_results()
    #################################
    # Cotan tests
    #
    cotanST = SphereCotanTest(
        test_dir="./output/sphere_cotan_tests",
        npz_dir="./output/half_edge_arrays",
        num_vertices=num_vertices,
        load_half_edge_meshes=load_half_edge_meshes,
        test_names=[
            "mean_curvature",
            "unit_normal",
            "lap_x",
            "lap_x_squared",
            "lap_exp_x_y",
        ],
    )

    # Td0 = ST.run_tests(overwrite=True)

    cotanTdict = cotanST.load_test_results()
    return belkinST, guckenbergerST, cotanST


def get_test_results0(
    test_name="mean_curvature",
    load_half_edge_meshes=False,
    N_diff_order=[1],
    N_snum=[0, 1, 2, 3, 4],
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
):
    """
    For tests with multiple param values, the first index corresponds to the
    parameter value and the second corresponds to the number of vertices.
    """
    # belkinST, guckenbergerST, cotanST = load_tests(
    #     load_half_edge_meshes=load_half_edge_meshes
    # )

    belkin_keys = [
        (n_diff_order, n_snum) for n_diff_order in N_diff_order for n_snum in N_snum
    ]
    guckenberger_keys = [n_diff_order for n_diff_order in N_diff_order]
    belkinST, guckenbergerST, cotanST = load_tests(
        load_half_edge_meshes=load_half_edge_meshes, num_vertices=num_vertices
    )

    test_results = dict()
    # M = load_test_surfs(run_name=run_name)
    if load_half_edge_meshes:
        M = cotanST.M
        test_results["M"] = M
    # test_results["noise_scale"] = M[0].cotan_laplacian_mcvec_results["noise_scale"]
    test_results["num_faces"] = cotanST.Tdict["mean_curvature"][0].independent_var[2:]
    test_results["mean_curvature_actual"] = cotanST.Tdict["mean_curvature"][
        0
    ].samples_actual[2:]
    test_results["unit_normal_actual"] = cotanST.Tdict["unit_normal"][0].samples_actual[
        2:
    ]
    test_results["lap_x_actual"] = cotanST.Tdict["lap_x"][0].samples_actual[2:]
    test_results["lap_x_squared_actual"] = cotanST.Tdict["lap_x_squared"][
        0
    ].samples_actual[2:]
    test_results["lap_exp_x_y_actual"] = cotanST.Tdict["lap_exp_x_y"][0].samples_actual[
        2:
    ]

    #
    #########
    # Cotan #
    #########
    cotanT = cotanST.Tdict[test_name][0]
    test_results["cotan_samples"] = cotanT.samples_numerical[2:]
    test_results["cotan_L2error"] = cotanT.normalized_L2_error[2:]
    test_results["cotan_Linftyerror"] = cotanT.Linfinity_error[2:]
    #########################################################
    # Guckenberger with local heat_param=Meyer's mixed area #
    #########################################################
    # test_results["samples"] = [m.guckenberger_laplacian_mcvec_results["mcvec"] for m in M]
    # test_results["guckenberger_L2error"] = np.array(
    #     [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M]
    # )
    # test_results["guckenberger_Linftyerror"] = np.array(
    #     [m.guckenberger_laplacian_mcvec_results["Linftyerror"] for m in M]
    # )
    guckenbergerTall = guckenbergerST.Tdict
    test_results["guckenberger_samples"] = [
        guckenbergerTall[key].samples_numerical[2:] for key in guckenberger_keys
    ]
    test_results["guckenberger_L2error"] = np.array(
        [guckenbergerTall[key].normalized_L2_error[2:] for key in guckenberger_keys]
    )
    test_results["guckenberger_Linftyerror"] = np.array(
        [guckenbergerTall[key].Linfinity_error[2:] for key in guckenberger_keys]
    )
    #######################################################
    # Belkin with fixed values for heat_param=[s0,s1,...] #
    #######################################################
    belkinTall = belkinST.Tdict[test_name]

    test_results["belkin_samples"] = [
        belkinTall[key].samples_numerical[2:] for key in belkin_keys
    ]
    test_results["belkin_fixed_heat_param"] = np.array(
        [belkinTall[key].params["s"] for key in belkin_keys]
    )
    # test_results["belkin_fixed_heat_param"] = np.array(
    #     [
    #         M[0].belkin_laplacian_mcvec_fixed_heat_param_results["heat_param"][n_param]
    #         for n_param in range(num_heat_param)
    #     ]
    # )
    test_results["belkin_fixed_L2error"] = np.array(
        [belkinTall[key].normalized_L2_error[2:] for key in belkin_keys]
    )
    test_results["belkin_fixed_Linftyerror"] = np.array(
        [belkinTall[key].Linfinity_error[2:] for key in belkin_keys]
    )

    return test_results


def get_test_results(
    test_name="mean_curvature",
    load_half_edge_meshes=False,
    N_diff_order=[1],
    N_snum=[0, 1, 2, 3, 4],
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
):
    """
    For tests with multiple param values, the first index corresponds to the
    parameter value and the second corresponds to the number of vertices.
    """
    # belkinST, guckenbergerST, cotanST = load_tests(
    #     load_half_edge_meshes=load_half_edge_meshes
    # )

    belkin_keys = [
        (n_diff_order, n_snum) for n_diff_order in N_diff_order for n_snum in N_snum
    ]
    guckenberger_keys = [n_diff_order for n_diff_order in N_diff_order]
    belkinST, guckenbergerST, cotanST = load_tests(
        load_half_edge_meshes=load_half_edge_meshes, num_vertices=num_vertices
    )

    test_results = dict()
    # M = load_test_surfs(run_name=run_name)
    if load_half_edge_meshes:
        M = cotanST.M
        test_results["M"] = M
    # test_results["noise_scale"] = M[0].cotan_laplacian_mcvec_results["noise_scale"]
    test_results["num_faces"] = cotanST.Tdict["mean_curvature"][0].independent_var
    test_results["mean_curvature_actual"] = cotanST.Tdict["mean_curvature"][
        0
    ].samples_actual
    test_results["unit_normal_actual"] = cotanST.Tdict["unit_normal"][0].samples_actual[
        2:
    ]
    test_results["lap_x_actual"] = cotanST.Tdict["lap_x"][0].samples_actual
    test_results["lap_x_squared_actual"] = cotanST.Tdict["lap_x_squared"][
        0
    ].samples_actual
    test_results["lap_exp_x_y_actual"] = cotanST.Tdict["lap_exp_x_y"][0].samples_actual[
        2:
    ]

    #
    #########
    # Cotan #
    #########
    cotanT = cotanST.Tdict[test_name][0]
    test_results["cotan_samples"] = cotanT.samples_numerical
    test_results["cotan_L2error"] = cotanT.normalized_L2_error
    test_results["cotan_Linftyerror"] = cotanT.Linfinity_error
    #########################################################
    # Guckenberger with local heat_param=Meyer's mixed area #
    #########################################################
    # test_results["samples"] = [m.guckenberger_laplacian_mcvec_results["mcvec"] for m in M]
    # test_results["guckenberger_L2error"] = np.array(
    #     [m.guckenberger_laplacian_mcvec_results["L2error"] for m in M]
    # )
    # test_results["guckenberger_Linftyerror"] = np.array(
    #     [m.guckenberger_laplacian_mcvec_results["Linftyerror"] for m in M]
    # )
    guckenbergerTall = guckenbergerST.Tdict
    test_results["guckenberger_samples"] = [
        guckenbergerTall[key].samples_numerical for key in guckenberger_keys
    ]
    test_results["guckenberger_L2error"] = np.array(
        [guckenbergerTall[key].normalized_L2_error for key in guckenberger_keys]
    )
    test_results["guckenberger_Linftyerror"] = np.array(
        [guckenbergerTall[key].Linfinity_error for key in guckenberger_keys]
    )
    #######################################################
    # Belkin with fixed values for heat_param=[s0,s1,...] #
    #######################################################
    belkinTall = belkinST.Tdict[test_name]

    test_results["belkin_samples"] = [
        belkinTall[key].samples_numerical for key in belkin_keys
    ]
    test_results["belkin_fixed_heat_param"] = np.array(
        [belkinTall[key].params["s"] for key in belkin_keys]
    )
    # test_results["belkin_fixed_heat_param"] = np.array(
    #     [
    #         M[0].belkin_laplacian_mcvec_fixed_heat_param_results["heat_param"][n_param]
    #         for n_param in range(num_heat_param)
    #     ]
    # )
    test_results["belkin_fixed_L2error"] = np.array(
        [belkinTall[key].normalized_L2_error for key in belkin_keys]
    )
    test_results["belkin_fixed_Linftyerror"] = np.array(
        [belkinTall[key].Linfinity_error for key in belkin_keys]
    )

    return test_results


def single_plot_L2_fit(
    results,
    belkin_fixed=True,
    cotan=True,
    guckenberger=True,
    Xlabel="",
    Ylabel="",
    suptitle="",
    fig_path="./output/fig.png",
    # num_v = np.array(range(len(_NUM_VERTS_)))
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
):
    num_vertices_all = [
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
    Nv_indices = [num_vertices_all.index(_) for _ in num_vertices]
    num_faces = np.array([results["num_faces"][_] for _ in Nv_indices])
    # cotan_error = np.array([results["cotan_L2error"][_] for _ in Nv_indices])
    # guckenberger_error = np.array(
    #     [results["guckenberger_L2error"][0][_] for _ in Nv_indices]
    # )
    # belkin_fixed_heat_param = results["belkin_fixed_heat_param"]
    #
    # belkin_fixed_error = np.array(
    #     [results["belkin_fixed_L2error"][:, _] for _ in Nv_indices]
    # ).T
    cotan_error = np.array([results["cotan_Linftyerror"][_] for _ in Nv_indices])
    guckenberger_error = np.array(
        [results["guckenberger_Linftyerror"][0][_] for _ in Nv_indices]
    )
    belkin_fixed_heat_param = results["belkin_fixed_heat_param"]

    belkin_fixed_error = np.array(
        [results["belkin_fixed_Linftyerror"][:, _] for _ in Nv_indices]
    ).T

    # Linftyerror
    #     return (num_faces,cotan_error,
    # guckenberger_error,
    # belkin_fixed_heat_param,
    # belkin_fixed_error,results["belkin_fixed_L2error"])
    # belkin_afe_heat_param = results["belkin_afe_heat_param"]
    # belkin_afe_error = results["belkin_afe_L2error"]

    N_fit = 0
    if belkin_fixed:
        N_fit += len(belkin_fixed_error)
    # if belkin_afe:
    #     N_fit += len(belkin_afe_error)
    if cotan:
        N_fit += len(cotan_error)
    if guckenberger:
        N_fit += 1  # len(guckenberger_error)
    color_marker_linestyle = get_plt_combos(N_fit)

    textsize = 16
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
            fit_dict = log_log_fit(num_faces, err)
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

    # if False:
    #     error = belkin_afe_error
    #     parameter = belkin_afe_heat_param
    #     sample_pow = [r"a^{1/2}", r"a", r"a^2"]
    #
    #     sample_labels = []
    #     for n_pow in range(3):
    #         a_pow = sample_pow[n_pow]
    #         valsp = parameter[n_pow]
    #         sample_label = r" $s_{afe}=" + a_pow + r"=("
    #         sample_tex = to_scinotation_tex(valsp, decimals=1, mode="plain")
    #         for tex in sample_tex:
    #             # sample_label += f"{round_to(param, n=3)}, "
    #             sample_label += tex + r", "
    #         sample_label = sample_label[:-2]
    #         sample_label += r")$"
    #         sample_labels.append(sample_label)
    #
    #     for _ in range(len(error)):
    #         # param = parameter[_]
    #         err = error[_]
    #         color, marker, linestyle = color_marker_linestyle[_]
    #         fit_dict = log_log_fit(num_vertices, err)
    #         m = fit_dict["m"]
    #         y_fit = fit_dict["F"]
    #         y = fit_dict["logY"]
    #         x = fit_dict["logX"]
    #         fit_label = (
    #             r"$O\left(" + Xlabel + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    #         )
    #         sample_label = sample_labels[_]
    #         ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
    #         ax.plot(x, y, marker, color=color, label=sample_label)
    #
    #         ymin = np.min([ymin, *y])
    #         ymax = np.max([ymax, *y])
    #         errmin = np.min([errmin, *err])
    #         errmax = np.max([errmax, *err])
    #         n_fit += 1
    #

    if guckenberger:
        err = guckenberger_error

        color, marker, linestyle = color_marker_linestyle[n_fit]
        fit_dict = log_log_fit(num_faces, err)
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
        fit_dict = log_log_fit(num_faces, err)
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

    ax.set_xlabel(r"$\text{Face count}\quad " + Xlabel + r"$")
    ax.set_xticks(x)
    ax.set_xticklabels(to_scinotation_tex(num_faces))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")
    yminmax = np.linspace(ymin, ymax, 4)
    errminmax = np.exp(yminmax)
    ax.set_yticks(yminmax)
    ax.set_yticklabels(to_scinotation_tex(errminmax, decimals=2))

    ax.legend()
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
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
    num_faces = results["num_faces"]
    cotan_error = results["cotan_Linftyerror"]
    # guckenberger_error = results["guckenberger_Linftyerror"]
    belkin_fixed_heat_param = results["belkin_fixed_heat_param"]
    belkin_fixed_error = results["belkin_fixed_Linftyerror"]
    # belkin_afe_heat_param = results["belkin_afe_heat_param"]
    # belkin_afe_error = results["belkin_afe_Linftyerror"]

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
            fit_dict = log_log_fit(num_faces, err)
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
            fit_dict = log_log_fit(num_faces, err)
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
        fit_dict = log_log_fit(num_faces, err)
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
        fit_dict = log_log_fit(num_faces, err)
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
    ax.set_xticklabels(to_scinotation_tex(num_faces))
    ax.set_ylabel(r"$\text{Error}\quad " + Ylabel + r"$")
    yminmax = np.linspace(ymin, ymax, 4)
    errminmax = np.exp(yminmax)
    ax.set_yticks(yminmax)
    ax.set_yticklabels(to_scinotation_tex(errminmax, decimals=2))

    ax.legend()
    plt.tight_layout()
    plt.show()


#
# noise_scales = [0.0]
# for n, noise_scale in enumerate(noise_scales):
# run_mcvec_tests(run_name="mcvec0", overwrite=True)

# dir(results)
# guckenbergerT = guckenbergerST.Tdict[test_name][0]
# test_results["guckenberger_samples"] = guckenbergerT.samples_numerical
# test_results["guckenberger_L2error"] = guckenbergerT.normalized_L2_error
# test_results["guckenberger_Linftyerror"] = guckenbergerT.Linfinity_error
# %%
p = 2
results = get_test_results(
    test_name="mean_curvature",
    load_half_edge_meshes=False,
    N_diff_order=[p],
    N_snum=[0, 1, 2, 3, 4],
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
)
plt_kwargs = {
    "Xlabel": r"N",
    "Ylabel": r"\varepsilon",
    "suptitle": r"{\bf{r}}(x),\quad"
    + r"\varepsilon=\frac{||\mathcal{D} {\bf{r}}-\Delta {\bf{r}}||_{2}}{||\Delta {\bf{r}}||_{2}}, \quad"
    + f" {p=}",
    "fig_path": f"./output/diff_order{p}.png",
    "num_vertices": [
        # 12,
        # 42,
        162,
        642,
        # 2562,
        # 10242,
        # 40962,
        # 163842,
        # 655362,
        # 2621442,
    ],
    "belkin_fixed": False,
    "guckenberger": True,
    "cotan": False,
}
single_plot_L2_fit(
    results,
    **plt_kwargs,
)
# (
#     num_faces,
#     cotan_error,
#     guckenberger_error,
#     belkin_fixed_heat_param,
#     belkin_fixed_error,
#     b,
# ) = a
# belkin_fixed_error.shape
# b.shape
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
