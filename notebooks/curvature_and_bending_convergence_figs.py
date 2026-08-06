import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

import numpy as np
import matplotlib.pyplot as plt
from src_python.time_series import (
    read_time_series,
    plot_log_log_fit,
    log_log_fit,
    round_to,
    MATPLOTLIB_COLORS,
    MATPLOTLIB_MARKERS,
    MATPLOTLIB_LINESTYLES,
    to_scinotation_tex,
)


def get_curvature_and_bending_convergence_data():
    output_dir = "../output/bending_force_test"

    Nf = np.array(
        [
            320,
            1280,
            5120,
            20480,
        ]
    )

    L_paths = [
        f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_average_edge_length.dat"
        for _ in Nf
    ]
    H_paths = [
        f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_mean_curvature_V.dat"
        for _ in Nf
    ]
    lapH_paths = [
        f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_lap_mean_curvature_V.dat"
        for _ in Nf
    ]
    K_paths = [
        f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_gaussian_curvature_V.dat"
        for _ in Nf
    ]
    A_paths = [
        f"{output_dir}/bending_force_test_{_:0>6}/raw_data/envelope_area_V.dat"
        for _ in Nf
    ]

    Ne = 3 * Nf / 2
    Nv = 2 + Nf / 2
    mean_edge_length = np.array([read_time_series(path)[0] for path in L_paths])

    H = [read_time_series(path)[0] for path in H_paths]
    lapH = [read_time_series(path)[0] for path in lapH_paths]
    K = [read_time_series(path)[0] for path in K_paths]
    F = [-2 * (laph + 2 * h * (h**2 - k)) for h, laph, k in zip(H, lapH, K)]
    A = [read_time_series(path)[0] for path in A_paths]

    H0 = -1.0
    lapH0 = 0.0
    K0 = 1.0
    F0 = -2 * (lapH0 + 2 * H0 * (H0**2 - K0))
    helfrich_energy0 = 8 * np.pi

    # B = 1.0
    helfrich_energy = np.array(
        [2 * np.einsum("v, v", Hv**2, Av) for Hv, Av in zip(H, A)]
    )
    err_helfrich_energy = np.abs(helfrich_energy - helfrich_energy0)

    eps_H = [np.abs(h - H0) for h in H]
    eps_lapH = [np.abs(laph - lapH0) for laph in lapH]
    eps_K = [np.abs(k - K0) for k in K]
    eps_F = [np.abs(f - F0) for f in F]

    # normalizedL2_err_H used by Belkin
    normalizedL2_err_H = np.array(
        [np.linalg.norm(h - H0) / np.linalg.norm(H0 * np.ones_like(h)) for h in H]
    )

    L2_eps_H = np.array([np.linalg.norm(e, 2) for e in eps_H])
    L2_eps_lapH = np.array([np.linalg.norm(e, 2) for e in eps_lapH])
    L2_eps_K = np.array([np.linalg.norm(e, 2) for e in eps_K])
    L2_eps_F = np.array([np.linalg.norm(e, 2) for e in eps_F])

    Linf_eps_H = np.array([np.linalg.norm(e, np.inf) for e in eps_H])
    Linf_eps_lapH = np.array([np.linalg.norm(e, np.inf) for e in eps_lapH])
    Linf_eps_K = np.array([np.linalg.norm(e, np.inf) for e in eps_K])
    Linf_eps_F = np.array([np.linalg.norm(e, np.inf) for e in eps_F])

    mean_eps_H = np.array([np.mean(e) for e in eps_H])
    mean_eps_lapH = np.array([np.mean(e) for e in eps_lapH])
    mean_eps_K = np.array([np.mean(e) for e in eps_K])
    mean_eps_F = np.array([np.mean(e) for e in eps_F])

    data_dict = {
        "Nf": Nf,
        "Ne": Ne,
        "Nv": Nv,
        "mean_edge_length": mean_edge_length,
        "normalizedL2_err_H": normalizedL2_err_H,
        "L2_eps_H": L2_eps_H,
        "L2_eps_lapH": L2_eps_lapH,
        "L2_eps_K": L2_eps_K,
        "L2_eps_F": L2_eps_F,
        "Linf_eps_H": Linf_eps_H,
        "Linf_eps_lapH": Linf_eps_lapH,
        "Linf_eps_K": Linf_eps_K,
        "Linf_eps_F": Linf_eps_F,
        "mean_eps_H": mean_eps_H,
        "mean_eps_lapH": mean_eps_lapH,
        "mean_eps_K": mean_eps_K,
        "mean_eps_F": mean_eps_F,
        "err_helfrich_energy": err_helfrich_energy,
    }
    return data_dict


def average_mean_curvature_error_plot(err_dict):
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    textsize = 12
    plt.rcParams.update(
        {
            "font.size": textsize,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            # "xtick.labeltop": False,
            # "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            # "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            # "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            # "ytick.right": True,
            # "text.usetex": False,
            "figure.titlesize": textsize,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            # "lines.linewidth": 5,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.handletextpad": 0.2,
            "legend.loc": "upper right",
            "legend.borderaxespad": 0.15,
            "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fit_dict = log_log_fit(err_dict["Nf"], err_dict["mean_eps_H"])
    Xlabel = r"$N_f$"
    Ylabel = r"$\varepsilon_H$"
    error_label = "Average " + r"$\varepsilon_H$"
    style_num = 0

    m = fit_dict["slope"]
    y_fit = fit_dict["fit_samples"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    fit_label = r"$O\left(" + "N_f" + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    color = MATPLOTLIB_COLORS[0]
    marker = MATPLOTLIB_MARKERS[0]
    linestyle = MATPLOTLIB_LINESTYLES[2]

    xticks = fit_dict["logX"]
    xticklabels_data = err_dict["Nf"]
    xticklabels = to_scinotation_tex(xticklabels_data)

    yticks = fit_dict["logY"]
    yticklabels_data = np.exp(yticks)
    yticklabels = to_scinotation_tex(yticklabels_data, decimals=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 3.25))
    fig.suptitle("Average mean curvature error")

    ax.set_xlabel(Xlabel)
    ax.set_ylabel(error_label)
    # ax.plot(x, y_fit, linestyle=linestyle, color=color, label=fit_label)
    # ax.plot(x, y, marker, color=color)
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels, rotation=-45)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticklabels)

    ax.loglog(
        np.exp(x), np.exp(y_fit), linestyle=linestyle, color=color, label=fit_label
    )
    ax.loglog(np.exp(x), np.exp(y), marker, color=color)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.rcParams.update(rcparams0)


def average_gaussian_curvature_error_plot(err_dict):
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    textsize = 12
    plt.rcParams.update(
        {
            "font.size": textsize,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            # "xtick.labeltop": False,
            # "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            # "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            # "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            # "ytick.right": True,
            # "text.usetex": False,
            "figure.titlesize": textsize,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            # "lines.linewidth": 5,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.handletextpad": 0.2,
            "legend.loc": "upper right",
            "legend.borderaxespad": 0.15,
            "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fit_dict = log_log_fit(err_dict["Nf"], err_dict["mean_eps_K"])
    Xlabel = r"$N_f$"
    Ylabel = r"$\varepsilon_K$"
    error_label = "Average " + r"$\varepsilon_K$"

    m = fit_dict["slope"]
    y_fit = fit_dict["fit_samples"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    fit_label = r"$O\left(" + "N_f" + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    color = MATPLOTLIB_COLORS[0]
    marker = MATPLOTLIB_MARKERS[0]
    linestyle = MATPLOTLIB_LINESTYLES[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 3.25))
    fig.suptitle("Average Gaussian curvature error")

    ax.set_xlabel(Xlabel)
    ax.set_ylabel(error_label)
    ax.loglog(
        np.exp(x), np.exp(y_fit), linestyle=linestyle, color=color, label=fit_label
    )
    ax.loglog(np.exp(x), np.exp(y), marker, color=color)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.rcParams.update(rcparams0)


def average_bending_stress_error_plot(err_dict):
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    textsize = 12
    plt.rcParams.update(
        {
            "font.size": textsize,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            # "xtick.labeltop": False,
            # "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            # "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            # "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            # "ytick.right": True,
            # "text.usetex": False,
            "figure.titlesize": textsize,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            # "lines.linewidth": 5,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.handletextpad": 0.2,
            "legend.loc": "upper right",
            "legend.borderaxespad": 0.15,
            "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fit_dict = log_log_fit(err_dict["Nf"], err_dict["mean_eps_F"])
    Xlabel = r"$N_f$"
    Ylabel = r"$\varepsilon_f$"
    error_label = "Average " + r"$\varepsilon_f$"

    m = fit_dict["slope"]
    y_fit = fit_dict["fit_samples"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    fit_label = r"$O\left(" + "N_f" + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    color = MATPLOTLIB_COLORS[0]
    marker = MATPLOTLIB_MARKERS[0]
    linestyle = MATPLOTLIB_LINESTYLES[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 3.25))
    fig.suptitle("Average bending stress error")

    ax.set_xlabel(Xlabel)
    ax.set_ylabel(error_label)
    ax.loglog(
        np.exp(x), np.exp(y_fit), linestyle=linestyle, color=color, label=fit_label
    )
    ax.loglog(np.exp(x), np.exp(y), marker, color=color)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.rcParams.update(rcparams0)


def bending_energy_error_plot(err_dict):
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    textsize = 12
    plt.rcParams.update(
        {
            "font.size": textsize,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            # "xtick.labeltop": False,
            # "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            # "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            # "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            # "ytick.right": True,
            # "text.usetex": False,
            "figure.titlesize": textsize,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            # "lines.linewidth": 5,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.handletextpad": 0.2,
            "legend.loc": "upper right",
            "legend.borderaxespad": 0.15,
            "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fit_dict = log_log_fit(err_dict["Nf"], err_dict["err_helfrich_energy"])
    Xlabel = r"$N_f$"
    Ylabel = r"$\varepsilon_f$"
    error_label = r"$U_{\text{bend}}$ error"

    m = fit_dict["slope"]
    y_fit = fit_dict["fit_samples"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    fit_label = r"$O\left(" + "N_f" + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    color = MATPLOTLIB_COLORS[0]
    marker = MATPLOTLIB_MARKERS[0]
    linestyle = MATPLOTLIB_LINESTYLES[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 3.25))
    fig.suptitle("Bending energy error")

    ax.set_xlabel(Xlabel)
    ax.set_ylabel(error_label)
    ax.loglog(
        np.exp(x), np.exp(y_fit), linestyle=linestyle, color=color, label=fit_label
    )
    ax.loglog(np.exp(x), np.exp(y), marker, color=color)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.rcParams.update(rcparams0)


def average_edge_length_vs_num_faces_plot(err_dict):
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    textsize = 12
    plt.rcParams.update(
        {
            "font.size": textsize,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            # "xtick.labeltop": False,
            # "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            # "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            # "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            # "ytick.right": True,
            # "text.usetex": False,
            "figure.titlesize": textsize,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            # "lines.linewidth": 5,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.handletextpad": 0.2,
            "legend.loc": "upper right",
            "legend.borderaxespad": 0.15,
            "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fit_dict = log_log_fit(err_dict["Nf"], err_dict["mean_edge_length"])
    Xlabel = r"$N_f$"
    error_label = r"$h$"

    m = fit_dict["slope"]
    y_fit = fit_dict["fit_samples"]
    y = fit_dict["logY"]
    x = fit_dict["logX"]
    fit_label = r"$O\left(" + "N_f" + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"
    color = MATPLOTLIB_COLORS[0]
    marker = MATPLOTLIB_MARKERS[0]
    linestyle = MATPLOTLIB_LINESTYLES[2]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 3.25))
    fig.suptitle("Average edge length vs num faces")

    ax.set_xlabel(Xlabel)
    ax.set_ylabel(error_label)
    ax.loglog(
        np.exp(x), np.exp(y_fit), linestyle=linestyle, color=color, label=fit_label
    )
    ax.loglog(np.exp(x), np.exp(y), marker, color=color)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.rcParams.update(rcparams0)


def average_combined_error_plot(err_dict):
    fig_path = "../output/average_combined_error_plot.svg"
    rcparams0 = dict(plt.rcParams)  # save original rcparams
    textsize = 12
    plt.rcParams.update(
        {
            "font.size": textsize,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "text.latex.preamble": "",
            "figure.dpi": 300.0,
            "xtick.direction": "in",
            "xtick.labelsize": textsize,
            # "xtick.labeltop": False,
            # "xtick.minor.ndivs": 4,
            "xtick.minor.visible": True,
            # "xtick.top": True,
            "ytick.direction": "in",
            "ytick.labelsize": textsize,
            # "ytick.minor.ndivs": 4,
            "ytick.minor.visible": True,
            # "ytick.right": True,
            # "text.usetex": False,
            "figure.titlesize": textsize,
            "axes.titlesize": textsize,
            "axes.labelsize": textsize,
            # "lines.linewidth": 5,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": textsize,
            # "legend.handletextpad": 0.2,
            # "legend.loc": "upper right",
            # "legend.borderaxespad": 0.15,
            # "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 3.25))
    fig.suptitle("Average combined error")
    ax.set_xlabel(r"$N_f$")
    ax.set_ylabel("error")

    # err_keys = ["mean_eps_H", "mean_eps_K", "mean_eps_F", "err_helfrich_energy"]
    # err_labels = [
    #     r"$\varepsilon_H$",
    #     r"$\varepsilon_K$",
    #     r"$\varepsilon_f$",
    #     r"$\varepsilon_U$",
    # ]
    err_keys = ["mean_eps_H", "mean_eps_F"]
    err_labels = [r"$\varepsilon_H$", r"$\varepsilon_f$"]

    for _ in range(len(err_keys)):
        err_key = err_keys[_]
        err_label = err_labels[_]
        color = MATPLOTLIB_COLORS[_]
        marker = MATPLOTLIB_MARKERS[_]
        linestyle = MATPLOTLIB_LINESTYLES[_]

        fit_dict = log_log_fit(err_dict["Nf"], err_dict[err_key])
        m = fit_dict["slope"]
        y_fit = fit_dict["fit_samples"]
        y = fit_dict["logY"]
        x = fit_dict["logX"]

        fit_label = r"$O\left(" + "N_f" + r"^{" + f"{round_to(m, n=3)}" + r"}\right)$"

        ax.loglog(
            np.exp(x), np.exp(y_fit), linestyle=linestyle, color=color, label=fit_label
        )
        ax.loglog(np.exp(x), np.exp(y), marker, color=color, label=err_label)

    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.show()
    plt.rcParams.update(rcparams0)


err_dict = get_curvature_and_bending_convergence_data()


average_mean_curvature_error_plot(err_dict)
# average_gaussian_curvature_error_plot(err_dict)
# average_bending_stress_error_plot(err_dict)
# bending_energy_error_plot(err_dict)
average_combined_error_plot(err_dict)
# average_edge_length_vs_num_faces_plot(err_dict)
