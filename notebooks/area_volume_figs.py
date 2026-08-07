import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import colormaps

from src_python.pretty_pictures import (
    scalars_to_rgba,
    get_cmap,
    MATPLOTLIB_COLORS,
    MATPLOTLIB_COLORMAPS,
)

from src_python.time_series import (
    read_time_series,
    plot_log_log_fit,
    get_sweep_num_rows_cols,
    plot_ne_point_cloud2d_run,
    plot_ne_point_cloud2d_sweep,
)


import yaml

#


def get_run_dirs_from_sweep_dir(sweep_dir):
    sweep_dir = Path(sweep_dir)
    num_rows, num_cols = get_sweep_num_rows_cols(sweep_dir)
    run_dirs = [
        sweep_dir / f"run_{row:0>2}_{col:0>2}"
        for row in range(num_rows)
        for col in range(num_cols)
    ]

    return run_dirs


def load_sweep_data(sweep_dir):
    sweep_dir = Path(sweep_dir)
    num_rows, num_cols = get_sweep_num_rows_cols(sweep_dir)
    run_dirs = [
        sweep_dir / f"run_{row:0>2}_{col:0>2}"
        for row in range(num_rows)
        for col in range(num_cols)
    ]

    time_list = [read_time_series(run_dir / "raw_data/t.dat") for run_dir in run_dirs]
    area_list = [
        read_time_series(run_dir / "raw_data/envelope_area.dat") for run_dir in run_dirs
    ]
    volume_list = [
        read_time_series(run_dir / "raw_data/envelope_volume.dat")
        for run_dir in run_dirs
    ]
    contact_radius_list = []
    for run_dir in run_dirs:
        with open(run_dir / "parameters.yaml", "r") as f:
            params = yaml.safe_load(f)
            contact_radius_list.append(params["spindle"]["spb1"]["contact_radius"])
    return {
        "time_list": time_list,
        "area_list": area_list,
        "volume_list": volume_list,
        "contact_radius_list": contact_radius_list,
    }


def load_data_from_run_dirs(run_dirs):
    run_dirs = [Path(path) for path in run_dirs]

    time_list = [read_time_series(run_dir / "raw_data/t.dat") for run_dir in run_dirs]
    area_list = [
        read_time_series(run_dir / "raw_data/envelope_area.dat") for run_dir in run_dirs
    ]
    volume_list = [
        read_time_series(run_dir / "raw_data/envelope_volume.dat")
        for run_dir in run_dirs
    ]
    contact_radius_list = []
    for run_dir in run_dirs:
        with open(run_dir / "parameters.yaml", "r") as f:
            params = yaml.safe_load(f)
            contact_radius_list.append(params["spindle"]["spb1"]["contact_radius"])
    return {
        "time_list": time_list,
        "area_list": area_list,
        "volume_list": volume_list,
        "contact_radius_list": contact_radius_list,
    }


def plot_area_change_sweep(
    sweep_dir,
    t_plot_interval=2.0,
    err_min=0.001,
    show_legend=True,
    xlims=None,
    ylims=None,
    fig_path=None,
    color_by_contact_radius=False,
):

    sweep_data = load_sweep_data(sweep_dir)

    time_list = sweep_data["time_list"]
    area_list = sweep_data["area_list"]
    # volume_list = sweep_data["volume_list"]
    contact_radius_list = sweep_data["contact_radius_list"]

    dA_A = [(A - A[0]) / A[0] for A in area_list]
    # dV_V = [(V - V[0]) / V[0] for V in volume_list]

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
            "legend.borderaxespad": 0.15,
            # "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    cmap = get_cmap(0.15, 0.65, "coolwarm")
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.25, 3.25),
        # sharey=True,
    )

    ax.set_title("Area change during mitosis")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$(A-A_0)/A_0$")

    num_things = len(time_list)
    for _ in range(num_things):
        r = contact_radius_list[_]
        t = time_list[_]
        da_a = dA_A[_]
        color = f"C{_}"  # MATPLOTLIB_COLORS[_]

        err_mask = da_a >= err_min
        t_start = t[err_mask][0]
        t_mask = t <= t_plot_interval + t_start
        plot_mask = np.logical_and(t_mask, err_mask)
        t = t[plot_mask] - t_start
        da_a = da_a[plot_mask]
        # print(f"{len(t)=}")

        if color_by_contact_radius:
            color = cmap(r)
        ax.plot(t, da_a, label=r"$\rho=" + f"{r}" + r"$", color=color)
    if show_legend:
        plt.legend()

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if fig_path is not None:
        fig.savefig(fig_path, dpi=600)
    plt.show()
    plt.rcParams.update(rcparams0)


def get_sweep_area_change_minmaxmean(
    sweep_dir,
    t_plot_interval=2.0,
    err_min=0.001,
):

    sweep_data = load_sweep_data(sweep_dir)

    time_list = sweep_data["time_list"]
    area_list = sweep_data["area_list"]
    contact_radius_list = sweep_data["contact_radius_list"]

    std_r = np.std(contact_radius_list)
    if std_r > 1e-6:
        raise ValueError("sweep uses different contact radii")
    contact_radius = np.mean(contact_radius_list)

    dA_A = [(A - A[0]) / A[0] for A in area_list]

    shifted_time_list = []
    shifted_dA_A_list = []

    for r, t, da_a in zip(contact_radius_list, time_list, dA_A):
        err_mask = da_a >= err_min
        t_start = t[err_mask][0]
        t_mask = t <= t_plot_interval + t_start
        plot_mask = np.logical_and(t_mask, err_mask)
        t = t[plot_mask] - t_start
        da_a = da_a[plot_mask]
        shifted_time_list.append(t)
        shifted_dA_A_list.append(da_a)
    Nt_min = min([len(_) for _ in shifted_time_list])
    shifted_time_list = [_[:Nt_min] for _ in shifted_time_list]
    shifted_dA_A_list = np.array([_[:Nt_min] for _ in shifted_dA_A_list])
    dA_A_min = np.min(shifted_dA_A_list, axis=0)
    dA_A_max = np.max(shifted_dA_A_list, axis=0)
    dA_A_mean = np.mean(shifted_dA_A_list, axis=0)
    t = shifted_time_list[0]
    return {
        "t": t,
        "dA_A_min": dA_A_min,
        "dA_A_max": dA_A_max,
        "dA_A_mean": dA_A_mean,
        "contact_radius": contact_radius,
    }


def get_run_dir_list_area_change_minmaxmean(
    run_dirs,
    t_plot_interval=2.0,
    err_min=0.001,
):

    sweep_data = load_data_from_run_dirs(run_dirs)

    time_list = sweep_data["time_list"]
    area_list = sweep_data["area_list"]
    contact_radius_list = sweep_data["contact_radius_list"]

    std_r = np.std(contact_radius_list)
    if std_r > 1e-6:
        raise ValueError("sweep uses different contact radii")
    contact_radius = np.mean(contact_radius_list)

    dA_A = [(A - A[0]) / A[0] for A in area_list]

    shifted_time_list = []
    shifted_dA_A_list = []

    for r, t, da_a in zip(contact_radius_list, time_list, dA_A):
        err_mask = da_a >= err_min
        t_start = t[err_mask][0]
        t_mask = t <= t_plot_interval + t_start
        plot_mask = np.logical_and(t_mask, err_mask)
        t = t[plot_mask] - t_start
        da_a = da_a[plot_mask]
        shifted_time_list.append(t)
        shifted_dA_A_list.append(da_a)
    Nt_min = min([len(_) for _ in shifted_time_list])
    shifted_time_list = [_[:Nt_min] for _ in shifted_time_list]
    shifted_dA_A_list = np.array([_[:Nt_min] for _ in shifted_dA_A_list])
    dA_A_min = np.min(shifted_dA_A_list, axis=0)
    dA_A_max = np.max(shifted_dA_A_list, axis=0)
    dA_A_mean = np.mean(shifted_dA_A_list, axis=0)
    t = shifted_time_list[0]
    return {
        "t": t,
        "dA_A_min": dA_A_min,
        "dA_A_max": dA_A_max,
        "dA_A_mean": dA_A_mean,
        "contact_radius": contact_radius,
    }


def plot_sweep_area_change_minmaxmean(
    sweep_dir,
    t_plot_interval=2.0,
    err_min=0.001,
    show_legend=True,
    xlims=None,
    ylims=None,
    fig_path=None,
    color_by_contact_radius=False,
):

    sweep_data = load_sweep_data(sweep_dir)
    minmaxmean_data = get_sweep_area_change_minmaxmean(
        sweep_dir,
        t_plot_interval=t_plot_interval,
        err_min=err_min,
    )
    t = minmaxmean_data["t"]
    dA_A_min = minmaxmean_data["dA_A_min"]
    dA_A_max = minmaxmean_data["dA_A_max"]
    dA_A_mean = minmaxmean_data["dA_A_mean"]
    contact_radius_list = sweep_data["contact_radius_list"]

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
            "legend.borderaxespad": 0.15,
            # "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.25, 3.25),
        # sharey=True,
    )

    ax.set_title("Area change during mitosis")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$(A-A_0)/A_0$")
    # ax.plot(t, dA_A_min, label="min")
    # ax.plot(t, dA_A_max, label="max")

    ax.plot(t, dA_A_mean, color="blue")
    ax.fill_between(t, dA_A_min, dA_A_max, color="blue", alpha=0.3)

    plt.show()
    plt.rcParams.update(rcparams0)


def plot_multisweep_area_change_minmaxmean(
    sweep_dirs,
    t_plot_interval=2.0,
    err_min=0.001,
    fig_path=None,
):

    minmaxmean_data = [
        get_sweep_area_change_minmaxmean(
            sweep_dir,
            t_plot_interval=t_plot_interval,
            err_min=err_min,
        )
        for sweep_dir in sweep_dirs
    ]
    big_t = [data["t"] for data in minmaxmean_data]
    big_dA_A_min = [data["dA_A_min"] for data in minmaxmean_data]
    big_dA_A_max = [data["dA_A_max"] for data in minmaxmean_data]
    big_dA_A_mean = [data["dA_A_mean"] for data in minmaxmean_data]
    big_r = [data["contact_radius"] for data in minmaxmean_data]

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
            "legend.borderaxespad": 0.15,
            # "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.25, 3.25),
        # sharey=True,
    )

    ax.set_title("Area change during mitosis")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$(A-A_0)/A_0$")

    num_things = len(big_t)

    for _ in range(num_things):
        t = big_t[_]
        # dA_A_mean = big_dA_A_mean[_] * (1 - 1 / (_ + 2))
        # dA_A_min = big_dA_A_min[_] * (1 - 1 / (_ + 2))
        # dA_A_max = big_dA_A_max[_] * (1 - 1 / (_ + 2))
        dA_A_mean = big_dA_A_mean[_]
        dA_A_min = big_dA_A_min[_]
        dA_A_max = big_dA_A_max[_]
        r = big_r[_]
        color = f"C{_}"
        label = r"$\rho=" + f"{r:.2g}" + r"$"

        ax.plot(t, dA_A_mean, color=color, label=label)
        ax.fill_between(t, dA_A_min, dA_A_max, color=color, alpha=0.3)

    plt.legend()
    if fig_path is not None:
        fig.savefig(fig_path, dpi=600)

    plt.show()
    plt.rcParams.update(rcparams0)


def plot_multiruns_area_change_minmaxmean(
    run_dirs_lists,
    t_plot_interval=2.0,
    err_min=0.001,
    fig_path=None,
):

    minmaxmean_data = [
        get_run_dir_list_area_change_minmaxmean(
            run_dirs,
            t_plot_interval=t_plot_interval,
            err_min=err_min,
        )
        for run_dirs in run_dirs_lists
    ]
    big_t = [data["t"] for data in minmaxmean_data]
    big_dA_A_min = [data["dA_A_min"] for data in minmaxmean_data]
    big_dA_A_max = [data["dA_A_max"] for data in minmaxmean_data]
    big_dA_A_mean = [data["dA_A_mean"] for data in minmaxmean_data]
    big_r = [data["contact_radius"] for data in minmaxmean_data]

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
            "legend.borderaxespad": 0.15,
            # "legend.borderpad": 0.0,
            "legend.handlelength": 0.75,
            "legend.handletextpad": 0.1,
            "legend.labelspacing": 0.0,
            "axes.labelpad": 0.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
        }
    )

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.25, 3.25),
        # sharey=True,
    )

    ax.set_title("Area change during mitosis")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$(A-A_0)/A_0$")

    num_things = len(big_t)

    for _ in range(num_things):
        t = big_t[_]
        # dA_A_mean = big_dA_A_mean[_] * (1 - 1 / (_ + 2))
        # dA_A_min = big_dA_A_min[_] * (1 - 1 / (_ + 2))
        # dA_A_max = big_dA_A_max[_] * (1 - 1 / (_ + 2))
        dA_A_mean = big_dA_A_mean[_]
        dA_A_min = big_dA_A_min[_]
        dA_A_max = big_dA_A_max[_]
        r = big_r[_]
        color = f"C{_}"
        label = r"$\rho=" + f"{r:.2g}" + r"$"

        ax.plot(t, dA_A_mean, color=color, label=label)
        ax.fill_between(t, dA_A_min, dA_A_max, color=color, alpha=0.3)

    plt.legend()
    if fig_path is not None:
        fig.savefig(fig_path, dpi=600)

    plt.show()
    plt.rcParams.update(rcparams0)


# %%

# plot_sweep_area_change_minmaxmean(
#     "../output/area_volume_test_mitosis_sweep_001280_rp45",
#     t_plot_interval=2.0,
#     err_min=0.001,
#     show_legend=True,
#     xlims=None,
#     ylims=None,
#     fig_path=None,
#     color_by_contact_radius=False,
# )
# plot_area_change_sweep(
#     "../output/area_volume_test_mitosis_sweep_001280_rp45",
#     t_plot_interval=2.0,
#     err_min=0.001,
#     show_legend=True,
#     xlims=None,
#     # ylims=[-0.01, 0.46],
#     fig_path="../output/area_test_mitosis.png",
#     color_by_contact_radius=True,
# )

plot_area_change_sweep(
    "../output/area_volume_test_mitosis_sweep_001280",
    t_plot_interval=2.0,
    err_min=0.001,
    show_legend=True,
    xlims=None,
    ylims=[-0.01, 0.46],
)
#
# plot_area_change_sweep(
#     "../output/area_volume_test_mitosis_001280_rp25_sweep",
#     t_plot_interval=2.0,
#     err_min=0.001,
#     show_legend=True,
#     xlims=None,
#     ylims=[-0.01, 0.46],
# )
#
plot_multisweep_area_change_minmaxmean(
    [
        # "../output/area_volume_test_mitosis_sweep_001280_rp45",
        "../output/area_volume_test_mitosis_001280_rp25_sweep0",
        "../output/area_volume_test_mitosis_001280_rp25_sweep",
        "../output/area_volume_test_mitosis_001280_rp45_sweep",
        "../output/area_volume_test_mitosis_001280_rp65_sweep",
    ],
    t_plot_interval=2.0,
    err_min=0.001,
    fig_path=None,
)

plot_multiruns_area_change_minmaxmean(
    [
        [
            *get_run_dirs_from_sweep_dir(
                "../output/area_volume_test_mitosis_001280_rp25_sweep0"
            ),
            *get_run_dirs_from_sweep_dir(
                "../output/area_volume_test_mitosis_001280_rp25_sweep"
            ),
        ],
        [
            *get_run_dirs_from_sweep_dir(
                "../output/area_volume_test_mitosis_001280_rp45_sweep"
            ),
        ],
        [
            *get_run_dirs_from_sweep_dir(
                "../output/area_volume_test_mitosis_001280_rp65_sweep"
            )
        ],
    ],
    t_plot_interval=2.0,
    err_min=0.001,
    fig_path=None,
)


# plot_ne_point_cloud2d_sweep(
#     "../output/area_volume_test_mitosis_001280_rp25_sweep",
#     t_start=0.01,
#     dt_sample=0.25,
#     markersize=2.5,
# )

plot_ne_point_cloud2d_sweep(
    "../output/area_volume_test_mitosis_001280_rp25_sweep0",
    t_start=0.01,
    dt_sample=0.25,
    markersize=2.5,
)

# plot_ne_point_cloud2d_run(
#     "../output/area_volume_test_mitosis_001280_rp25_sweep/run_05_00",
#     t_start=0.01,
#     dt_sample=0.25,
#     markersize=2.5,
# )


# %%
