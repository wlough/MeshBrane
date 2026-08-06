import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from src_python.time_series import read_time_series, plot_log_log_fit

# import sympy as sp

import yaml


def get_sweep_num_rows_cols(sweep_dir):
    sweep_dir = Path(sweep_dir)
    num_rows = 0
    num_cols = 0
    while True:
        run_dir = sweep_dir / f"run_{num_rows:0>2}_{num_cols:0>2}"
        if not run_dir.exists():
            break
        num_rows += 1
    while True:
        run_dir = sweep_dir / f"run_{0:0>2}_{num_cols:0>2}"
        if not run_dir.exists():
            break
        num_cols += 1
    return num_rows, num_cols


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
    cmap01 = colormaps["plasma"]
    cmap = lambda x: cmap01((x - 0.15) / (0.65 - 0.15))
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.25, 3.25),
        # sharey=True,
    )

    ax.set_title("Area change during mitosis")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$(A-A_0)/A_0$")

    for r, t, da_a in zip(contact_radius_list, time_list, dA_A):
        err_mask = da_a >= err_min
        t_start = t[err_mask][0]
        t_mask = t <= t_plot_interval + t_start
        plot_mask = np.logical_and(t_mask, err_mask)
        t = t[plot_mask] - t_start
        da_a = da_a[plot_mask]
        # print(f"{len(t)=}")
        color = None
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


def get_sweep_area_change_envelope(
    sweep_dir,
    t_plot_interval=2.0,
    err_min=0.001,
):

    sweep_data = load_sweep_data(sweep_dir)

    time_list = sweep_data["time_list"]
    area_list = sweep_data["area_list"]
    contact_radius_list = sweep_data["contact_radius_list"]

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
    return {"t": t, "dA_A_min": dA_A_min, "dA_A_max": dA_A_max, "dA_A_mean": dA_A_mean}


def plot_sweep_area_change_envelope(
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
    envelope_data = get_sweep_area_change_envelope(
        sweep_dir,
        t_plot_interval=t_plot_interval,
        err_min=err_min,
    )
    t = envelope_data["t"]
    dA_A_min = envelope_data["dA_A_min"]
    dA_A_max = envelope_data["dA_A_max"]
    dA_A_mean = envelope_data["dA_A_mean"]
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
    ax.fill_between(t, dA_A_min, dA_A_max, color="blue", alpha=0.4)

    plt.show()
    plt.rcParams.update(rcparams0)


plot_sweep_area_change_envelope(
    "../output/area_volume_test_mitosis_sweep_001280_rp45",
    t_plot_interval=2.0,
    err_min=0.001,
    show_legend=True,
    xlims=None,
    ylims=None,
    fig_path=None,
    color_by_contact_radius=False,
)
plot_area_change_sweep(
    "../output/area_volume_test_mitosis_sweep_001280_rp45",
    t_plot_interval=2.0,
    err_min=0.01,
    show_legend=True,
    xlims=None,
    # ylims=[-0.01, 0.46],
    fig_path="../output/area_test_mitosis.png",
    color_by_contact_radius=True,
)

# plot_area_change_sweep(
#     "../output/area_volume_test_mitosis_sweep_001280",
#     t_plot_interval=2.0,
#     err_min=0.001,
#     show_legend=True,
#     xlims=None,
#     ylims=[-0.01, 0.46],
# )
# %%
########################################
########################################
# Area and volume conservation MITOSIS #
########################################
########################################

output_dir = "../output/area_volume_test_mitosis"
output_dir = "../output/area_volume_test_mitosis_sweep_001280"
output_dir = "../output/area_volume_test_mitosis_sweep_001280_rp45"


# rows = [
#     0,
#     1,
#     2,
#     3,
# ]
# cols = [0]
# t_max = 2.0
# nt_skip = 1
# err_min = 0.001

# R_contact = np.array(
#     [
#         0.65,
#         0.45,
#         0.25,
#         0.15,
#     ]
# )

rows = list(range(12))
cols = [0]
t_max = 2.0
nt_skip = 1
err_min = 0.001


R_contact = np.array(12 * [0.45])

time_paths = [
    f"{output_dir}/run_{row:0>2}_{col:0>2}/raw_data/t.dat"
    for row in rows
    for col in cols
]
area_paths = [
    f"{output_dir}/run_{row:0>2}_{col:0>2}/raw_data/envelope_area.dat"
    for row in rows
    for col in cols
]
volume_paths = [
    f"{output_dir}/run_{row:0>2}_{col:0>2}/raw_data/envelope_volume.dat"
    for row in rows
    for col in cols
]

time = [read_time_series(path)[::nt_skip] for path in time_paths]
area = [read_time_series(path)[::nt_skip] for path in area_paths]
volume = [read_time_series(path)[::nt_skip] for path in volume_paths]


# A_tube = 2 * np.pi * 0.1 * 2
# dA_A = [(A - A_tube - A[0]) / A[0] for A in area]
dA_A = [(A - A[0]) / A[0] for A in area]
dV_V = [(V - V[0]) / V[0] for V in volume]


# len(time[0][time[0]<t_max])
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(8, 4),
    # sharey=True,
)
axes[0].set_title(r"$\Delta A/A_0$")
axes[1].set_title(r"$\Delta V/V_0$")

for r, t, da_a, dv_v in zip(R_contact, time, dA_A, dV_V):
    # print(f"dt={t[1]-t[0]:.2g}")
    t_mask = t <= t_max
    err_mask = da_a >= err_min
    plot_mask = np.logical_and(t_mask, err_mask)
    tt = t[plot_mask]
    tt -= tt[0]
    a_err = da_a[plot_mask]
    v_err = dv_v[plot_mask]
    # axes[0].plot(t[t_mask], da_a[t_mask], label=r"$\rho=" + f"{r}" + r"$")
    # axes[1].plot(t[t_mask], dv_v[t_mask], label=r"$\rho=" + f"{r}" + r"$")
    axes[0].plot(tt, a_err, label=r"$\rho=" + f"{r}" + r"$")
    axes[1].plot(tt, v_err, label=r"$\rho=" + f"{r}" + r"$")
plt.legend()
# plt.xlim([0, 2])
# plt.ylim([0, 0.46])
axes[0].set_xlim([0, 2])
axes[1].set_xlim([0, 2])
axes[0].set_ylim([-0.01, 0.46])
axes[1].set_ylim([-0.01, 0.46])


fig.savefig("../output/area_volume_test_mitosis.png", dpi=600)
plt.show()
# %%
################
################
# Energy #
################
################


output_dir = "../output"

Nf = np.array(
    [
        320,
        1280,
        5120,
        20480,
    ]
)

L_paths = [
    f"{output_dir}/fluctuations_test_{_:0>6}_no_graph/raw_data/envelope_average_edge_length.dat"
    for _ in Nf
]
H_paths = [
    f"{output_dir}/fluctuations_test_{_:0>6}_no_graph/raw_data/envelope_mean_curvature_V.dat"
    for _ in Nf
]
lapH_paths = [
    f"{output_dir}/fluctuations_test_{_:0>6}_no_graph/raw_data/envelope_lap_mean_curvature_V.dat"
    for _ in Nf
]
K_paths = [
    f"{output_dir}/fluctuations_test_{_:0>6}_no_graph/raw_data/envelope_gaussian_curvature_V.dat"
    for _ in Nf
]
A_paths = [
    f"{output_dir}/fluctuations_test_{_:0>6}_no_graph/raw_data/envelope_area_V.dat"
    for _ in Nf
]

Ne = 3 * Nf / 2
Nv = 2 + Nf / 2
L = np.array([read_time_series(path)[0] for path in L_paths])
1 / L


big_H = [read_time_series(path)[0] for path in H_paths]
big_A = [read_time_series(path)[0] for path in A_paths]
big_lapH = [read_time_series(path)[0] for path in lapH_paths]
big_K = [read_time_series(path)[0] for path in K_paths]
big_F = [-2 * (laph + 2 * h * (h**2 - k)) for h, laph, k in zip(H, lapH, K)]

B = 1.0

E_actual = 8 * np.pi * B

E = np.array([2 * B * np.einsum("v, v", H**2, dA) for H, dA in zip(big_H, big_A)])

err_E = abs(E - E_actual) / E_actual


plot_log_log_fit(Nf, err_E)
