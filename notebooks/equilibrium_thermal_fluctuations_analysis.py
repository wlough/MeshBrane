import sys
from pathlib import Path

nb_dir = Path().resolve()
proj_dir = (nb_dir / "..").resolve()
sys.path.insert(0, str(proj_dir))

import numpy as np
import matplotlib.pyplot as plt
from pymathutils import thetaphi_from_xyz
from pymathutils.special import (
    Ylm,
    compute_all_Ylm,
    spherical_harmonic_index_n_LM,
    spherical_harmonic_index_lm_N,
)


from src_python.time_series import read_time_series, plot_log_log_fit
import sympy as sp
from scipy.optimize import least_squares

import yaml

######################################
# required for surface tension stats #
######################################
# save_final_surface_tension_stats
# compute_concatenated_run_average_surface_tension_stats
# compute_run_spatial_average_surface_tension_time_series
# read_time_series


# ***
def compute_run_spatial_average_surface_tension_time_series(
    output_dir,
    t_start=0.1,
    dt_sample=0.03,
):
    print(f"compute_run_average_surface_tension_time_series for {output_dir}")
    output_dir = Path(output_dir)

    with open(output_dir / "parameters.yaml", "r") as file:
        params = yaml.safe_load(file)
        k = float(params["envelope"]["area_stiffness"])
        gamma0 = float(params["envelope"]["surface_tension_constant"])

    big_t = read_time_series(output_dir / "raw_data/t.dat")
    dt0 = big_t[1] - big_t[0]
    nt_skip = int(dt_sample / dt0) + 1
    nt_start = int(t_start / dt0)
    big_t = big_t[nt_start::nt_skip]
    if len(big_t) < 2:
        return dict()
    dt = big_t[1] - big_t[0]

    big_area_F = read_time_series(output_dir / "raw_data/envelope_area_F.dat")[
        nt_start::nt_skip
    ]
    Nt, Nf = big_area_F.shape
    big_area = read_time_series(output_dir / "raw_data/envelope_area.dat")
    A0 = big_area[0] / Nf
    big_area = big_area[nt_start::nt_skip]

    # should_be_zeros = big_area - np.sum(big_area_F, axis=1)
    # return should_be_zeros
    # big_surface_tension_F = read_time_series(
    #     output_dir / "raw_data/envelope_surface_tension_F.dat"
    # )[nt_start::nt_skip]
    big_surface_tension_F = gamma0 + k * (big_area_F - A0) / A0

    big_spatial_average_surface_tension = np.einsum(
        "tf,tf,t->t", big_surface_tension_F, big_area_F, 1 / big_area
    )
    big_unweighted_average_surface_tension = np.mean(big_surface_tension_F, axis=1)

    return {
        "big_spatial_average_surface_tension": big_spatial_average_surface_tension,
        "big_unweighted_average_surface_tension": big_unweighted_average_surface_tension,
        "dt": dt,
        "Nf": Nf,
    }


# ***
def compute_concatenated_run_average_surface_tension_stats(
    output_dirs,
    t_start=0.1,
    dt_sample=0.03,
):

    data0 = compute_run_spatial_average_surface_tension_time_series(
        output_dirs[0],
        t_start=t_start,
        dt_sample=dt_sample,
    )

    dt_list = [data0["dt"]]
    Nf = data0["Nf"]

    big_spatial_average_surface_tension = data0["big_spatial_average_surface_tension"]

    for path in output_dirs[1:]:
        data = compute_run_spatial_average_surface_tension_time_series(
            path,
            t_start=t_start,
            dt_sample=dt_sample,
        )
        if len(data) == 0:
            print(f"skipping {path} due to insufficient data.")
            continue
        if Nf != data["Nf"]:
            raise ValueError("runs have different number of faces")
        dt_list.append(data["dt"])
        big_spatial_average_surface_tension = np.concatenate(
            [
                big_spatial_average_surface_tension,
                data["big_spatial_average_surface_tension"],
            ],
            axis=0,
        )

    mean_gamma = np.mean(big_spatial_average_surface_tension)
    Nt = len(big_spatial_average_surface_tension)
    std_gamma = np.std(big_spatial_average_surface_tension)
    dt_list = np.array(dt_list)

    return {
        "average_surface_tension": mean_gamma,
        "std_surface_tension": std_gamma,
        "Nt": Nt,
        "dt_list": dt_list,
        "Nf": Nf,
    }


#########################################################
# required for chunked fluctuations samples and weights #
#########################################################
# save_final_chunked_samples_and_weights
# save_chunked_samples_and_weights_from_concatenated_runs
# get_concatenated_run_Qlj_data
# get_run_Qlj_data


def get_run_Qlj_data(
    output_dir,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
    R0=None,
    xyz0=None,
):
    print(f"get_run_Qlj_data for {output_dir}")
    output_dir = Path(output_dir)
    big_t = read_time_series(output_dir / "raw_data/t.dat")
    dt0 = big_t[1] - big_t[0]
    nt_skip = int(dt_sample / dt0) + 1
    nt_start = int(t_start / dt0) + 1
    big_t = big_t[nt_start::nt_skip]
    dt = big_t[1] - big_t[0]

    big_xyz_coord_V = read_time_series(
        output_dir / "raw_data/envelope_xyz_coord_V.dat"
    )[nt_start::nt_skip]
    big_area_V = read_time_series(output_dir / "raw_data/envelope_area_V.dat")[
        nt_start::nt_skip
    ]
    big_normal_V = read_time_series(output_dir / "raw_data/envelope_normal_V.dat")[
        nt_start::nt_skip
    ]

    Nt, Nv, _ = big_xyz_coord_V.shape

    if R0 is None:
        initial_volume = read_time_series(output_dir / "raw_data/envelope_volume.dat")[
            nt_start
        ]
        R0 = (3 * initial_volume / (4 * np.pi)) ** (1 / 3)
    # if xyz0 is None:
    #     xyz0 = np.einsum(
    #         "v, vx->x",
    #         big_area_V[0] / np.sum(big_area_V[0], axis=1, keepdims=True),
    #         big_xyz_coord_V[0],
    #     )
    #
    big_xyz_coord_V_com = np.einsum(
        "tv, tvx->tx",
        big_area_V / np.sum(big_area_V, axis=1, keepdims=True),
        big_xyz_coord_V,
    )

    big_xyz_coord_V = np.array(
        [
            [xyz - xyz_com for xyz in XYZt]
            for XYZt, xyz_com in zip(big_xyz_coord_V, big_xyz_coord_V_com)
        ]
    )

    big_R = np.linalg.norm(big_xyz_coord_V, axis=2)
    big_r = big_R / R0 - 1.0

    big_solid_angle_V = np.einsum(
        "tvx, tvx, tv, tv -> tv",
        big_xyz_coord_V,
        big_normal_V,
        big_area_V,
        1 / big_R**3,
    )

    big_thetaphi_coord_V = np.array([thetaphi_from_xyz(xyz) for xyz in big_xyz_coord_V])

    big_Un = []
    for solid_angle_V, r, thetaphi_coord_V in zip(
        big_solid_angle_V, big_r, big_thetaphi_coord_V
    ):
        Yn = compute_all_Ylm(l_max, thetaphi_coord_V)
        big_Un.append(np.einsum("v,v,vn->n", solid_angle_V, r, Yn.conjugate()))
    big_Un = np.array(big_Un)

    big_sqr_Un = np.abs(big_Un) ** 2
    big_Q = np.zeros((Nt, l_max + 1))

    for l in range(l_max + 1):
        # n_l0 = spherical_harmonic_index_n_LM(l, 0)
        # n_ll = spherical_harmonic_index_n_LM(l, l)
        # big_Q[:, l] = np.sum(big_sqr_Un[:, n_l0 : n_ll + 1], axis=1) / (l + 1)
        n_start = spherical_harmonic_index_n_LM(l, -l)
        n_stop = spherical_harmonic_index_n_LM(l, l)
        big_Q[:, l] = np.mean(big_sqr_Un[:, n_start : n_stop + 1], axis=1)

    return {
        "big_Q": big_Q,
        "R0": R0,
        "dt": dt,
        "Nt": Nt,
        "Nv": Nv,
    }


def get_concatenated_run_Qlj_data(
    output_dirs,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
):
    data0 = get_run_Qlj_data(
        output_dirs[0],
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
        R0=None,
    )
    R0 = data0["R0"]
    Nv = data0["Nv"]

    big_Q = data0["big_Q"]

    dt_list = [data0["dt"]]

    for output_dir in output_dirs[1:]:
        data = get_run_Qlj_data(
            output_dir,
            l_max=l_max,
            t_start=t_start,
            dt_sample=dt_sample,
            R0=R0,
        )
        if Nv != data["Nv"]:
            raise ValueError("runs have different number of vertices")
        big_Q = np.concatenate([big_Q, data["big_Q"]], axis=0)
        dt_list.append(data["dt"])

    Nt = len(big_Q)
    dt_list = np.array(dt_list)
    return {
        "big_Q": big_Q,
        "R0": R0,
        "Nv": Nv,
        "Nt": Nt,
        "dt_list": dt_list,
    }


def save_chunked_samples_and_weights_from_concatenated_runs(
    output_dirs,
    l_max=20,
    t_start=0.1,
    dt_sample=0.1,
    Nt_chunk=200,
    save_path=None,
):
    concatenated_run_Qlj_data = get_concatenated_run_Qlj_data(
        output_dirs,
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
    )
    big_Q = concatenated_run_Qlj_data["big_Q"]
    big_Nt = len(big_Q)
    Nchunks = big_Nt // Nt_chunk
    big_Q = big_Q[: Nchunks * Nt_chunk]

    chunked_Q = big_Q.reshape((Nchunks, Nt_chunk, l_max + 1))
    # chunked_Q = np.zeros((Nchunks, Nt_chunk, l_max))
    # for n_chunk in range(Nchunks):
    #     chunked_Q[n_chunk] = big_Q[n_chunk * Nt_chunk : (n_chunk + 1) * Nt_chunk]
    chunked_y = np.mean(chunked_Q, axis=1)
    chunked_s = np.std(chunked_Q, axis=1, ddof=1) / np.sqrt(Nt_chunk)

    Nv = concatenated_run_Qlj_data["Nv"]
    dt_list = concatenated_run_Qlj_data["dt_list"]
    R0 = concatenated_run_Qlj_data["R0"]

    data = {
        "chunked_y": chunked_y,
        "chunked_s": chunked_s,
        "l_max": l_max,
        "t_start": t_start,
        "dt_sample": dt_sample,
        "Nt_chunk": Nt_chunk,
        "Nv": Nv,
        "dt_list": dt_list,
        "R0": R0,
    }
    np.savez(save_path, **data)


#########################################
# required for plotting unweighted fits #
#########################################
# plot_unweighted_fit_B_gamma_vs_l_max
# fit_saved_chunked_samples_and_weights_data


def fit_saved_chunked_samples_and_weights_data(
    path,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
    l_max=20,
):

    chunked_samples_and_weights_data = dict(np.load(path, allow_pickle=True))
    R0 = chunked_samples_and_weights_data["R0"]
    kBT = 1.0291e-2
    B_gamma_actual = np.array([B_actual, gamma_actual])
    gamma_red_guess = gamma_guess * R0**2 / B_guess
    B_gamma_red_guess = np.array([B_guess, gamma_red_guess])

    l_max_data = chunked_samples_and_weights_data["l_max"]
    if l_max > l_max_data:
        raise ValueError("l_max > l_max_data")
    l_range_for_fit = range(2, l_max + 1)
    X = np.array(l_range_for_fit)
    chunked_Y = chunked_samples_and_weights_data["chunked_y"][:, X]
    chunked_S = chunked_samples_and_weights_data["chunked_s"][:, X]

    # N_chunks, _ = chunked_Y.shape
    # chunked_Y = np.array([chunked_Y[c] for c in range(N_chunks) if c != 0 and c != 3])
    # chunked_S = np.array([chunked_S[c] for c in range(N_chunks) if c != 0 and c != 3])

    N_chunks, _ = chunked_Y.shape
    fit_B_gamma_chunked = []
    err_B_gamma_chunked = []

    for n_chunk in range(N_chunks):
        Y = chunked_Y[n_chunk]

        if inverse_variance_weights:
            S = chunked_S[n_chunk]
            # S *= np.sqrt(X**6)
        else:
            S = np.ones_like(X)

        def res_B_gamma_red(beta):
            B, gamma_red = beta
            return (
                np.array(
                    [
                        y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                        for l, y in zip(X, Y)
                    ]
                )
                / S
            )

        lsqr_out_B_gamma_red = least_squares(
            res_B_gamma_red,
            B_gamma_red_guess,
            ftol=tol,
            gtol=tol,
            xtol=tol,
            max_nfev=100000,
        )
        fit_B_gamma_red = lsqr_out_B_gamma_red.x
        fit_B_gamma = np.array(
            [fit_B_gamma_red[0], fit_B_gamma_red[1] * fit_B_gamma_red[0] / R0**2]
        )

        err_B_gamma = np.abs(fit_B_gamma - B_gamma_actual) / B_gamma_actual

        fit_B_gamma_chunked.append(fit_B_gamma)
        err_B_gamma_chunked.append(err_B_gamma)

    fit_B_gamma_chunked = np.array(fit_B_gamma_chunked)
    err_B_gamma_chunked = np.array(err_B_gamma_chunked)

    fit_B_gamma = np.mean(fit_B_gamma_chunked, axis=0)
    err_B_gamma = np.abs(fit_B_gamma - B_gamma_actual) / B_gamma_actual
    fit_B_gamma_std = np.std(fit_B_gamma_chunked, axis=0, ddof=1)
    fit_B_gamma_serror = fit_B_gamma_std / np.sqrt(N_chunks)

    return {
        "fit_B_gamma": fit_B_gamma,
        "err_B_gamma": err_B_gamma,
        "fit_B_gamma_std": fit_B_gamma_std,
        "fit_B_gamma_serror": fit_B_gamma_serror,
        "err_B_gamma_chunked": err_B_gamma_chunked,
        "fit_B_gamma_chunked": fit_B_gamma_chunked,
        "N_chunks": N_chunks,
        "t_start": chunked_samples_and_weights_data["t_start"],
        "dt_sample": chunked_samples_and_weights_data["dt_sample"],
        "Nt_chunk": chunked_samples_and_weights_data["Nt_chunk"],
        "dt_list": chunked_samples_and_weights_data["dt_list"],
    }


#########################################
#########################################
#########################################

t_start = 0.06  # let system thermalize for t_start before sampling
dt_sample_stats = 0.03  # dt between samples for surface tension stats

dt_sample_chunked = 0.06
Nt_chunk = 200
l_max_save = 20  # max spherical harmonic index to compute samples for


fluctuations_analysis_dir = Path("../output/fluctuations_test_analysis_with_drift")

# runs used to compute actual surface tension stats
surface_tension_stats_runs_001280 = [
    fluctuations_analysis_dir / f"fluctuations_test_001280_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]
surface_tension_stats_runs_005120 = [
    fluctuations_analysis_dir / f"fluctuations_test_005120_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]

# path to save surface_tension_stats
surface_tension_stats_path = fluctuations_analysis_dir / "surface_tension_stats.npz"

# runs used to compute samples and weights used for fits for each Nf value
samples_and_weights_runs_001280 = [
    fluctuations_analysis_dir / f"fluctuations_test_001280_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]
samples_and_weights_runs_005120 = [
    fluctuations_analysis_dir / f"fluctuations_test_005120_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]


# paths to save samples and weights data
chunked_100_samples_and_weights_path_001280 = (
    fluctuations_analysis_dir
    / f"chunked_100_samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max_save}.npz"
)
chunked_200_samples_and_weights_path_001280 = (
    fluctuations_analysis_dir
    / f"chunked_200_samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max_save}.npz"
)
chunked_300_samples_and_weights_path_001280 = (
    fluctuations_analysis_dir
    / f"chunked_300_samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max_save}.npz"
)
#
#
#
chunked_50_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_50_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
chunked_100_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_100_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
chunked_200_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_200_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
chunked_300_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_300_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)


# %%
##############################
# Save surface tension stats #
##############################


def save_final_surface_tension_stats():

    # surface_tension_stats_001280 = (
    #     compute_concatenated_run_average_surface_tension_stats(
    #         surface_tension_stats_runs_001280,
    #         t_start=t_start,
    #         dt_sample=dt_sample_stats,
    #     )
    # )

    surface_tension_stats_005120 = (
        compute_concatenated_run_average_surface_tension_stats(
            surface_tension_stats_runs_005120,
            t_start=t_start,
            dt_sample=dt_sample_stats,
        )
    )

    all_stats = [
        # surface_tension_stats_001280,
        surface_tension_stats_005120,
    ]

    Nf = np.array([stats["Nf"] for stats in all_stats])
    Nt = np.array([stats["Nt"] for stats in all_stats])
    average_surface_tension = np.array(
        [stats["average_surface_tension"] for stats in all_stats]
    )
    std_surface_tension = np.array(
        [stats["std_surface_tension"] for stats in all_stats]
    )
    surface_tension_stats_data = {
        "Nf": Nf,
        "Nt": Nt,
        "average_surface_tension": average_surface_tension,
        "std_surface_tension": std_surface_tension,
    }

    np.savez(surface_tension_stats_path, **surface_tension_stats_data)
    return surface_tension_stats_data


# save_final_surface_tension_stats()
# s = compute_concatenated_run_average_surface_tension_stats(
#     surface_tension_stats_runs_005120,
#     t_start=0.0,
#     dt_sample=1e-3,
# )
#
# # 12:45 -
# t_sim = 1.03
# t_elapsed_min = 4 * 60 + 3
# t_run_hrs = (5 / t_sim) * t_elapsed_min / 60
# t_run_hrs
# t_fin = int(13 + t_run_hrs) % 24
# t_fin

# %%
#################################################
# Save chunked fluctuations samples and weights #
#################################################


def save_final_chunked_samples_and_weights():

    # save_chunked_samples_and_weights_from_concatenated_runs(
    #     samples_and_weights_runs_001280,
    #     l_max=l_max_save,
    #     t_start=t_start,
    #     dt_sample=dt_sample_chunked,
    #     Nt_chunk=Nt_chunk,
    #     save_path=chunked_samples_and_weights_path_001280,
    # )

    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=50,
        save_path=chunked_50_samples_and_weights_path_005120,
    )
    #
    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=100,
        save_path=chunked_100_samples_and_weights_path_005120,
    )
    #
    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=200,
        save_path=chunked_200_samples_and_weights_path_005120,
    )
    #
    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=300,
        save_path=chunked_300_samples_and_weights_path_005120,
    )


# save_final_chunked_samples_and_weights()

# %%


# %%
def plot_unweighted_fit_B_gamma_vs_l_max():
    show_all_fits = False
    inverse_variance_weights = False

    fig_path = fluctuations_analysis_dir / "unweighted_fit_B_gamma_vs_l_max.svg"

    Nf = 5120
    # chunked_samples_and_weights_path = chunked_50_samples_and_weights_path_005120
    chunked_samples_and_weights_path = chunked_100_samples_and_weights_path_005120
    # chunked_samples_and_weights_path = chunked_200_samples_and_weights_path_005120
    # chunked_samples_and_weights_path = chunked_300_samples_and_weights_path_005120
    B_actual = 1.0
    B_guess = 1.2
    gamma_guess = 23.0
    fit_tol = 1e-15
    l_min = 2
    l_max = 12
    l_max_range = np.arange(l_min, l_max + 1, dtype="int")
    surface_tension_stats = dict(np.load(surface_tension_stats_path))
    n_Nf = np.where(surface_tension_stats["Nf"] == Nf)[0][0]
    gamma_actual = surface_tension_stats["average_surface_tension"][n_Nf]
    fits = [
        fit_saved_chunked_samples_and_weights_data(
            chunked_samples_and_weights_path,
            B_actual=B_actual,
            gamma_actual=gamma_actual,
            B_guess=B_guess,
            gamma_guess=gamma_guess,
            tol=fit_tol,
            inverse_variance_weights=inverse_variance_weights,
            l_max=l_max,
        )
        for l_max in l_max_range
    ]

    dt_sample = fits[0]["dt_sample"]
    N_chunks = fits[0]["N_chunks"]

    fit_B = np.array([fit["fit_B_gamma"][0] for fit in fits])
    fit_B_std = np.array([fit["fit_B_gamma_std"][0] for fit in fits])
    fit_B_serror = np.array([fit["fit_B_gamma_serror"][0] for fit in fits])

    fit_gamma = np.array([fit["fit_B_gamma"][1] for fit in fits])
    fit_gamma_std = np.array([fit["fit_B_gamma_std"][1] for fit in fits])
    fit_gamma_serror = np.array([fit["fit_B_gamma_serror"][1] for fit in fits])

    # l,c->c,l
    fit_B_chunked = np.array([fit["fit_B_gamma_chunked"][:, 0] for fit in fits]).T
    fit_gamma_chunked = np.array([fit["fit_B_gamma_chunked"][:, 1] for fit in fits]).T

    rcparams0 = dict(plt.rcParams)  # save original rcparams
    # print(rcparams0)
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
            "xtick.minor.ndivs": 5,
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
            # "legend.handlelength": 0.75,
            # "legend.handletextpad": 0.1,
            # "legend.labelspacing": 0.0,
            # "axes.labelpad": 0.0,
            # "xtick.major.pad": 1.5,
            # "ytick.major.pad": 1.5,
            # 'axes.grid': True,
            # 'axes.grid.axis': 'both',
            # 'axes.grid.which': 'minor',
        }
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.25, 5),
        # sharex=True
    )
    # axes[0].set_xlabel(r"$\ell_{\text{max}}$")
    axes[0].set_ylabel(r"$(B_{\text{fit}}-B)/B$")
    axes[1].set_xlabel(r"$\ell_{\text{max}}$")
    axes[1].set_ylabel(
        r"$(\sigma_{\text{fit}}-\langle\overline{\gamma}\rangle)/\langle\overline{\gamma}\rangle$"
    )

    Nf_str = f"{Nf}"
    dt_str = f"{dt_sample:.3g}"
    N_chunks_str = f"{N_chunks}"
    fig.suptitle(
        r"$N_f="
        + Nf_str
        + r"$, "
        + r"$N_{chunks}="
        + N_chunks_str
        + r"$, "
        + r"$dt="
        + dt_str
        + r"$"
    )

    # axes[0].set_title("Stiffness")
    axes[0].errorbar(
        l_max_range,
        (fit_B - B_actual) / B_actual,
        yerr=fit_B_serror / B_actual,
        capsize=3,
        # label=r"$B_{\text{fit}}$",
        marker="o",
    )
    # axes[0].plot(
    #     l_max_range,
    #     0 * l_max_range + 0.0 - 0.1,
    #     "--",
    #     label=r"$B$",
    # )
    if show_all_fits:
        for c in range(N_chunks):
            # if c == 0 or c==3:
            #     continue
            axes[0].plot(
                l_max_range,
                (fit_B_chunked[c] - B_actual) / B_actual,
                "--",
                # label=f"{c=}",
            )
            axes[1].plot(
                l_max_range,
                (fit_gamma_chunked[c] - gamma_actual) / gamma_actual,
                "--",
                # label=f"{c=}",
            )
    #
    axes[0].set_xticks([5, 10])
    # ax.set_xticklabels(xticklabels, rotation=-45)

    # axes[0].legend()

    # axes[1].set_title("Tension")
    axes[1].errorbar(
        l_max_range,
        (fit_gamma - gamma_actual) / gamma_actual,
        yerr=fit_gamma_serror / gamma_actual,
        capsize=3,
        # label=r"$\sigma_{\text{fit}}$",
        marker="o",
    )

    axes[1].set_xticks([5, 10])
    # axes[1].legend()
    # plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.show()
    plt.rcParams.update(rcparams0)


plot_unweighted_fit_B_gamma_vs_l_max()
