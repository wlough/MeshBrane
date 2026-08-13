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


def get_auto_corr(x):
    mean_x = np.mean(x)
    z = x - mean_x
    corr_x = []
    T = len(x)
    corr_x = np.array(
        [
            np.dot(z[: T - tau], np.roll(z, -tau)[: T - tau])
            / np.dot(z[: T - tau], z[: T - tau])
            for tau in range(T)
        ]
    )
    return corr_x


def plot_autocors():
    # output_dir = "../output/fluctuations_test_000320_no_graph"
    output_dir = "../output/fluctuations_test_001280_no_graph"
    # output_dir = "../output/fluctuations_test_005120_no_graph"
    # output_dir = "../output/fluctuations_test_020480_no_graph"
    l_max = 5
    t_start = 0.01
    nt_skip = 1

    T_cor = 5.0
    T_plot = 0.075

    t_stop = t_start + T_cor

    xyz_coord_V_path = f"{output_dir}/raw_data/envelope_xyz_coord_V.dat"
    area_V_path = f"{output_dir}/raw_data/envelope_area_V.dat"
    volume_path = f"{output_dir}/raw_data/envelope_volume.dat"
    normal_V_path = f"{output_dir}/raw_data/envelope_normal_V.dat"
    t_path = f"{output_dir}/raw_data/t.dat"

    big_t = read_time_series(t_path)[::nt_skip]
    t_mask = np.logical_and(big_t >= t_start, big_t <= t_stop)
    big_t = big_t[t_mask][::nt_skip]
    t_plot_mask = big_t <= T_plot
    t_plot = big_t[t_plot_mask] - big_t[0]

    big_xyz_coord_V = read_time_series(xyz_coord_V_path)[::nt_skip][t_mask]
    big_area_V = read_time_series(area_V_path)[::nt_skip][t_mask]
    big_volume = read_time_series(volume_path)[::nt_skip][t_mask]  # ***
    big_normal_V = read_time_series(normal_V_path)[::nt_skip][t_mask]

    Nt, Nv, _ = big_xyz_coord_V.shape
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)

    R0 = (3 * big_volume[0] / (4 * np.pi)) ** (1 / 3)

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

    cor_sqr_Un = np.array([get_auto_corr(u) for u in big_sqr_Un.T])
    cor_sqr_Ulm = [
        [cor_sqr_Un[spherical_harmonic_index_n_LM(l, m)] for m in range(-l, l + 1)]
        for l in range(l_max)
    ]

    for l_plot in range(2, 4):
        for m_plot in range(0, l_plot + 1):
            _ = plt.plot(
                t_plot,
                cor_sqr_Ulm[l_plot][m_plot + l_plot][t_plot_mask],
                label=r"$\ell,m=" + f"{l_plot},{m_plot}" + r"$",
            )
    _ = plt.plot(
        t_plot,
        cor_sqr_Ulm[l_plot][m_plot + l_plot][t_plot_mask] * 0 + 1 / np.e,
    )
    _ = plt.plot(
        t_plot,
        cor_sqr_Ulm[l_plot][m_plot + l_plot][t_plot_mask] * 0,
    )
    _ = plt.legend()

    def get_time_constant(l_plot, m_plot):
        tol = 1e-15
        samps = cor_sqr_Ulm[l_plot][m_plot + l_plot][t_plot_mask]

        def fun(tau):
            return samps - np.exp(-t_plot / tau)

        lsqr_out = least_squares(
            fun,
            0.01,
            ftol=tol,
            gtol=tol,
            xtol=tol,
            max_nfev=100000,
        )
        tau = lsqr_out.x[0]
        return tau

    for l_plot in range(2, 4):
        for m_plot in range(0, l_plot + 1):
            tau = get_time_constant(l_plot, m_plot)
            print(30 * "-")
            print(f"l, m = {l_plot}, {m_plot}")
            print(f"    tau={tau:.2g}")


def get_run_data0(
    output_dir,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
    R0=None,
):
    print(f"get_run_data0 for {output_dir}")
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

    return {
        "big_sqr_Un": big_sqr_Un,
        "R0": R0,
        "dt": dt,
        "Nt": Nt,
        "Nv": Nv,
    }


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


def plot_run_autocor_data(
    output_dir,
    l_max=4,
    t_start=0.1,
    T_cor=5.0,
    T_plot=0.075,
    R0=None,
):
    print(f"plot_run_autocor_data for {output_dir}")
    output_dir = Path(output_dir)
    big_t = read_time_series(output_dir / "raw_data/t.dat")
    dt = big_t[1] - big_t[0]
    # nt_skip = int(dt_sample / dt0) + 1
    nt_start = int(t_start / dt)
    nt_stop = int((t_start + T_cor) / dt) + 1
    big_t = big_t[nt_start:nt_stop]
    big_lag = big_t - big_t[0]

    big_xyz_coord_V = read_time_series(
        output_dir / "raw_data/envelope_xyz_coord_V.dat"
    )[nt_start:nt_stop]
    big_area_V = read_time_series(output_dir / "raw_data/envelope_area_V.dat")[
        nt_start:nt_stop:
    ]
    big_normal_V = read_time_series(output_dir / "raw_data/envelope_normal_V.dat")[
        nt_start:nt_stop
    ]

    Nt, Nv, _ = big_xyz_coord_V.shape

    if R0 is None:
        initial_volume = read_time_series(output_dir / "raw_data/envelope_volume.dat")[
            nt_start
        ]
        R0 = (3 * initial_volume / (4 * np.pi)) ** (1 / 3)
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
        n_l0 = spherical_harmonic_index_n_LM(l, 0)
        n_ll = spherical_harmonic_index_n_LM(l, l)
        big_Q[:, l] = np.sum(big_sqr_Un[:, n_l0 : n_ll + 1], axis=1) / (l + 1)

    big_cor_Q = np.array([get_auto_corr(Q) for Q in big_Q.T]).T
    t_mask = big_lag <= T_plot
    lag = big_lag[t_mask]
    cor_Q = big_cor_Q[t_mask]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(3.25, 3.25),
    )
    for l in range(2, l_max + 1):
        ax.plot(lag, cor_Q[:, l], label=r"$\ell=" + f"{l}" + r"$")
    ax.legend()

    return {
        "big_Q": big_Q,
        "cor_Q": big_cor_Q,
        "R0": R0,
        "dt": dt,
        "Nt": Nt,
        "Nv": Nv,
        "big_lag": big_lag,
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


def get_samples_and_weights_from_concatenated_runs(
    output_dirs,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
):
    # data0 = get_run_Qlj_data(
    #     output_dirs[0],
    #     l_max=l_max,
    #     t_start=t_start,
    #     dt_sample=dt_sample,
    #     R0=None,
    # )
    # R0 = data0["R0"]
    # Nv = data0["Nv"]
    #
    # big_Q = data0["big_Q"]
    #
    # for output_dir in output_dirs[1:]:
    #     data = get_run_Qlj_data(
    #         output_dir,
    #         l_max=l_max,
    #         t_start=t_start,
    #         dt_sample=dt_sample,
    #         R0=R0,
    #     )
    #     if Nv != data["Nv"]:
    #         raise ValueError("runs have different number of vertices")
    #     big_Q = np.concatenate([big_Q, data["big_Q"]], axis=0)
    concatenated_run_Qlj_data = get_concatenated_run_Qlj_data(
        output_dirs,
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
    )
    big_Q = concatenated_run_Qlj_data["big_Q"]
    Nv = concatenated_run_Qlj_data["Nv"]
    R0 = concatenated_run_Qlj_data["R0"]
    Nt = len(big_Q)
    y = np.sum(big_Q, axis=0) / Nt

    s = np.array(
        [
            np.sqrt(np.sum((big_Q[:, l] - y[l]) ** 2) / (Nt * (Nt - 1)))
            for l in range(l_max + 1)
        ]
    )

    return {
        "y": y,
        "s": s,
        "R0": R0,
        "Nv": Nv,
        "Nt": Nt,
    }


def save_samples_and_weights_from_concatenated_runs(
    output_dirs,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
    save_path=None,
):
    concatenated_run_Qlj_data = get_concatenated_run_Qlj_data(
        output_dirs,
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
    )
    big_Q = concatenated_run_Qlj_data["big_Q"]
    Nv = concatenated_run_Qlj_data["Nv"]
    R0 = concatenated_run_Qlj_data["R0"]
    Nt = len(big_Q)
    # y = np.sum(big_Q, axis=0) / Nt
    y = np.mean(big_Q, axis=0)

    # s = np.array(
    #     [
    #         np.sqrt(np.sum((big_Q[:, l] - y[l]) ** 2) / (Nt * (Nt - 1)))
    #         for l in range(l_max + 1)
    #     ]
    # )
    s = np.std(big_Q, axis=0, ddof=1) / np.sqrt(Nt)
    # print(1 / s**2)
    # print(np.linalg.norm(s - ss))
    # print(np.linalg.norm(y - yy))

    data = concatenated_run_Qlj_data | {
        "y": y,
        "s": s,
        "l_max": l_max,
        "t_start": t_start,
        "dt_sample": dt_sample,
        # "output_dirs": output_dirs,
    }
    np.savez(save_path, **data)


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


def fit_saved_samples_and_weights_data(
    path,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
    l_max=20,
):
    samples_and_weights_data = dict(np.load(path, allow_pickle=True))

    R0 = samples_and_weights_data["R0"]
    Nv = samples_and_weights_data["Nv"]
    Nt = samples_and_weights_data["Nt"]
    l_max_data = samples_and_weights_data["l_max"]
    if l_max > l_max_data:
        raise ValueError("l_max > l_max_data")

    l_range_for_fit = range(2, l_max + 1)

    X = np.array(l_range_for_fit)
    Y = samples_and_weights_data["y"][X]

    if inverse_variance_weights:
        S = samples_and_weights_data["s"][X]
    else:
        S = np.ones_like(X)
    kBT = 1.0291e-2

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

    def res_B(beta):
        (B,) = beta
        gamma_red = gamma_actual * R0**2 / B_actual
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            / S
        )

    def res_gamma_red(beta):
        (gamma_red,) = beta
        B = 1.0
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            / S
        )

    gamma_red_guess = gamma_guess * R0**2 / B_guess
    B_gamma_red_guess = np.array([B_guess, gamma_red_guess])
    # print("fitting (B, gamma)")
    # print(f"{B_gamma_red_guess=}")
    lsqr_out_B_gamma_red = least_squares(
        res_B_gamma_red,
        B_gamma_red_guess,
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_B_gamma_red = lsqr_out_B_gamma_red.x
    # print(f"{fit_B_gamma_red=}")
    fit_B_gamma = np.array(
        [fit_B_gamma_red[0], fit_B_gamma_red[1] * fit_B_gamma_red[0] / R0**2]
    )
    # print(f"{fit_B_gamma=}")
    B_gamma_actual = np.array([B_actual, gamma_actual])
    err_B_gamma = np.abs(fit_B_gamma - B_gamma_actual) / B_gamma_actual
    resids_B_gamma_red = res_B_gamma_red(fit_B_gamma_red)
    # print(f"{resids_B_gamma_red=}")
    # print("fitting (B, gamma) -done")

    # print("fitting B")
    # lsqr_out_B = least_squares(
    #     res_B,
    #     np.array([B_guess]),
    #     ftol=tol,
    #     gtol=tol,
    #     xtol=tol,
    #     max_nfev=100000,
    # )
    # fit_B = lsqr_out_B.x[0]
    # err_B = np.abs(fit_B - B_actual) / B_actual
    # print("fitting B -done")
    #
    # print("fitting gamma")
    # lsqr_out_gamma_red = least_squares(
    #     res_gamma_red,
    #     np.array([gamma_red_guess]),
    #     ftol=tol,
    #     gtol=tol,
    #     xtol=tol,
    #     max_nfev=100000,
    # )
    # fit_gamma_red = lsqr_out_gamma_red.x[0]
    # fit_gamma = fit_gamma_red * B_actual / R0**2
    # err_gamma = np.abs(fit_gamma - gamma_actual) / gamma_actual
    # print("fitting gamma -done")

    return {
        "err_B_gamma": err_B_gamma,
        # "err_B": err_B,
        # "err_gamma": err_gamma,
        "fit_B_gamma": fit_B_gamma,
        # "fit_B": fit_B,
        # "fit_gamma": fit_gamma,
        "lsqr_out_B_gamma_red": lsqr_out_B_gamma_red,
    } | samples_and_weights_data


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


def fit_concatenated_runs(
    output_dirs,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
):
    print("Computing samples and weights")
    samples_and_weights_data = get_samples_and_weights_from_concatenated_runs(
        output_dirs,
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
    )
    print("Computing samples and weights -done")

    R0 = samples_and_weights_data["R0"]
    Nv = samples_and_weights_data["Nv"]
    Nt = samples_and_weights_data["Nt"]

    l_range_for_fit = range(2, l_max + 1)

    X = np.array(l_range_for_fit)
    Y = samples_and_weights_data["y"][X]

    if inverse_variance_weights:
        S = samples_and_weights_data["s"][X]
    else:
        S = np.ones_like(X)
    kBT = 1.0291e-2

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

    def res_B(beta):
        (B,) = beta
        gamma_red = gamma_actual * R0**2 / B_actual
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            / S
        )

    def res_gamma_red(beta):
        (gamma_red,) = beta
        B = 1.0
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            / S
        )

    gamma_red_guess = gamma_guess * R0**2 / B_guess

    print("fitting (B, gamma)")
    lsqr_out_B_gamma_red = least_squares(
        res_B_gamma_red,
        np.array([B_guess, gamma_red_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_B_gamma_red = lsqr_out_B_gamma_red.x
    fit_B_gamma = np.array(
        [fit_B_gamma_red[0], fit_B_gamma_red[1] * fit_B_gamma_red[0] / R0**2]
    )
    B_gamma_actual = np.array([B_actual, gamma_actual])
    err_B_gamma = np.abs(fit_B_gamma - B_gamma_actual) / B_gamma_actual
    print("fitting (B, gamma) -done")

    print("fitting B")
    lsqr_out_B = least_squares(
        res_B,
        np.array([B_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_B = lsqr_out_B.x[0]
    err_B = np.abs(fit_B - B_actual) / B_actual
    print("fitting B -done")

    print("fitting gamma")
    lsqr_out_gamma_red = least_squares(
        res_gamma_red,
        np.array([gamma_red_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_gamma_red = lsqr_out_gamma_red.x[0]
    fit_gamma = fit_gamma_red * B_actual / R0**2
    err_gamma = np.abs(fit_gamma - gamma_actual) / gamma_actual
    print("fitting gamma -done")

    return {
        "err_B_gamma": err_B_gamma,
        "err_B": err_B,
        "err_gamma": err_gamma,
        "fit_B_gamma": fit_B_gamma,
        "fit_B": fit_B,
        "fit_gamma": fit_gamma,
    } | samples_and_weights_data


def get_concatenated_run_data0(
    output_dirs,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
):
    data0 = get_run_data0(
        output_dirs[0],
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
        R0=None,
    )
    R0 = data0["R0"]
    Nv = data0["Nv"]

    big_sqr_Un = data0["big_sqr_Un"]

    for output_dir in output_dirs[1:]:
        data = get_run_data0(
            output_dir,
            l_max=l_max,
            t_start=t_start,
            dt_sample=dt_sample,
            R0=R0,
        )
        if Nv != data["Nv"]:
            raise ValueError("runs have different number of vertices")
        big_sqr_Un = np.concatenate([big_sqr_Un, data["big_sqr_Un"]], axis=0)

    Nt, num_harmonics = big_sqr_Un.shape
    mean_sqr_Un = np.mean(big_sqr_Un, axis=0)
    serror_sqr_Un = np.std(big_sqr_Un, axis=0) / np.sqrt(Nt)

    mean_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]
    serror_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            n = spherical_harmonic_index_n_LM(l, m)
            mean_sqr_Ulm[l][m] = mean_sqr_Un[n]
            serror_sqr_Ulm[l][m] = serror_sqr_Un[n]

    mean_sqr_Ul = np.array([np.mean(mean_sqr_Ulm[l][l:]) for l in range(l_max + 1)])
    serror_sqr_Ul = np.array([np.mean(serror_sqr_Ulm[l][l:]) for l in range(l_max + 1)])
    # serror_sqr_Ul = np.array(
    #     [np.linalg.norm(serror_sqr_Ulm[l]) / (l + 1) for l in range(l_max + 1)]
    # )

    return {
        "mean_sqr_Ul": mean_sqr_Ul,
        "serror_sqr_Ul": serror_sqr_Ul,
        "mean_sqr_Un": mean_sqr_Un,
        "serror_sqr_Un": serror_sqr_Un,
        "R0": R0,
        "Nv": Nv,
        "Nt": Nt,
    }


def fit_concatenated_run_samples0(
    output_dirs,
    l_max=20,
    t_start=0.01,
    dt_sample=0.01,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
):
    concatenated_run_data = get_concatenated_run_data0(
        output_dirs,
        l_max=l_max,
        t_start=t_start,
        dt_sample=dt_sample,
    )

    R0 = concatenated_run_data["R0"]
    Nv = concatenated_run_data["Nv"]
    Nt = concatenated_run_data["Nt"]

    mean_sqr_Ul = concatenated_run_data["mean_sqr_Ul"]
    serror_sqr_Ul = concatenated_run_data["serror_sqr_Ul"]

    l_range_for_fit = range(2, l_max + 1)

    X = np.array(l_range_for_fit)
    Y = mean_sqr_Ul[X]
    if inverse_variance_weights:
        W = 1 / serror_sqr_Ul[X]
    else:
        W = np.ones_like(X)
    kBT = 1.0291e-2

    def res_B_gamma_red(beta):
        B, gamma_red = beta
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            * W
        )

    def res_B(beta):
        (B,) = beta
        gamma_red = gamma_actual * R0**2 / B_actual
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            * W
        )

    def res_gamma_red(beta):
        (gamma_red,) = beta
        B = 1.0
        return (
            np.array(
                [
                    y - kBT / (B * (l - 1) * (l + 2) * (gamma_red + l * (l + 1)))
                    for l, y in zip(X, Y)
                ]
            )
            * W
        )

    gamma_red_guess = gamma_guess * R0**2 / B_guess

    lsqr_out_B_gamma_red = least_squares(
        res_B_gamma_red,
        np.array([B_guess, gamma_red_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_B_gamma_red = lsqr_out_B_gamma_red.x
    fit_B_gamma = np.array(
        [fit_B_gamma_red[0], fit_B_gamma_red[1] * fit_B_gamma_red[0] / R0**2]
    )

    lsqr_out_B = least_squares(
        res_B,
        np.array([B_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_B = lsqr_out_B.x[0]

    lsqr_out_gamma_red = least_squares(
        res_gamma_red,
        np.array([gamma_red_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_gamma_red = lsqr_out_gamma_red.x[0]
    fit_gamma = fit_gamma_red * B_actual / R0**2

    return {
        "fit_B_gamma": fit_B_gamma,
        "fit_B": fit_B,
        "fit_gamma": fit_gamma,
    } | concatenated_run_data


def save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
    output_dirs,
    save_paths,
    dt_samples,
    l_max=20,
    t_start=0.1,
):

    for dt_sample, save_path in zip(dt_samples, save_paths):
        print(f"{dt_sample=}")
        save_samples_and_weights_from_concatenated_runs(
            output_dirs,
            l_max=l_max,
            t_start=t_start,
            dt_sample=dt_sample,
            save_path=save_path,
        )


# plot_run_autocor_data(
#     "../output/fluctuations_test_analysis/fluctuations_test_005120_no_graph",
#     l_max=4,
#     t_start=0.1,
#     T_cor=5.0,
#     T_plot=0.075,
#     R0=None,
# )

t_start = 0.06  # let system thermalize for t_start before sampling
dt_sample_stats = 0.03  # dt between samples for surface tension stats
dt_samples_save = 1 / np.linspace(
    1 / 1.0,
    1 / 0.02,
    20,
)  # dt's between samples

dt_sample_chunked = 0.06
Nt_chunk = 200
l_max_save = 20  # max spherical harmonic index to compute samples for


fluctuations_analysis_dir = Path("../output/fluctuations_test_analysis")

# runs used to compute actual surface tension stats
surface_tension_stats_runs_000320 = [
    fluctuations_analysis_dir / f"fluctuations_test_000320_sweep1/run_{_:0>2}_00"
    for _ in range(16)
]
surface_tension_stats_runs_001280 = [
    fluctuations_analysis_dir / f"fluctuations_test_001280_sweep1/run_{_:0>2}_00"
    for _ in range(16)
]
surface_tension_stats_runs_005120 = [
    fluctuations_analysis_dir / f"fluctuations_test_005120_sweep1/run_{_:0>2}_00"
    for _ in range(16)
]
surface_tension_stats_runs_020480 = [
    Path("../output/fluctuations_test_020480_sweep") / f"run_{_:0>2}_00"
    for _ in range(16)
]

# path to save surface_tension_stats
surface_tension_stats_path = fluctuations_analysis_dir / "surface_tension_stats.npz"

# runs used to compute samples and weights used for fits
samples_and_weights_runs_000320 = [
    fluctuations_analysis_dir / "fluctuations_test_000320_no_graph"
]
samples_and_weights_runs_000320 += [
    fluctuations_analysis_dir / f"fluctuations_test_000320_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]
samples_and_weights_runs_000320 += [
    fluctuations_analysis_dir / f"fluctuations_test_000320_sweep1/run_{_:0>2}_00"
    for _ in range(16)
]

samples_and_weights_runs_001280 = [
    fluctuations_analysis_dir / "fluctuations_test_001280_no_graph"
]
samples_and_weights_runs_001280 += [
    fluctuations_analysis_dir / f"fluctuations_test_001280_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]
samples_and_weights_runs_001280 += [
    fluctuations_analysis_dir / f"fluctuations_test_001280_sweep1/run_{_:0>2}_00"
    for _ in range(16)
]

samples_and_weights_runs_005120 = [
    fluctuations_analysis_dir / "fluctuations_test_005120_no_graph"
]
samples_and_weights_runs_005120 += [
    fluctuations_analysis_dir / f"fluctuations_test_005120_sweep0/run_{_:0>2}_00"
    for _ in range(16)
]
samples_and_weights_runs_005120 += [
    fluctuations_analysis_dir / f"fluctuations_test_005120_sweep1/run_{_:0>2}_00"
    for _ in range(16)
]

samples_and_weights_runs_020480 = [
    fluctuations_analysis_dir / "fluctuations_test_020480_no_graph"
]
samples_and_weights_runs_020480 += [
    Path("../output/fluctuations_test_020480_sweep") / f"run_{_:0>2}_00"
    for _ in range(16)
]


# get_concatenated_run_Qlj_data(
#     samples_and_weights_runs_020480,
#     l_max=20,
#     t_start=0.01,
#     dt_sample=0.01,
# )


# paths to save samples and weights data
samples_and_weights_paths_000320 = [
    fluctuations_analysis_dir
    / f"samples_and_weights_data_Nf_{320:0>6}_l_max_{l_max_save}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples_save))
]
samples_and_weights_paths_001280 = [
    fluctuations_analysis_dir
    / f"samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max_save}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples_save))
]
samples_and_weights_paths_005120 = [
    fluctuations_analysis_dir
    / f"samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples_save))
]
samples_and_weights_paths_020480 = [
    fluctuations_analysis_dir
    / f"samples_and_weights_data_Nf_{20480:0>6}_l_max_{l_max_save}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples_save))
]


chunked_samples_and_weights_path_000320 = (
    fluctuations_analysis_dir
    / f"chunked_samples_and_weights_data_Nf_{320:0>6}_l_max_{l_max_save}.npz"
)
chunked_samples_and_weights_path_001280 = (
    fluctuations_analysis_dir
    / f"chunked_samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max_save}.npz"
)
chunked_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
#
#
#
chunked_100_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_100_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
#
#
#
chunked_200_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_200_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
#
#
#
chunked_300_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_300_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)
#
#
#
chunked_500_samples_and_weights_path_005120 = (
    fluctuations_analysis_dir
    / f"chunked_500_samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max_save}.npz"
)


# %%
##############################
# Save surface tension stats #
##############################


def save_final_surface_tension_stats():
    surface_tension_stats_000320 = (
        compute_concatenated_run_average_surface_tension_stats(
            surface_tension_stats_runs_000320,
            t_start=t_start,
            dt_sample=dt_sample_stats,
        )
    )

    surface_tension_stats_001280 = (
        compute_concatenated_run_average_surface_tension_stats(
            surface_tension_stats_runs_001280,
            t_start=t_start,
            dt_sample=dt_sample_stats,
        )
    )

    surface_tension_stats_005120 = (
        compute_concatenated_run_average_surface_tension_stats(
            surface_tension_stats_runs_005120,
            t_start=t_start,
            dt_sample=dt_sample_stats,
        )
    )

    # surface_tension_stats_020480 = (
    #     compute_concatenated_run_average_surface_tension_stats(
    #         surface_tension_stats_runs_020480,
    #         t_start=0.0,
    #         dt_sample=0.01,
    #     )
    # )

    all_stats = [
        surface_tension_stats_000320,
        surface_tension_stats_001280,
        surface_tension_stats_005120,
        # surface_tension_stats_020480,
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

# %%
#########################################
# Save fluctuations samples and weights #
#########################################


def save_final_samples_and_weights_for_dt_samples_range():

    save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
        samples_and_weights_runs_000320,
        samples_and_weights_paths_000320,
        dt_samples_save,
        l_max=l_max_save,
        t_start=t_start,
    )

    save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
        samples_and_weights_runs_001280,
        samples_and_weights_paths_001280,
        dt_samples_save,
        l_max=l_max_save,
        t_start=t_start,
    )

    save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
        samples_and_weights_runs_005120,
        samples_and_weights_paths_005120,
        dt_samples_save,
        l_max=l_max_save,
        t_start=t_start,
    )


# save_final_samples_and_weights_for_dt_samples_range()
# %%
#################################################
# Save chunked fluctuations samples and weights #
#################################################


def save_final_chunked_samples_and_weights():

    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_000320,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=Nt_chunk,
        save_path=chunked_samples_and_weights_path_000320,
    )

    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_001280,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=Nt_chunk,
        save_path=chunked_samples_and_weights_path_001280,
    )

    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=Nt_chunk,
        save_path=chunked_samples_and_weights_path_005120,
    )
    #
    #
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
    #
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
    #
    #
    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=300,
        save_path=chunked_300_samples_and_weights_path_005120,
    )

    save_chunked_samples_and_weights_from_concatenated_runs(
        samples_and_weights_runs_005120,
        l_max=l_max_save,
        t_start=t_start,
        dt_sample=dt_sample_chunked,
        Nt_chunk=500,
        save_path=chunked_500_samples_and_weights_path_005120,
    )


save_final_chunked_samples_and_weights()


# %%
def plot_unweighted_fit_B_gamma_vs_l_max():

    B_actual = 1.0
    B_guess = 1.2
    gamma_guess = 23.0
    fit_tol = 1e-15
    l_min = 4
    l_max = 12
    l_max_range = np.arange(l_min, l_max + 1, dtype="int")
    n_Nf = 2
    fig_path = fluctuations_analysis_dir / "unweighted_fit_B_gamma_vs_l_max.svg"

    chunked_samples_and_weights_paths = [
        chunked_samples_and_weights_path_000320,
        chunked_samples_and_weights_path_001280,
        chunked_samples_and_weights_path_005120,
    ]
    Nf_list = [320, 1280, 5120]
    Nf = Nf_list[n_Nf]
    surface_tension_stats = dict(np.load(surface_tension_stats_path))
    gamma_actual = surface_tension_stats["average_surface_tension"][n_Nf]
    chunked_samples_and_weights_path = chunked_samples_and_weights_paths[n_Nf]

    fits = [
        fit_saved_chunked_samples_and_weights_data(
            chunked_samples_and_weights_path,
            B_actual=B_actual,
            gamma_actual=gamma_actual,
            B_guess=B_guess,
            gamma_guess=gamma_guess,
            tol=fit_tol,
            inverse_variance_weights=False,
            l_max=l_max,
        )
        for l_max in l_max_range
    ]
    # return fits

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
    # fig.suptitle(
    #     r"$N_f="
    #     + Nf_str
    #     + r"$, "
    #     + r"$N_{chunks}="
    #     + N_chunks_str
    #     + r"$, "
    #     + r"$dt="
    #     + dt_str
    #     + r"$"
    # )

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
    #     0 * l_max_range + 1.0,
    #     "--",
    #     label=r"$B$",
    # )
    #
    # for c in range(N_chunks):
    #     # if c == 0 or c==3:
    #     #     continue
    #     axes[0].plot(
    #         l_max_range,
    #         fit_B_chunked[c],
    #         ".",
    #         # label=f"{c=}",
    #     )
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
    # axes[1].plot(
    #     l_max_range,
    #     0 * l_max_range + gamma_actual,
    #     "--",
    #     label=r"$\left\langle\overline{\gamma}\right\rangle$",
    # )
    #
    # for c in range(N_chunks):
    #     # if c == 0 or c==3:
    #     #     continue
    #     axes[1].plot(
    #         l_max_range,
    #         fit_gamma_chunked[c],
    #         ".",
    #         # label=f"{c=}",
    #     )
    #
    axes[1].set_xticks([5, 10])
    # axes[1].legend()
    # plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.show()
    plt.rcParams.update(rcparams0)


plot_unweighted_fit_B_gamma_vs_l_max()


# %%
def plot_weighted_and_unweighted_fit_B_gamma_vs_l_max():
    B_guess = 1.2
    gamma_guess = 23.0
    fit_tol = 1e-15
    l_max_range = np.arange(4, 13)
    n_Nf = 2

    chunked_samples_and_weights_paths = [
        chunked_samples_and_weights_path_000320,
        chunked_samples_and_weights_path_001280,
        chunked_500_samples_and_weights_path_005120,
    ]
    Nf_list = [320, 1280, 5120]
    Nf = Nf_list[n_Nf]
    surface_tension_stats = dict(np.load(surface_tension_stats_path))
    gamma_actual = surface_tension_stats["average_surface_tension"][n_Nf]
    chunked_samples_and_weights_path = chunked_samples_and_weights_paths[n_Nf]

    weighted_fits = [
        fit_saved_chunked_samples_and_weights_data(
            chunked_samples_and_weights_path,
            B_actual=1.0,
            gamma_actual=gamma_actual,
            B_guess=B_guess,
            gamma_guess=gamma_guess,
            tol=fit_tol,
            inverse_variance_weights=True,
            l_max=l_max,
        )
        for l_max in l_max_range
    ]

    weighted_B = np.array([fit["fit_B_gamma"][0] for fit in weighted_fits])
    weighted_B_std = np.array([fit["fit_B_gamma_std"][0] for fit in weighted_fits])
    weighted_B_serror = np.array(
        [fit["fit_B_gamma_serror"][0] for fit in weighted_fits]
    )

    weighted_gamma = np.array([fit["fit_B_gamma"][1] for fit in weighted_fits])
    weighted_gamma_std = np.array([fit["fit_B_gamma_std"][1] for fit in weighted_fits])
    weighted_gamma_serror = np.array(
        [fit["fit_B_gamma_serror"][1] for fit in weighted_fits]
    )

    unweighted_fits = [
        fit_saved_chunked_samples_and_weights_data(
            chunked_samples_and_weights_path,
            B_actual=1.0,
            gamma_actual=gamma_actual,
            B_guess=B_guess,
            gamma_guess=gamma_guess,
            tol=fit_tol,
            inverse_variance_weights=False,
            l_max=l_max,
        )
        for l_max in l_max_range
    ]

    dt_sample = unweighted_fits[0]["dt_sample"]
    N_chunks = unweighted_fits[0]["N_chunks"]

    unweighted_B = np.array([fit["fit_B_gamma"][0] for fit in unweighted_fits])
    unweighted_B_std = np.array([fit["fit_B_gamma_std"][0] for fit in unweighted_fits])
    unweighted_B_serror = np.array(
        [fit["fit_B_gamma_serror"][0] for fit in unweighted_fits]
    )

    unweighted_gamma = np.array([fit["fit_B_gamma"][1] for fit in unweighted_fits])
    unweighted_gamma_std = np.array(
        [fit["fit_B_gamma_std"][1] for fit in unweighted_fits]
    )
    unweighted_gamma_serror = np.array(
        [fit["fit_B_gamma_serror"][1] for fit in unweighted_fits]
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(6.75, 3.25),
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

    axes[0].set_title("Stiffness")
    axes[0].errorbar(
        l_max_range,
        weighted_B,
        yerr=weighted_B_serror,
        capsize=3,
        label="weighted",
    )
    axes[0].errorbar(
        l_max_range,
        unweighted_B,
        yerr=unweighted_B_serror,
        capsize=3,
        label="unweighted",
    )
    axes[0].plot(
        l_max_range,
        0 * l_max_range + 1.0,
        "--",
        label=r"$B$",
    )
    axes[0].legend()

    axes[1].set_title("Tension")
    axes[1].errorbar(
        l_max_range,
        weighted_gamma,
        yerr=weighted_gamma_serror,
        capsize=3,
        label="weighted",
    )
    axes[1].errorbar(
        l_max_range,
        unweighted_gamma,
        yerr=unweighted_gamma_serror,
        capsize=3,
        label="unweighted",
    )
    axes[1].plot(
        l_max_range,
        0 * l_max_range + gamma_actual,
        "--",
        label=r"$\gamma$",
    )
    axes[1].legend()
    plt.tight_layout()
    plt.show()


plot_weighted_and_unweighted_fit_B_gamma_vs_l_max()


# %%
def plot_err_B_gamma_for_dt_samples_range():
    B_guess = 1.2
    gamma_guess = 23.0
    fit_tol = 1e-15
    l_max_list = [4, 6, 10]

    surface_tension_stats = dict(np.load(surface_tension_stats_path))

    samples_and_weights_paths_list = [
        samples_and_weights_paths_000320,
        samples_and_weights_paths_001280,
        samples_and_weights_paths_005120,
    ]
    Nf_list = surface_tension_stats["Nf"]
    gamma_actual_list = surface_tension_stats["average_surface_tension"]

    # Nf -by- weighted yes/no -by- dt_sample
    all_fit_data = [
        [
            [
                fit_saved_samples_and_weights_data(
                    path,
                    B_actual=1.0,
                    gamma_actual=gamma_actual,
                    B_guess=B_guess,
                    gamma_guess=gamma_guess,
                    tol=fit_tol,
                    inverse_variance_weights=False,
                    l_max=l_max,
                )
                for path in samples_and_weights_paths
            ],
            [
                fit_saved_samples_and_weights_data(
                    path,
                    B_actual=1.0,
                    gamma_actual=gamma_actual,
                    B_guess=B_guess,
                    gamma_guess=gamma_guess,
                    tol=fit_tol,
                    inverse_variance_weights=True,
                    l_max=l_max,
                )
                for path in samples_and_weights_paths
            ],
        ]
        for samples_and_weights_paths, gamma_actual, l_max in zip(
            samples_and_weights_paths_list, gamma_actual_list, l_max_list
        )
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(10, 15),
        # sharey=True,
    )
    # fig.suptitle(f"{N}")

    for n in range(3):
        Nf = Nf_list[n]
        l_max = l_max_list[n]
        fit_data_unweighted = all_fit_data[n][0]
        fit_data_weighted = all_fit_data[n][1]

        Nt_samps_unweighted = np.array([dat["Nt"] for dat in fit_data_unweighted])
        err_B_unweighted = np.array(
            [dat["err_B_gamma"][0] for dat in fit_data_unweighted]
        )
        err_gamma_unweighted = np.array(
            [dat["err_B_gamma"][1] for dat in fit_data_unweighted]
        )

        Nt_samps_weighted = np.array([dat["Nt"] for dat in fit_data_weighted])
        err_B_weighted = np.array([dat["err_B_gamma"][0] for dat in fit_data_weighted])
        err_gamma_weighted = np.array(
            [dat["err_B_gamma"][1] for dat in fit_data_weighted]
        )

        ax_B, ax_gamma = axes[n]

        ax_B.plot(Nt_samps_unweighted, err_B_unweighted, "-o", label="unweighted")
        ax_B.plot(Nt_samps_weighted, err_B_weighted, "-o", label="weighted")
        ax_B.set_title(
            r"Relative $B$ error, "
            + r"$N_f="
            + f"{Nf}"
            + r"$, "
            + r"$l_{\text{max}}="
            + f"{l_max}"
            + r"$"
        )
        ax_B.set_xlabel(r"$N_t$")
        ax_B.set_ylabel("error")
        ax_B.legend()

        ax_gamma.plot(
            Nt_samps_unweighted, err_gamma_unweighted, "-o", label="unweighted"
        )
        ax_gamma.plot(Nt_samps_weighted, err_gamma_weighted, "-o", label="weighted")
        ax_gamma.set_title(
            r"Relative $\gamma$ error, "
            + r"$N_f="
            + f"{Nf}"
            + r"$, "
            + r"$l_{\text{max}}="
            + f"{l_max}"
            + r"$"
        )
        ax_gamma.set_xlabel(r"$N_t$")
        ax_gamma.set_ylabel("error")
        ax_gamma.legend()


plot_err_B_gamma_for_dt_samples_range()
# %%


def plot_mean_sqr_amplitudes_vs_l_for_Nf_range():
    l_range = np.array(range(2, 10))
    n_dt_sample = 5
    samples_and_weights_paths_list = [
        samples_and_weights_paths_000320,
        samples_and_weights_paths_001280,
        samples_and_weights_paths_005120,
    ]
    all_samples_and_weights_data = [
        [dict(np.load(path, allow_pickle=True)) for path in samples_and_weights_paths]
        for samples_and_weights_paths in samples_and_weights_paths_list
    ]

    samples_and_weights_data_000320 = [
        dict(np.load(path, allow_pickle=True))
        for path in samples_and_weights_paths_000320
    ]
    samples_and_weights_data_001280 = [
        dict(np.load(path, allow_pickle=True))
        for path in samples_and_weights_paths_001280
    ]
    samples_and_weights_data_005120 = [
        dict(np.load(path, allow_pickle=True))
        for path in samples_and_weights_paths_005120
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(6.75, 3.25),
    )

    Nf_list = [320, 1280, 5120]
    for n_Nf in range(3):
        samples_and_weights_data = all_samples_and_weights_data[n_Nf][n_dt_sample]
        dt_sample = samples_and_weights_data["dt_sample"]
        Nt = samples_and_weights_data["Nt"]
        # Nf = samples_and_weights_data["Nf"]
        Nf = Nf_list[n_Nf]
        # print(samples_and_weights_data.keys())
        mean_sqr_U = samples_and_weights_data["y"]
        std_mean_sqr_U = samples_and_weights_data["s"]

        Nt_str = f"{Nt}"
        dt_str = f"{dt_sample:.3g}"
        fig.suptitle(r"$N_t=" + Nt_str + r"$, " + r"$dt=" + dt_str + r"$")

        axes[0].plot(
            l_range,
            mean_sqr_U[l_range],
            "-o",
            label=r"$N_f=" + f"{Nf}" + r"$",
        )
        axes[0].set_xlabel(r"$\ell$")
        axes[0].set_ylabel(r"$\overline{Q_{\ell}}$")
        axes[0].legend()
        axes[1].plot(
            l_range,
            std_mean_sqr_U[l_range],
            "-o",
            label=r"$N_f=" + f"{Nf}" + r"$",
        )
        axes[1].set_xlabel(r"$\ell$")
        axes[1].set_ylabel(r"$\sigma_{\overline{Q_{\ell}}}$")
        axes[1].legend()

    # ax.errorbar(l_range, mean_sqr_U[l_range], yerr=10 * std_mean_sqr_U[l_range])
    # for l in l_range:
    #     ax.errorbar()
    plt.tight_layout()
    plt.show()


# %%
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# %%

samples_and_weights_runs_020480 = [
    fluctuations_analysis_dir / "fluctuations_test_020480_no_graph"
]
samples_and_weights_runs_020480 += [
    Path("../output/fluctuations_test_020480_sweep") / f"run_{_:0>2}_00"
    for _ in range(16)
]
fit_dat = fit_concatenated_runs(
    samples_and_weights_runs_020480,
    l_max=20,
    t_start=0.0,
    dt_sample=0.01,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
)

# get_run_data(
#     f"../output/fluctuations_test_000320_no_graph",
#     l_max=20,
#     t_start=0.01,
#     dt_sample=0.01,
#     R0=None,
# )
# %%
# output_dirs_000320 = [f"../output/fluctuations_test_000320_no_graph"]
# output_dirs_000320 += [
#     f"../output/fluctuations_test_000320_sweep0/run_{_:0>2}_00" for _ in range(16)
# ]
# output_dirs_000320 += [
#     f"../output/fluctuations_test_000320_sweep/run_{_:0>2}_00" for _ in range(16)
# ]
# output_dirs_001280 = [f"../output/fluctuations_test_001280_no_graph"]
# output_dirs_001280 += [
#     f"../output/fluctuations_test_001280_sweep0/run_{_:0>2}_00" for _ in range(16)
# ]
# # output_dirs_001280 += [
# #     f"../output/fluctuations_test_001280_sweep/run_{_:0>2}_00" for _ in range(16)
# # ]
# output_dirs_005120 = [f"../output/fluctuations_test_005120_no_graph"]
# output_dirs_005120 += [
#     f"../output/fluctuations_test_005120_sweep0/run_{_:0>2}_00" for _ in range(16)
# ]
# # output_dirs_005120 += [
# #     f"../output/fluctuations_test_005120_sweep/run_{_:0>2}_00" for _ in range(16)
# # ]
#
#
# dt_samples = 1 / np.linspace(1 / 1.0, 1 / 0.02, 20)
# l_max = 20
# t_start = 0.1
#
# save_paths_000320 = [
#     f"../output/fluctuations_test_analysis/samples_and_weights_data_Nf_{320:0>6}_l_max_{l_max}_dt_{n_dt}.npz"
#     for n_dt in range(len(dt_samples))
# ]
# save_paths_001280 = [
#     f"../output/fluctuations_test_analysis/samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max}_dt_{n_dt}.npz"
#     for n_dt in range(len(dt_samples))
# ]
# save_paths_005120 = [
#     f"../output/fluctuations_test_analysis/samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max}_dt_{n_dt}.npz"
#     for n_dt in range(len(dt_samples))
# ]
#
#
# save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
#     output_dirs_000320,
#     save_paths_000320,
#     dt_samples,
#     l_max=l_max,
#     t_start=t_start,
# )
#
# save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
#     output_dirs_001280,
#     save_paths_001280,
#     dt_samples,
#     l_max=l_max,
#     t_start=t_start,
# )
#
# save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
#     output_dirs_005120,
#     save_paths_005120,
#     dt_samples,
#     l_max=l_max,
#     t_start=t_start,
# )

# %%
Nf = np.array([320, 1280, 5120, 20480])
radius_over_edge_length = np.array([3.34077128, 6.63439233, 13.2451913, 26.4785837])
L_max = np.array([3, 6, 10])

# save_paths = save_paths_001280
save_paths = samples_and_weights_paths_005120
B_guess = 1.0
gamma_guess = 32.5
weighted = False
tol = 1e-15
l_max = 10

Nt_list = []
err_B_list = []
for path in save_paths:
    fit_data = fit_saved_samples_and_weights_data(
        path,
        B_actual=1.0,
        gamma_actual=32.5,
        B_guess=B_guess,
        gamma_guess=gamma_guess,
        tol=tol,
        inverse_variance_weights=False,
        l_max=l_max,
    )
    Nt_list.append(fit_data["Nt"])
    err_B_list.append(fit_data["err_B_gamma"][0])

plt.plot(Nt_list, err_B_list, "-o", label="unweighted")

Nt_list = []
err_B_list = []
for path in save_paths:
    fit_data = fit_saved_samples_and_weights_data(
        path,
        B_actual=1.0,
        gamma_actual=32.5,
        B_guess=B_guess,
        gamma_guess=gamma_guess,
        tol=tol,
        inverse_variance_weights=True,
        l_max=l_max,
    )
    Nt_list.append(fit_data["Nt"])
    err_B_list.append(fit_data["err_B_gamma"][0])
    # print(f"dt={fit_data["dt_list"]}")

plt.plot(Nt_list, err_B_list, "-o", label="weighted")

plt.legend()
# %%


fit_concatenated_run_samples0(
    output_dirs_001280,
    l_max=20,
    t_start=0.1,
    dt_sample=0.5,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=B_guess,
    gamma_guess=gamma_guess,
    tol=tol,
    inverse_variance_weights=True,
)

# %%

fit_data_001280_weighted = fit_concatenated_runs(
    output_dirs_001280,
    l_max=20,
    t_start=0.1,
    dt_sample=1.0,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
)
fit_data_001280_not_weighted = fit_concatenated_runs(
    output_dirs_001280,
    l_max=20,
    t_start=0.03,
    dt_sample=0.03,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=False,
)

fit_data_005120_weighted = fit_concatenated_runs(
    output_dirs_005120,
    l_max=20,
    t_start=0.03,
    dt_sample=0.3,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=True,
)
fit_data_005120_not_weighted = fit_concatenated_runs(
    output_dirs_005120,
    l_max=20,
    t_start=0.1,
    dt_sample=1.0,
    B_actual=1.0,
    gamma_actual=32.5,
    B_guess=1.0,
    gamma_guess=32.5,
    tol=1e-15,
    inverse_variance_weights=False,
)


# %%
def get_fluctuation_fit_data(output_dir):
    # output_dir = "../output/fluctuations_test_000320_no_graph"
    # output_dir = "../output/fluctuations_test_001280_no_graph"
    # output_dir = "../output/fluctuations_test_005120_no_graph"
    # output_dir = "../output/fluctuations_test_020480_no_graph"

    l_max = 20
    t_start = 0.01
    t_stop = 2.5
    nt_skip = 25
    tol = 1e-15

    l_max = 20
    t_start = 2.51
    t_stop = 5.0
    nt_skip = 25
    tol = 1e-15

    xyz_coord_V_path = f"{output_dir}/raw_data/envelope_xyz_coord_V.dat"
    area_V_path = f"{output_dir}/raw_data/envelope_area_V.dat"
    volume_path = f"{output_dir}/raw_data/envelope_volume.dat"
    normal_V_path = f"{output_dir}/raw_data/envelope_normal_V.dat"
    t_path = f"{output_dir}/raw_data/t.dat"

    big_t = read_time_series(t_path)

    t_mask = np.logical_and(big_t >= t_start, big_t <= t_stop)
    #
    big_t = big_t[t_mask][::nt_skip]
    big_xyz_coord_V = read_time_series(xyz_coord_V_path)[t_mask][::nt_skip]
    big_area_V = read_time_series(area_V_path)[t_mask][::nt_skip]
    big_volume = read_time_series(volume_path)[t_mask][::nt_skip]  # ***
    big_normal_V = read_time_series(normal_V_path)[t_mask][::nt_skip]

    Nt, Nv, _ = big_xyz_coord_V.shape
    n_max = spherical_harmonic_index_n_LM(l_max, l_max)
    print(f"dt={big_t[1]-big_t[0]:.2g}, Nt={Nt}")

    R0 = (3 * big_volume[0] / (4 * np.pi)) ** (1 / 3)

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

    mean_sqr_Un = np.mean(big_sqr_Un, axis=0)

    mean_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            n = spherical_harmonic_index_n_LM(l, m)
            mean_sqr_Ulm[l][m] = mean_sqr_Un[n]
            # std_sqr_Ulm[l][m] = std_sqr_Un[n]

    mean_sqr_Ul = np.array([np.mean(mean_sqr_Ulm[l]) for l in range(l_max + 1)])

    ell_range = [l for l in range(2, l_max + 1)]

    X = np.array(ell_range)
    Y = mean_sqr_Ul[X]
    kBT = 1.0291e-2

    def res_B_gamma(beta):
        B, gamma = beta
        return np.array(
            [
                y - kBT / ((l - 1) * (l + 2) * (R0**2 * gamma + B * l * (l + 1)))
                for l, y in zip(X, Y)
            ]
        )

    def res_B(beta):
        (B,) = beta
        gamma = 32.5
        return np.array(
            [
                y - kBT / ((l - 1) * (l + 2) * (R0**2 * gamma + B * l * (l + 1)))
                for l, y in zip(X, Y)
            ]
        )

    def res_gamma(beta):
        (gamma,) = beta
        B = 1.0
        return np.array(
            [
                y - kBT / ((l - 1) * (l + 2) * (R0**2 * gamma + B * l * (l + 1)))
                for l, y in zip(X, Y)
            ]
        )

    B_guess = 0.8
    gamma_guess = 25.0

    lsqr_out_B_gamma = least_squares(
        res_B_gamma,
        np.array([B_guess, gamma_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )

    fit_B_gamma = lsqr_out_B_gamma.x

    lsqr_out_B = least_squares(
        res_B,
        np.array([B_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )

    fit_B = lsqr_out_B.x[0]

    lsqr_out_gamma = least_squares(
        res_gamma,
        np.array([gamma_guess]),
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )

    fit_gamma = lsqr_out_gamma.x[0]

    return {
        "mean_sqr_Ul": mean_sqr_Ul,
        "R0": R0,
        "Nt": Nt,
        "fit_B_gamma": fit_B_gamma,
        "fit_B": fit_B,
        "fit_gamma": fit_gamma,
    }


Nf = np.array(
    [
        # 320,
        1280,
        # 5120,
        # 20480,
    ]
)
Ne = 3 * Nf / 2
Nv = 2 + Nf / 2

B_actual = 1.0
gamma_actual = 32.5

B_guess = 0.8
gamma_guess = 25.0

B_gamma_actual = np.array([B_actual, gamma_actual])

output_dirs = [f"../output/fluctuations_test_{nf:0>6}_no_graph" for nf in Nf]
ell_fit = np.arange(2, 21)

fluctuation_fit_data = [get_fluctuation_fit_data(path) for path in output_dirs]
mean_sqr_Ul = np.array([data["mean_sqr_Ul"] for data in fluctuation_fit_data])

fit_B_gamma = np.array([data["fit_B_gamma"] for data in fluctuation_fit_data])
fit_B = np.array([data["fit_B"] for data in fluctuation_fit_data])
fit_gamma = np.array([data["fit_gamma"] for data in fluctuation_fit_data])

# %%
err_B = np.abs(fit_B - B_actual) / B_actual
err_gamma = np.abs(fit_gamma - gamma_actual) / gamma_actual
err_B_gamma = np.linalg.norm((fit_B_gamma - B_gamma_actual) / B_gamma_actual, axis=1)
plot_log_log_fit(Nf, err_B)
plot_log_log_fit(Nf, err_gamma)
plot_log_log_fit(Nf, err_B_gamma)

err_B = np.abs(fit_B[:-1] - fit_B[-1]) / fit_B[-1]
err_gamma = np.abs(fit_gamma[:-1] - fit_gamma[-1]) / fit_gamma[-1]
err_B_gamma = np.linalg.norm(
    (fit_B_gamma[:-1] - fit_B_gamma[-1]) / fit_B_gamma[-1], axis=1
)
plot_log_log_fit(Nf[:-1], err_B)
plot_log_log_fit(Nf[:-1], err_gamma)
plot_log_log_fit(Nf[:-1], err_B_gamma)
# %%

# %%
Nf = np.array(
    [
        320,
        1280,
        5120,
        20480,
    ]
)
Ne = 3 * Nf / 2
Nv = 2 + Nf / 2
