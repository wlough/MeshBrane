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

#


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


# plot_autocors()


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
        "output_dirs": output_dirs,
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
    samples_and_weights_data = dict(np.load(path))

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
    print("fitting (B, gamma)")
    print(f"{B_gamma_red_guess=}")
    lsqr_out_B_gamma_red = least_squares(
        res_B_gamma_red,
        B_gamma_red_guess,
        ftol=tol,
        gtol=tol,
        xtol=tol,
        max_nfev=100000,
    )
    fit_B_gamma_red = lsqr_out_B_gamma_red.x
    print(f"{fit_B_gamma_red=}")
    fit_B_gamma = np.array(
        [fit_B_gamma_red[0], fit_B_gamma_red[1] * fit_B_gamma_red[0] / R0**2]
    )
    print(f"{fit_B_gamma=}")
    B_gamma_actual = np.array([B_actual, gamma_actual])
    err_B_gamma = np.abs(fit_B_gamma - B_gamma_actual) / B_gamma_actual
    resids_B_gamma_red = res_B_gamma_red(fit_B_gamma_red)
    print(f"{resids_B_gamma_red=}")
    print("fitting (B, gamma) -done")

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


# get_run_data(
#     f"../output/fluctuations_test_000320_no_graph",
#     l_max=20,
#     t_start=0.01,
#     dt_sample=0.01,
#     R0=None,
# )
# %%
output_dirs_000320 = [f"../output/fluctuations_test_000320_no_graph"]
output_dirs_000320 += [
    f"../output/fluctuations_test_000320_sweep/run_{_:0>2}_00" for _ in range(16)
]
output_dirs_001280 = [f"../output/fluctuations_test_001280_no_graph"]
output_dirs_001280 += [
    f"../output/fluctuations_test_001280_sweep/run_{_:0>2}_00" for _ in range(16)
]
output_dirs_005120 = [f"../output/fluctuations_test_005120_no_graph"]
output_dirs_005120 += [
    f"../output/fluctuations_test_005120_sweep/run_{_:0>2}_00" for _ in range(16)
]


dt_samples = 1 / np.linspace(1 / 1.0, 1 / 0.02, 20)
l_max = 20
t_start = 0.1

save_paths_000320 = [
    f"../output/fluctuations_test_analysis/samples_and_weights_data_Nf_{320:0>6}_l_max_{l_max}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples))
]
save_paths_001280 = [
    f"../output/fluctuations_test_analysis/samples_and_weights_data_Nf_{1280:0>6}_l_max_{l_max}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples))
]
save_paths_005120 = [
    f"../output/fluctuations_test_analysis/samples_and_weights_data_Nf_{5120:0>6}_l_max_{l_max}_dt_{n_dt}.npz"
    for n_dt in range(len(dt_samples))
]


save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
    output_dirs_000320,
    save_paths_000320,
    dt_samples,
    l_max=l_max,
    t_start=t_start,
)

save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
    output_dirs_001280,
    save_paths_001280,
    dt_samples,
    l_max=l_max,
    t_start=t_start,
)

save_samples_and_weights_from_concatenated_runs_for_dt_samples_range(
    output_dirs_005120,
    save_paths_005120,
    dt_samples,
    l_max=l_max,
    t_start=t_start,
)

# %%
Nf = np.array([320, 1280, 5120, 20480])
radius_over_edge_length = np.array([3.34077128, 6.63439233, 13.2451913, 26.4785837])
L_max = np.array([3, 6, 13])
# save_paths = save_paths_001280
save_paths = save_paths_005120
B_guess = 1.2
gamma_guess = 20.0
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
################
################
# Fluctuations #
################
################
# output_dir = "../output/fluctuations_test_000320_no_graph"
# output_dir = "../output/fluctuations_test_001280_no_graph"
output_dir = "../output/fluctuations_test_005120_no_graph"
# output_dir = "../output/fluctuations_test_020480_no_graph"

# output_dir = "../output/fluctuations_test/fluctuations_test_000320"
# output_dir = "../output/fluctuations_test/fluctuations_test_001280"
# output_dir = "../output/fluctuations_test/fluctuations_test_005120"

l_max = 20
t_start = 0.01
t_stop = 2.5
nt_skip = 25

# l_max = 20
# t_start = 0.01
# t_stop = 1.5
# nt_skip = 15


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

Nt
# %%

R0 = (3 * big_volume[0] / (4 * np.pi)) ** (1 / 3)
# big_xyz_coord_V_com = np.mean(big_xyz_coord_V, axis=1)
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
# big_solid_angle_V = big_area_V / big_R**2

big_solid_angle_V = np.einsum(
    "tvx, tvx, tv, tv -> tv", big_xyz_coord_V, big_normal_V, big_area_V, 1 / big_R**3
)


big_thetaphi_coord_V = np.array([thetaphi_from_xyz(xyz) for xyz in big_xyz_coord_V])


# tvn
# big_Yn = np.array([compute_all_Ylm(l_max, th_ph) for th_ph in big_thetaphi_coord_V])
# big_Un = np.einsum("tv, tv, tvn->tn", big_area_V, big_r, big_Yn.conjugate())

big_Un = []
for solid_angle_V, r, thetaphi_coord_V in zip(
    big_solid_angle_V, big_r, big_thetaphi_coord_V
):
    Yn = compute_all_Ylm(l_max, thetaphi_coord_V)
    big_Un.append(np.einsum("v,v,vn->n", solid_angle_V, r, Yn.conjugate()))
big_Un = np.array(big_Un)

# np.abs(big_Un.mean(axis=0))
big_sqr_Un = np.abs(big_Un) ** 2
# var_sqr_Un = np.var(big_sqr_Un, axis=0)


# %%

mean_sqr_Un = np.mean(big_sqr_Un, axis=0)

# std_sqr_Un = np.std(big_sqr_Un, axis=0)

mean_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]

# std_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]

for l in range(l_max + 1):
    for m in range(-l, l + 1):
        n = spherical_harmonic_index_n_LM(l, m)
        mean_sqr_Ulm[l][m] = mean_sqr_Un[n]
        # std_sqr_Ulm[l][m] = std_sqr_Un[n]

mean_sqr_Ul = np.array([np.mean(mean_sqr_Ulm[l]) for l in range(l_max + 1)])

# std_sqr_Ul = np.array([np.mean(std_sqr_Ulm[l]) for l in range(l_max + 1)])
#
# std_sqr_Ul / mean_sqr_Ul

#


ell_range = [l for l in range(2, l_max + 1)]

X = np.array(ell_range)
Y = mean_sqr_Ul[X]
kBT = 1.0291e-2
beta0 = np.array([1.0, 32.5]) / 1.03
beta_actual = np.array([1.0, 32.5])
tol = 1e-15
gamma0 = np.array([32.5]) / 13
gamma_actual = np.array([32.5])
B0 = np.array([1.0]) / 13
B_actual = np.array([1.0])


def fun(beta):
    B, gamma = beta
    return np.array(
        [
            y - kBT / (B * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


# def fun(beta):
#     B, gamma = beta
#     return np.array(
#         [
#             y - kBT / ((l - 1) * (l + 2) * (R0**2 * gamma + B * l * (l + 1)))
#             for l, y in zip(X, Y)
#         ]
#     )


lsqr_out = least_squares(
    fun,
    beta0,
    ftol=tol,
    gtol=tol,
    xtol=tol,
    max_nfev=100000,
)
beta = lsqr_out.x
beta[1] *= beta[0] / R0**2
(beta - beta_actual) / beta_actual


def fun_tension(gamma):
    B = 1.0
    return np.array(
        [
            y - kBT / (B * (l - 1) * (l + 2) * (gamma[0] + l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


def fun_tension(gamma):
    B = 1.0
    return np.array(
        [
            y - kBT / ((l - 1) * (l + 2) * (R0**2 * gamma[0] + B * l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


def fun_stiffness(B):
    gamma = 32.5
    return np.array(
        [
            y - kBT / (B[0] * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


def fun_stiffness(B):
    gamma = 32.5
    return np.array(
        [
            y - kBT / ((l - 1) * (l + 2) * (R0**2 * gamma + B[0] * l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


lsqr_out = least_squares(
    fun_tension,
    gamma0,
    ftol=tol,
    gtol=tol,
    xtol=tol,
    max_nfev=100000,
)

gamma = lsqr_out.x
fun_tension(gamma)
(gamma - gamma_actual) / gamma_actual


lsqr_out = least_squares(
    fun_stiffness,
    B0,
    ftol=tol,
    gtol=tol,
    xtol=tol,
    max_nfev=100000,
)

B = lsqr_out.x
fun_stiffness(B)
(B - B_actual) / B_actual

# %%
##
kBT = 1.0291e-2
B = 1.0
l = np.arange(l_max + 1)
gamma = kBT / (B * (l - 1) * (l + 2) * mean_sqr_Ul) - l * (l + 1)
gamma[2:]
##

##
kBT = 1.0291e-2
gamma = 32.5
l = np.arange(l_max + 1)
B = kBT / (mean_sqr_Ul * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
B[2:]
# ##

# %%
#
# ######
# gamma, kBT, B = sp.symbols(r"\gamma k_{B}T B")
# # eqns = [
# #     mean_sqr_Ul[l] - kBT / (B * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
# #     for l in range(2, 5)
# # ]
# eqns = sp.Array(
#     [
#         mean_sqr_Ul[l] * (B * (l - 1) * (l + 2) * (gamma + l * (l + 1))) - kBT
#         for l in range(2, 5)
#     ]
# )
# vars = sp.Array([B, gamma, kBT])
# #
# sp.solve(eqns, vars)
# #######
#
#
# B, gamma = sp.symbols(r"B \gamma")
# kBT = 1.0291e-2
# eqns = sp.Array(
#     [
#         mean_sqr_Ul[l] * (B * (l - 1) * (l + 2) * (gamma + l * (l + 1))) - kBT
#         for l in range(5, 7)
#     ]
# )
# vars = sp.Array([B, gamma])
# sp.solve(eqns, vars)

t_stop = 1.0
nt_skip = 1

xyz_coord_V_path = f"{output_dir}/raw_data/envelope_xyz_coord_V.dat"
area_V_path = f"{output_dir}/raw_data/envelope_area_V.dat"
volume_path = f"{output_dir}/raw_data/envelope_volume.dat"
normal_V_path = f"{output_dir}/raw_data/envelope_normal_V.dat"
t_path = f"{output_dir}/raw_data/t.dat"

big_t = read_time_series(t_path)

t_mask = np.logical_and(big_t >= t_start, big_t <= t_stop)
big_t = big_t[t_mask][::nt_skip]
big_xyz_coord_V = read_time_series(xyz_coord_V_path)[t_mask][::nt_skip]
big_area_V = read_time_series(area_V_path)[t_mask][::nt_skip]
big_volume = read_time_series(volume_path)[t_mask][::nt_skip]  # ***
Nt, Nv, _ = big_xyz_coord_V.shape
n_max = spherical_harmonic_index_n_LM(l_max, l_max)


R0 = (3 * big_volume[0] / (4 * np.pi)) ** (1 / 3)
# big_xyz_coord_V_com = np.mean(big_xyz_coord_V, axis=1)
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
big_solid_angle_V = big_area_V / big_R**2


big_thetaphi_coord_V = np.array([thetaphi_from_xyz(xyz) for xyz in big_xyz_coord_V])


# tvn
# big_Yn = np.array([compute_all_Ylm(l_max, th_ph) for th_ph in big_thetaphi_coord_V])
# big_Un = np.einsum("tv, tv, tvn->tn", big_area_V, big_r, big_Yn.conjugate())

big_Un = []
for area_V, r, thetaphi_coord_V in zip(big_area_V, big_r, big_thetaphi_coord_V):
    Yn = compute_all_Ylm(l_max, thetaphi_coord_V)
    big_Un.append(np.einsum("v,v,vn->n", area_V, r, Yn.conjugate()))
big_Un = np.array(big_Un)

# np.abs(big_Un.mean(axis=0))
big_sqr_Un = np.abs(big_Un) ** 2
# var_sqr_Un = np.var(big_sqr_Un, axis=0)


# %%
from scipy.signal import correlate
from scipy.fft import fft


def get_auto_corr0(x):
    mean_x = np.mean(x)
    z = x - mean_x
    corr_x = []
    T = len(x)
    for tau in range(T):
        corr_x_tau = np.sum([z[t] * z[t + tau] for t in range(T - tau)]) / np.sum(
            z[: T - tau] ** 2
        )
        corr_x.append(corr_x_tau)
    return np.array(corr_x)


def get_auto_corr1(x):
    mean_x = np.mean(x)
    z = x - mean_x
    corr_x = []
    T = len(x)
    for tau in range(T):
        corr_x_tau = (z[: T - tau] @ np.roll(z, -tau)[: T - tau]) / (
            z[: T - tau] @ z[: T - tau]
        )
        corr_x.append(corr_x_tau)
    return np.array(corr_x)


def get_auto_corr2(x):
    mean_x = np.mean(x)
    z = x - mean_x
    corr_x = []
    T = len(x)
    for tau in range(T):
        corr_x_tau = np.dot(z[: T - tau], np.roll(z, -tau)[: T - tau]) / np.dot(
            z[: T - tau], z[: T - tau]
        )
        corr_x.append(corr_x_tau)
    return np.array(corr_x)


def get_auto_corr3(x):
    mean_x = np.mean(x)
    z = x - mean_x
    var_x = np.mean(z**2)
    corr_x = []
    T = len(x)
    corr_x = (
        np.array(
            [np.mean(z[: T - tau] * np.roll(z, -tau)[: T - tau]) for tau in range(T)]
        )
        / var_x
    )
    return corr_x


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


def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = correlate(x, x, mode="full")[-n:]
    # assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


def autocor(x):
    # x_exp = sum(x) / len(x)
    x_exp = np.mean(x)
    z = x - x_exp
    lag = np.array([h for h in range(1 - len(z), len(z))])
    cor = np.array(
        [np.sum([z[n] * z[n + abs(h)] for n in range(len(z) - abs(h))]) for h in lag]
    ) / np.dot(z, z)
    # return lag, cor
    return cor


def autocor_lag(x):
    x_exp = sum(x) / len(x)
    z = x - x_exp
    lag = np.array([h for h in range(1 - len(z), len(z))])
    return lag


# cor = get_auto_corr(big_volume) - get_auto_corr3(big_volume)
# N_plot = Nt // 2
N_plot = 10

cor = get_auto_corr(big_volume)[:N_plot]
_ = plt.plot(cor)
_ = plt.plot(cor * 0 + 1 / np.e)

# %%
# cor_sqr_Un = np.array([estimated_autocorrelation(u) for u in big_sqr_Un.T])
cor_sqr_Un = np.array([get_auto_corr(u) for u in big_sqr_Un.T])
cor_sqr_Ulm = [
    [cor_sqr_Un[spherical_harmonic_index_n_LM(l, m)] for m in range(-l, l + 1)]
    for l in range(l_max)
]

l_plot = 5
m_plot = 1
plt.plot(cor_sqr_Ulm[l_plot][m_plot + l_plot][:1000])
plt.plot(cor_sqr_Ulm[l_plot][m_plot + l_plot][:1000] * 0 + 1 / np.e)

# %%
mean_sqr_Un = np.mean(big_sqr_Un, axis=0)

# std_sqr_Un = np.std(big_sqr_Un, axis=0)

mean_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]

# std_sqr_Ulm = [[0.0 for m in range(-l, l + 1)] for l in range(l_max + 1)]

for l in range(l_max + 1):
    for m in range(-l, l + 1):
        n = spherical_harmonic_index_n_LM(l, m)
        mean_sqr_Ulm[l][m] = mean_sqr_Un[n]
        # std_sqr_Ulm[l][m] = std_sqr_Un[n]

mean_sqr_Ul = np.array([np.mean(mean_sqr_Ulm[l]) for l in range(l_max + 1)])

# std_sqr_Ul = np.array([np.mean(std_sqr_Ulm[l]) for l in range(l_max + 1)])
#
# std_sqr_Ul / mean_sqr_Ul

#
from scipy.optimize import least_squares

ell_range = [l for l in range(2, l_max + 1)]

X = np.array(ell_range)
Y = mean_sqr_Ul[X]
kBT = 1.0291e-2
beta0 = np.array([1.0, 32.5])
beta_actual = np.array([1.0, 32.5])
tol = 1e-15
gamma0 = np.array([32.5])
gamma_actual = np.array([32.5])


def fun(beta):
    B, gamma = beta
    return np.array(
        [
            y - kBT / (B * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


# def fun(beta):
#     B, gamma = beta
#     return np.array(
#         [
#             y * (B * (l - 1) * (l + 2) * (gamma + l * (l + 1))) - kBT
#             for l, y in zip(X, Y)
#         ]
#     )


lsqr_out = least_squares(
    fun,
    beta0,
    ftol=tol,
    gtol=tol,
    xtol=tol,
    max_nfev=100000,
)
beta = lsqr_out.x
fun(beta)
(beta - beta_actual) / beta_actual


def fun_tension(gamma):
    B = 1.0
    return np.array(
        [
            y - kBT / (B * (l - 1) * (l + 2) * (gamma[0] + l * (l + 1)))
            for l, y in zip(X, Y)
        ]
    )


lsqr_out = least_squares(
    fun_tension,
    gamma0,
    ftol=tol,
    gtol=tol,
    xtol=tol,
    max_nfev=100000,
)

gamma = lsqr_out.x
fun_tension(gamma)
(gamma - gamma_actual) / gamma_actual

# %%
##
kBT = 1.0291e-2
B = 1.0
l = np.arange(l_max + 1)
gamma = kBT / (B * (l - 1) * (l + 2) * mean_sqr_Ul) - l * (l + 1)
gamma[2:]
##

##
kBT = 1.0291e-2
gamma = 32.5
l = np.arange(l_max + 1)
B = kBT / (mean_sqr_Ul * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
B[2:]
# ##

# %%
#
# ######
# gamma, kBT, B = sp.symbols(r"\gamma k_{B}T B")
# # eqns = [
# #     mean_sqr_Ul[l] - kBT / (B * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
# #     for l in range(2, 5)
# # ]
# eqns = sp.Array(
#     [
#         mean_sqr_Ul[l] * (B * (l - 1) * (l + 2) * (gamma + l * (l + 1))) - kBT
#         for l in range(2, 5)
#     ]
# )
# vars = sp.Array([B, gamma, kBT])
# #
# sp.solve(eqns, vars)
# #######
#
#
# B, gamma = sp.symbols(r"B \gamma")
# kBT = 1.0291e-2
# eqns = sp.Array(
#     [
#         mean_sqr_Ul[l] * (B * (l - 1) * (l + 2) * (gamma + l * (l + 1))) - kBT
#         for l in range(5, 7)
#     ]
# )
# vars = sp.Array([B, gamma])
# sp.solve(eqns, vars)
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
