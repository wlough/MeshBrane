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
    err_helfrich_energy = abs(helfrich_energy - helfrich_energy0) / helfrich_energy0

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


err_dict = get_curvature_and_bending_convergence_data()
