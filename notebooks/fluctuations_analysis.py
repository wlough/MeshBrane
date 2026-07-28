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

#


################
################
# Fluctuations #
################
################
# output_dir = "../output/fluctuations_test_000320_no_graph"
# output_dir = "../output/fluctuations_test_001280_no_graph"
# output_dir = "../output/fluctuations_test_005120_no_graph"
output_dir = "../output/fluctuations_test_020480_no_graph"

# output_dir = "../output/fluctuations_test/fluctuations_test_000320"
# output_dir = "../output/fluctuations_test/fluctuations_test_001280"
# output_dir = "../output/fluctuations_test/fluctuations_test_005120"
1.25e-4 / 1e-5
5 / 4
l_max = 20
t_start = 0.01
t_stop = 2.5
nt_skip = 25

l_max = 20
t_start = 0.01
t_stop = 5
nt_skip = 10


xyz_coord_V_path = f"{output_dir}/raw_data/envelope_xyz_coord_V.dat"
area_V_path = f"{output_dir}/raw_data/envelope_area_V.dat"
volume_path = f"{output_dir}/raw_data/envelope_volume.dat"
normal_V_path = f"{output_dir}/raw_data/envelope_normal_V.dat"
t_path = f"{output_dir}/raw_data/t.dat"

big_t = read_time_series(t_path)


t_mask = np.logical_and(big_t >= t_start, big_t <= t_stop)
big_t.size / (200 * 1e3)
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
N_plot = Nt // 2
N_cor = 20000
N_plot = 25
N_plot_start = 0

# cor = get_auto_corr(big_volume[N_plot_start : N_plot_start + N_cor])[:N_plot]
# _ = plt.plot(cor)
# _ = plt.plot(cor * 0 + 1 / np.e)
# _ = plt.plot(cor * 0)
# #

# np.mean(cor[300:])
#
# cor_sqr_Un = np.array([estimated_autocorrelation(u) for u in big_sqr_Un.T])
cor_sqr_Un = np.array(
    [
        get_auto_corr(u[N_plot_start : N_plot_start + N_cor])[:N_plot]
        for u in big_sqr_Un.T
    ]
)
cor_sqr_Ulm = [
    [cor_sqr_Un[spherical_harmonic_index_n_LM(l, m)] for m in range(-l, l + 1)]
    for l in range(l_max)
]

l_plot = 2
m_plot = 0
plt.plot(
    big_t[: cor_sqr_Ulm[l_plot][m_plot + l_plot].size] - big_t[0],
    cor_sqr_Ulm[l_plot][m_plot + l_plot],
)
plt.plot(
    big_t[: cor_sqr_Ulm[l_plot][m_plot + l_plot].size] - big_t[0],
    cor_sqr_Ulm[l_plot][m_plot + l_plot] * 0 + 1 / np.e,
)
plt.plot(
    big_t[: cor_sqr_Ulm[l_plot][m_plot + l_plot].size] - big_t[0],
    cor_sqr_Ulm[l_plot][m_plot + l_plot] * 0,
)

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
#             y * (B * (l - 1) * (l + 2) * (gamma + l * (l + 1))) / kBT - 1.0
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


def fun_stiffness(B):
    gamma = 32.5
    return np.array(
        [
            y - kBT / (B[0] * (l - 1) * (l + 2) * (gamma + l * (l + 1)))
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
