import numpy as np


def _unit_bump(s):
    val = np.zeros_like(s)
    I = np.abs(s) < 1.0
    val[I] = np.exp(1 + -1 / (1 - s[I] ** 2))
    return val


def _bump3(xyz, center, radius):
    s = np.linalg.norm(xyz - center, axis=-1) / radius
    return unit_bump(s)


def unit_bump(s):
    val = np.zeros_like(s)
    I = np.abs(s) < 1.0
    val[I] = np.exp(1 + -1 / (1 - s[I] ** 2))
    return val


def bump3(xyz, center, radius):
    s = np.linalg.norm(xyz - center, axis=-1) / radius
    return unit_bump(s)


def dimensionless_tethering_potential(xi, lam, mu, nu):
    abs_xi = np.abs(xi)
    val = np.zeros_like(xi)
    I = abs_xi > mu
    val[I] = lam * np.exp(-nu * (abs_xi[I] - mu)) / (1 - abs_xi[I])
    return val


def tethering_potential(
    s,
    preferred_length,
    repulsive_onset=0.8,
    repulsive_singularity=0.6,
    attractive_onset=1.2,
    attractive_singularity=1.4,
    length_unit=1.0,
):
    L0 = preferred_length / length_unit
    Lmin = repulsive_singularity * preferred_length / length_unit
    Lmax = attractive_singularity * preferred_length / length_unit
    Lrep = repulsive_onset * preferred_length / length_unit
    Latt = attractive_onset * preferred_length / length_unit

    U = np.zeros_like(s)
    Irep = s < Lrep
    srep = s[Irep]
    U[Irep] = np.exp(1 / (srep - Lrep)) / (srep - Lmin)
    Iatt = s > Latt
    satt = s[Iatt]
    U[Iatt] = np.exp(1 / (Latt - satt)) / (Lmax - satt)
    return U


def tethering_potential_vutukuri0(s, L0):
    Lstar = 1.0
    alpha = 2 / 5
    dL = L0 * alpha
    lam = Lstar / dL
    mu = 1 / 2
    nu = Lstar / dL
    return tethering_potential(s, lam, mu, nu, alpha, L0)


def tethering_potential_vutukuri(s, L0):
    Lmin = 0.6 * L0
    Lmax = 1.4 * L0
    Lrep = 0.8 * L0
    Latt = 1.2 * L0

    abs_s = np.abs(s)
    U = np.zeros_like(s)

    Iatt = abs_s > Latt
    U[Iatt] = np.exp(1 / (Latt - abs_s[Iatt])) / (Lmax - abs_s[Iatt])

    Irep = abs_s < Lrep
    U[Irep] = np.exp(1 / (abs_s[Irep] - Lrep)) / (abs_s[Irep] - Lmin)
    return U
