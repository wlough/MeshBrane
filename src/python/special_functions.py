import numpy as np


def unit_bump(s):
    val = np.zeros_like(s)
    I = np.abs(s) < 1.0
    val[I] = np.exp(1 + -1 / (1 - s[I] ** 2))
    return val


def bump3(xyz, center, radius):
    s = np.linalg.norm(xyz - center, axis=-1) / radius
    return unit_bump(s)



