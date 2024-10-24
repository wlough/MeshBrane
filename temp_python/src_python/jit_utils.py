# from numba import njit
import numpy as np

from numba import njit, float64, int32, int64, boolean, prange

# from numba.experimental import jitclass
from numba.typed import List, Dict


@njit
def jitnorm(V, axis=-1):
    return np.sqrt(np.sum(V**2, axis=axis))


@njit
def fib_sphere(Npoints=100):
    xyz = np.zeros((Npoints, 3))
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    for i in range(Npoints):
        z = 1.0 - 2 * i / (Npoints - 1)  # -1<=z<=1
        rad = np.sqrt(1.0 - z**2)  # radius at z

        theta = ga * i  # angle increment

        x = rad * np.cos(theta)
        y = rad * np.sin(theta)

        xyz[i] = np.array([x, y, z])

    return xyz


# @njit
def uniform_sphere(Npoints=100):
    V = np.random.randn(3, Npoints)
    V /= jitnorm(V, axis=0)
    return V.T


@njit
def jit_get_halfedge_index_of_twin(H, h):
    """
    Find the half-edge twin to h in the list of half-edges H.

    Parameters
    ----------
    H : list
        List of half-edges [[v0, v1], ...]
    h : int
        Index of half-edge in H

    Returns
    -------
    h_twin : int
        Index of H[h_twin]=[v1,v0] in H, where H[h]=[v0,v1]. Returns -1 if twin not found.
    """
    Nhedges = len(H)
    v0 = H[h][0]
    v1 = H[h][1]
    for h_twin in range(Nhedges):
        if H[h_twin][0] == v1 and H[h_twin][1] == v0:
            return h_twin

    return -1


@njit(parallel=True)
def jit_vf_samples_to_he_samples(V, F):
    # (V, F) = source_samples
    Nfaces = len(F)
    Nvertices = len(V)

    H = []
    h_out_V = Nvertices * [-1]
    v_origin_H = []
    h_next_H = []
    f_left_H = []
    h_bound_F = np.zeros(Nfaces, dtype=np.int32)

    # h = 0
    for f in range(Nfaces):
        h_bound_F[f] = 3 * f
        for i in range(3):
            h = 3 * f + i
            h_next = 3 * f + (i + 1) % 3
            v0 = F[f][i]
            v1 = F[f][(i + 1) % 3]
            H.append([v0, v1])
            v_origin_H.append(v0)
            f_left_H.append(f)
            h_next_H.append(h_next)
            if h_out_V[v0] == -1:
                h_out_V[v0] = h
    need_twins = set([_ for _ in range(len(H))])
    need_next = set()
    h_twin_H = len(H) * [-2]  # -2 means not set
    while need_twins:
        h = need_twins.pop()
        if h_twin_H[h] == -2:  # if twin not set
            h_twin = jit_get_halfedge_index_of_twin(
                H, h
            )  # returns -1 if twin not found
            if h_twin == -1:  # if twin not found
                h_twin = len(H)
                v0, v1 = H[h]
                H.append([v1, v0])
                v_origin_H.append(v1)
                need_next.add(h_twin)
                h_twin_H[h] = h_twin
                h_twin_H.append(h)
                f_left_H.append(-1)
            else:
                h_twin_H[h], h_twin_H[h_twin] = h_twin, h
                need_twins.remove(h_twin)

    h_next_H.extend([-1] * len(need_next))
    while need_next:
        h = need_next.pop()
        h_next = h_twin_H[h]
        # rotate ccw around origin of twin until we find nex h on boundary
        while f_left_H[h_next] != -1:
            h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
        h_next_H[h] = h_next

    # find and enumerate boundaries -1,-2,...
    H_need2visit = set([h for h in range(len(H)) if f_left_H[h] < 0])
    bdry_count = 0
    while H_need2visit:
        bdry_count += 1
        h_start = H_need2visit.pop()
        f_left_H[h_start] = -bdry_count
        h = h_next_H[h_start]
        while h != h_start:
            H_need2visit.remove(h)
            f_left_H[h] = -bdry_count
            h = h_next_H[h]

    target_samples = (
        V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )
    return target_samples


@njit
def vertex_pair_key(v0, v1):
    return min(v0, v1) * 1000000 + max(v0, v1)


@njit
def jit_refine_icososphere(Vm1, Fm1, r):
    # Nfaces = len(Fm1)
    # F = np.zeros((4*Nfaces, 3), dtype=np.int32)
    F = []
    V = [xyz for xyz in Vm1]
    v_midpt_vv = Dict.empty(key_type=int64, value_type=int32)
    nf = 0
    for tri in Fm1:
        v0, v1, v2 = tri
        key01 = vertex_pair_key(v0, v1)
        key12 = vertex_pair_key(v1, v2)
        key20 = vertex_pair_key(v2, v0)
        v01 = v_midpt_vv.get(key01, int32(-1))
        v12 = v_midpt_vv.get(key12, int32(-1))
        v20 = v_midpt_vv.get(key20, int32(-1))
        if v01 == int32(-1):
            v01 = int32(len(V))
            xyz01 = (V[v0] + V[v1]) / 2
            xyz01 *= r / np.linalg.norm(xyz01)
            V.append(xyz01)
            v_midpt_vv[key01] = v01
        if v12 == int32(-1):
            v12 = int32(len(V))
            xyz12 = (V[v1] + V[v2]) / 2
            xyz12 *= r / np.linalg.norm(xyz12)
            V.append(xyz12)
            v_midpt_vv[key12] = v12
        if v20 == int32(-1):
            v20 = int32(len(V))
            xyz20 = (V[v2] + V[v0]) / 2
            xyz20 *= r / np.linalg.norm(xyz20)
            V.append(xyz20)
            v_midpt_vv[key20] = v20
        F.append([v0, v01, v20])
        F.append([v01, v1, v12])
        F.append([v20, v12, v2])
        F.append([v01, v12, v20])

    return V, F


@njit
def vf_orientation_correct(a, b, c, cm):
    abc = (a + b + c) / 3
    n = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    return np.dot(n, abc - cm) > 0


@njit
def check_vf_list_orientation(V, F0):
    cm = np.sum(V, axis=0) / len(V)
    F = np.zeros_like(F0)
    for f in range(len(F0)):
        i, j, k = F0[f]
        a, b, c = V[i], V[j], V[k]

        if vf_orientation_correct(a, b, c, cm):
            F[f, :] = np.array([i, j, k])
        else:
            F[f, :] = np.array([i, k, j])
    return F
