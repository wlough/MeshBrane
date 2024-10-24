import numpy as np
from numba import jit, vectorize, prange
from numba.typed import (
    List,
    Dict,
)
from numba.types import (
    unicode_type,  # str
    boolean,
    int32,
    int64,
    float64,
    ListType,
    DictType,
    Array,
)
from numba.experimental import jitclass
import pickle
from time import time
import os
from temp_python.src_python.utilities.misc_utils import save_npz


@jit
def jitnorm(V, axis=-1):
    return np.sqrt(np.sum(V**2, axis=axis))


@jit
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


@jit
def uniform_sphere(Npoints=100):
    V = np.random.randn(3, Npoints)
    V /= jitnorm(V, axis=0)
    return V.T


@jit("int32(int32[:,:], int32)")
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

    return int32(-1)


@jit
def jit_vf_samples_to_he_samples(V, F):
    # (V, F) = source_samples
    Nfaces = len(F)
    Nvertices = len(V)
    _Nhedges = 3 * Nfaces * 2
    _H = np.zeros((_Nhedges, 2), dtype=np.int32)
    h_out_V = -np.ones(Nvertices, dtype=np.int32)
    _v_origin_H = np.zeros(_Nhedges, dtype=np.int32)
    _h_next_H = -np.ones(_Nhedges, dtype=np.int32)
    _f_left_H = np.zeros(_Nhedges, dtype=np.int32)
    h_bound_F = np.zeros(Nfaces, dtype=np.int32)

    # h_count = 0
    for f in range(Nfaces):
        h_bound_F[f] = 3 * f
        for i in range(3):
            h = 3 * f + i
            h_next = 3 * f + (i + 1) % 3
            v0 = F[f][i]
            v1 = F[f][(i + 1) % 3]
            _H[h] = [v0, v1]
            _v_origin_H[h] = v0
            _f_left_H[h] = f
            _h_next_H[h] = h_next
            if h_out_V[v0] == -1:
                h_out_V[v0] = h
    h_count = 3 * Nfaces
    need_twins = set([int32(_) for _ in range(h_count)])
    need_next = set([int32(0)])
    need_next.pop()
    _h_twin_H = -2 * np.ones(_Nhedges, dtype=np.int32)  # -2 means not set
    while need_twins:
        h = need_twins.pop()
        if _h_twin_H[h] == -2:  # if twin not set
            h_twin = jit_get_halfedge_index_of_twin(
                _H, h
            )  # returns -1 if twin not found
            if h_twin == -1:  # if twin not found
                h_twin = int32(h_count)
                h_count += 1
                v0, v1 = _H[h]
                _H[h_twin] = [v1, v0]
                _v_origin_H[h_twin] = v1
                need_next.add(h_twin)
                _h_twin_H[h] = h_twin
                _h_twin_H[h_twin] = h
                _f_left_H[h_twin] = -1
            else:
                _h_twin_H[h], _h_twin_H[h_twin] = h_twin, h
                need_twins.remove(h_twin)

    Nhedges = h_count
    # H = _H[:Nhedges]
    v_origin_H = _v_origin_H[:Nhedges]
    h_next_H = _h_next_H[:Nhedges]
    f_left_H = _f_left_H[:Nhedges]
    h_twin_H = _h_twin_H[:Nhedges]
    # h_next_H.extend([-1] * len(need_next))
    while need_next:
        h = need_next.pop()
        h_next = h_twin_H[h]
        # rotate ccw around origin of twin until we find nex h on boundary
        while f_left_H[h_next] != -1:
            h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
        h_next_H[h] = h_next

    # find and enumerate boundaries -1,-2,...
    H_need2visit = set([int32(h) for h in range(Nhedges) if f_left_H[h] == -1])
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

    return (
        V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )


@jit("int64(int32, int32)")
def jit_vertex_pair_key(v0, v1):
    return min(v0, v1) * 1000000 + max(v0, v1)


@jit
def jit_refine_icososphere(Vm1, Fm1, r=1.0):
    num_refine = int32(np.log2(len(Fm1) // 20) // 2 + 1)
    Nv = 10 * 4**num_refine + 2
    Nf = 20 * 4**num_refine
    V = np.zeros((Nv, 3), dtype=np.float64)
    V[: len(Vm1)] = Vm1
    F = np.zeros((Nf, 3), dtype=np.int32)
    v_midpt_vv = Dict.empty(key_type=int64, value_type=int32)
    v_count = len(Vm1)
    f_count = 0
    for tri in Fm1:
        v0, v1, v2 = tri
        key01 = jit_vertex_pair_key(v0, v1)
        key12 = jit_vertex_pair_key(v1, v2)
        key20 = jit_vertex_pair_key(v2, v0)
        v01 = v_midpt_vv.get(key01, int32(-1))
        v12 = v_midpt_vv.get(key12, int32(-1))
        v20 = v_midpt_vv.get(key20, int32(-1))
        if v01 == int32(-1):
            v01 = int32(v_count)
            xyz01 = (V[v0] + V[v1]) / 2
            xyz01 *= r / np.linalg.norm(xyz01)
            V[v_count] = xyz01
            v_count += 1
            v_midpt_vv[key01] = v01
        if v12 == int32(-1):
            v12 = int32(v_count)
            xyz12 = (V[v1] + V[v2]) / 2
            xyz12 *= r / np.linalg.norm(xyz12)
            V[v_count] = xyz12
            v_count += 1
            v_midpt_vv[key12] = v12
        if v20 == int32(-1):
            v20 = int32(v_count)
            xyz20 = (V[v2] + V[v0]) / 2
            xyz20 *= r / np.linalg.norm(xyz20)
            V[v_count] = xyz20
            v_count += 1
            v_midpt_vv[key20] = v20
        F[f_count] = [v0, v01, v20]
        f_count += 1
        F[f_count] = [v01, v1, v12]
        f_count += 1
        F[f_count] = [v20, v12, v2]
        f_count += 1
        F[f_count] = [v01, v12, v20]
        f_count += 1

    return V, F


@jit("boolean(float64[:], float64[:], float64[:], float64[:])")
def vf_orientation_is_correct(a, b, c, cm):
    abc = (a + b + c) / 3
    n = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
    return np.dot(n, abc - cm) > 0


@jit("int32[:, :](float64[:, :], int32[:, :])")
def check_vf_list_orientation(V, F0):
    cm = np.sum(V, axis=0) / len(V)
    F = np.zeros_like(F0)
    for f in range(len(F0)):
        i, j, k = F0[f]
        a, b, c = V[i], V[j], V[k]

        if vf_orientation_is_correct(a, b, c, cm):
            F[f, :] = np.array([i, j, k])
        else:
            F[f, :] = np.array([i, k, j])
    return F


@jit(parallel=True)
def cotan_laplacian(
    Q, xyz_coord_V, h_out_V, v_origin_H, h_next_H, h_twin_H, f_left_H, h_bound_F
):
    """
    Computes the cotan Laplacian of Q at each vertex
    """
    Nv = len(xyz_coord_V)
    lapQ = np.zeros_like(Q)
    for vi in prange(Nv):
        Atot = 0.0
        ri = xyz_coord_V[vi]
        qi = Q[vi]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        h_start = h_out_V[vi]
        hij = h_start
        while True:
            hijm1 = h_next_H[h_twin_H[hij]]
            hijp1 = h_twin_H[h_next_H[h_next_H[hij]]]
            vjm1 = v_origin_H[h_twin_H[hijm1]]
            vj = v_origin_H[h_twin_H[hij]]
            vjp1 = v_origin_H[h_twin_H[hijp1]]

            qj = Q[vj]

            rjm1 = xyz_coord_V[vjm1]
            rj = xyz_coord_V[vj]
            rjp1 = xyz_coord_V[vjp1]

            rjm1_rjm1 = rjm1[0] ** 2 + rjm1[1] ** 2 + rjm1[2] ** 2
            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            ri_rjm1 = ri[0] * rjm1[0] + ri[1] * rjm1[1] + ri[2] * rjm1[2]
            rj_rjm1 = rj[0] * rjm1[0] + rj[1] * rjm1[1] + rj[2] * rjm1[2]
            ri_rjp1 = ri[0] * rjp1[0] + ri[1] * rjp1[1] + ri[2] * rjp1[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]

            Lijm1 = np.sqrt(ri_ri - 2 * ri_rjm1 + rjm1_rjm1)
            Ljjm1 = np.sqrt(rj_rj - 2 * rj_rjm1 + rjm1_rjm1)
            Lijp1 = np.sqrt(ri_ri - 2 * ri_rjp1 + rjp1_rjp1)
            Ljjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            Lij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)

            cos_thetam = (ri_rj + rjm1_rjm1 - rj_rjm1 - ri_rjm1) / (Lijm1 * Ljjm1)

            cos_thetap = (ri_rj + rjp1_rjp1 - ri_rjp1 - rj_rjp1) / (Lijp1 * Ljjp1)

            cot_thetam = cos_thetam / np.sqrt(1 - cos_thetam**2)
            cot_thetap = cos_thetap / np.sqrt(1 - cos_thetap**2)

            Atot += Lij**2 * (cot_thetam + cot_thetap) / 8
            lapQ[vi] += (cot_thetam + cot_thetap) * (qj - qi) / 2
            hij = h_next_H[h_twin_H[hij]]
            if hij == h_start:
                break
        lapQ[vi] /= Atot

    return lapQ


def barcell_area(self, v):
    """area of the barycentric cell dual to vertex v"""
    r = self.xyz_coord_v(v)
    A = 0.0
    for h in self.generate_H_out_v_clockwise(v):
        if self.complement_boundary_contains_h(h):
            continue
        # r1 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(h)))
        # r2 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(self.h_next_h(h))))
        # A_face_vec = (
        #     np.cross(r, r1) / 2 + np.cross(r1, r2) / 2 + np.cross(r2, r) / 2
        # )
        # A_face = np.sqrt(
        #     A_face_vec[0] ** 2 + A_face_vec[1] ** 2 + A_face_vec[2] ** 2
        # )

        A += self.area_f(self.f_left_h(h)) / 3

    return A


@jit(parallel=True)
def belkin_laplacian(
    Q,
    s,
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
):
    """
    ...
    """
    Nv = len(xyz_coord_V)
    Nf = len(h_bound_F)
    lapQ = np.zeros_like(Q)
    for v in prange(Nv):
        xyz = xyz_coord_V[v]
        for f in range(Nf):
            h0 = h_bound_F[f]
            h1 = h_next_H[h0]
            h2 = h_next_H[h1]
            v0 = v_origin_H[h0]
            v1 = v_origin_H[h1]
            v2 = v_origin_H[h2]
            xyz0 = xyz_coord_V[v0]
            xyz1 = xyz_coord_V[v1]
            xyz2 = xyz_coord_V[v2]
            Avec = np.cross(xyz0, xyz1) + np.cross(xyz1, xyz2) + np.cross(xyz2, xyz0)
            Af = np.linalg.norm(Avec) / 2
            lapQ[v] += (
                (1 / (4 * np.pi * s**2))
                * (Af / 3)
                * np.exp(-np.linalg.norm(xyz - xyz0) ** 2 / (4 * s))
                * (Q[v0] - Q[v])
            )
            lapQ[v] += (
                (1 / (4 * np.pi * s**2))
                * (Af / 3)
                * np.exp(-np.linalg.norm(xyz - xyz1) ** 2 / (4 * s))
                * (Q[v1] - Q[v])
            )
            lapQ[v] += (
                (1 / (4 * np.pi * s**2))
                * (Af / 3)
                * np.exp(-np.linalg.norm(xyz - xyz2) ** 2 / (4 * s))
                * (Q[v2] - Q[v])
            )
    return lapQ


class SphereFactory:
    """
    subdivides icosahedron (20 triangles; 12 vertices) to create meshes of unit sphere

    after k refinements we have

    |V|=10*4^k+2
    |E|=30*4^k
    |F|=20*4^k
    """

    def __init__(self, name="unit_sphere", r=1.0):
        self._name = name
        self.radius = r
        phi = (1.0 + np.sqrt(5.0)) * 0.5  # golden ratio
        _a = 1.0
        _b = 1.0 / phi
        a = r * _a / np.sqrt(_a**2 + _b**2)
        b = r * _b / np.sqrt(_a**2 + _b**2)
        V0 = np.array(
            [
                [0.0, b, -a],
                [b, a, 0.0],
                [-b, a, 0.0],
                [0.0, b, a],
                [0.0, -b, a],
                [-a, 0.0, b],
                [0.0, -b, -a],
                [a, 0.0, -b],
                [a, 0.0, b],
                [-a, 0.0, -b],
                [b, -a, 0.0],
                [-b, -a, 0.0],
            ],
            dtype=np.float64,
        )
        F0 = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
                [5, 4, 3],
                [4, 8, 3],
                [7, 6, 0],
                [6, 9, 0],
                [11, 10, 4],
                [10, 11, 6],
                [9, 5, 2],
                [5, 9, 11],
                [8, 7, 1],
                [7, 8, 10],
                [2, 5, 3],
                [8, 1, 3],
                [9, 2, 0],
                [1, 7, 0],
                [11, 9, 6],
                [7, 10, 6],
                [5, 11, 4],
                [10, 8, 4],
            ],
            dtype=np.int32,
        )

        self.F = [F0]
        (
            self.V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = jit_vf_samples_to_he_samples(V0, F0)

        self.h_out_V = [_h_out_V]
        self.v_origin_H = [_v_origin_H]
        self.h_next_H = [_h_next_H]
        self.h_twin_H = [_h_twin_H]
        self.f_left_H = [_f_left_H]
        self.h_bound_F = [_h_bound_F]
        self._num_vertices = [len(self.V)]
        self.V0 = V0
        self.F0 = F0
        self.t_refine = []
        self.refine_level = 0

    def VF(self, level=-1):
        return self.V[: self.num_vertices(level)], self.F[level]

    def HE(self, level=-1):
        return (
            self.V[: self.num_vertices(level)],
            self.h_out_V[level],
            self.v_origin_H[level],
            self.h_next_H[level],
            self.h_twin_H[level],
            self.f_left_H[level],
            self.h_bound_F[level],
        )

    def num_vertices(self, level=-1):
        return self._num_vertices[level]

    def next_num_vertices(self):
        level = len(self._num_vertices)
        # return self._NUM_VERTICES_[level]
        return 10 * 4**level + 2

    @property
    def name(self):
        return self._name

    def refine(self, timeit=True):
        t = time()
        Fm1 = self.F[-1]
        Vm1 = self.V
        V, F = jit_refine_icososphere(Vm1, Fm1, r=self.radius)
        (
            self.V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = jit_vf_samples_to_he_samples(V, F)

        self.F.append(F)
        self.h_out_V.append(_h_out_V)
        self.v_origin_H.append(_v_origin_H)
        self.h_next_H.append(_h_next_H)
        self.h_twin_H.append(_h_twin_H)
        self.f_left_H.append(_f_left_H)
        self.h_bound_F.append(_h_bound_F)
        self._num_vertices.append(len(self.V))
        t = time() - t
        self.t_refine.append(t)
        self.refine_level += 1

    def save_data_arrays(
        self, path, level=-1, compressed=False, chunk=False, remove_unchunked=False
    ):
        """
        Save data arrays to npz file

        Args:
            path (str): path to save file
        """
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        ) = self.HE(level=level)
        h_comp_B = np.array([], dtype=np.int32)
        data = {
            "xyz_coord_V": xyz_coord_V,
            "h_out_V": h_out_V,
            "v_origin_H": v_origin_H,
            "h_next_H": h_next_H,
            "h_twin_H": h_twin_H,
            "f_left_H": f_left_H,
            "h_bound_F": h_bound_F,
            "h_comp_B": h_comp_B,
        }

        save_npz(
            data,
            path,
            compressed=compressed,
            chunk=chunk,
            remove_unchunked=remove_unchunked,
        )

    def pickle_data_arrays(
        self, path, level=-1, compressed=False, chunk=False, remove_unchunked=False
    ):
        """
        Save data arrays to pickle file

        Args:
            path (str): path to save file
        """
        (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        ) = self.HE(level=level)
        h_comp_B = np.array([], dtype=np.int32)
        data = {
            "xyz_coord_V": xyz_coord_V,
            "h_out_V": h_out_V,
            "v_origin_H": v_origin_H,
            "h_next_H": h_next_H,
            "h_twin_H": h_twin_H,
            "f_left_H": f_left_H,
            "h_bound_F": h_bound_F,
            "h_comp_B": h_comp_B,
        }
        save_pkl(
            data,
            path,
            compressed=compressed,
            chunk=chunk,
            remove_unchunked=remove_unchunked,
        )

    def write_plys(self, level=-1):
        if isinstance(level, int):
            vf_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_vf.ply"
            )
            he_path = (
                f"./data/ply/binary/{self.name}_{self.num_vertices(level):06d}_he.ply"
            )
            print(f"Writing vertex-face ply to {vf_path}")
            self.v2h[level].write_source_ply(vf_path, use_ascii=False)
            print(f"Writing half-edge ply to {he_path}")
            self.v2h[level].write_target_ply(he_path, use_ascii=False)

        elif level == "all":
            for level in range(len(self.F)):
                self.write_plys(level=level)
            print(f"Done writing {self.name} plys.")

    @classmethod
    def build_fibonacci_test_plys(cls, num_refine=5, noise_scale=0.01):
        b = cls()
        b._name = "fibonacci_sphere"
        b.V = b.r * fib_sphere(12)
        b.F
        b.v2h = [VertTri2HalfEdgeConverter.from_source_samples(*b.VF())]
        b.write_plys(level=0)
        for level in range(1, num_refine + 1):
            b.refine()
            b.write_plys(level=level)
        print("Done.")


sf_spec = [
    ("V", float64[:, :, :]),  # 3D array of float64
    (
        "attributes",
        DictType(unicode_type, float64),
    ),  # Dict with unicode keys and float64 values
]


@jitclass(sf_spec)
class _jitSphereFactory:
    """
    subdivides icosahedron (20 triangles; 12 vertices) to create meshes of unit sphere

    after k refinements we have

    |V|=10*4^k+2
    |E|=30*4^k
    |F|=20*4^k
    """

    def __init__(self, name="unit_sphere", r=1.0):
        phi = (1.0 + np.sqrt(5.0)) * 0.5  # golden ratio
        _a = 1.0
        _b = 1.0 / phi
        a = r * _a / np.sqrt(_a**2 + _b**2)
        b = r * _b / np.sqrt(_a**2 + _b**2)
        V0 = np.array(
            [
                [0.0, b, -a],
                [b, a, 0.0],
                [-b, a, 0.0],
                [0.0, b, a],
                [0.0, -b, a],
                [-a, 0.0, b],
                [0.0, -b, -a],
                [a, 0.0, -b],
                [a, 0.0, b],
                [-a, 0.0, -b],
                [b, -a, 0.0],
                [-b, -a, 0.0],
            ],
            dtype=np.float64,
        )
        F0 = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
                [5, 4, 3],
                [4, 8, 3],
                [7, 6, 0],
                [6, 9, 0],
                [11, 10, 4],
                [10, 11, 6],
                [9, 5, 2],
                [5, 9, 11],
                [8, 7, 1],
                [7, 8, 10],
                [2, 5, 3],
                [8, 1, 3],
                [9, 2, 0],
                [1, 7, 0],
                [11, 9, 6],
                [7, 10, 6],
                [5, 11, 4],
                [10, 8, 4],
            ],
            dtype=np.int32,
        )

        (
            V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = jit_vf_samples_to_he_samples(V0, F0)
        #############################################
        self.V = V
        self.F = [F0]
        self._name = name
        self.radius = r
        self.h_out_V = [_h_out_V]
        self.v_origin_H = [_v_origin_H]
        self.h_next_H = [_h_next_H]
        self.h_twin_H = [_h_twin_H]
        self.f_left_H = [_f_left_H]
        self.h_bound_F = [_h_bound_F]
        self._num_vertices = [len(self.V)]
        self.V0 = V0
        self.F0 = F0
        self.t_refine = []
        self.refine_level = 0

    def VF(self, level=-1):
        return self.V[: self.num_vertices(level)], self.F[level]

    def HE(self, level=-1):
        return (
            self.V[: self.num_vertices(level)],
            self.h_out_V[level],
            self.v_origin_H[level],
            self.h_next_H[level],
            self.h_twin_H[level],
            self.f_left_H[level],
            self.h_bound_F[level],
        )

    def num_vertices(self, level=-1):
        return self._num_vertices[level]

    def next_num_vertices(self):
        level = len(self._num_vertices)
        # return self._NUM_VERTICES_[level]
        return 10 * 4**level + 2

    @property
    def name(self):
        return self._name

    def refine(self, timeit=True):
        t = time()
        Fm1 = self.F[-1]
        Vm1 = self.V
        V, F = jit_refine_icososphere(Vm1, Fm1, r=self.radius)
        (
            self.V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = jit_vf_samples_to_he_samples(V, F)

        self.F.append(F)
        self.h_out_V.append(_h_out_V)
        self.v_origin_H.append(_v_origin_H)
        self.h_next_H.append(_h_next_H)
        self.h_twin_H.append(_h_twin_H)
        self.f_left_H.append(_f_left_H)
        self.h_bound_F.append(_h_bound_F)
        self._num_vertices.append(len(self.V))
        t = time() - t
        self.t_refine.append(t)
        self.refine_level += 1


def refine_icososphere_and_save(
    num_refine,
    output_dir="./output/sphere_builder",
    save_name="refinement",
    overwrite=False,
):
    # num_refine = 10
    # data_paths = [f"./output/{save_name}_{n:06d}.pickle" for n in range(num_refine)]
    if os.path.exists(output_dir) and overwrite:
        os.system(f"rm -r {output_dir}")
    elif not os.path.exists(output_dir):
        pass
    else:
        raise ValueError(f"{output_dir} already exists.")
    os.system(f"mkdir -p {output_dir}")
    sf = SphereFactory()
    n = 0
    data_path = f"./{output_dir}/{save_name}_{n:06d}.pickle"
    with open(data_path, "wb") as f:
        pickle.dump(sf, f)
    for n in range(1, num_refine + 1):
        print("------------------")
        print(f"refinement {n=}")
        # t = time()
        sf.refine()
        # t = time() - t
        t = sf.t_refine[-1]
        print(f"refine time {t=}")
        data_path = f"./{output_dir}/{save_name}_{n:06d}.pickle"
        t = time()
        with open(data_path, "wb") as f:
            pickle.dump(sf, f)
        t = time() - t
        print(f"pickle time {t=}")
        print("------------------")


def load_refine_icososphere_and_save_output(
    num_refine,
    output_dir="./output/sphere_builder",
    save_name="refinement",
):
    # num_refine = 10
    data_paths = [
        f"./{output_dir}/{save_name}_{n:06d}.pickle" for n in range(num_refine + 1)
    ]
    SF = []
    for data_path in data_paths:
        with open(data_path, "rb") as f:
            sf = pickle.load(f)
        SF.append(sf)
    return SF


def save_new_from_old():
    SF = load_refine_icososphere_and_save_output(
        9,
        output_dir="./output/sphere_builder",
        save_name="refinement",
    )
    SF[-1].num_vertices()
    sf_old = SF[-1]
    sf_new = SphereFactory()

    sf_new._num_vertices = sf_old._num_vertices
    sf_new.refine_level = 9
    sf_new.V = sf_old.V
    sf_new.h_out_V = [np.array(arr, dtype=np.int32) for arr in sf_old.h_out_V]
    sf_new.v_origin_H = [np.array(arr, dtype=np.int32) for arr in sf_old.v_origin_H]
    sf_new.h_next_H = [np.array(arr, dtype=np.int32) for arr in sf_old.h_next_H]
    sf_new.h_twin_H = [np.array(arr, dtype=np.int32) for arr in sf_old.h_twin_H]
    sf_new.f_left_H = [np.array(arr, dtype=np.int32) for arr in sf_old.f_left_H]
    sf_new.h_bound_F = [np.array(arr, dtype=np.int32) for arr in sf_old.h_bound_F]
    levels = [_ for _ in range(10)]
    _NUM_VERTS_ = [
        12,
        42,
        162,
        642,
        2562,
        10242,
        40962,
        163842,
        655362,
        2621442,
    ]  # [12, 42, 162, 642, 2562, 10242, 40962, 163842]
    # _SURF_NAMES_ = [f"unit_sphere_{N:07d}" for N in _NUM_VERTS_]
    paths = [f"./data/half_edge_arrays/unit_sphere_{N:07d}" for N in _NUM_VERTS_]
    cpaths = [
        f"./data/half_edge_arrays/compressed_unit_sphere_{N:07d}" for N in _NUM_VERTS_
    ]
    for level in range(len(paths)):
        path = paths[level]
        cpath = cpaths[level]
        # path_pickle = path + ".pkl"
        path_npz = path + ".npz"
        print(path_npz)
        sf_new.save_data_arrays(
            path_npz, level=level, compressed=False, chunk=False, remove_unchunked=False
        )
        cpath_npz = cpath + ".npz"
        print(cpath_npz)
        # sf_new.pickle_data_arrays(path_pickle, level=level)
        sf_new.save_data_arrays(
            cpath_npz, level=level, compressed=True, chunk=True, remove_unchunked=True
        )
