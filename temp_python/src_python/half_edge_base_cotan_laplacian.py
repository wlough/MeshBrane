import numpy as np
from numba import jit, prange

_NUMPY_INT_ = np.int64
_NUMPY_FLOAT_ = np.float64


@jit(parallel=True)
def pcotan_laplacian(Q):
    """
    Computes the cotan Laplacian of Q at each vertex
    """
    Nv = self.num_vertices
    lapQ = np.zeros_like(Q)
    for vi in prange(Nv):
        Atot = 0.0
        ri = self.xyz_coord_v(vi)
        qi = Q[vi]
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        for hij in self.generate_H_out_v_clockwise(vi):
            hijm1 = self.h_next_h(self.h_twin_h(hij))
            hijp1 = self.h_twin_h(self.h_prev_h(hij))
            vjm1 = self.v_head_h(hijm1)
            vj = self.v_head_h(hij)
            vjp1 = self.v_head_h(hijp1)

            qj = Q[vj]

            rjm1 = self.xyz_coord_v(vjm1)
            rj = self.xyz_coord_v(vj)
            rjp1 = self.xyz_coord_v(vjp1)

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
        lapQ[vi] /= Atot

    return lapQ


import numpy as np

try:
    from temp_python.src_python.half_edge_base_numba_utils import (
        numba_vf_samples_to_he_samples,
        numba_find_h_right_B,
        numba_he_samples_to_vf_samples,
        numba_V_of_F,
    )
except ImportError:
    pass


def make_half_edge_base_cotan_laplacian(
    module_name="half_edge_base_cotan_laplacian", output_directory="./src/python"
):
    from numba import jit, typeof
    from numba.pycc import CC

    _NUMBA_INT_ = typeof(_NUMPY_INT_(1))
    _NUMBA_FLOAT_ = typeof(_NUMPY_FLOAT_(1))
    cc = CC(module_name)
    cc.output_dir = output_directory

    @jit
    def get_halfedge_index_of_twin(H, h):
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
        hedge_twin = np.flip(H[h])
        ht_arr = np.where((hedge_twin[0] == H[:, 0]) * (hedge_twin[1] == H[:, 1]))[0]
        if ht_arr.size > 0:
            return _NUMBA_INT_(ht_arr[0])
        return _NUMBA_INT_(-1)

    @jit(
        (
            _NUMBA_FLOAT_[:, :],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
        ),
    )
    def jit_V_of_F(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
    ):
        Nf = len(h_bound_F)
        V_of_F = np.zeros((Nf, 3), dtype=_NUMPY_INT_)
        for f in range(Nf):
            h = h_bound_F[f]
            h_start = h
            _v = 0
            while True:
                V_of_F[f, _v] = v_origin_H[h]
                h = h_next_H[h]
                _v += 1
                if h == h_start:
                    break
        return V_of_F

    @cc.export(
        "numba_V_of_F",
        (
            _NUMBA_FLOAT_[:, :],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
        ),
    )
    def numba_V_of_F(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
    ):
        # Nf = len(h_bound_F)
        # V_of_F = np.zeros((Nf, 3), dtype=_NUMPY_INT_)
        # for f in range(Nf):
        #     h = h_bound_F[f]
        #     h_start = h
        #     _v = 0
        #     while True:
        #         V_of_F[f, _v] = v_origin_H[h]
        #         h = h_next_H[h]
        #         _v += 1
        #         if h == h_start:
        #             break
        # return V_of_F
        return jit_V_of_F(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )

    @cc.export(
        "numba_find_h_right_B",
        (
            _NUMBA_FLOAT_[:, :],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
        ),
    )
    def numba_find_h_right_B(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    ):
        """
        Find a half-edge in each boundary of mesh
        """
        Nhedges = len(v_origin_H)
        # find and enumerate boundaries -1,-2,...
        H_need2visit = set([_NUMBA_INT_(h) for h in range(Nhedges) if f_left_H[h] < 0])
        _h_right_B = []
        while H_need2visit:
            b = len(_h_right_B)
            h_start = H_need2visit.pop()
            f_left_H[h_start] = -(b + 1)
            h = _NUMBA_INT_(h_next_H[h_start])
            _h_right_B.append(h)
            while h != h_start:
                H_need2visit.remove(h)
                f_left_H[h] = -(b + 1)
                h = h_next_H[h]
        h_right_B = np.array(_h_right_B, dtype=_NUMPY_INT_)
        return h_right_B

    @cc.export(
        "numba_vf_samples_to_he_samples",
        (
            _NUMBA_FLOAT_[:, :],
            _NUMBA_INT_[:, :],
        ),
    )
    def numba_vf_samples_to_he_samples(xyz_coord_V, vvv_of_F):
        Nfaces = len(vvv_of_F)
        Nvertices = len(xyz_coord_V)
        _Nhedges = 3 * Nfaces * 2
        _H = np.zeros((_Nhedges, 2), dtype=_NUMPY_INT_)
        h_out_V = -np.ones(Nvertices, dtype=_NUMPY_INT_)
        _v_origin_H = np.zeros(_Nhedges, dtype=_NUMPY_INT_)
        _h_next_H = -np.ones(_Nhedges, dtype=_NUMPY_INT_)
        _f_left_H = np.zeros(_Nhedges, dtype=_NUMPY_INT_)
        h_bound_F = np.zeros(Nfaces, dtype=_NUMPY_INT_)

        # h_count = 0
        for f in range(Nfaces):
            h_bound_F[f] = 3 * f
            for i in range(3):
                h = 3 * f + i
                h_next = 3 * f + (i + 1) % 3
                v0 = vvv_of_F[f, i]
                v1 = vvv_of_F[f, (i + 1) % 3]
                _H[h] = [v0, v1]
                _v_origin_H[h] = v0
                _f_left_H[h] = f
                _h_next_H[h] = h_next
                if h_out_V[v0] == -1:
                    h_out_V[v0] = h
        h_count = 3 * Nfaces
        need_twins = set([_NUMBA_INT_(_) for _ in range(h_count)])
        need_next = set([_NUMBA_INT_(0)])
        need_next.pop()
        _h_twin_H = _NUMBA_INT_(-2) * np.ones(
            _Nhedges, dtype=_NUMPY_INT_
        )  # -2 means not set
        while need_twins:
            h = need_twins.pop()
            if _h_twin_H[h] == -2:  # if twin not set
                h_twin = get_halfedge_index_of_twin(
                    _H, h
                )  # returns -1 if twin not found
                if h_twin == -1:  # if twin not found
                    h_twin = _NUMBA_INT_(h_count)
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
                    need_twins.remove(_NUMBA_INT_(h_twin))

        Nhedges = h_count
        # H = _H[:Nhedges]
        v_origin_H = _v_origin_H[:Nhedges]
        h_next_H = _h_next_H[:Nhedges]
        f_left_H = _f_left_H[:Nhedges]
        h_twin_H = _h_twin_H[:Nhedges]
        while need_next:
            h = need_next.pop()
            h_next = h_twin_H[h]
            # rotate ccw around origin of twin until we find nex h on boundary
            while f_left_H[h_next] != -1:
                h_next = h_twin_H[h_next_H[h_next_H[h_next]]]
            h_next_H[h] = h_next

        # find and enumerate boundaries -1,-2,...
        H_need2visit = set(
            [_NUMBA_INT_(h) for h in range(Nhedges) if f_left_H[h] == -1]
        )
        _h_right_B = []
        while H_need2visit:
            b = len(_h_right_B)
            h_start = H_need2visit.pop()
            f_left_H[h_start] = -(b + 1)
            h = _NUMBA_INT_(h_next_H[h_start])
            _h_right_B.append(h)
            while h != h_start:
                H_need2visit.remove(h)
                f_left_H[h] = -(b + 1)
                h = h_next_H[h]
        h_right_B = np.array(_h_right_B, dtype=_NUMPY_INT_)
        return (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )

    @cc.export(
        "numba_he_samples_to_vf_samples",
        (
            _NUMBA_FLOAT_[:, :],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
            _NUMBA_INT_[:],
        ),
    )
    def numba_he_samples_to_vf_samples(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
    ):
        vvv_of_F = jit_V_of_F(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_right_B,
        )

        return (
            xyz_coord_V,
            vvv_of_F,
        )

    # if __name__ == "__main__":
    cc.compile()
    # os.system(f"mv ./{module_name}.cpython-310-x86_64-linux-gnu.so {output_directory}")


def vf_samples_to_he_samples(xyz_coord_V, vvv_of_F):
    """
    Convert vertex-face mesh data to half-edge mesh data

    Parameters
    ----------
    xyz_coord_V : ndarray[:, :] of _NUMPY_FLOAT_
        Nvx3 array of vertex Cartesian coordinates
    vvv_of_F : numpy.ndarray[:, :] of _NUMPY_INT_
        Nfx3 array of vertex indices of faces

    Returns
    -------
    xyz_coord_V : ndarray[:, :] of _NUMPY_FLOAT_
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of _NUMPY_INT_
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of _NUMPY_INT_
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of _NUMPY_INT_
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of _NUMPY_INT_
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of _NUMPY_INT_
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of _NUMPY_INT_
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of _NUMPY_INT_
        h_right_B[n] = half-edge to the right of boundary n.
    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=_NUMPY_FLOAT_)
    vvv_of_F = np.array(vvv_of_F, dtype=_NUMPY_INT_)
    return numba_vf_samples_to_he_samples(xyz_coord_V, vvv_of_F)


def he_samples_to_vf_samples(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    h_right_B,
):
    """
    Convert vertex-face mesh data to half-edge mesh data

    Parameters
    -------
    xyz_coord_V : ndarray[:, :] of _NUMPY_FLOAT_
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of _NUMPY_INT_
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of _NUMPY_INT_
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of _NUMPY_INT_
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of _NUMPY_INT_
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of _NUMPY_INT_
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of _NUMPY_INT_
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of _NUMPY_INT_
        h_right_B[n] = half-edge to the right of boundary n.

    Returns
    ----------
    xyz_coord_V : ndarray[:, :] of _NUMPY_FLOAT_
        Nvx3 array of vertex Cartesian coordinates
    vvv_of_F : numpy.ndarray[:, :] of _NUMPY_INT_
        Nfx3 array of vertex indices of faces

    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=_NUMPY_FLOAT_)
    h_out_V = np.array(h_out_V, dtype=_NUMPY_INT_)
    v_origin_H = np.array(v_origin_H, dtype=_NUMPY_INT_)
    h_next_H = np.array(h_next_H, dtype=_NUMPY_INT_)
    h_twin_H = np.array(h_twin_H, dtype=_NUMPY_INT_)
    f_left_H = np.array(f_left_H, dtype=_NUMPY_INT_)
    h_bound_F = np.array(h_bound_F, dtype=_NUMPY_INT_)
    h_right_B = np.array(h_right_B, dtype=_NUMPY_INT_)
    return numba_he_samples_to_vf_samples(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
    )


def V_of_F(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
    h_right_B,
):
    """
    Convert he mesh data to...

    Parameters
    -------
    xyz_coord_V : ndarray[:, :] of _NUMPY_FLOAT_
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of _NUMPY_INT_
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of _NUMPY_INT_
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of _NUMPY_INT_
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of _NUMPY_INT_
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of _NUMPY_INT_
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of _NUMPY_INT_
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of _NUMPY_INT_
        h_right_B[n] = half-edge to the right of boundary n.

    Returns
    ----------
    vvv_of_F : numpy.ndarray[:, :] of _NUMPY_INT_
        Nfx3 array of vertex indices of faces

    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=_NUMPY_FLOAT_)
    h_out_V = np.array(h_out_V, dtype=_NUMPY_INT_)
    v_origin_H = np.array(v_origin_H, dtype=_NUMPY_INT_)
    h_next_H = np.array(h_next_H, dtype=_NUMPY_INT_)
    h_twin_H = np.array(h_twin_H, dtype=_NUMPY_INT_)
    f_left_H = np.array(f_left_H, dtype=_NUMPY_INT_)
    h_bound_F = np.array(h_bound_F, dtype=_NUMPY_INT_)
    h_right_B = np.array(h_right_B, dtype=_NUMPY_INT_)
    return numba_V_of_F(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_right_B,
    )


def find_h_right_B(
    xyz_coord_V,
    h_out_V,
    v_origin_H,
    h_next_H,
    h_twin_H,
    f_left_H,
    h_bound_F,
):
    """

    Parameters
    ----------
    xyz_coord_V : ndarray[:, :] of _NUMPY_FLOAT_
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of _NUMPY_INT_
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of _NUMPY_INT_
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of _NUMPY_INT_
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of _NUMPY_INT_
        h_twin_H[j] = half-edge antiparalel to half-edge j
    f_left_H : ndarray[:] of _NUMPY_INT_
        f_left_H[j] = face to the left of half-edge j
    h_bound_F : ndarray[:] of _NUMPY_INT_
        h_bound_F[k] = some half-edge on the ccw boudary of face k

    Returns
    -------
    h_right_B : ndarray[:] of _NUMPY_INT_
        h_right_B[n] = half-edge in complement boundary n of the mesh
    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=_NUMPY_FLOAT_)
    h_out_V = np.array(h_out_V, dtype=_NUMPY_INT_)
    v_origin_H = np.array(v_origin_H, dtype=_NUMPY_INT_)
    h_next_H = np.array(h_next_H, dtype=_NUMPY_INT_)
    h_twin_H = np.array(h_twin_H, dtype=_NUMPY_INT_)
    f_left_H = np.array(f_left_H, dtype=_NUMPY_INT_)
    h_bound_F = np.array(h_bound_F, dtype=_NUMPY_INT_)
    return numba_find_h_right_B(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )
