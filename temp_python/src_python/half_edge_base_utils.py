import numpy as np

try:
    from temp_python.src_python.half_edge_base_numba_utils import (
        numba_vf_samples_to_he_samples,
        numba_find_h_right_B,
        numba_he_samples_to_vf_samples,
        numba_V_of_F,
    )
except ImportError:
    print("Could not import numba utils")
    # make_half_edge_base_numba_utils(
    #     module_name="half_edge_base_numba_utils", output_directory="./src/python"
    # )
    pass
# _NUMPY_INT_ = np.int64
# _NUMPY_FLOAT_ = np.float64
from temp_python.src_python.global_vars import (
    INT_TYPE,
    FLOAT_TYPE,
    _NUMBA_INT_,
    _NUMBA_FLOAT_,
)


def make_half_edge_base_numba_utils(
    module_name="half_edge_base_numba_utils", output_directory="./src/python"
):
    from numba import jit  # , typeof
    from numba.pycc import CC

    # _NUMBA_INT_ = typeof(INT_TYPE(1))
    # _NUMBA_FLOAT_ = typeof(FLOAT_TYPE(1))
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
        V_of_F = np.zeros((Nf, 3), dtype=INT_TYPE)
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
        # V_of_F = np.zeros((Nf, 3), dtype=INT_TYPE)
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
        h_right_B = np.array(_h_right_B, dtype=INT_TYPE)
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
        _H = np.zeros((_Nhedges, 2), dtype=INT_TYPE)
        h_out_V = -np.ones(Nvertices, dtype=INT_TYPE)
        _v_origin_H = np.zeros(_Nhedges, dtype=INT_TYPE)
        _h_next_H = -np.ones(_Nhedges, dtype=INT_TYPE)
        _f_left_H = np.zeros(_Nhedges, dtype=INT_TYPE)
        h_bound_F = np.zeros(Nfaces, dtype=INT_TYPE)

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
            _Nhedges, dtype=INT_TYPE
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
        h_right_B = np.array(_h_right_B, dtype=INT_TYPE)
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
    xyz_coord_V : ndarray[:, :] of FLOAT_TYPE
        Nvx3 array of vertex Cartesian coordinates
    vvv_of_F : numpy.ndarray[:, :] of INT_TYPE
        Nfx3 array of vertex indices of faces

    Returns
    -------
    xyz_coord_V : ndarray[:, :] of FLOAT_TYPE
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of INT_TYPE
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of INT_TYPE
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of INT_TYPE
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of INT_TYPE
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of INT_TYPE
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of INT_TYPE
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of INT_TYPE
        h_right_B[n] = half-edge to the right of boundary n.
    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=FLOAT_TYPE)
    vvv_of_F = np.array(vvv_of_F, dtype=INT_TYPE)
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
    xyz_coord_V : ndarray[:, :] of FLOAT_TYPE
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of INT_TYPE
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of INT_TYPE
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of INT_TYPE
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of INT_TYPE
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of INT_TYPE
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of INT_TYPE
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of INT_TYPE
        h_right_B[n] = half-edge to the right of boundary n.

    Returns
    ----------
    xyz_coord_V : ndarray[:, :] of FLOAT_TYPE
        Nvx3 array of vertex Cartesian coordinates
    vvv_of_F : numpy.ndarray[:, :] of INT_TYPE
        Nfx3 array of vertex indices of faces

    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=FLOAT_TYPE)
    h_out_V = np.array(h_out_V, dtype=INT_TYPE)
    v_origin_H = np.array(v_origin_H, dtype=INT_TYPE)
    h_next_H = np.array(h_next_H, dtype=INT_TYPE)
    h_twin_H = np.array(h_twin_H, dtype=INT_TYPE)
    f_left_H = np.array(f_left_H, dtype=INT_TYPE)
    h_bound_F = np.array(h_bound_F, dtype=INT_TYPE)
    h_right_B = np.array(h_right_B, dtype=INT_TYPE)
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
    xyz_coord_V : ndarray[:, :] of FLOAT_TYPE
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of INT_TYPE
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of INT_TYPE
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of INT_TYPE
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of INT_TYPE
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of INT_TYPE
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of INT_TYPE
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of INT_TYPE
        h_right_B[n] = half-edge to the right of boundary n.

    Returns
    ----------
    vvv_of_F : numpy.ndarray[:, :] of INT_TYPE
        Nfx3 array of vertex indices of faces

    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=FLOAT_TYPE)
    h_out_V = np.array(h_out_V, dtype=INT_TYPE)
    v_origin_H = np.array(v_origin_H, dtype=INT_TYPE)
    h_next_H = np.array(h_next_H, dtype=INT_TYPE)
    h_twin_H = np.array(h_twin_H, dtype=INT_TYPE)
    f_left_H = np.array(f_left_H, dtype=INT_TYPE)
    h_bound_F = np.array(h_bound_F, dtype=INT_TYPE)
    h_right_B = np.array(h_right_B, dtype=INT_TYPE)
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
    xyz_coord_V : ndarray[:, :] of FLOAT_TYPE
        xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : ndarray[:] of INT_TYPE
        h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : ndarray[:] of INT_TYPE
        v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : ndarray[:] of INT_TYPE
        h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : ndarray[:] of INT_TYPE
        h_twin_H[j] = half-edge antiparalel to half-edge j
    f_left_H : ndarray[:] of INT_TYPE
        f_left_H[j] = face to the left of half-edge j
    h_bound_F : ndarray[:] of INT_TYPE
        h_bound_F[k] = some half-edge on the ccw boudary of face k

    Returns
    -------
    h_right_B : ndarray[:] of INT_TYPE
        h_right_B[n] = half-edge in complement boundary n of the mesh
    """
    xyz_coord_V = np.array(xyz_coord_V, dtype=FLOAT_TYPE)
    h_out_V = np.array(h_out_V, dtype=INT_TYPE)
    v_origin_H = np.array(v_origin_H, dtype=INT_TYPE)
    h_next_H = np.array(h_next_H, dtype=INT_TYPE)
    h_twin_H = np.array(h_twin_H, dtype=INT_TYPE)
    f_left_H = np.array(f_left_H, dtype=INT_TYPE)
    h_bound_F = np.array(h_bound_F, dtype=INT_TYPE)
    return numba_find_h_right_B(
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    )
