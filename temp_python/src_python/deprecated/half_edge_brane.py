from temp_python.src_python.ply_tools import VertTri2HalfEdgeConverter
from temp_python.src_python.jit_brane_utils import (
    _NUMPY_FLOAT_,
    _NUMPY_INT_,
    _NUMBA_FLOAT_,
    _NUMBA_INT_,
    xyz_coord_V_numba_type,
    h_out_V_numba_type,
    v_origin_H_numba_type,
    h_next_H_numba_type,
    h_twin_H_numba_type,
    f_left_H_numba_type,
    h_bound_F_numba_type,
    h_comp_B_numba_type,
    jit_vf_samples_to_he_samples,
    half_edge_arrays_to_dicts,
    half_edge_dicts_to_arrays,
    find_h_comp_B_array,
)
from numba import jit, prange
import numpy as np
from numba.experimental import jitclass
from numba.typed import Dict
from numba.types import unicode_type


@jit(parallel=True)
def pcotan_laplacian(self, Q):
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


@jit(parallel=True)
def pV_of_F(self):
    """
    vertices of faces
    """
    Fkeys = sorted(self.h_bound_F.keys())
    F = np.zeros((len(Fkeys), 3), dtype=_NUMPY_INT_)
    Nf = len(Fkeys)
    for _f in prange(Nf):
        f = Fkeys[_f]
        h0 = self.h_bound_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        F[_f] = [self.v_origin_h(h0), self.v_origin_h(h1), self.v_origin_h(h2)]

    return F


@jit(parallel=True)
def pgaussian_curvature_V(self):
    """
    Compute the Gaussian curvature at each vertex in parallel
    """
    Nv = self.num_vertices
    K = np.zeros(Nv, dtype=_NUMPY_FLOAT_)
    Vkeys = sorted(self.xyz_coord_V.keys())
    for _i in prange(Nv):
        i = Vkeys[_i]
        K[_i] = self.gaussian_curvature_v(i)
    return K


@jit(parallel=True)
def pcompute_curvature_data(self):
    """
    Compute the mean curvature vector at all vertices
    """
    Vkeys = sorted(self.xyz_coord_V.keys())
    X = self.xyz_array
    lapX = self.laplacian(X)
    H = np.zeros_like(X[:, 0])
    K = np.zeros_like(X[:, 0])
    n = np.zeros_like(X)
    for _i in prange(len(Vkeys)):
        i = Vkeys[_i]
        mcvec = lapX[_i]
        f = self.f_left_h(self.h_out_v(i))
        af_vec = self.vec_area_f(f)
        mcvec_sign = np.sign(np.dot(mcvec, af_vec))
        n[_i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
        H[_i] = np.dot(n[_i], mcvec) / 2
        K[_i] = self.gaussian_curvature_v(i)

    lapH = self.laplacian(H)
    return H, K, lapH, n


brane_spec = [
    ("_name", unicode_type),
    ("_xyz_coord_V", xyz_coord_V_numba_type),
    ("_h_out_V", h_out_V_numba_type),
    ("_v_origin_H", v_origin_H_numba_type),
    ("_h_next_H", h_next_H_numba_type),
    ("_h_twin_H", h_twin_H_numba_type),
    ("_f_left_H", f_left_H_numba_type),
    ("_h_bound_F", h_bound_F_numba_type),
    ("_h_comp_B", h_comp_B_numba_type),
    ###########################
    ("bending_modulus", _NUMBA_FLOAT_),
    ("splay_modulus", _NUMBA_FLOAT_),
    ("length_reg_stiffness", _NUMBA_FLOAT_),
    ("area_reg_stiffness", _NUMBA_FLOAT_),
    ("volume_reg_stiffness", _NUMBA_FLOAT_),
    ("linear_drag_coeff", _NUMBA_FLOAT_),
    ("spontaneous_curvature", _NUMBA_FLOAT_),
    ###########################
    # geometric stuff
    ("preferred_total_volume", _NUMBA_FLOAT_),
    ("preferred_total_area", _NUMBA_FLOAT_),
    ("preferred_face_area", _NUMBA_FLOAT_),
    ("preferred_cell_volume", _NUMBA_FLOAT_),
    ("preferred_edge_length", _NUMBA_FLOAT_),
]


@jitclass(brane_spec)
class Brane:
    """
    h_bound_F-->h_adjacent_F
    Dict-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.arrays of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as dicts of vertex/half-edge/face labels. Each data dict has a name of the form "a_description_B", where "a" denotes the type of object associated with the dict elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "B" denotes the type of object associated with the dict indices ("V" for vertex, "H" for half-edge, or "F" for face), and "description" is a description of information represented by the dict. For example, "_v_origin_H" is a dict of vertices at the origin of each half-edge. The i-th element of data dict "a_description_B" can be accessed using the "a_description_b(i)" method.

    Properties
    ----------
    xyz_coord_V : dict of numpy.array
        _xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : dict of int
        _h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : dict of int
        _v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : dict of int
        _h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : dict of int
        _h_twin_H[j] = half-edge antiparalel to half-edge j
    f_left_H : dict of int
        _f_left_H[j] = face to the left of half-edge j
    h_bound_F : dict of int
        _h_bound_F[k] = some half-edge on the ccw boudary of face k
    h_comp_B : dict of int
        _h_comp_B[n] = half-edge in complement boundary n of the mesh

    Initialization
    ---------------
    The HalfEdgeMesh class can be initialized in several ways:
    - Directly from half-edge mesh data dicts:
        HalfEdgeMesh(xyz_coord_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_bound_F)
    - From a dict of vertex positions and a dict of face vertices:
        HalfEdgeMesh.from_vert_face_dict(xyz_coord_V, vvv_of_F)
    - From a ply file (binary/ascii) containing vertex/face data:
        HalfEdgeMesh.from_vertex_face_ply(ply_path)
        * See HalfEdgeMeshBuilder for more details about ply format.
    - From a ply file (binary/ascii) containing half-edge mesh data:
        HalfEdgeMesh.from_half_edge_ply(ply_path)
        * See HalfEdgeMeshBuilder for more details about ply format.
    - From C-Glass binary data:
        HalfEdgeMesh.from_cglass_binary(data)
        * To be implemented...
    """

    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_comp_B,
        find_h_comp_B=False,
        bending_modulus=1.0,
        splay_modulus=1.0,
        length_reg_stiffness=1.0,
        area_reg_stiffness=1.0,
        volume_reg_stiffness=1.0,
        linear_drag_coeff=1.0,
        spontaneous_curvature=1.0,
    ):

        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.h_bound_F = h_bound_F
        self.f_left_H = f_left_H
        if find_h_comp_B:
            self.h_comp_B = self.find_h_comp_B()
        else:
            self.h_comp_B = h_comp_B

        self.bending_modulus = bending_modulus
        self.splay_modulus = splay_modulus
        self.length_reg_stiffness = length_reg_stiffness
        self.area_reg_stiffness = area_reg_stiffness
        self.volume_reg_stiffness = volume_reg_stiffness
        self.linear_drag_coeff = linear_drag_coeff
        self.spontaneous_curvature = spontaneous_curvature

    #######################################################
    def run_tests(self):
        v, h, f = 1, 2, 3
        # Combinatorial maps
        self.xyz_coord_v(v)
        self.h_out_v(v)
        self.h_bound_f(f)
        self.v_origin_h(h)
        self.f_left_h(h)
        self.h_next_h(h)
        self.h_twin_h(h)
        #
        self.v_head_h(h)
        # self.f_right_h(h)
        # self.h_out_cw_from_h(h)
        self.h_in_cw_from_h(h)
        self.h_prev_h(h)
        self.h_prev_h_by_rot(h)
        # Predicates
        self.complement_boundary_contains_h(h)
        self.interior_boundary_contains_h(h)
        self.boundary_contains_h(h)
        self.boundary_contains_v(v)
        self.h_is_locally_delaunay(h)
        self.h_is_flippable(h)
        # Generators
        self.generate_H_out_v_clockwise(v)
        self.generate_H_in_cw_from_h(h)
        self.generate_H_bound_f(f)
        self.generate_H_next_h(h)
        self.generate_V_of_f(f)
        # Data exporters
        self.xyz_array
        self.V_of_F
        self.data_arrays
        self.data_dicts
        # Simplicial operations
        V, H, F = self.star_of_vertex(v)
        self.star_of_vertex(v)
        # self.star_of_edge(h)
        self.star(V, H, F)
        self.closure(V, H, F)
        self.link(V, H, F)
        self.find_h_comp_B()
        # geometry
        self.vec_area_f(f)
        self.area_f(f)
        # self.total_area_of_faces()
        self.barcell_area(v)
        # self.total_area_of_dual_barcells()
        self.vorcell_area(v)
        # self.total_area_of_dual_vorcells()
        self.meyercell_area(v)
        # self.total_area_of_dual_meyercells()
        # Misc helper functions
        self.valence_v(v)
        self.num_vertices
        self.num_edges
        self.num_faces
        self.num_boundaries
        self.genus
        self.euler_characteristic

    #######################################################
    # Fundamental accessors
    @property
    def xyz_coord_V(self):
        return self._xyz_coord_V

    @xyz_coord_V.setter
    def xyz_coord_V(self, value):
        if isinstance(value, Dict):
            self._xyz_coord_V = value
        else:
            raise TypeError("xyz_coord_V must be a numba.typed.Dict.")

    @property
    def h_out_V(self):
        return self._h_out_V

    @h_out_V.setter
    def h_out_V(self, value):
        if isinstance(value, Dict):
            self._h_out_V = value
        else:
            raise TypeError("h_out_V must be a numba.typed.Dict.")

    @property
    def v_origin_H(self):
        return self._v_origin_H

    @v_origin_H.setter
    def v_origin_H(self, value):
        if isinstance(value, Dict):
            self._v_origin_H = value
        else:
            raise TypeError("v_origin_H must be a numba.typed.Dict.")

    @property
    def h_next_H(self):
        return self._h_next_H

    @h_next_H.setter
    def h_next_H(self, value):
        if isinstance(value, Dict):
            self._h_next_H = value
        else:
            raise TypeError("h_next_H must be a numba.typed.Dict.")

    @property
    def h_twin_H(self):
        return self._h_twin_H

    @h_twin_H.setter
    def h_twin_H(self, value):
        if isinstance(value, Dict):
            self._h_twin_H = value
        else:
            raise TypeError("h_twin_H must be a numba.typed.Dict.")

    @property
    def f_left_H(self):
        return self._f_left_H

    @f_left_H.setter
    def f_left_H(self, value):
        if isinstance(value, Dict):
            self._f_left_H = value
        else:
            raise TypeError("f_left_H must be a numba.typed.Dict.")

    @property
    def h_bound_F(self):
        return self._h_bound_F

    @h_bound_F.setter
    def h_bound_F(self, value):
        if isinstance(value, Dict):
            self._h_bound_F = value
        else:
            raise TypeError("h_bound_F must be a numba.typed.Dict.")

    @property
    def h_comp_B(self):
        return self._h_comp_B

    @h_comp_B.setter
    def h_comp_B(self, value):
        if isinstance(value, Dict):
            self._h_comp_B = value
        else:
            raise TypeError("h_comp_B must be a numba.typed.Dict.")

    #######################################################
    # Combinatorial maps #
    #######################################################
    def xyz_coord_v(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self._xyz_coord_V[v]

    def h_out_v(self, v):
        """
        get index of an outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        return self._h_out_V[v]

    def h_bound_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        return self._h_bound_F[f]

    def v_origin_h(self, h):
        """get index of the vertex at the origin of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self._v_origin_H[h]

    def f_left_h(self, h):
        """get index of the face to the left of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: face index
        """
        return self._f_left_H[h]

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face cycle

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self._h_next_H[h]

    def h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        return self._h_twin_H[h]

    def h_comp_b(self, b):
        """get index of a half-edge contained in boundary b

        Args:
            b (int): boundary index

        Returns:
            int: half-edge index
        """
        return self._h_comp_B[b]

    # # Derived combinatorial maps
    def v_head_h(self, h):
        """get index of the vertex at the head of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        return self.v_origin_h(self.h_twin_h(h))

    def f_right_h(self, h):
        return self.f_left_h(self.h_twin_h(h))

    def h_out_cw_from_h(self, h):
        return self.h_next_h(self.h_twin_h(h))

    def h_in_cw_from_h(self, h):
        return self.h_twin_h(self.h_next_h(h))

    def h_prev_h(self, h):
        """
        Finds half-edge previous to h by following next cycle.
        Safe for half-edges of non-triangle faces and boundaries.
        """
        h_next = self.h_next_h(h)

        while h_next != h:
            h_prev = h_next
            h_next = self.h_next_h(h_prev)
        return h_prev

    def h_prev_h_by_rot(self, h):
        """
        Finds half-edge previous to h by rotating around origin of h. Faster when length of next cycle is much larger than valence of origin of h.
        """
        p_h = self.h_twin_h(h)
        n_h = self.h_next_h(p_h)
        while n_h != h:
            p_h = self.h_twin_h(n_h)
            n_h = self.h_next_h(p_h)
        return p_h

    def h_out_ccw_from_h(self, h):
        return self.h_twin_h(self.h_prev_h(h))

    ######################################################
    # Predicates
    ######################################################
    def complement_boundary_contains_h(self, h):
        """check if half-edge h is in the boundary of the mesh"""
        return self.f_left_h(h) < 0

    def interior_boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.f_left_h(self.h_twin_h(h)) < 0

    def boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return self.f_left_h(h) < 0 or self.f_left_h(self.h_twin_h(h)) < 0

    def boundary_contains_v(self, v):
        """check if vertex v is on the boundary of the mesh"""
        for h in self.generate_H_out_v_clockwise(v):
            if self.f_left_h(h) < 0 or self.f_left_h(self.h_twin_h(h)) < 0:
                return True
        return False

    def h_is_locally_delaunay(self, h):
        r"""
        Checks if edge is locally Delaunay (the circumcircle of the triangle to one side of the edge does not contain the vertex opposite the edge on the triangle's other side)
             vj
             /|\
           vk | vi
             \|/
             vl
        """
        vi = self.v_head_h(self.h_next_h(self.h_twin_h(h)))
        vj = self.v_head_h(h)
        vk = self.v_head_h(self.h_next_h(h))
        vl = self.v_origin_h(h)

        rij = self.xyz_coord_v(vj) - self.xyz_coord_v(vi)
        ril = self.xyz_coord_v(vl) - self.xyz_coord_v(vi)
        rkj = self.xyz_coord_v(vj) - self.xyz_coord_v(vk)
        rkl = self.xyz_coord_v(vl) - self.xyz_coord_v(vk)

        alphai = np.arccos(
            np.dot(rij, ril) / (np.linalg.norm(rij) * np.linalg.norm(ril))
        )
        alphak = np.arccos(
            np.dot(rkl, rkj) / (np.linalg.norm(rkl) * np.linalg.norm(rkj))
        )

        return alphai + alphak <= np.pi

    def h_is_flippable(self, h):
        r"""
        edge flip hlj-->hki is allowed unless hlj is on a boundary or vi and vk are already neighbors
        vj
        /|\
      vk | vi
        \|/
        vl
        """
        if self.boundary_contains_h(h):
            return False
        hlj = h
        hjk = self.h_next_h(hlj)
        # hjl = self.h_twin_h(hlj)
        hli = self.h_next_h(self.h_twin_h(hlj))
        vi = self.v_head_h(hli)
        vk = self.v_head_h(hjk)

        for him in self.generate_H_out_v_clockwise(vi):
            if self.v_head_h(him) == vk:
                return False
        return True

    #######################################################
    # Generators
    def generate_H_out_v_clockwise(self, v):
        """
        Generate outgoing half-edges from vertex v in clockwise order until the starting half-edge is reached again
        """
        h = self.h_out_v(v)
        h_start = h
        while True:
            yield h
            h = self.h_next_h(self.h_twin_h(h))
            if h == h_start:
                break

    def generate_H_in_cw_from_h(self, h):
        """ """
        h_start = h
        while True:
            yield h
            h = self.h_in_cw_from_h(h)
            if h == h_start:
                break

    def generate_H_bound_f(self, f):
        """Generate half-edges on the boundary of face f"""
        h = self.h_bound_f(f)
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_H_next_h(self, h):
        """Generate half-edges in the face/boundary cycle containing half-edge h"""
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_V_of_f(self, f):
        h = self.h_bound_f(f)
        h_start = h
        while True:
            yield self.v_origin_h(h)
            h = self.h_next_h(h)
            if h == h_start:
                break

    #######################################################
    # Data exporters
    @property
    def xyz_array(self):
        Vkeys = sorted(self.xyz_coord_V.keys())
        Nv = len(Vkeys)
        xyz_array = np.zeros((Nv, 3), _NUMPY_FLOAT_)
        for i, v in enumerate(Vkeys):
            xyz_array[i] = self.xyz_coord_v(v)
        return xyz_array

    @property
    def V_of_F(self):
        return pV_of_F(self)

    def rekey_half_edge_dicts(self):
        """
        Reorders and updates keys to be contiguous integers starting
        """
        xyz_coord_V = self.xyz_coord_V
        h_out_V = self.h_out_V
        v_origin_H = self.v_origin_H
        h_next_H = self.h_next_H
        h_twin_H = self.h_twin_H
        f_left_H = self.f_left_H
        h_bound_F = self.h_bound_F
        h_comp_B = self.h_comp_B
        old_V_new = sorted(xyz_coord_V.keys())
        new_V_old = {val: key for key, val in enumerate(old_V_new) if val != key}
        old_H_new = sorted(h_next_H.keys())
        new_H_old = {val: key for key, val in enumerate(old_H_new) if val != key}
        old_F_new = sorted(h_bound_F.keys())
        new_F_old = {val: key for key, val in enumerate(old_F_new) if val != key}
        ##
        # oldBkeys = [-b1, -b2, -b3, ...]
        # newBkeys = [-1, -2, -3, ...]
        oldBkeys = sorted(h_comp_B.keys(), reverse=True)
        newBkeys = np.array(
            [-(_key + 1) for _key, val in enumerate(oldBkeys)], dtype=_NUMPY_INT_
        )
        new_B_old = {
            k_old: k_new for k_old, k_new in zip(oldBkeys, newBkeys) if k_old != k_new
        }
        new_F_old.update(new_B_old)

        if new_V_old or new_H_old or new_F_old:
            for k_old, k_new in new_V_old.items():
                xyz = xyz_coord_V.pop(k_old)
                xyz_coord_V[k_new] = xyz
                h_out_v_old = h_out_V.pop(k_old)
                if h_out_v_old in new_H_old:
                    h_out_v = new_H_old[h_out_v_old]
                    h_out_V[k_new] = h_out_v
                else:
                    h_out_V[k_new] = h_out_v_old
            for k_old, k_new in new_H_old.items():
                v_origin_h_old = v_origin_H.pop(k_old)
                if v_origin_h_old in new_V_old:
                    v_origin_h = new_V_old[v_origin_h_old]
                    v_origin_H[k_new] = v_origin_h
                else:
                    v_origin_H[k_new] = v_origin_h_old
                h_next_h_old = h_next_H.pop(k_old)
                if h_next_h_old in new_H_old:
                    h_next_h = new_H_old[h_next_h_old]
                    h_next_H[k_new] = h_next_h
                else:
                    h_next_H[k_new] = h_next_h_old
                h_twin_h_old = h_twin_H.pop(k_old)
                if h_twin_h_old in new_H_old:
                    h_twin_h = new_H_old[h_twin_h_old]
                    h_twin_H[k_new] = h_twin_h
                else:
                    h_twin_H[k_new] = h_twin_h_old
                f_left_h_old = f_left_H.pop(k_old)
                if f_left_h_old in new_F_old:
                    f_left_h = new_F_old[f_left_h_old]
                    f_left_H[k_new] = f_left_h
                else:
                    f_left_H[k_new] = f_left_h_old
            for _k_old, _k_new in new_F_old.items():
                k_old = _k_old
                k_new = _k_new
                h_bound_f_old = h_bound_F.pop(k_old)
                if h_bound_f_old in new_H_old:
                    h_bound_f = new_H_old[h_bound_f_old]
                    h_bound_F[k_new] = h_bound_f
                else:
                    h_bound_F[k_new] = h_bound_f_old

            for k_old, k_new in new_B_old.items():
                h_comp_b_old = h_comp_B.pop(k_old)
                if h_comp_b_old in new_H_old:
                    h_comp_b = new_H_old[h_comp_b_old]
                    h_comp_B[k_new] = h_comp_b
                else:
                    h_comp_B[k_new] = h_comp_b_old

            self.xyz_coord_V = xyz_coord_V
            self.h_out_V = h_out_V
            self.v_origin_H = v_origin_H
            self.h_next_H = h_next_H
            self.h_twin_H = h_twin_H
            self.f_left_H = f_left_H
            self.h_bound_F = h_bound_F
            self.h_comp_B = h_comp_B

    @property
    def data_dicts(self):
        """
        get dicts of vertex positions and connectivity data and required to reconstruct mesh or write to ply file
        """
        return (
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
            self.h_comp_B,
        )

    @property
    def data_arrays(self):
        """
        Get lists of vertex positions and connectivity data and required to reconstruct mesh or write to ply file. Vertex/half-edge/face indices are sorted in ascending order and relabeled so that the first index is 0, the second index is 1, etc...
        """
        self.rekey_half_edge_dicts()
        Nv = len(self.xyz_coord_V)
        Nh = len(self.h_next_H)
        Nf = len(self.h_bound_F)
        Nb = len(self.h_comp_B)
        xyz_coord_V_array = np.zeros((Nv, 3), dtype=_NUMPY_FLOAT_)
        h_out_V_array = np.zeros(Nv, dtype=_NUMPY_INT_)
        v_origin_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
        h_next_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
        h_twin_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
        f_left_H_array = np.zeros(Nh, dtype=_NUMPY_INT_)
        h_bound_F_array = np.zeros(Nf, dtype=_NUMPY_INT_)
        h_comp_B_array = np.zeros(Nb, dtype=_NUMPY_INT_)
        for v_arr in range(Nv):
            v_dic = _NUMPY_INT_(v_arr)
            xyz_coord_V_array[v_arr] = self.xyz_coord_v(v_dic)
            h_out_V_array[v_arr] = self.h_out_v(v_dic)  #
        for h_arr in range(Nh):
            h_dic = _NUMPY_INT_(h_arr)
            v_origin_H_array[h_arr] = self.v_origin_h(h_dic)
            h_next_H_array[h_arr] = self.h_next_h(h_dic)
            h_twin_H_array[h_arr] = self.h_twin_h(h_dic)
            f_left_H_array[h_arr] = self.f_left_h(h_dic)
        for f_arr in range(Nf):
            f_dic = _NUMPY_INT_(f_arr)
            h_bound_F_array[f_arr] = self.h_bound_f(f_dic)
        for b_arr in range(Nb):
            b_dic = _NUMPY_INT_(-(b_arr + 1))
            h_comp_B_array[b_arr] = self.h_comp_b(b_dic)
        return (
            xyz_coord_V_array,
            h_out_V_array,
            v_origin_H_array,
            h_next_H_array,
            h_twin_H_array,
            f_left_H_array,
            h_bound_F_array,
            h_comp_B_array,
        )

    ######################################################
    # The star St(s) of a k-simplex s consists of: s and all (n>k)-simplices that contain s.
    # The closure Cl(s) of a k-simplex s consists of: s and all (n<k)-simplices that are proper faces of s.
    # The link Lk(s)=Cl(St(s))-St(Cl(s)).
    def star_of_vertex(self, v):
        """Star of a vertex is the set of all simplices that contain the vertex."""
        V = {v}
        H = set()
        F = set()
        for h in self.generate_H_out_v_clockwise(v):
            ht = self.h_twin_h(h)
            H.update([h, ht])
            if not self.complement_boundary_contains_h(h):
                F.add(self.f_left_h(h))

        return V, H, F

    def star(self, V_in, H_in, F_in):
        """The star of a single simplex is the set of all simplices that have the simplex as a face."""
        F = F_in.copy()
        H = H_in.copy()
        V = V_in.copy()

        for h in H_in:
            ht = self.h_twin_h(h)
            H.add(ht)
            if not self.complement_boundary_contains_h(h):
                F.add(self.f_left_h(h))
            if not self.complement_boundary_contains_h(ht):
                F.add(self.f_left_h(ht))
        for v in V_in:
            for h in self.generate_H_out_v_clockwise(v):
                H.add(h)
                H.add(self.h_twin_h(h))
                if not self.complement_boundary_contains_h(h):
                    F.add(self.f_left_h(h))
        return V, H, F

    def closure(self, V_in, H_in, F_in):
        """The closure of a single simplex is the set of all simplices that contain the simplex as a subset of their vertices."""
        F = F_in.copy()
        H = H_in.copy()
        V = V_in.copy()

        for f in F_in:
            for h in self.generate_H_bound_f(f):
                H.add(h)
                H.add(self.h_twin_h(h))
                V.add(self.v_origin_h(h))
                V.add(self.v_origin_h(self.h_twin_h(h)))
        for h in H_in:
            V.add(self.v_origin_h(h))
            V.add(self.v_origin_h(self.h_twin_h(h)))
        return V, H, F

    def link(self, V, H, F):
        """"""
        StCl_V, StCl_H, StCl_F = self.star(*self.closure(V, H, F))
        ClSt_V, ClSt_H, ClSt_F = self.closure(*self.star(V, H, F))
        return ClSt_V - StCl_V, ClSt_H - StCl_H, ClSt_F - StCl_F

    def find_h_comp_B(self):
        """Find a half-edge in each boundary of mesh"""
        h_comp_B = dict()
        complement_boundary_contains_H = set()
        F_need2check = set(self.h_bound_F)  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            for h in self.generate_H_bound_f(f):
                if self.interior_boundary_contains_h(h):
                    complement_boundary_contains_H.add(self.h_twin_h(h))
        while complement_boundary_contains_H:
            h = complement_boundary_contains_H.pop()
            bdry = self.f_left_h(h)
            h_comp_B[bdry] = h
            for h in self.generate_H_next_h(h):
                complement_boundary_contains_H.discard(h)
        return h_comp_B

    # ######################################################
    # # Geometry

    def vec_area_f(self, f):
        h0 = self.h_bound_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        r0 = self.xyz_coord_v(self.v_origin_h(h0))
        r1 = self.xyz_coord_v(self.v_origin_h(h1))
        r2 = self.xyz_coord_v(self.v_origin_h(h2))
        vec_area = 0.5 * (np.cross(r0, r1) + np.cross(r1, r2) + np.cross(r2, r0))
        return vec_area

    def area_f(self, f):
        return np.linalg.norm(self.vec_area_f(f))

    def area_F(self):
        keys = sorted(self.h_bound_F.keys())
        N = len(keys)
        A = np.zeros(N, dtype=_NUMPY_FLOAT_)
        for _, k in enumerate(keys):
            A[_] = self.area_f(k)
        return A

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

    def barcell_area_V(self):
        keys = sorted(self.xyz_coord_V.keys())
        N = len(keys)
        A = np.zeros(N, dtype=_NUMPY_FLOAT_)
        for _, k in enumerate(keys):
            A[_] = self.barcell_area(k)
        return A

    def vorcell_area(self, v):
        r"""area of voronoi cell dual to vertex v
                  v=v0
                //  \\
               // |  \\
              // /|   \\
             //   ||   \\
            //    || h20\\
           //     ||     \\
        ...       ||h01    v2
           \\     ||     //
            \\    || h12//
             \\   ||   //
              \\  ||/ //
               \\ |  //
                \\| //
                  v1
        """
        Atot = 0.0
        r0 = self.xyz_coord_v(v)
        for h in self.generate_H_out_v_clockwise(v):
            if self.complement_boundary_contains_h(h):
                continue
            r1 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(h)))
            r2 = self.xyz_coord_v(self.v_origin_h(self.h_next_h(self.h_next_h(h))))
            r01 = r1 - r0
            r12 = r2 - r1
            r20 = r0 - r2

            norm_r20 = np.linalg.norm(r20)
            norm_r01 = np.linalg.norm(r01)
            norm_r12 = np.linalg.norm(r12)
            cos_210 = -np.dot(r12, r01) / (norm_r12 * norm_r01)
            cos_021 = -np.dot(r20, r12) / (norm_r20 * norm_r12)

            cot_210 = cos_210 / np.sqrt(1 - cos_210**2)
            cot_021 = cos_021 / np.sqrt(1 - cos_021**2)
            Atot += norm_r20**2 * cot_210 / 8 + norm_r01**2 * cot_021 / 8

        return Atot

    def vorcell_area_V(self):
        keys = sorted(self.xyz_coord_V.keys())
        N = len(keys)
        A = np.zeros(N, dtype=_NUMPY_FLOAT_)
        for _, k in enumerate(keys):
            A[_] = self.vorcell_area(k)
        return A

    def signed_volume_f(self, f):

        h1 = self.h_bound_f(f)
        h2 = self.h_next_h(h1)
        h3 = self.h_next_h(h2)
        u = self.xyz_coord_v(self.v_origin_h(h1))
        v = self.xyz_coord_v(self.v_origin_h(h2))
        w = self.xyz_coord_v(self.v_origin_h(h3))
        return (
            u[1] * v[2] * w[0]
            - u[2] * v[1] * w[0]
            + u[2] * v[0] * w[1]
            - u[0] * v[2] * w[1]
            + u[0] * v[1] * w[2]
            - u[1] * v[0] * w[2]
        )

    def volume_of_mesh(self):
        vol = 0.0
        Fkeys = self.h_bound_F.keys()
        for f in Fkeys:
            vol += self.signed_volume_f(f)
        return abs(vol)

    def area_of_mesh(self):
        return np.sum(self.area_F())

    ######################################################
    # Misc helper functions
    def valence_v(self, v):
        """get the valence of vertex v"""
        valence = 0
        for h in self.generate_H_out_v_clockwise(v):
            valence += 1
        return valence

    @property
    def num_vertices(self):
        return len(self._xyz_coord_V)

    @property
    def num_edges(self):
        return len(self._v_origin_H) // 2

    @property
    def num_faces(self):
        return len(self._h_bound_F)

    @property
    def num_boundaries(self):
        return len(self._h_comp_B)

    @property
    def genus(self):
        return 1 - (self.euler_characteristic + self.num_boundaries) // 2

    @property
    def euler_characteristic(self):
        return self.num_vertices - self.num_edges + self.num_faces

    ######################################################
    ######################################################
    ######################################################
    def _cotan_laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex
        """
        Nv = self.num_vertices
        lapQ = np.zeros_like(Q)
        for vi in range(Nv):
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

    def cotan_laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex in parallel
        """
        return pcotan_laplacian(self, Q)

    def laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex in parallel
        """
        return pcotan_laplacian(self, Q)

    ######################################################
    # forces
    def preferred_geometric_defaults(self):
        Nf = self.num_faces
        Nv = self.num_vertices
        volume = self.volume_of_mesh()
        area = self.area_of_mesh()

        Rv = (3 * volume / (4 * np.pi)) ** (1 / 3)
        Ra = np.sqrt(area / (4 * np.pi))
        w = 0.75
        R = (1 - w) * Ra + w * Rv
        preferred_total_area = 4 * np.pi * R**2
        preferred_face_area = preferred_total_area / Nf
        preferred_cell_area = preferred_face_area * Nf / Nv
        preferred_total_volume = 4 * np.pi * R**3 / 3
        preferred_edge_length = 4 * R * np.sqrt(np.pi / (Nf * np.sqrt(3)))

        return (
            preferred_edge_length,
            preferred_cell_area,
            preferred_face_area,
            preferred_total_volume,
            preferred_total_area,
        )

    def angle_defect_v(self, v):
        """
        2*pi - sum_f (angle_f)
        """
        r0 = self.xyz_coord_v(v)
        defect = 2 * np.pi
        for h in self.generate_H_out_v_clockwise(v):
            h_rot = self.h_next_h(self.h_twin_h(h))
            r1 = self.xyz_coord_v(self.v_head_h(h))
            r2 = self.xyz_coord_v(self.v_head_h(h_rot))
            e1 = r1 - r0
            e2 = r2 - r0
            norm_e1 = np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            norm_e2 = np.sqrt(e2[0] ** 2 + e2[1] ** 2 + e2[2] ** 2)
            cos_angle = (e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2]) / (
                norm_e1 * norm_e2
            )
            defect -= np.arccos(cos_angle)

        return defect

    def gaussian_curvature_v(self, v):
        """
        Compute the Gaussian curvature at vertex v
        """
        area_v = self.barcell_area(v)
        angle_defect_v = self.angle_defect_v(v)
        return angle_defect_v / area_v

    def gaussian_curvature_V(self):
        """
        Compute the Gaussian curvature at all vertices
        """
        return pgaussian_curvature_V(self)

    def mean_curvature_unit_normal_V(self):
        """
        Compute the mean curvature vector at all vertices
        """
        Vkeys = sorted(self.xyz_coord_V.keys())
        X = self.xyz_array
        lapX = self.laplacian(X)
        H = np.zeros_like(X[:, 0])
        n = np.zeros_like(X)
        for _i in range(len(Vkeys)):
            i = Vkeys[_i]
            mcvec = lapX[_i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[_i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
            H[_i] = np.dot(n[_i], mcvec) / 2

        return H, n

    def compute_curvature_data(self):
        """
        Compute the Gaussian curvature at all vertices
        """

        return pcompute_curvature_data(self)

    ######################################################
    # to be deprecated


class BraneBuilder:
    """
    h_bound_F-->h_adjacent_F
    Dict-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.arrays of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as dicts of vertex/half-edge/face labels. Each data dict has a name of the form "a_description_B", where "a" denotes the type of object associated with the dict elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "B" denotes the type of object associated with the dict indices ("V" for vertex, "H" for half-edge, or "F" for face), and "description" is a description of information represented by the dict. For example, "_v_origin_H" is a dict of vertices at the origin of each half-edge. The i-th element of data dict "a_description_B" can be accessed using the "a_description_b(i)" method.

    Properties
    ----------
    xyz_coord_V : dict of numpy.array
        _xyz_coord_V[i] = xyz coordinates of vertex i
    h_out_V : dict of int
        _h_out_V[i] = some outgoing half-edge incident on vertex i
    v_origin_H : dict of int
        _v_origin_H[j] = vertex at the origin of half-edge j
    h_next_H : dict of int
        _h_next_H[j] next half-edge after half-edge j in the face cycle
    h_twin_H : dict of int
        _h_twin_H[j] = half-edge antiparalel to half-edge j
    f_left_H : dict of int
        _f_left_H[j] = face to the left of half-edge j
    h_bound_F : dict of int
        _h_bound_F[k] = some half-edge on the ccw boudary of face k
    h_comp_B : dict of int
        _h_comp_B[n] = half-edge in complement boundary n of the mesh

    Initialization
    ---------------
    The HalfEdgeMesh class can be initialized in several ways:
    - Directly from half-edge mesh data dicts:
        HalfEdgeMesh(xyz_coord_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_bound_F)
    - From a dict of vertex positions and a dict of face vertices:
        HalfEdgeMesh.from_vert_face_dict(xyz_coord_V, vvv_of_F)
    - From a ply file (binary/ascii) containing vertex/face data:
        HalfEdgeMesh.from_vertex_face_ply(ply_path)
        * See HalfEdgeMeshBuilder for more details about ply format.
    - From a ply file (binary/ascii) containing half-edge mesh data:
        HalfEdgeMesh.from_half_edge_ply(ply_path)
        * See HalfEdgeMeshBuilder for more details about ply format.
    - From C-Glass binary data:
        HalfEdgeMesh.from_cglass_binary(data)
        * To be implemented...
    """

    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
        h_comp_B=None,
        bending_modulus=1e-1,
        splay_modulus=1e0,
        length_reg_stiffness=1e0,
        area_reg_stiffness=1e-3,
        volume_reg_stiffness=1e1,
        linear_drag_coeff=1e3,
        spontaneous_curvature=0,
        mesh_class=Brane,
    ):
        self.mesh_class = mesh_class
        # Nv = len(xyz_coord_V)
        self.xyz_coord_V_array = np.array(xyz_coord_V, dtype=_NUMPY_FLOAT_)
        # self.xyz_coord_V_array = np.zeros((Nv, 3), dtype=_NUMPY_FLOAT_)
        self.h_out_V_array = np.array(h_out_V, dtype=_NUMPY_INT_)
        self.v_origin_H_array = np.array(v_origin_H, dtype=_NUMPY_INT_)
        self.h_next_H_array = np.array(h_next_H, dtype=_NUMPY_INT_)
        self.h_twin_H_array = np.array(h_twin_H, dtype=_NUMPY_INT_)
        self.f_left_H_array = np.array(f_left_H, dtype=_NUMPY_INT_)
        self.h_bound_F_array = np.array(h_bound_F, dtype=_NUMPY_INT_)
        if h_comp_B is None:
            self.h_comp_B_array = self.find_h_comp_B_array()
        else:
            self.h_comp_B_array = np.array(h_comp_B, dtype=_NUMPY_INT_)
        (
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
            self.h_comp_B,
        ) = half_edge_arrays_to_dicts(
            self.xyz_coord_V_array,
            self.h_out_V_array,
            self.v_origin_H_array,
            self.h_next_H_array,
            self.h_twin_H_array,
            self.f_left_H_array,
            self.h_bound_F_array,
            self.h_comp_B_array,
        )
        self.bending_modulus = bending_modulus
        self.splay_modulus = splay_modulus
        self.length_reg_stiffness = length_reg_stiffness
        self.area_reg_stiffness = area_reg_stiffness
        self.volume_reg_stiffness = volume_reg_stiffness
        self.linear_drag_coeff = linear_drag_coeff
        self.spontaneous_curvature = spontaneous_curvature

    @staticmethod
    def default_brane_kwargs():
        brane_kwargs = {
            "length_reg_stiffness": 1e-9,
            "area_reg_stiffness": 1e-3,
            "volume_reg_stiffness": 1e1,
            "bending_modulus": 1e-1,
            "splay_modulus": 1.0,
            "spontaneous_curvature": 0.0,
            "linear_drag_coeff": 1e3,
        }
        return brane_kwargs

    #######################################################
    # Initilization methods

    @classmethod
    def from_half_edge_ply(
        cls,
        ply_path,
        default_brane_kwargs=True,
        mesh_class=Brane,
        bending_modulus=1e-1,
        splay_modulus=1e0,
        length_reg_stiffness=1e0,
        area_reg_stiffness=1e-3,
        volume_reg_stiffness=1e1,
        linear_drag_coeff=1e3,
        spontaneous_curvature=0,
    ):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        if default_brane_kwargs:
            brane_kwargs = cls.default_brane_kwargs()
        else:
            brane_kwargs = {
                "bending_modulus": bending_modulus,
                "splay_modulus": splay_modulus,
                "length_reg_stiffness": length_reg_stiffness,
                "area_reg_stiffness": area_reg_stiffness,
                "volume_reg_stiffness": volume_reg_stiffness,
                "linear_drag_coeff": linear_drag_coeff,
                "spontaneous_curvature": spontaneous_curvature,
            }
        c = cls(
            *VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples,
            mesh_class=mesh_class,
            **brane_kwargs,
        )
        return c.build()

    @classmethod
    def from_data_arrays(
        cls,
        path,
        default_brane_kwargs=True,
        bending_modulus=1e-1,
        splay_modulus=1e0,
        length_reg_stiffness=1e0,
        area_reg_stiffness=1e-3,
        volume_reg_stiffness=1e1,
        linear_drag_coeff=1e3,
        spontaneous_curvature=0,
    ):
        """Initialize a half-edge mesh from npz file containing data arrays."""
        if default_brane_kwargs:
            brane_kwargs = cls.default_brane_kwargs()
        else:
            brane_kwargs = {
                "bending_modulus": bending_modulus,
                "splay_modulus": splay_modulus,
                "length_reg_stiffness": length_reg_stiffness,
                "area_reg_stiffness": area_reg_stiffness,
                "volume_reg_stiffness": volume_reg_stiffness,
                "linear_drag_coeff": linear_drag_coeff,
                "spontaneous_curvature": spontaneous_curvature,
            }
        data = np.load(path)
        c = cls(
            data["xyz_coord_V"],
            data["h_out_V"],
            data["v_origin_H"],
            data["h_next_H"],
            data["h_twin_H"],
            data["f_left_H"],
            data["h_bound_F"],
            data["h_comp_B"],
            **brane_kwargs,
        )
        return c.build()

    @classmethod
    def load_test_spheres(cls, npz_paths=None):
        """Load test spheres from npz files."""
        if npz_paths is None:
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
            ]
            npz_paths = [
                f"./output/half_edge_arrays/unit_sphere_{N:07d}.npz"
                for N in _NUM_VERTS_
            ]
        M = [
            cls.from_data_arrays(path, default_brane_kwargs=True) for path in npz_paths
        ]
        return M

    @classmethod
    def load_test_sphere(cls, n_v=5):
        """Load a test sphere from npz file."""
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
        ]
        N = _NUM_VERTS_[n_v]
        npz_path = f"./output/half_edge_arrays/unit_sphere_{N:07d}.npz"

        return cls.from_data_arrays(npz_path, default_brane_kwargs=True)

    @classmethod
    def load_oblate_sphere(cls, n_v=5):
        """Load a test sphere from npz file."""
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
        ]
        N = _NUM_VERTS_[n_v]
        npz_path = f"./output/half_edge_arrays/unit_sphere_{N:07d}.npz"

        b = cls.from_data_arrays(npz_path, default_brane_kwargs=True)
        for i in range(b.num_vertices):
            b._xyz_coord_V[i][-1] *= 0.8
        return b

    #######################################################
    @property
    def numba_dicts(self):
        return (
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
            self.h_comp_B,
        )

    @property
    def numpy_arrays(self):
        return (
            self.xyz_coord_V_array,
            self.h_out_V_array,
            self.v_origin_H_array,
            self.h_next_H_array,
            self.h_twin_H_array,
            self.f_left_H_array,
            self.h_bound_F_array,
            self.h_comp_B_array,
        )

    def find_h_comp_B_array(self):
        return find_h_comp_B_array(
            self.xyz_coord_V_array,
            self.h_out_V_array,
            self.v_origin_H_array,
            self.h_next_H_array,
            self.h_twin_H_array,
            self.f_left_H_array,
            self.h_bound_F_array,
        )

    def brane_kwargs(self):
        return {
            "bending_modulus": self.bending_modulus,
            "splay_modulus": self.splay_modulus,
            "length_reg_stiffness": self.length_reg_stiffness,
            "area_reg_stiffness": self.area_reg_stiffness,
            "volume_reg_stiffness": self.volume_reg_stiffness,
            "linear_drag_coeff": self.linear_drag_coeff,
            "spontaneous_curvature": self.spontaneous_curvature,
        }

    def build(self, brane_kwargs=None):
        if brane_kwargs is None:
            brane_kwargs = self.brane_kwargs()
        return self.mesh_class(*self.numba_dicts, find_h_comp_B=True, **brane_kwargs)
