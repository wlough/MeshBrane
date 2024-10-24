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
def pbelkin_laplacian(self, Q, s):
    """
    Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
    defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
    constant timelike parameter s.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (
                    (Af[f] / 3)
                    * (Q[j] - Q[i])
                    * np.exp(
                        -(
                            (V[j, 0] - V[i, 0]) ** 2
                            + (V[j, 1] - V[i, 1]) ** 2
                            + (V[j, 2] - V[i, 2]) ** 2
                        )
                        / (4 * s)
                    )
                    / (4 * np.pi * s**2)
                )
    return lapQ


@jit(parallel=True)
def pguckenberger_laplacian(self, Q):
    """
    Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
    Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
    This is a modification of Belkin et al's which replaces the constant time-like
    parameter s with the area the dual cell at each vertex.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    Av = self.barcell_area_V()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (
                    (Af[f] / 3)
                    * (Q[j] - Q[i])
                    * np.exp(
                        -(
                            (V[j, 0] - V[i, 0]) ** 2
                            + (V[j, 1] - V[i, 1]) ** 2
                            + (V[j, 2] - V[i, 2]) ** 2
                        )
                        / (4 * Av[i])
                    )
                    / (4 * np.pi * Av[i] ** 2)
                )
    return lapQ


@jit
def heat_kernel(x, y, s):
    # return np.exp(
    #     -((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2) / (4 * s)
    # ) / (4 * np.pi * s)
    return np.exp(-np.sum((y - x) ** 2, axis=-1) / (4 * s)) / (4 * np.pi * s)


@jit
def lap_kernel1(x, y, ds):
    return heat_kernel(x, y, ds) / ds


@jit
def lap_kernel2(x, y, ds):
    c1, c2 = 2, -1 / 2
    s1, s2 = ds, 2 * ds
    return c1 * heat_kernel(x, y, s1) / ds + c2 * heat_kernel(x, y, s2) / ds


@jit
def lap_kernel3(x, y, ds):
    c1, c2, c3 = 3, -3 / 2, 1 / 3
    s1, s2, s3 = ds, 2 * ds, 3 * ds
    return (
        c1 * heat_kernel(x, y, s1) / ds
        + c2 * heat_kernel(x, y, s2) / ds
        + c3 * heat_kernel(x, y, s3) / ds
    )


@jit(parallel=True)
def belkin_laplacian1(self, Q, s):
    """
    Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
    defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
    constant timelike parameter s.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (Af[f] / 3) * (Q[j] - Q[i]) * lap_kernel1(V[i], V[j], s)
    return lapQ


@jit(parallel=True)
def belkin_laplacian2(self, Q, s):
    """
    Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
    defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
    constant timelike parameter s.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (Af[f] / 3) * (Q[j] - Q[i]) * lap_kernel2(V[i], V[j], s)
    return lapQ


@jit(parallel=True)
def belkin_laplacian3(self, Q, s):
    """
    Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
    defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
    constant timelike parameter s.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (Af[f] / 3) * (Q[j] - Q[i]) * lap_kernel3(V[i], V[j], s)
    return lapQ


@jit(parallel=True)
def guckenberger_laplacian1(self, Q):
    """
    Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
    Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
    This is a modification of Belkin et al's which replaces the constant time-like
    parameter s with the area the dual cell at each vertex.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    Av = self.barcell_area_V()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (Af[f] / 3) * (Q[j] - Q[i]) * lap_kernel1(V[i], V[j], Av[i])
    return lapQ


@jit(parallel=True)
def guckenberger_laplacian2(self, Q):
    """
    Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
    Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
    This is a modification of Belkin et al's which replaces the constant time-like
    parameter s with the area the dual cell at each vertex.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    Av = self.barcell_area_V()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (Af[f] / 3) * (Q[j] - Q[i]) * lap_kernel2(V[i], V[j], Av[i])
    return lapQ


@jit(parallel=True)
def guckenberger_laplacian3(self, Q):
    """
    Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
    Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
    This is a modification of Belkin et al's which replaces the constant time-like
    parameter s with the area the dual cell at each vertex.
    """
    Fkeys = self.h_bound_F.keys()
    Af = self.area_F()
    Av = self.barcell_area_V()
    F = self.V_of_F
    V = self.xyz_array
    Nv = len(V)
    lapQ = np.zeros_like(Q)
    for i in prange(Nv):
        for f in Fkeys:
            for j in F[f]:
                lapQ[i] += (Af[f] / 3) * (Q[j] - Q[i]) * lap_kernel3(V[i], V[j], Av[i])
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


half_edge_mesh_spec = [
    ("_name", unicode_type),
    ("_xyz_coord_V", xyz_coord_V_numba_type),
    ("_h_out_V", h_out_V_numba_type),
    ("_v_origin_H", v_origin_H_numba_type),
    ("_h_next_H", h_next_H_numba_type),
    ("_h_twin_H", h_twin_H_numba_type),
    ("_f_left_H", f_left_H_numba_type),
    ("_h_bound_F", h_bound_F_numba_type),
    ("_h_comp_B", h_comp_B_numba_type),
]


@jitclass(half_edge_mesh_spec)
class HalfEdgeMeshBase:
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
    ):

        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.h_bound_F = h_bound_F
        self.f_left_H = f_left_H
        # self.h_comp_B = h_comp_B
        # self.h_comp_B = self.find_h_comp_B()
        if find_h_comp_B:
            self.h_comp_B = self.find_h_comp_B()
        else:
            self.h_comp_B = h_comp_B

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

    # def meyercell_area(self, v):
    #     """Meyer's mixed area of cell dual to vertex v"""
    #     Atot = 0.0
    #     ri = self.xyz_coord_v(v)
    #     ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
    #     # h_start = self.V_hedge[v]
    #     # hij = h_start
    #     # while True:
    #     for hij in self.generate_H_out_v_clockwise(v):
    #         if self.complement_boundary_contains_h(hij):
    #             continue
    #         hjjp1 = self.h_next_h(hij)
    #         hjp1i = self.h_next_h(hjjp1)
    #         vj = self.v_origin_h(hjjp1)
    #         rj = self.xyz_coord_v(vj)
    #         vjp1 = self.v_origin_h(hjp1i)
    #         rjp1 = self.xyz_coord_v(vjp1)

    #         rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
    #         rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
    #         ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
    #         rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
    #         rjp1_ri = rjp1[0] * ri[0] + rjp1[1] * ri[1] + rjp1[2] * ri[2]

    #         normDrij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
    #         # normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
    #         normDrjjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
    #         # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
    #         normDrjp1i = np.sqrt(rjp1_rjp1 - 2 * rjp1_ri + ri_ri)
    #         cos_thetajijp1 = (ri_ri + rj_rjp1 - ri_rj - rjp1_ri) / (
    #             normDrij * normDrjp1i
    #         )
    #         cos_thetajp1ji = (rj_rj + rjp1_ri - rj_rjp1 - ri_rj) / (
    #             normDrij * normDrjjp1
    #         )
    #         cos_thetaijp1j = (rjp1_rjp1 + ri_rj - rj_rjp1 - rjp1_ri) / (
    #             normDrjp1i * normDrjjp1
    #         )
    #         if cos_thetajijp1 < 0:
    #             semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
    #             Atot += (
    #                 np.sqrt(
    #                     semiP
    #                     * (semiP - normDrij)
    #                     * (semiP - normDrjjp1)
    #                     * (semiP - normDrjp1i)
    #                 )
    #                 / 2
    #             )
    #             # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 4
    #         elif cos_thetajp1ji < 0 or cos_thetaijp1j < 0:
    #             semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
    #             Atot += (
    #                 np.sqrt(
    #                     semiP
    #                     * (semiP - normDrij)
    #                     * (semiP - normDrjjp1)
    #                     * (semiP - normDrjp1i)
    #                 )
    #                 / 4
    #             )
    #             # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 8
    #         else:
    #             cot_thetaijp1j = cos_thetaijp1j / np.sqrt(1 - cos_thetaijp1j**2)
    #             cot_thetajp1ji = cos_thetajp1ji / np.sqrt(1 - cos_thetajp1ji**2)
    #             Atot += (
    #                 normDrij**2 * cot_thetaijp1j / 8
    #                 + normDrjp1i**2 * cot_thetajp1ji / 8
    #             )

    #     return Atot

    # def meyercell_area_V(self):
    #     keys = sorted(self.xyz_coord_V.keys())
    #     N = len(keys)
    #     A = np.zeros(N, dtype=_NUMPY_FLOAT_)
    #     for _, k in enumerate(keys):
    #         A[_] = self.meyercell_area(k)
    #     return A

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
    def patch_from_seed_vertex(self, v_seed):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If v_seed is not in a boundary of supermesh, the patch will be a disk centered at v_seed.

        Parameters:
            v_seed (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = self.closure(*self.star_of_vertex(v_seed))

        return V, H, F

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

    def belkin_laplacian(self, Q, s):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        # Fkeys = self.h_bound_F.keys()
        # Vkeys = sorted(self.xyz_coord_V.keys())
        # lapQ = np.zeros_like(Q)
        # for i in Vkeys:
        #     x = self.xyz_coord_v(i)
        #     qx = Q[i]
        #     for f in Fkeys:
        #         af = self.area_f(f)
        #         for j in self.generate_V_of_f(f):
        #             qy = Q[j]
        #             y = self.xyz_coord_v(j)
        #             lapQ[i] += (
        #                 (af / 3)
        #                 * (qy - qx)
        #                 * np.exp(-np.linalg.norm(y - x) ** 2 / (4 * s))
        #                 / (4 * np.pi * s**2)
        #             )
        # return lapQ
        Fkeys = self.h_bound_F.keys()
        Af = self.area_F()
        F = self.V_of_F
        V = self.xyz_array
        Nv = len(V)
        lapQ = np.zeros_like(Q)
        for i in range(Nv):
            for f in Fkeys:
                for j in F[f]:
                    lapQ[i] += (
                        (Af[f] / 3)
                        * (Q[j] - Q[i])
                        * np.exp(
                            -(
                                (V[j, 0] - V[i, 0]) ** 2
                                + (V[j, 1] - V[i, 1]) ** 2
                                + (V[j, 2] - V[i, 2]) ** 2
                            )
                            / (4 * s)
                        )
                        / (4 * np.pi * s**2)
                    )
        return lapQ

    def pbelkin_laplacian(self, Q, s):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        return pbelkin_laplacian(self, Q, s)

    def guckenberger_laplacian(self, Q):
        """
        Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
        Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
        This is a modification of Belkin et al's which replaces the constant time-like
        parameter s with the area the dual cell at each vertex.
        """
        Fkeys = self.h_bound_F.keys()
        Af = self.area_F()
        Av = self.barcell_area_V()
        F = self.V_of_F
        V = self.xyz_array
        Nv = len(V)
        lapQ = np.zeros_like(Q)
        for i in range(Nv):
            for f in Fkeys:
                for j in F[f]:
                    lapQ[i] += (
                        (Af[f] / 3)
                        * (Q[j] - Q[i])
                        * np.exp(
                            -(
                                (V[j, 0] - V[i, 0]) ** 2
                                + (V[j, 1] - V[i, 1]) ** 2
                                + (V[j, 2] - V[i, 2]) ** 2
                            )
                            / (4 * Av[i])
                        )
                        / (4 * np.pi * Av[i] ** 2)
                    )
        return lapQ

    def pguckenberger_laplacian(self, Q):
        """
        Computes the heat kernel Laplacian of Q at each vertex using 'Method D' from
        Guckenberger et al 2016 'On the bending algorithms for soft objects in flows'.
        This is a modification of Belkin et al's which replaces the constant time-like
        parameter s with the area the dual cell at each vertex.
        """
        return pguckenberger_laplacian(self, Q)

    def order_p_belkin_laplacian(self, Q, s, p):
        if p == 1:
            return belkin_laplacian1(self, Q, s)
        if p == 2:
            return belkin_laplacian2(self, Q, s)
        if p == 3:
            return belkin_laplacian3(self, Q, s)

    def order_p_guckenberger_laplacian(self, Q, p):
        if p == 1:
            return guckenberger_laplacian1(self, Q)
        if p == 2:
            return guckenberger_laplacian2(self, Q)
        if p == 3:
            return guckenberger_laplacian3(self, Q)

    ######################################################
    # to be deprecated


class HalfEdgeMeshBuilder:
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
        mesh_class=HalfEdgeMeshBase,
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
        # if h_comp_B is None:
        #     self.h_comp_B = self.find_h_comp_B()
        # else:
        #     self.h_comp_B = h_comp_B

    #######################################################
    # Initilization methods
    @classmethod
    def from_vert_face_list(cls, xyz_coord_V, vvv_of_F):
        """
        Initialize a half-edge mesh from vertex/face data.

        Parameters:
        ----------
        xyz_coord_V : list of numpy.array
            xyz_coord_V[i] = xyz coordinates of vertex i
        vvv_of_F : list of lists of integers
            vvv_of_F[j] = [v0, v1, v2] = vertices in face j.

        Returns:
        -------
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with the given vertices and faces.
        """
        HEarr = jit_vf_samples_to_he_samples(xyz_coord_V, vvv_of_F)
        HEdict = half_edge_arrays_to_dicts(*HEarr)
        c = cls(*HEarr)
        return c.mesh_class(*HEdict)

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        c = cls(
            *VertTri2HalfEdgeConverter.jit_from_source_samples(ply_path).target_samples
        )
        return c.build()

    @classmethod
    def from_half_edge_ply(cls, ply_path, mesh_class=HalfEdgeMeshBase):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        c = cls(
            *VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples,
            mesh_class=mesh_class,
        )
        return c.build()

    @classmethod
    def builder_from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        c = cls(*VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples)
        return c

    @classmethod
    def from_data_arrays(cls, path):
        """Initialize a half-edge mesh from npz file containing data arrays."""
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
        M = [cls.from_data_arrays(path) for path in npz_paths]
        return M

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

    def build(self):
        # HEdicts = self.numba_dicts
        # return HalfEdgeMeshBase(*self.numba_dicts)
        # return HalfEdgeMeshBase(*HEdicts)
        return self.mesh_class(*self.numba_dicts, find_h_comp_B=True)


half_edge_patch_spec = [
    ("_name", unicode_type),
    ("_xyz_coord_V", xyz_coord_V_numba_type),
    ("_h_out_V", h_out_V_numba_type),
    ("_v_origin_H", v_origin_H_numba_type),
    ("_h_next_H", h_next_H_numba_type),
    ("_h_twin_H", h_twin_H_numba_type),
    ("_f_left_H", f_left_H_numba_type),
    ("_h_bound_F", h_bound_F_numba_type),
    ("_h_comp_B", h_comp_B_numba_type),
]


# @jitclass(half_edge_mesh_spec)
class HalfEdgePatch:
    """
    A submanifold of a HalfEdgeMesh topologically equivalent to a disk.
    """

    def __init__(
        self,
        supermesh,
        V,
        H,
        F,
        h_comp_B=None,
        V_bdry=None,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        if h_comp_B is None:
            self.h_comp_B = self.find_h_comp_B()
        if V_bdry is None:
            self.V_bdry = set(self.generate_V_cw_B())

    @property
    def V(self):
        return self._V

    @V.setter
    def V(self, value):
        if isinstance(value, set):
            self._V = value
        elif hasattr(value, "__iter__"):
            self._V = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        if isinstance(value, set):
            self._H = value
        elif hasattr(value, "__iter__"):
            self._H = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        if isinstance(value, set):
            self._F = value
        elif hasattr(value, "__iter__"):
            self._F = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    @property
    def V_bdry(self):
        return self._V_bdry

    @V_bdry.setter
    def V_bdry(self, value):
        if isinstance(value, set):
            self._V_bdry = value
        elif hasattr(value, "__iter__"):
            self._V_bdry = set(value)
        else:
            raise ValueError("Argument must be set or iterable.")

    ##############################################
    # @classmethod
    # def from_seed_vertex(cls, v_seed, supermesh):
    #     """
    #     Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If v_seed is not in a boundary of supermesh, the patch will be a disk centered at v_seed.

    #     Parameters:
    #         v_seed (int): vertex index
    #         supermesh (HalfEdgeMesh): mesh from which the patch is extracted
    #     """
    #     V, H, F = supermesh.closure(*supermesh.star_of_vertex(v_seed))
    #     self = cls(supermesh, V, H, F)
    #     # self.h_comp_B = self.find_h_comp_B()

    #     return self

    def complement_boundary_contains_h(self, h):
        """check if half-edge h is in the boundary of the mesh"""
        return h in self.H and self.supermesh.f_left_h(h) not in self.F

    def interior_boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        return (
            h in self.H
            and self.supermesh.f_left_h(self.supermesh.h_twin_h(h)) not in self.F
        )

    def boundary_contains_h(self, h):
        """check if half-edge h is on the boundary of the mesh"""
        if h in self.H:
            if self.supermesh.f_left_h(h) not in self.F:
                return True
            if self.supermesh.f_left_h(self.supermesh.h_twin_h(h)) not in self.F:
                return True
        return False

    ##############################################
    def xyz_coord_v(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self.supermesh.xyz_coord_v(v)

    def h_out_v(self, v):
        """
        get index of an non-boundary outgoing half-edge incident on vertex v

        Args:
            v (int): vertex index

        Returns:
            int: half-edge index
        """
        if v not in self.V:
            raise ValueError("Vertex not in patch.")
        for h in self.supermesh.generate_H_out_v_clockwise(v):
            if h not in self.H:
                continue
            elif self.supermesh.f_left_h(h) in self.F:
                return h

    def v_origin_h(self, h):
        """get index of the vertex at the origin of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: vertex index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        return self.supermesh.v_origin_h(h)

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face/boundary cycle
        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        elif self.supermesh.f_left_h(h) in self.F:
            return self.supermesh.h_next_h(h)
        n = self.supermesh.h_next_h(h)
        while n not in self.H:
            n = self.supermesh.h_out_cw_from_h(n)
        return n

    def h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        return self.supermesh.h_twin_h(h)

    def f_left_h(self, h):
        """get index of the face to the left of half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: face index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        elif self.supermesh.f_left_h(h) in self.F:
            return self.supermesh.f_left_h(h)
        else:
            return -1

    def h_bound_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        if f not in self.F:
            raise ValueError("Face not in patch.")
        return self.supermesh.h_bound_f(f)

    def generate_H_next_h(self, h_start):
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_H_cw_B(self):
        for bdry, h in self.h_comp_B.items():
            for h in self.generate_H_next_h(h):
                yield h

    def generate_V_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.v_origin_h(h)

    def generate_F_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.f_left_h(self.h_twin_h(h))

    def find_h_comp_B(self, F_need2check=None):
        h_comp_B = dict()
        bdry_count = 0
        H_in_cw_boundary = set()
        # boundary_is_right_of_H = set()
        if F_need2check is None:
            F_need2check = self.F.copy()  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            h = self.h_bound_f(f)
            for h in self.generate_H_next_h(h):
                if self.interior_boundary_contains_h(h):
                    H_in_cw_boundary.add(self.h_twin_h(h))
        while H_in_cw_boundary:
            bdry_count += 1
            h = H_in_cw_boundary.pop()
            bdry = -bdry_count
            h_comp_B[bdry] = h
            for h in self.generate_H_next_h(h):
                H_in_cw_boundary.discard(h)
        return h_comp_B

    ##############################################
    # ****************************************** #
    ##############################################
    def expand_boundary(self):
        """
        **slow but actually works***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices

        V
        H
        F
        generate_V_cw_B()
        find_h_comp_B()
            h_bound_f()
            generate_H_next_h()
            interior_boundary_contains_h()
            h_twin_h()
            generate_H_next_h
        supermesh.closure()
        supermesh.star()
        """
        new_boundary_verts = set()
        V_bdry_old = set(self.generate_V_cw_B())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V.update(V)
        self.H.update(H)
        self.F.update(F)
        self.h_comp_B = self.find_h_comp_B(F_need2check=F)
        return V_bdry_new

    ##############################################
    # ****************************************** #
    ##############################################

    ##############################################
    def to_half_edge_mesh(self):
        V = sorted(self.V)
        F = sorted(self.F)
        xyz_coord_V = [self.xyz_coord_v(v) for v in V]
        F = [
            [
                V.index(self.v_origin_h(h))
                for h in self.generate_H_next_h(self.h_bound_f(f))
            ]
            for f in F
        ]
        return HalfEdgeMesh.from_vert_face_list(xyz_coord_V, F)

    @property
    def data_lists(self):
        """ """
        V = sorted(self.V)
        H = sorted(self.H)
        F = sorted(self.F)
        # [x if x<.5 else 33 for x in X]
        xyz_coord_V = [self.xyz_coord_v(v) for v in V]
        h_out_V = [H.index(self.h_out_v(v)) for v in V]
        v_origin_H = [V.index(self.v_origin_h(h)) for h in H]
        h_next_H = [H.index(self.h_next_h(h)) for h in H]
        h_twin_H = [H.index(self.h_twin_h(h)) for h in H]
        f_left_H = [
            self.f_left_h(h) if self.f_left_h(h) < 0 else F.index(self.f_left_h(h))
            for h in H
        ]
        h_bound_F = [H.index(self.h_bound_f(f)) for f in F]

        return (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

    ##############################################
    ##############################################
    # to be deprecated
