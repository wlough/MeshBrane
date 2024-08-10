from src.python.ply_tools import VertTri2HalfEdgeConverter
from src.python.jit_brane_utils import (
    _NUMPY_FLOAT_,
    _NUMPY_INT_,
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
)
from numba import jit, prange
import numpy as np

# import warnings

# from numba import jit, prange
from numba.experimental import jitclass

# from numba import from_dtype, typeof
from numba.typed import (
    # List,
    Dict,
    # Tuple,
)
from numba.types import (
    unicode_type,
    # boolean,
    # int32,
    # int64,
    # float64,
    # ListType,
    # DictType,
    # Array,
)


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
    # V = self.xyz_array
    Vkeys = sorted(self.xyz_coord_V)
    Nv = len(Vkeys)
    # A = np.array([self.barcell_area(v) for v in Vkeys])
    lapQ = np.zeros_like(Q)
    for _i in prange(Nv):
        i = Vkeys[_i]
        qx = Q[_i]
        x = self.xyz_coord_v(i)
        ax = self.barcell_area(i)
        for _j in range(Nv):
            j = Vkeys[_j]
            qy = Q[_j]
            y = self.xyz_coord_v(j)
            lapQ[_i] += (
                ax
                * (qy - qx)
                * np.exp(-np.linalg.norm(y - x) ** 2 / (4 * s))
                / (4 * np.pi * s)
            )
    return lapQ


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
        # h_comp_B=None,
    ):

        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.h_bound_F = h_bound_F
        self.f_left_H = f_left_H
        self.h_comp_B = self.find_h_comp_B()
        # if h_comp_B is None:
        #     self.h_comp_B = self.find_h_comp_B()
        # else:
        #     self.h_comp_B = h_comp_B

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
        self.f_right_h(h)
        self.h_out_cw_from_h(h)
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
        # self.Vkeys
        # self.Hkeys
        # self.Fkeys
        self.xyz_array
        # self.V_of_F
        # self.V_of_H
        # self.data_lists
        # self.data_dicts
        # Simplical operations
        V, H, F = self.star_of_vertex(v)
        self.star_of_vertex(v)
        # self.star_of_edge(h)
        self.star(V, H, F)
        self.closure(V, H, F)
        self.link(V, H, F)
        self.find_h_comp_B()
        # geometry
        self.xyz_com_f(f)
        self.vec_area_f(f)
        self.area_f(f)
        self.total_area_of_faces()
        self.barcell_area(v)
        self.total_area_of_dual_barcells()
        self.vorcell_area(v)
        self.total_area_of_dual_vorcells()
        self.meyercell_area(v)
        self.total_area_of_dual_meyercells()
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
    # @property
    # def Vkeys(self):
    #     return sorted(self._xyz_coord_V.keys())

    # @property
    # def Hkeys(self):
    #     return sorted(self._v_origin_H.keys())

    # @property
    # def Fkeys(self):
    #     return sorted(self._h_bound_F.keys())

    @property
    def xyz_array(self):
        Vkeys = sorted(self.xyz_coord_V.keys())
        Nv = len(Vkeys)
        xyz_array = np.zeros((Nv, 3), _NUMPY_FLOAT_)
        for i, v in enumerate(Vkeys):
            xyz_array[i] = self.xyz_coord_v(v)
        return xyz_array

    # @property
    # def V_of_F(self):
    #     # return [list(self.generate_V_of_f(f)) for f in self.h_bound_F.keys()]
    #     # Vkeys = sorted(self.xyz_coord_V.keys())
    #     Fkeys = sorted(self.h_bound_F.keys())
    #     Nf = len(Fkeys)
    #     V_of_F = np.zeros((Nf, 3), dtype=_NUMPY_INT_)
    #     for i, f in enumerate(Fkeys):
    #         V_of_F[i] = list(self.generate_V_of_f(f))
    #     return V_of_F

    # @property
    # def V_of_H(self):
    #     Hkeys = sorted(self.v_origin_H.keys())
    #     Nh = len(Hkeys)
    #     V_of_H = np.zeros((Nh, 3), dtype=_NUMPY_INT_)
    #     for i, h in enumerate(Hkeys):
    #         V_of_H[i] = [self.v_origin_h(h), self.v_head_h(h)]
    #     return V_of_F

    @property
    def data_arrays(self):
        """
        Get lists of vertex positions and connectivity data and required to reconstruct mesh or write to ply file. Vertex/half-edge/face indices are sorted in ascending order and relabeled so that the first index is 0, the second index is 1, etc...
        """

        return half_edge_dicts_to_arrays(
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
        )

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

    # def star_of_edge(self, h):
    #     """Star of an edge is the set of all simplices that contain the edge."""
    #     V = set()
    #     H = {h, self.h_twin_h(h)}
    #     F = set()
    #     for hi in H:
    #         if not self.complement_boundary_contains_h(hi):
    #             F.add(self.f_left_h(hi))

    #     return V, H, F

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

    def xyz_com_f(self, f):
        h0 = self.h_bound_f(f)
        h1 = self.h_next_h(h0)
        h2 = self.h_next_h(h1)
        r0 = self.xyz_coord_v(self.v_origin_h(h0))
        r1 = self.xyz_coord_v(self.v_origin_h(h1))
        r2 = self.xyz_coord_v(self.v_origin_h(h2))
        return (r0 + r1 + r2) / 3

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

    def total_area_of_faces(self):
        Atot = 0.0
        for f in self.h_bound_F.keys():
            Atot += self.area_f(f)

        return Atot

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

    def total_area_of_dual_barcells(self):
        Atot = 0.0
        for v in self.xyz_coord_V.keys():
            Atot += self.barcell_area(v)

        return Atot

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

    def total_area_of_dual_vorcells(self):
        Atot = 0.0
        for v in self.xyz_coord_V.keys():
            Atot += self.vorcell_area(v)

        return Atot

    def meyercell_area(self, v):
        """Meyer's mixed area of cell dual to vertex v"""
        Atot = 0.0
        ri = self.xyz_coord_v(v)
        ri_ri = ri[0] ** 2 + ri[1] ** 2 + ri[2] ** 2
        # h_start = self.V_hedge[v]
        # hij = h_start
        # while True:
        for hij in self.generate_H_out_v_clockwise(v):
            if self.complement_boundary_contains_h(hij):
                continue
            hjjp1 = self.h_next_h(hij)
            hjp1i = self.h_next_h(hjjp1)
            vj = self.v_origin_h(hjjp1)
            rj = self.xyz_coord_v(vj)
            vjp1 = self.v_origin_h(hjp1i)
            rjp1 = self.xyz_coord_v(vjp1)

            rj_rj = rj[0] ** 2 + rj[1] ** 2 + rj[2] ** 2
            rjp1_rjp1 = rjp1[0] ** 2 + rjp1[1] ** 2 + rjp1[2] ** 2
            ri_rj = ri[0] * rj[0] + ri[1] * rj[1] + ri[2] * rj[2]
            rj_rjp1 = rj[0] * rjp1[0] + rj[1] * rjp1[1] + rj[2] * rjp1[2]
            rjp1_ri = rjp1[0] * ri[0] + rjp1[1] * ri[1] + rjp1[2] * ri[2]

            normDrij = np.sqrt(ri_ri - 2 * ri_rj + rj_rj)
            # normu1 = np.sqrt(r1_r1 - 2 * r_r1 + r_r)  # jitnorm(u1)
            normDrjjp1 = np.sqrt(rj_rj - 2 * rj_rjp1 + rjp1_rjp1)
            # normu2 = np.sqrt(r2_r2 - 2 * r1_r2 + r1_r1)  # jitnorm(u2)
            normDrjp1i = np.sqrt(rjp1_rjp1 - 2 * rjp1_ri + ri_ri)
            cos_thetajijp1 = (ri_ri + rj_rjp1 - ri_rj - rjp1_ri) / (
                normDrij * normDrjp1i
            )
            cos_thetajp1ji = (rj_rj + rjp1_ri - rj_rjp1 - ri_rj) / (
                normDrij * normDrjjp1
            )
            cos_thetaijp1j = (rjp1_rjp1 + ri_rj - rj_rjp1 - rjp1_ri) / (
                normDrjp1i * normDrjjp1
            )
            if cos_thetajijp1 < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += (
                    np.sqrt(
                        semiP
                        * (semiP - normDrij)
                        * (semiP - normDrjjp1)
                        * (semiP - normDrjp1i)
                    )
                    / 2
                )
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 4
            elif cos_thetajp1ji < 0 or cos_thetaijp1j < 0:
                semiP = (normDrij + normDrjjp1 + normDrjp1i) / 2
                Atot += (
                    np.sqrt(
                        semiP
                        * (semiP - normDrij)
                        * (semiP - normDrjjp1)
                        * (semiP - normDrjp1i)
                    )
                    / 4
                )
                # Atot += normDrij * normDrjp1i * np.sqrt(1 - cos_thetajijp1**2) / 8
            else:
                cot_thetaijp1j = cos_thetaijp1j / np.sqrt(1 - cos_thetaijp1j**2)
                cot_thetajp1ji = cos_thetajp1ji / np.sqrt(1 - cos_thetajp1ji**2)
                Atot += (
                    normDrij**2 * cot_thetaijp1j / 8
                    + normDrjp1i**2 * cot_thetajp1ji / 8
                )

        return Atot

    def total_area_of_dual_meyercells(self):
        Atot = 0.0
        for v in self.xyz_coord_V.keys():
            Atot += self.meyercell_area(v)

        return Atot

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

    def patch_from_seed_vertex(self, v_seed):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If v_seed is not in a boundary of supermesh, the patch will be a disk centered at v_seed.

        Parameters:
            v_seed (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = self.closure(*self.star_of_vertex(v_seed))

        return V, H, F

    def cotan_laplacian(self, Q):
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

    def pcotan_laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex
        """
        return pcotan_laplacian(self, Q)

    def belkin_laplacian(self, Q, s):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        Fkeys = self.h_bound_F.keys()
        Vkeys = sorted(self.xyz_coord_V.keys())
        lapQ = np.zeros_like(Q)
        for i in Vkeys:
            x = self.xyz_coord_v(i)
            qx = Q[i]
            for f in Fkeys:
                af = self.area_f(f)
                for j in self.generate_V_of_f(f):
                    qy = Q[j]
                    y = self.xyz_coord_v(j)
                    lapQ[i] += (
                        (af / 3)
                        * (qy - qx)
                        * np.exp(-np.linalg.norm(y - x) ** 2 / (4 * s))
                        / (4 * np.pi * s)
                    )
        return lapQ

    def obelkin_laplacian(self, Q, s):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        # V = self.xyz_array
        Vkeys = sorted(self.xyz_coord_V)
        Nv = len(Vkeys)
        # A = np.array([self.barcell_area(v) for v in Vkeys])
        lapQ = np.zeros_like(Q)
        for _i in range(Nv):
            i = Vkeys[_i]
            qx = Q[_i]
            x = self.xyz_coord_v(i)
            ax = self.barcell_area(i)
            for _j in range(Nv):
                j = Vkeys[_j]
                qy = Q[_j]
                y = self.xyz_coord_v(j)
                lapQ[_i] += (
                    ax
                    * (qy - qx)
                    * np.exp(-np.linalg.norm(y - x) ** 2 / (4 * s))
                    / (4 * np.pi * s)
                )
        return lapQ

    def pbelkin_laplacian(self, Q, s):
        """
        Computes the heat kernel Laplacian of Q at each vertex using the 'mesh Laplacian'
        defined in Belkin et al 2008 'Discrete laplace operator on meshed surfaces' with
        constant timelike parameter s.
        """
        return pbelkin_laplacian(self, Q, s)

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
        # h_comp_B=None,
        # mesh_class=HalfEdgeMeshBase,
    ):
        # self.mesh_class = mesh_class
        # Nv = len(xyz_coord_V)
        self.xyz_coord_V_array = np.array(xyz_coord_V, dtype=_NUMPY_FLOAT_)
        # self.xyz_coord_V_array = np.zeros((Nv, 3), dtype=_NUMPY_FLOAT_)
        self.h_out_V_array = np.array(h_out_V, dtype=_NUMPY_INT_)
        self.v_origin_H_array = np.array(v_origin_H, dtype=_NUMPY_INT_)
        self.h_next_H_array = np.array(h_next_H, dtype=_NUMPY_INT_)
        self.h_twin_H_array = np.array(h_twin_H, dtype=_NUMPY_INT_)
        self.f_left_H_array = np.array(f_left_H, dtype=_NUMPY_INT_)
        self.h_bound_F_array = np.array(h_bound_F, dtype=_NUMPY_INT_)

        (
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
        ) = half_edge_arrays_to_dicts(
            self.xyz_coord_V_array,
            self.h_out_V_array,
            self.v_origin_H_array,
            self.h_next_H_array,
            self.h_twin_H_array,
            self.f_left_H_array,
            self.h_bound_F_array,
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
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        c = cls(*VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples)
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
        )

    def build(self):
        # return self.mesh_class(*self.numba_dicts)
        HEdicts = self.numba_dicts
        # return HalfEdgeMeshBase(*self.numba_dicts)
        return HalfEdgeMeshBase(*HEdicts)
