from src.python.half_edge_base_ply_tools import VertTri2HalfEdgeMeshConverter
from src.python.half_edge_base_utils import V_of_F
import numpy as np

_NUMPY_INT_ = np.int64
_NUMPY_FLOAT_ = np.float64


class HalfEdgeMeshBase:
    """
    Array-based half-edge mesh data structure
    ----------------------------------------
    HalfEdgeMesh uses two basic data types: numpy.ndarray of Cartesian coordinates for vertex position and integer-valued labels for vertices/half-edges/faces. Mesh connectivity data are stored as ndarrays of vertex/half-edge/face labels. Each data array has a name of the form "a_description_Q", where "a" denotes the type of object associated with the elements ("xyz" for position, "v" for vertex, "h" for half-edge, or "f" for face), "Q" denotes the type of object associated with the indices ("V" for vertex, "H" for half-edge, "F" for face, or "B" for boundary), and "description" is a description of information represented by the data. For example, "_v_origin_H" is an array of vertices at the origin of each half-edge. The i-th element of data array "a_description_Q" can be accessed using the "a_description_q(i)" method.

    Properties
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
        h_twin_H[j] = half-edge antiparallel to half-edge j
    f_left_H : ndarray[:] of _NUMPY_INT_
        f_left_H[j] = face to the left of half-edge j, if j in interior(M) or a positively oriented boundary of M
        f_left_H[j] = boundary to the left of half-edge j, if j in a negatively oriented boundary
    h_bound_F : ndarray[:] of _NUMPY_INT_
        h_bound_F[k] = some half-edge on the boudary of face k.
    h_right_B : ndarray[:] of _NUMPY_INT_
        h_right_B[n] = half-edge to the right of boundary n.

    Initialization
    ---------------
    The HalfEdgeMesh class can be initialized in several ways:
    - Directly from half-edge mesh data arrays:
        HalfEdgeMesh(xyz_coord_V,
                     h_out_V,
                     v_origin_H,
                     h_next_H,
                     h_twin_H,
                     f_left_H,
                     h_bound_F)
    - From an npz file containing data arrays:
        HalfEdgeMesh.from_he_data(npz_path)
    - From an array of vertex positions and an array of face vertices:
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
        h_right_B,
    ):
        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.h_bound_F = h_bound_F
        self.f_left_H = f_left_H
        self.h_right_B = h_right_B

    #######################################################
    # Initilization methods
    @classmethod
    def from_he_data(cls, path):
        """Initialize a half-edge mesh from npz file containing data arrays."""
        data = np.load(path)
        return cls(
            data["xyz_coord_V"],
            data["h_out_V"],
            data["v_origin_H"],
            data["h_next_H"],
            data["h_twin_H"],
            data["f_left_H"],
            data["h_bound_F"],
            data["h_right_B"],
        )

    @classmethod
    def from_vf_data(cls, xyz_coord_V, vvv_of_F):
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
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_source_samples(
                xyz_coord_V, vvv_of_F
            ).target_samples
        )

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_source_ply(ply_path).target_samples
        )

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_target_ply(ply_path).target_samples
        )

    #######################################################
    # Fundamental accessors
    @property
    def xyz_coord_V(self):
        return self._xyz_coord_V

    @xyz_coord_V.setter
    def xyz_coord_V(self, value):
        self._xyz_coord_V = np.array(value, dtype=_NUMPY_FLOAT_)

    @property
    def h_out_V(self):
        return self._h_out_V

    @h_out_V.setter
    def h_out_V(self, value):
        self._h_out_V = np.array(value, dtype=_NUMPY_INT_)

    @property
    def v_origin_H(self):
        return self._v_origin_H

    @v_origin_H.setter
    def v_origin_H(self, value):
        self._v_origin_H = np.array(value, dtype=_NUMPY_INT_)

    @property
    def h_next_H(self):
        return self._h_next_H

    @h_next_H.setter
    def h_next_H(self, value):
        self._h_next_H = np.array(value, dtype=_NUMPY_INT_)

    @property
    def h_twin_H(self):
        return self._h_twin_H

    @h_twin_H.setter
    def h_twin_H(self, value):
        self._h_twin_H = np.array(value, dtype=_NUMPY_INT_)

    @property
    def f_left_H(self):
        return self._f_left_H

    @f_left_H.setter
    def f_left_H(self, value):
        self._f_left_H = np.array(value, dtype=_NUMPY_INT_)

    @property
    def h_bound_F(self):
        return self._h_bound_F

    @h_bound_F.setter
    def h_bound_F(self, value):
        self._h_bound_F = np.array(value, dtype=_NUMPY_INT_)

    @property
    def h_right_B(self):
        return self._h_right_B

    @h_right_B.setter
    def h_right_B(self, value):
        self._h_right_B = np.array(value, dtype=_NUMPY_INT_)

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

    def h_right_b(self, b):
        """get index of a half-edge contained in boundary b

        Args:
            b (int): boundary index

        Returns:
            int: half-edge index
        """
        return self._h_right_B[b]

    # Derived combinatorial maps
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
    #######################################################
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
    # Generators and data exporters
    @property
    def V_of_F(self):
        return V_of_F(*self.data_arrays)

    @property
    def V_of_H(self):
        return np.array(
            [[self.v_origin_h(h), self.v_head_h(h)] for h in range(self.num_half_edges)]
        )

    @property
    def data_arrays(self):
        """
        Get lists of vertex positions and connectivity data and required to reconstruct mesh or write to ply file. Vertex/half-edge/face indices are sorted in ascending order and relabeled so that the first index is 0, the second index is 1, etc...
        """

        return (
            self.xyz_coord_V,
            self.h_out_V,
            self.v_origin_H,
            self.h_next_H,
            self.h_twin_H,
            self.f_left_H,
            self.h_bound_F,
            self.h_right_B,
        )

    def save_data_arrays(self, path):
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
            h_right_B,
        ) = self.data_arrays
        np.savez(
            path,
            xyz_coord_V=xyz_coord_V,
            h_out_V=h_out_V,
            v_origin_H=v_origin_H,
            h_next_H=h_next_H,
            h_twin_H=h_twin_H,
            f_left_H=f_left_H,
            h_bound_F=h_bound_F,
            h_right_B=h_right_B,
        )

    def generate_H_out_v_clockwise(self, v, h_start=None):
        """
        Generate outgoing half-edges from vertex v in clockwise order until the starting half-edge is reached again
        """
        if h_start is None:
            h_start = self.h_out_v(v)
        elif self.v_origin_h(h_start) != v:
            raise ValueError("Starting half-edge does not originate at vertex v")
        h = h_start
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

    def generate_V_nearest_v_clockwise(self, v, h_start=None):
        """Generates nearest neighbor vertices of vertex v in clockwise order"""
        for h in self.generate_H_out_v_clockwise(v, h_start=h_start):
            yield self.v_head_h(h)

    @property
    def num_vertices(self):
        return len(self._xyz_coord_V)

    @property
    def num_edges(self):
        return len(self._v_origin_H) // 2

    @property
    def num_half_edges(self):
        return len(self._v_origin_H)

    @property
    def num_faces(self):
        return len(self._h_bound_F)

    @property
    def num_boundaries(self):
        return len(self._h_right_B)

    @property
    def genus(self):
        return 1 - (self.euler_characteristic + self.num_boundaries) // 2

    @property
    def euler_characteristic(self):
        return self.num_vertices - self.num_edges + self.num_faces

    def valence_v(self, v):
        """get the valence of vertex v"""
        valence = 0
        for h in self.generate_H_out_v_clockwise(v):
            valence += 1
        return valence

    ######################################################
    # Simplical operations
    #
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

    def star_of_edge(self, h):
        """Star of an edge is the set of all simplices that contain the edge."""
        V = set()
        H = {h, self.h_twin_h(h)}
        F = set()
        for hi in H:
            if not self.complement_boundary_contains_h(hi):
                F.add(self.f_left_h(hi))

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

    ######################################################
    # Geometry
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
        for f in range(self.num_faces):
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
        for v in range(self.num_vertices):
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
        for v in range(self.num_vertices):
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
        for v in range(self.num_vertices):
            Atot += self.meyercell_area(v)

        return Atot

    def barcell_area_V(self):

        N = self.num_vertices
        A = np.zeros(N, dtype=_NUMPY_FLOAT_)
        for k in range(N):
            A[k] = self.barcell_area(k)
        return A

    ######################################################
    # experimental
    ######################################################
    # unit normal methods
    def normal_some_face_of_v(self, i):
        h = self.h_out_v(i)
        f = self.f_left_h(h)
        if f < 0:
            h = self.h_out_cw_from_h(h)
            f = self.f_left_h(h)
        avec = self.vec_area_f(f)
        n = avec / np.linalg.norm(avec)
        return n

    def normal_some_face_of_V(self):
        n = np.zeros((self.num_vertices, 3), dtype=_NUMPY_FLOAT_)
        for i in range(self.num_vertices):
            n[i] = self.normal_some_face_of_v(i)
        return n

    def normal_other_weighted_v(self, i):
        """Weights for Computing Vertex Normals from Facet Normals Max99"""
        n = np.zeros(3)
        # x = self.xyz_coord_v(i)
        # neighbors = self.generate_V_nearest_v_clockwise(i)
        # j = next(neighbors)
        # r = self.xyz_coord_v(j) - x
        # for jrot in neighbors:
        defect = 2 * np.pi
        x = self.xyz_coord_v(i)
        h = self.h_out_v(i)
        r = self.xyz_coord_v(self.v_head_h(h)) - x
        h = self.h_out_cw_from_h(h)
        for jrot in self.generate_V_nearest_v_clockwise(i, h_start=h):
            rrot = self.xyz_coord_v(jrot) - x
            n += np.cross(rrot, r) / (np.dot(r, r) * np.dot(rrot, rrot))
            r = rrot
        n /= np.linalg.norm(n)
        return n

    def normal_other_weighted_V(self):
        n = np.zeros((self.num_vertices, 3), dtype=_NUMPY_FLOAT_)
        for i in range(self.num_vertices):
            n[i] = self.normal_other_weighted_v(i)
        return n

    # Curvature
    def angle_defect_v(self, i):
        """
        2*pi - sum_f (angle_f)
        """
        defect = 2 * np.pi
        x = self.xyz_coord_v(i)
        h = self.h_out_v(i)
        r = self.xyz_coord_v(self.v_head_h(h)) - x
        h = self.h_out_cw_from_h(h)
        for jrot in self.generate_V_nearest_v_clockwise(i, h_start=h):
            rrot = self.xyz_coord_v(jrot) - x
            cos_angle = np.dot(r, rrot) / (np.linalg.norm(r) * np.linalg.norm(rrot))
            defect -= np.arccos(cos_angle)
            r = rrot
        return defect

    def gaussian_curvature_v(self, i):
        """
        2*pi - sum_f (angle_f)
        """
        area = 0.0
        defect = 2 * np.pi
        x = self.xyz_coord_v(i)
        h = self.h_out_v(i)
        r = self.xyz_coord_v(self.v_head_h(h)) - x
        norm_r = np.linalg.norm(r)
        h = self.h_out_cw_from_h(h)
        # for jrot in self.generate_V_nearest_v_clockwise(i, h_start=h):
        for hrot in self.generate_H_out_v_clockwise(i, h_start=h):
            jrot = self.v_head_h(hrot)
            rrot = self.xyz_coord_v(jrot) - x
            norm_rrot = np.linalg.norm(rrot)
            r_dot_rrot = np.dot(r, rrot)
            cos_angle = r_dot_rrot / (norm_r * norm_rrot)
            if self.complement_boundary_contains_h(hrot):
                # do boundary geodesic curvature stuff
                continue
            defect -= np.arccos(cos_angle)
            area += norm_r * norm_rrot * np.sqrt(1 - cos_angle**2) / 6

            r = rrot
            norm_r = norm_rrot
        return defect / area

    def _angle_defect_v(self, v):
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

    def _gaussian_curvature_v(self, v):
        """
        Compute the Gaussian curvature at vertex v
        """
        area_v = self.barcell_area(v)
        angle_defect_v = self.angle_defect_v(v)
        return angle_defect_v / area_v

    def laplacian(self, Q):
        """
        Computes the cotan Laplacian of Q at each vertex
        """
        # Nv = self.num_vertices
        lapQ = np.zeros_like(Q)
        for vi in range(self.num_vertices):
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

    def compute_curvature_data(self):
        """
        Compute the mean curvature vector at all vertices
        """

        X = self.xyz_coord_V
        lapX = self.laplacian(X)
        H = np.zeros_like(X[:, 0])
        K = np.zeros_like(X[:, 0])
        n = np.zeros_like(X)
        for i in range(self.num_vertices):

            mcvec = lapX[i]
            f = self.f_left_h(self.h_out_v(i))
            af_vec = self.vec_area_f(f)
            mcvec_sign = np.sign(np.dot(mcvec, af_vec))
            n[i] = mcvec_sign * mcvec / np.linalg.norm(mcvec)
            H[i] = np.dot(n[i], mcvec) / 2
            K[i] = self.gaussian_curvature_v(i)

        lapH = self.laplacian(H)
        return H, K, lapH, n

    def update_vertex(self, v, xyz=None, h_out=None):
        if xyz is not None:
            self._xyz_coord_V[v] = xyz
        if h_out is not None:
            self._h_out_V[v] = h_out

    def update_hedge(self, h, h_next=None, h_twin=None, v_origin=None, f_left=None):
        if h_next is not None:
            self._h_next_H[h] = h_next
        if h_twin is not None:
            self._h_twin_H[h] = h_twin
        if v_origin is not None:
            self._v_origin_H[h] = v_origin
        if f_left is not None:
            self._f_left_H[h] = f_left

    def update_face(self, f, h_bound=None):
        if h_bound is not None:
            self._h_bound_F[f] = h_bound

    def flip_edge(self, h):
        r"""
        h cannot be on boundary!
                v1                           v1
              /    \                       /  |  \
             /      \                     /   |   \
            /h3    h2\                   /h3  |  h2\
           /    f0    \                 /     |     \
          /            \               /  f0  |  f1  \
         /      h0      \             /       |       \
        v2--------------v0  |----->  v2     h0|h1     v0
         \      h1      /             \       |       /
          \            /               \      |      /
           \    f1    /                 \     |     /
            \h4    h5/                   \h4  |  h5/
             \      /                     \   |   /
              \    /                       \  |  /
                v3                           v3
        v0
        --
        h_out
            pre-flip: may be h1
            post-flip: set to h2 if needed
        v2
        --
            pre-flip: may be h0
            post-flip: set to h4 if needed
        h0
        --
        v_origin_h(h0)
            pre-flip: v2
            post-flip: v3
        h_next
            pre-flip: h2
            post-flip: h3
        h_twin
            unchanged
        f_left
            unchanged
        h1
        --
        v_origin
            pre-flip: v0
            post-flip: v1
        h_next
            pre-flip: h4
            post-flip: h5
        h_twin
            unchanged
        f_left
            unchanged
        h2
        --
        v_origin
            unchanged
        h_next
            pre-flip: h3
            post-flip: h1
        h_twin
            unchanged
        f_left
            pre-flip: f0
            post-flip: f1
        h3
        --
        v_origin
            unchanged
        h_next
            pre-flip: h0
            post-flip: h4
        h_twin
            unchanged
        f_left
            unchanged
        h4
        --
        v_origin
            unchanged
        h_next
            pre-flip: h5
            post-flip: h0
        h_twin
            unchanged
        f_left
            pre-flip: f1
            post-flip: f0
        h5
        --
        v_origin
            unchanged
        h_next
            pre-flip: h1
            post-flip: h2
        h_twin
            unchanged
        f_left
            unchanged
        f0
        --
        h_bound
            pre-flip: may be h2
            post-flip: set to h3 if needed
        f1
        --
        h_bound
            pre-flip: may be h4
            post-flip: set to h5 if needed
        """
        # get involved half-edges/vertices/faces
        h0 = h
        h1 = self.h_twin_h(h0)
        h2 = self.h_next_h(h0)
        h3 = self.h_next_h(h2)
        h4 = self.h_next_h(h1)
        h5 = self.h_next_h(h4)

        v0 = self.v_origin_h(h1)
        v1 = self.v_origin_h(h3)
        v2 = self.v_origin_h(h0)
        v3 = self.v_origin_h(h5)

        f0 = self.f_left_h(h0)
        f1 = self.f_left_h(h1)

        # update vertices
        if self.h_out_v(v0) == h1:
            self.update_vertex(v0, h_out=h2)
        if self.h_out_v(v2) == h0:
            self.update_vertex(v2, h_out=h4)
        # update half-edges
        self.update_hedge(h0, v_origin=v3, h_next=h3)
        self.update_hedge(h1, v_origin=v1, h_next=h5)
        self.update_hedge(h2, h_next=h1, f_left=f1)
        self.update_hedge(h3, h_next=h4)
        self.update_hedge(h4, h_next=h0, f_left=f0)
        self.update_hedge(h5, h_next=h2)
        # update faces
        if self.h_bound_f(f0) == h2:
            self.update_face(f0, h_bound=h3)
        if self.h_bound_f(f1) == h4:
            self.update_face(f1, h_bound=h5)

    def flip_non_delaunay(self):
        flip_count = 0
        for h in range(self.num_half_edges):
            if not self.h_is_locally_delaunay(h):
                if self.h_is_flippable(h):
                    self.flip_edge(h)
                    flip_count += 1
        return flip_count

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
        self.V_of_F
        self.V_of_H
        self.data_arrays
        # Simplical operations
        V, H, F = self.star_of_vertex(v)
        self.star_of_vertex(v)
        # self.star_of_edge(h)
        self.star(V, H, F)
        self.closure(V, H, F)
        self.link(V, H, F)
        # geometry
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


class HalfEdgePatchBase:
    """
    A submanifold of a HalfEdgeMesh topologically equivalent to a disk.
    """

    def __init__(
        self,
        supermesh,
        V,
        H,
        F,
        h_right_B=None,
        V_bdry=None,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        if h_right_B is None:
            self.h_right_B = self.find_h_right_B()
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
    @classmethod
    def from_seed_vertex(cls, v_seed, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If v_seed is not in a boundary of supermesh, the patch will be a disk centered at v_seed.

        Parameters:
            v_seed (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = supermesh.closure(*supermesh.star_of_vertex(v_seed))
        self = cls(supermesh, V, H, F)
        # self.h_right_B = self.find_h_right_B()

        return self

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

    # def boundary_contains_v(self, v):
    #     """check if vertex v is on the boundary of the mesh"""
    #     return self.boundary_contains_h(self.supermesh.h_out_v(v))

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
        for bdry, h in self.h_right_B.items():
            for h in self.generate_H_next_h(h):
                yield h

    def generate_V_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.v_origin_h(h)

    def generate_F_cw_B(self):
        for h in self.generate_H_cw_B():
            yield self.f_left_h(self.h_twin_h(h))

    def find_h_right_B(self, F_need2check=None):
        h_right_B = dict()
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
            h_right_B[bdry] = h
            for h in self.generate_H_next_h(h):
                H_in_cw_boundary.discard(h)
        return h_right_B

    def expand_boundary(self):
        """
        **slow but actually works***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V_bdry_old = set(self.generate_V_cw_B())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V.update(V)
        self.H.update(H)
        self.F.update(F)
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_bdry_new

    def _expand_boundary(self):
        """
        ***this screws up something with boundaries, maybe in generate_H_cw_B/generate_F_cw_B?***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V, H, F = set(), set(), set()
        V_bdry_old = set(self.generate_V_cw_B())
        for h_start in self.generate_H_cw_B():
            if self.supermesh.complement_boundary_contains_h(h_start):
                continue
            for h in self.supermesh.generate_H_in_cw_from_h(h_start):
                f = self.supermesh.f_left_h(h)
                if f in self.F:
                    break
                n = self.supermesh.h_next_h(h)
                v = self.supermesh.v_origin_h(h)
                V.add(v)
                H.update([h, n])
                if f >= 0:
                    nn = self.supermesh.h_next_h(n)
                    H.update([nn, self.supermesh.h_twin_h(nn)])
                    F.add(f)

        # V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V.update(V)
        self.H.update(H)
        self.F.update(F)
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_bdry_new

    def expand_boundary_doesnt_work_yet(self):
        """
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V, H, F = set(), set(), set()
        V_bdry_old = set(self.generate_V_cw_B())
        for h_start in self.generate_H_cw_B():
            if self.supermesh.complement_boundary_contains_h(h_start):
                continue
            for h in self.supermesh.generate_H_in_cw_from_h(h_start):
                f = self.supermesh.f_left_h(h)
                if f in self.F:
                    break
                n = self.supermesh.h_next_h(h)
                v = self.supermesh.v_origin_h(h)
                V.add(v)
                H.update([h, n])
                if f >= 0:
                    nn = self.supermesh.h_next_h(n)
                    H.update([nn, self.supermesh.h_twin_h(nn)])
                    F.add(f)

        # V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V = V
        self.H = H
        self.F = F.copy()
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_bdry_new

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


######################################
######################################
class HalfEdgeMeshBuilder:
    def __init__(
        self,
        mesh_type=HalfEdgeMeshBase,
    ):
        self.mesh_type = mesh_type

    #######################################################
    # Initilization methods
    @classmethod
    def from_he_data(cls, path, mesh_type=HalfEdgeMeshBase):
        """Initialize an instance of mesh_type from npz file containing data arrays."""
        data = np.load(path)
        # cls(
        #     data["xyz_coord_V"],
        #     data["h_out_V"],
        #     data["v_origin_H"],
        #     data["h_next_H"],
        #     data["h_twin_H"],
        #     data["f_left_H"],
        #     data["h_bound_F"],
        #     data["h_right_B"],
        # )
        return mesh_type(**data)

    @classmethod
    def from_vf_data(cls, xyz_coord_V, vvv_of_F, mesh_type=HalfEdgeMeshBase):
        """
        Initialize a half-edge mesh from vertex/face data.

        Parameters:
        ----------
        xyz_coord_V : list of numpy.ndarray
            xyz_coord_V[i] = xyz coordinates of vertex i
        vvv_of_F : list of lists of integers
            vvv_of_F[j] = [v0, v1, v2] = vertices in face j.

        Returns:
        -------
            HalfEdgeMeshBase: An instance of the HalfEdgeMeshBase class with the given vertices and faces.
        """
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_source_samples(
                xyz_coord_V, vvv_of_F
            ).target_samples
        )

    @classmethod
    def from_vertex_face_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing vertex/face data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_source_ply(ply_path).target_samples
        )

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_target_ply(ply_path).target_samples
        )

    @classmethod
    def from_half_edge_ply_no_bdry_twin(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data using the h_twin = -1 convention for boundary edges.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(
            *VertTri2HalfEdgeMeshConverter.from_target_ply(ply_path).source_samples
        )
