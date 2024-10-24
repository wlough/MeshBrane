from temp_python.src_python.ply_tools import VertTri2HalfEdgeConverter
import numpy as np
import concurrent.futures
import warnings

_NUMPY_INT_ = np.int64
_NUMPY_FLOAT_ = np.float64


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
        h_comp_B=None,
    ):
        self.xyz_coord_V = xyz_coord_V
        self.h_out_V = h_out_V
        self.v_origin_H = v_origin_H
        self.h_next_H = h_next_H
        self.h_twin_H = h_twin_H
        self.h_bound_F = h_bound_F
        self.f_left_H = f_left_H
        if h_comp_B is None:
            self.h_comp_B = self.find_h_comp_B()
        else:
            self.h_comp_B = h_comp_B

    #######################################################
    # Initilization methods
    @classmethod
    def from_data_arrays(cls, path):
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
            data["h_comp_B"],
        )

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
        return cls(
            *VertTri2HalfEdgeConverter.from_source_samples(
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
        return cls(*VertTri2HalfEdgeConverter.from_source_ply(ply_path).target_samples)

    @classmethod
    def from_half_edge_ply(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(*VertTri2HalfEdgeConverter.from_target_ply(ply_path).target_samples)

    @classmethod
    def from_half_edge_ply_no_bdry_twin(cls, ply_path):
        """Initialize a half-edge mesh from a ply file containing half-edge mesh data using the h_twin = -1 convention for boundary edges.

        Args:
            ply_path (str): path to ply file

        Returns:
            HalfEdgeMesh: An instance of the HalfEdgeMesh class, initialized with data from the ply file.
        """
        return cls(*VertTri2HalfEdgeConverter.from_target_ply(ply_path).source_samples)

    #######################################################
    # Fundamental accessors
    @property
    def xyz_coord_V(self):
        return self._xyz_coord_V

    @xyz_coord_V.setter
    def xyz_coord_V(self, value):
        if isinstance(value, dict):
            self._xyz_coord_V = value
        elif hasattr(value, "__iter__"):
            self._xyz_coord_V = dict(enumerate(value))
        else:
            raise TypeError("xyz_coord_V must be a dictionary or an iterable.")

    @property
    def h_out_V(self):
        return self._h_out_V

    @h_out_V.setter
    def h_out_V(self, value):
        if isinstance(value, dict):
            self._h_out_V = value
        elif hasattr(value, "__iter__"):
            self._h_out_V = dict(enumerate(value))
        else:
            raise TypeError("h_out_V must be a dictionary or an iterable.")

    @property
    def v_origin_H(self):
        return self._v_origin_H

    @v_origin_H.setter
    def v_origin_H(self, value):
        if isinstance(value, dict):
            self._v_origin_H = value
        elif hasattr(value, "__iter__"):
            self._v_origin_H = dict(enumerate(value))
        else:
            raise TypeError("v_origin_H must be a dictionary or an iterable.")

    @property
    def h_next_H(self):
        return self._h_next_H

    @h_next_H.setter
    def h_next_H(self, value):
        if isinstance(value, dict):
            self._h_next_H = value
        elif hasattr(value, "__iter__"):
            self._h_next_H = dict(enumerate(value))
        else:
            raise TypeError("h_next_H must be a dictionary or an iterable.")

    @property
    def h_twin_H(self):
        return self._h_twin_H

    @h_twin_H.setter
    def h_twin_H(self, value):
        if isinstance(value, dict):
            self._h_twin_H = value
        elif hasattr(value, "__iter__"):
            self._h_twin_H = dict(enumerate(value))
        else:
            raise TypeError("h_twin_H must be a dictionary or an iterable.")

    @property
    def f_left_H(self):
        return self._f_left_H

    @f_left_H.setter
    def f_left_H(self, value):
        if isinstance(value, dict):
            self._f_left_H = value
        elif hasattr(value, "__iter__"):
            self._f_left_H = dict(enumerate(value))
        else:
            raise TypeError("f_left_H must be a dictionary or an iterable.")

    @property
    def h_bound_F(self):
        return self._h_bound_F

    @h_bound_F.setter
    def h_bound_F(self, value):
        if isinstance(value, dict):
            self._h_bound_F = value
        elif hasattr(value, "__iter__"):
            self._h_bound_F = dict(enumerate(value))
        else:
            raise TypeError("h_bound_F must be a dictionary or an iterable.")

    @property
    def h_comp_B(self):
        return self._h_comp_B

    @h_comp_B.setter
    def h_comp_B(self, value):
        if isinstance(value, dict):
            self._h_comp_B = value
        elif hasattr(value, "__iter__"):
            self._h_comp_B = {-(_key + 1): val for _key, val in enumerate(value)}

        else:
            raise TypeError("h_comp_B must be a dictionary or an iterable.")

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
    def Vkeys(self):
        return sorted(self._xyz_coord_V.keys())

    @property
    def Hkeys(self):
        return sorted(self._v_origin_H.keys())

    @property
    def Fkeys(self):
        return sorted(self._h_bound_F.keys())

    @property
    def xyz_array(self):
        return np.array([self.xyz_coord_v(v) for v in sorted(self.xyz_coord_V.keys())])

    @property
    def V_of_F(self):
        # return [list(self.generate_V_of_f(f)) for f in self.h_bound_F.keys()]

        return np.array(
            [list(self.generate_V_of_f(f)) for f in sorted(self.h_bound_F.keys())]
        )

    @property
    def V_of_H(self):
        return np.array(
            [
                [self.v_origin_h(h), self.v_head_h(h)]
                for h in sorted(self.v_origin_H.keys())
            ]
        )

    @property
    def data_arrays(self):
        """
        Get lists of vertex positions and connectivity data and required to reconstruct mesh or write to ply file. Vertex/half-edge/face indices are sorted in ascending order and relabeled so that the first index is 0, the second index is 1, etc...

        new_Xkeys_old[key_old] = old_Xkeys_new.index(key_old)
        """
        old_Vkeys_new = sorted(self._xyz_coord_V.keys())
        old_Hkeys_new = sorted(self._v_origin_H.keys())
        old_Fkeys_new = sorted(self._h_bound_F.keys())

        new_Vkeys_old = {val: key for key, val in enumerate(old_Vkeys_new)}
        new_Hkeys_old = {val: key for key, val in enumerate(old_Hkeys_new)}
        new_Fkeys_old = {val: key for key, val in enumerate(old_Fkeys_new)}
        ##
        # oldBkeys = [-b1, -b2, -b3, ...]
        # newBkeys = [-1, -2, -3, ...]
        oldBkeys = sorted(self.h_comp_B.keys(), reverse=True)
        new_Bkeys_old = {val: -(_key + 1) for _key, val in enumerate(oldBkeys)}
        # add boundary keys as negative face keys
        new_Fkeys_old.update(new_Bkeys_old)
        ##
        xyz_coord_V = np.array(
            [self.xyz_coord_v(v) for v in old_Vkeys_new], dtype=_NUMPY_FLOAT_
        )
        h_out_V = np.array(
            [new_Hkeys_old[self.h_out_v(v)] for v in old_Vkeys_new], dtype=_NUMPY_INT_
        )
        v_origin_H = np.array(
            [new_Vkeys_old[self.v_origin_h(h)] for h in old_Hkeys_new],
            dtype=_NUMPY_INT_,
        )
        h_next_H = np.array(
            [new_Hkeys_old[self.h_next_h(h)] for h in old_Hkeys_new], dtype=_NUMPY_INT_
        )
        h_twin_H = np.array(
            [new_Hkeys_old[self.h_twin_h(h)] for h in old_Hkeys_new], dtype=_NUMPY_INT_
        )
        ###
        # oldBkeys = sorted(self.h_comp_B.keys(), reverse=True)
        # new_Bkeys = [-(_key + 1) for _key, val in enumerate(old_Bkeys)]
        # old_Bkeys_new = {n: o for o, n in zip(old_Bkeys, new_Bkeys)}
        # new_Bkeys_old = {o: n for o, n in zip(old_Bkeys, new_Bkeys)}
        # new_Bkeys_old = {val: -(_key + 1) for _key, val in enumerate(old_Bkeys_new)}
        # new_Bkeys_old = {val: -(_key + 1) for _key, val in enumerate(oldBkeys)}
        h_comp_B = np.array(
            [new_Hkeys_old[self.h_comp_b(b)] for b in oldBkeys], dtype=_NUMPY_INT_
        )
        ###
        # f_left_H = np.array([self.f_left_h(h) for h in old_Hkeys_new])
        # f_left_H = np.array(
        #     [new_Bkeys_old[f] if f < 0 else new_Fkeys_old[f] for f in f_left_H]
        # )
        f_left_H = np.array(
            [new_Fkeys_old[self.f_left_h(h)] for h in old_Hkeys_new], dtype=_NUMPY_INT_
        )
        h_bound_F = np.array(
            [new_Hkeys_old[self.h_bound_f(f)] for f in old_Fkeys_new], dtype=_NUMPY_INT_
        )

        return (
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
            h_comp_B,
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
            h_comp_B,
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
            h_comp_B=h_comp_B,
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
    # to be deprecated
    # @property
    # def V(self, sorted=False):
    #     warnings.warn(
    #         "V is deprecated and will be replaced by Vkeys in a future version.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     if sorted:
    #         return sorted(self._xyz_coord_V.keys())
    #     else:
    #         return self._xyz_coord_V.keys()

    # @property
    # def H(self, sorted=False):
    #     warnings.warn(
    #         "H is deprecated and will be replaced by Hkeys in a future version.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     if sorted:
    #         return sorted(self._v_origin_H.keys())
    #     else:
    #         return self._v_origin_H.keys()

    # @property
    # def F(self, sorted=False):
    #     warnings.warn(
    #         "F is deprecated and will be replaced by Fkeys in a future version.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     if sorted:
    #         return sorted(self._h_bound_F.keys())
    #     else:
    #         return self._h_bound_F.keys()

    # @property
    # def data_lists(self):
    #     """
    #     Get lists of vertex positions and connectivity data and required to reconstruct mesh or write to ply file. Vertex/half-edge/face indices are sorted in ascending order and relabeled so that the first index is 0, the second index is 1, etc...
    #     """
    #     warnings.warn(
    #         "data_lists is deprecated and will be replaced by data_arrays in a future version.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     V = sorted(self._xyz_coord_V.keys())
    #     H = sorted(self._v_origin_H.keys())
    #     F = sorted(self._h_bound_F.keys())

    #     xyz_coord_V = [self.xyz_coord_v(v) for v in V]
    #     h_out_V = [H.index(self.h_out_v(v)) for v in V]
    #     v_origin_H = [V.index(self.v_origin_h(h)) for h in H]
    #     h_next_H = [H.index(self.h_next_h(h)) for h in H]
    #     h_twin_H = [H.index(self.h_twin_h(h)) for h in H]
    #     f_left_H = [self.f_left_h(h) for h in H]
    #     f_left_H = [f if f < 0 else F.index(f) for f in f_left_H]
    #     # f_left_H = [
    #     #     self.f_left_h(h) if self.f_left_h(h) < 0 else F.index(self.f_left_h(h))
    #     #     for h in H
    #     # ]
    #     h_bound_F = [H.index(self.h_bound_f(f)) for f in F]

    #     return (
    #         xyz_coord_V,
    #         h_out_V,
    #         v_origin_H,
    #         h_next_H,
    #         h_twin_H,
    #         f_left_H,
    #         h_bound_F,
    #     )


######################################
######################################
class HalfEdgeMesh(HalfEdgeMeshBase):
    def __init__(
        self,
        xyz_coord_V,
        h_out_V,
        v_origin_H,
        h_next_H,
        h_twin_H,
        f_left_H,
        h_bound_F,
    ):
        super().__init__(
            xyz_coord_V,
            h_out_V,
            v_origin_H,
            h_next_H,
            h_twin_H,
            f_left_H,
            h_bound_F,
        )

    ######################################################
    # mesh mutation

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
        for h in self.Hkeys:
            if not self.h_is_locally_delaunay(h):
                if self.h_is_flippable(h):
                    self.flip_edge(h)
                    flip_count += 1
        return flip_count

    @property
    def v_new(self):
        return max(self._xyz_coord_V.keys()) + 1

    @property
    def h_new(self):
        return max(self._v_origin_H.keys()) + 1

    @property
    def f_new(self):
        return max(self._h_bound_F.keys()) + 1

    def add_vertex(self, xyz=None, h_out=None):
        v_new = self.v_new
        self._xyz_coord_V[v_new] = xyz
        self._h_out_V[v_new] = h_out
        return v_new

    def update_vertex(self, v, xyz=None, h_out=None):
        if xyz is not None:
            self._xyz_coord_V[v] = xyz
        if h_out is not None:
            self._h_out_V[v] = h_out

    def add_face(self, h_bound=None):
        f_new = self.f_new
        self._h_bound_F[f_new] = h_bound
        return f_new

    def update_face(self, f, h_bound=None):
        if h_bound is not None:
            self._h_bound_F[f] = h_bound

    def add_hedge(self, h_next=None, h_twin=None, v_origin=None, f_left=None):
        h_new = self.h_new
        self._v_origin_H[h_new] = v_origin
        self._h_next_H[h_new] = h_next
        self._h_twin_H[h_new] = h_twin
        self._f_left_H[h_new] = f_left
        return h_new

    def update_hedge(self, h, h_next=None, h_twin=None, v_origin=None, f_left=None):
        if h_next is not None:
            self._h_next_H[h] = h_next
        if h_twin is not None:
            self._h_twin_H[h] = h_twin
        if v_origin is not None:
            self._v_origin_H[h] = v_origin
        if f_left is not None:
            self._f_left_H[h] = f_left

    def split_edge(self, h):
        # choose hij = h or twin(h) so that its face is contained in the mesh
        if not self.complement_boundary_contains_h(h):
            hij = h
        else:
            hij = self.h_twin_h(h)
        hji = self.h_twin_h(hij)
        vi = self.v_origin_h(hij)
        vj = self.v_origin_h(hji)
        fijk = self.f_left_h(hij)
        hjk = self.h_next_h(hij)
        hki = self.h_next_h(hjk)
        vk = self.v_origin_h(hki)

        ##############################
        # add new stuff
        vm = self.add_vertex()
        him = hij  # re-use hij label
        hmj = self.add_hedge()
        hmk = self.add_hedge()
        hkm = self.add_hedge()
        fimk = fijk  # re-use fijk label
        fmjk = self.add_face()
        # maybe bdry
        hmi = self.add_hedge()
        hjm = hji  # re-use hji label

        self.update_vertex(
            vm, xyz=0.5 * (self.xyz_coord_v(vi) + self.xyz_coord_v(vj)), h_out=hmj
        )

        self.update_hedge(him, h_next=hmk, h_twin=hmi, v_origin=vi, f_left=fimk)
        self.update_hedge(hmj, h_next=hjk, h_twin=hjm, v_origin=vm, f_left=fmjk)
        self.update_hedge(hmk, h_next=hki, h_twin=hkm, v_origin=vm, f_left=fimk)
        self.update_hedge(hkm, h_next=hmj, h_twin=hmk, v_origin=vk, f_left=fmjk)

        self.update_face(fimk, h_bound=him)
        self.update_face(fmjk, h_bound=hmj)

        ##############################
        if not self.complement_boundary_contains_h(hji):
            filj = self.f_left_h(hji)
            hil = self.h_next_h(hji)
            hlj = self.h_next_h(hil)
            vl = self.v_origin_h(hlj)
            # new stuff
            hlm = self.add_hedge()
            hml = self.add_hedge()
            film = self.add_face()
            fmlj = filj  # re-use filj label
            ############################

            self.update_hedge(hmi, h_next=hil, h_twin=him, v_origin=vm, f_left=film)
            self.update_hedge(hjm, h_next=hml, h_twin=hmj, v_origin=vj, f_left=fmlj)
            self.update_hedge(hlm, h_next=hmi, h_twin=hlm, v_origin=vl, f_left=film)
            self.update_hedge(hml, h_next=hlj, h_twin=hlm, v_origin=vm, f_left=fmlj)

            self.update_face(film, h_bound=hmi)
            self.update_face(fmlj, h_bound=hjm)

        else:
            f_bdry = self.f_left_h(hji)
            hmi_next = self.h_next_h(hji)
            hjm_next = hmi
            hjm_prev = self.h_prev_h(hji)
            self.update_hedge(
                hmi, h_next=hmi_next, h_twin=him, v_origin=vm, f_left=f_bdry
            )
            self.update_hedge(
                hjm, h_next=hjm_next, h_twin=hmj, v_origin=vj, f_left=f_bdry
            )

            self.update_hedge(hjm_prev, h_next=hjm)

    ######################################################
    # misc testing functions
    def average_face_area(self):
        # A = 0.0
        # for f in self.h_bound_F.keys():
        #     A += self.area_f(f)
        # return A / self.num_faces
        return np.mean([self.area_f(f) for f in self.h_bound_F.keys()])

    def average_dual_barcell_area(self):
        # A = 0.0
        # for v in self.xyz_coord_V.keys():
        #     A += self.barcell_area(v)
        # return A / self.num_vertices
        return np.mean([self.barcell_area(v) for v in self.xyz_coord_V.keys()])

    def Avorcell_array(self):
        def compute_for_vertex(v):
            return v, self.vorcell_area(v)

        vertex_indices = range(len(self.vertices))
        vorcell_areas = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(compute_for_vertex, v): v for v in vertex_indices
            }
            for future in concurrent.futures.as_completed(futures):
                v, area = future.result()
                vorcell_areas[v] = area

        return vorcell_areas

    def Abarcell_array(self):
        return np.array([self.barcell_area(v) for v in self.xyz_coord_V.keys()])

    def Aface_array(m):
        eps = np.zeros((3, 3, 3))
        for i, j, k in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
            eps[i, j, k] = 1
        for i, j, k in [[1, 0, 2], [2, 1, 0], [0, 2, 1]]:
            eps[i, j, k] = -1
        Fkeys = m.Fkeys
        V = m.xyz_array
        vvvF = np.array([list(m.generate_V_of_f(_)) for _ in Fkeys])
        VF = V[vvvF]
        vecAf = np.einsum("ijk,lmn,fjm,fkn->fl", eps, eps, VF, VF) / 4
        return np.linalg.norm(vecAf, axis=-1)
