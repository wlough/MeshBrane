from src.python.half_edge_base_mesh import HalfEdgeMeshBase


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
            n = self.supermesh.h_rotcw_h(n)
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
