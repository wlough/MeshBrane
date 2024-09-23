from src.python.half_edge_base_mesh import HalfEdgeMeshBase
import numpy as np


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
        h_right_B=None,
        find_h_right_B=True,
        seed_vertex=None,
    ):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        if h_right_B is None and find_h_right_B:
            self.h_right_B = self.find_h_right_B()
        else:
            self.h_right_B = h_right_B
        self.seed_vertex = seed_vertex
        self.V_frontier = set()
        self.H_frontier = set()
        self.F_frontier = set()

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
    def from_seed_vertex(cls, seed_vertex, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If seed_vertex is not in a boundary of supermesh, the patch will be a disk centered at seed_vertex.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = supermesh.closure(*supermesh.star_of_vertex(seed_vertex))
        self = cls(supermesh, V, H, F, seed_vertex=seed_vertex)

        return self

    @classmethod
    def from_seed_to_radius(cls, seed_vertex, supermesh, radius):
        self = cls.from_seed_vertex(seed_vertex, supermesh)
        self.expand_to_radius(supermesh.xyz_coord_v(seed_vertex), radius)
        return self

    @classmethod
    def from_seed_to_cylinder(cls, seed_vertex, supermesh, p0, r_max, ez):
        """
        x0 is point on the cylinder axis
        r_max is the radius
        ez is parallel with the axis
        """
        if not cls.is_p_in_cylinder(supermesh.xyz_coord_v(seed_vertex), p0, r_max, ez):
            raise ValueError("Seed vertex not in cylinder.")
        self = cls.from_seed_vertex(seed_vertex, supermesh)
        dNf, dNh, dNv = 1, 1, 1
        while dNf != 0:
            dNf, dNh, dNv = self.move_towards_cylinder(p0, r_max, ez)
        return self

    @classmethod
    def from_vertex_set(cls, V, supermesh):
        self = cls(
            supermesh,
            V,
            set(),
            set(),
            find_h_right_B=False,
        )
        self.update_from_V()
        return self

    @staticmethod
    def is_p_in_cylinder(p, p0, r_max, ez):
        """
        Test if point p is contained in a cylinder or radius r_max with axis through p0 and direction ez.
        """
        r = np.linalg.norm(np.cross(ez, p - p0))
        return r <= r_max

    ##############################################
    def negative_boundary_contains_h(self, h):
        """check if half-edge h is in the boundary of the mesh"""
        return h in self.H and self.supermesh.f_left_h(h) not in self.F

    def positive_boundary_contains_h(self, h):
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
            b = self.b_left_h(h)
            return -(b + 1)

    def b_left_h(self, h):
        """get index of the negative boundary containing half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: boundary index
        """
        # if h not in self.H:
        #     raise ValueError("Half-edge not in patch.")
        # elif self.supermesh.f_left_h(h) in self.F:
        #     raise ValueError("Half-edge not in negative boundary.")
        # else:
        for b, h_start in enumerate(self.h_right_B):
            if h in self.generate_H_next_h(h_start):
                return b
        raise ValueError("Half-edge not in negative boundary.")

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

    def _find_h_right_B(self, F_need2check=None):
        h_right_B = []
        bdry_count = 0
        H_in_positive_boundary = set()
        # boundary_is_right_of_H = set()
        if F_need2check is None:
            F_need2check = self.F.copy()  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()

            h = self.h_bound_f(f)
            for h in self.generate_H_next_h(h):
                if self.positive_boundary_contains_h(h):
                    H_in_positive_boundary.add(self.h_twin_h(h))
        while H_in_positive_boundary:
            bdry_count += 1
            h = H_in_positive_boundary.pop()
            bdry = -bdry_count
            # h_right_B[bdry] = h
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                H_in_positive_boundary.discard(h)
        return np.array(h_right_B, dtype="int32")

    def find_h_right_B(self, F_need2check=None):
        """works"""
        h_right_B = []
        bdry_count = 0
        H_in_positive_boundary = set()

        if F_need2check is None:
            F_need2check = self.F.copy()  # set of faces that need to be checked
        while F_need2check:
            f = F_need2check.pop()
            h = self.h_bound_f(f)
            for h in self.generate_H_next_h(h):
                if self.positive_boundary_contains_h(h):
                    H_in_positive_boundary.add(self.h_twin_h(h))
        while H_in_positive_boundary:
            bdry_count += 1
            h = H_in_positive_boundary.pop()
            bdry = -bdry_count
            # h_right_B[bdry] = h
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                H_in_positive_boundary.discard(h)
        return np.array(h_right_B, dtype="int32")

    def find_h_right_B_from_VHF(self, V_check=None, H_check=None, F_check=None):
        """
        Check supermesh.closure(V_check, H_check, F_check) for h_right_B
        """
        if V_check is None:
            V_check = set()
        if H_check is None:
            H_check = self.H.copy()
        if F_check is None:
            F_check = set()  # set of faces that need to be checked
        H_right_B = set()
        while V_check:
            v = V_check.pop()
            for hv in self.supermesh.generate_H_out_v_clockwise(v):
                for h in [h, self.supermesh.h_next_h(h)]:
                    ht = self.supermesh.h_twin_h(h)
                    f = self.supermesh.f_left_h(h)
                    ft = self.supermesh.f_left_h(ht)
                    f_in_F = f in self.F
                    ft_in_F = ft in self.F
                    if f_in_F and (not ft_in_F):
                        H_right_B.add(ht)
                    elif (not f_in_F) and ft_in_F:
                        H_right_B.add(h)
                    H_check.discard(h)
                    H_check.discard(ht)
                    F_check.discard(f)

        while F_check:
            f = F_check.pop()
            f_in_F = f in self.F
            for h in self.supermesh.generate_H_bound_f(f):
                ht = self.supermesh.h_twin_h(h)
                ft = self.supermesh.f_left_h(ht)
                ft_in_F = ft in self.F
                if f_in_F and (not ft_in_F):
                    H_right_B.add(ht)
                elif (not f_in_F) and ft_in_F:
                    H_right_B.add(h)
                H_check.discard(h)
                H_check.discard(ht)

        while H_check:
            h = H_check.pop()
            ht = self.supermesh.h_twin_h(h)
            f_in_F = f in self.F
            ft_in_F = ft in self.F
            if f_in_F and (not ft_in_F):
                H_right_B.add(ht)
            elif (not f_in_F) and ft_in_F:
                H_right_B.add(h)
            H_check.discard(ht)

        h_right_B = []
        while H_right_B:
            h = H_right_B.pop()
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                H_right_B.discard(h)
        return np.array(h_right_B, dtype="int32")

    def expand_by_one_ring(self):
        """
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new vertices
        """
        V_bdry = set(self.generate_V_negative_bdry())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry, set(), set()))
        V_new = V - self.V
        H_new = H - self.H
        F_new = F - self.F
        self.V.update(V_new)
        self.H.update(H_new)
        self.F.update(F_new)
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_new, H_new, F_new

    def get_frontier(self):
        V_bdry = set(self.generate_V_negative_bdry())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry, set(), set()))
        return V, H, F

    def expand_within_radius(self, xyz_center, radius):
        """ """
        V_bdry = set(self.generate_V_negative_bdry())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry, set(), set()))
        # V_new = V - self.V
        # H_new = H - self.H
        F_new = F - self.F
        F_keep = set()
        for f in F_new:
            h0 = self.supermesh.h_bound_f(f)
            h1 = self.supermesh.h_next_h(h0)
            h2 = self.supermesh.h_next_h(h1)
            i0 = self.supermesh.v_origin_h(h0)
            i1 = self.supermesh.v_origin_h(h1)
            i2 = self.supermesh.v_origin_h(h2)
            x0 = self.supermesh.xyz_coord_v(i0)
            x1 = self.supermesh.xyz_coord_v(i1)
            x2 = self.supermesh.xyz_coord_v(i2)
            x = (x0 + x1 + x2) / 3
            if np.linalg.norm(x - xyz_center) < radius:
                F_keep.add(f)
        V, H, F = self.supermesh.closure(set(), set(), F_keep)
        V_new = V - self.V
        H_new = H - self.H
        F_new = F - self.F
        self.V.update(V_new)
        self.H.update(H_new)
        self.F.update(F_new)
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_new, H_new, F_new

    def expand_to_radius(self, xyz_center, radius):
        while True:
            V_new, H_new, F_new = self.expand_within_radius(xyz_center, radius)
            if not V_new:
                break

    def move_towards_cylinder(self, p0, r_max, ez):
        """ """
        num_faces = len(self.F)
        num_half_edges = len(self.H)
        num_vertices = len(self.V)
        # V_bdry, H_bdry = self.get_VH_bdry()
        V_frontier, H_frontier, F_frontier = self.get_VHF_frontier()
        # F = self.F - F_frontier  # open boundary
        # H = self.H - H_frontier
        self.H.difference_update(H_frontier)
        self.V.difference_update(V_frontier)
        for f in F_frontier:
            h1 = self.supermesh.h_bound_f(f)
            h2 = self.supermesh.h_next_h(h1)
            h3 = self.supermesh.h_next_h(h2)
            i1 = self.supermesh.v_origin_h(h1)
            i2 = self.supermesh.v_origin_h(h2)
            i3 = self.supermesh.v_origin_h(h3)
            x1 = self.supermesh.xyz_coord_v(i1)
            x2 = self.supermesh.xyz_coord_v(i2)
            x3 = self.supermesh.xyz_coord_v(i3)
            x = (x1 + x2 + x3) / 3

            if self.is_p_in_cylinder(x, p0, r_max, ez):
                self.V.update([i1, i2, i3])
                self.H.update([h1, h2, h3])
                self.H.update(
                    [
                        self.supermesh.h_twin_h(h1),
                        self.supermesh.h_twin_h(h2),
                        self.supermesh.h_twin_h(h3),
                    ]
                )
                self.F.add(f)
        H_right_B = set()
        while H_frontier:
            h = H_frontier.pop()
            ht = self.supermesh.h_twin_h(h)
            H_frontier.discard(ht)
            f = self.supermesh.f_left_h(h)
            ft = self.supermesh.f_left_h(ht)
            f_in_F = f in self.F
            ft_in_F = ft in self.F
            if f_in_F and (not ft_in_F):
                H_right_B.add(ht)
            elif (not f_in_F) and ft_in_F:
                H_right_B.add(h)
            else:
                continue
            v = self.supermesh.v_origin_h(h)
            vt = self.supermesh.v_origin_h(ht)
            self.V.update([v, vt])
            self.H.update([h, ht])

        h_right_B = []
        while H_right_B:
            h = H_right_B.pop()
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                H_right_B.discard(h)
        self.h_right_B = np.array(h_right_B, dtype="int32")

        delta_num_faces = len(self.F) - num_faces
        delta_num_half_edges = len(self.H) - num_half_edges
        delta_num_vertices = len(self.V) - num_vertices
        return delta_num_faces, delta_num_half_edges, delta_num_vertices

    ##############################################
    def he_samples(self):

        V = sorted(self.V)
        H = sorted(self.H)
        F = sorted(self.F)
        xyz_coord_V = np.array([self.xyz_coord_v(v) for v in V])
        h_out_V = np.array([H.index(self.h_out_v(v)) for v in V], dtype="int32")
        v_origin_H = np.array([V.index(self.v_origin_h(h)) for h in H], dtype="int32")
        h_next_H = np.array([H.index(self.h_next_h(h)) for h in H], dtype="int32")
        h_twin_H = np.array([H.index(self.h_twin_h(h)) for h in H], dtype="int32")
        f_left_H = np.array(
            [
                self.f_left_h(h) if self.f_left_h(h) < 0 else F.index(self.f_left_h(h))
                for h in H
            ],
            dtype="int32",
        )
        h_bound_F = np.array([H.index(self.h_bound_f(f)) for f in F], dtype="int32")
        h_right_B = np.array([H.index(h) for h in self.h_right_B], dtype="int32")
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

    ##############################################
    def generate_H_negative_bdry(self):
        for h in self.h_right_B:
            for h in self.generate_H_next_h(h):
                yield h

    def generate_V_negative_bdry(self):
        for h in self.generate_H_negative_bdry():
            yield self.v_origin_h(h)

    def get_VH_bdry(self):
        V = set()
        H = set()
        for h in self.generate_H_negative_bdry():
            V.add(self.v_origin_h(h))
            H.add(h)
            H.add(self.h_twin_h(h))
        return V, H

    def get_VHF_frontier(self):
        """
        closed star of patch boundary vertices
        """
        V = set()
        H = set()
        F = set()
        for h_bdry in self.generate_H_negative_bdry():
            v1 = self.v_origin_h(h_bdry)
            V.add(v1)
            for h1 in self.supermesh.generate_H_out_v_clockwise(v1):
                h1t = self.supermesh.h_twin_h(h1)
                v2 = self.supermesh.v_origin_h(h1t)
                H.add(h1)
                H.add(h1t)
                V.add(v2)
                f = self.supermesh.f_left_h(h1)
                if f < 0:
                    continue
                h2 = self.supermesh.h_next_h(h1)
                h3 = self.supermesh.h_next_h(h2)
                h2t = self.supermesh.h_twin_h(h2)
                h3t = self.supermesh.h_twin_h(h3)
                v3 = self.supermesh.v_origin_h(h2t)
                F.add(f)
                H.update([h2, h2t, h3, h3t])
                V.add(v3)
        return V, H, F

    ##############################################
    def update_from_V(self):
        """
        Recompute everything from the current vertex set
        """
        F = set()
        for i in self.V:
            for h in self.supermesh.generate_H_out_v_clockwise(i):
                j = self.supermesh.v_head_h(h)
                k = self.supermesh.v_head_h(self.supermesh.h_next_h(h))
                f = self.supermesh.f_left_h(h)
                if j in self.V and k in self.V and f >= 0:
                    F.add(f)
        V, H, F = self.supermesh.closure(self.V, set(), F)
        if self.V != V:
            raise ValueError("Vertices have changed.")
        self.H = H
        self.F = F
        self.h_right_B = self.find_h_right_B()

    ##############################################
    # to be deprecated
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

    def _generate_H_negative_bdry(self):
        for bdry, h in self.h_right_B.items():
            for h in self.generate_H_next_h(h):
                yield h

    def _generate_V_negative_bdry(self):
        for h in self.generate_H_negative_bdry():
            yield self.v_origin_h(h)

    def _generate_F_cw_B(self):
        for h in self.generate_H_negative_bdry():
            yield self.f_left_h(self.h_twin_h(h))

    def _expand_boundary(self):
        """
        ***this screws up something with boundaries, maybe in generate_H_negative_bdry/generate_F_cw_B?***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V, H, F = set(), set(), set()
        V_bdry_old = set(self.generate_V_negative_bdry())
        for h_start in self.generate_H_negative_bdry():
            if self.supermesh.negative_boundary_contains_h(h_start):
                continue
            # for h in self.supermesh.generate_H_in_cw_from_h(h_start): # ***
            for ht in self.supermesh.generate_H_out_v_clockwise(self, v, h_start=None):
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

    def _expand_boundary_doesnt_work_yet(self):
        """
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V, H, F = set(), set(), set()
        V_bdry_old = set(self.generate_V_negative_bdry())
        for h_start in self.generate_H_negative_bdry():
            if self.supermesh.negative_boundary_contains_h(h_start):
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

    def _expand_boundary(self):
        """
        **slow but actually works***
        Expand the boundary of the patch by one ring of vertices, edges, and faces.

        Returns:
            set: set of new boundary vertices
        """
        new_boundary_verts = set()
        V_bdry_old = set(self.generate_V_negative_bdry())
        V, H, F = self.supermesh.closure(*self.supermesh.star(V_bdry_old, set(), set()))
        V_bdry_new = V - self.V
        self.V.update(V)
        self.H.update(H)
        self.F.update(F)
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_bdry_new

    def _expand_towards_cylinder(self, x0, r_max, ez):
        """ """
        # V_bdry = set(self.generate_V_negative_bdry())
        V_bdry, H_bdry = self.get_VH_bdry()
        V_frontier, H_frontier, F_frontier = self.get_VHF_frontier()
        F_test = F_frontier - self.F
        V_add = set()
        H_add = set()
        F_add = set()
        for f in F_test:
            h1 = self.supermesh.h_bound_f(f)
            h2 = self.supermesh.h_next_h(h1)
            h3 = self.supermesh.h_next_h(h2)
            i1 = self.supermesh.v_origin_h(h1)
            i2 = self.supermesh.v_origin_h(h2)
            i3 = self.supermesh.v_origin_h(h3)
            x1 = self.supermesh.xyz_coord_v(i1)
            x2 = self.supermesh.xyz_coord_v(i2)
            x3 = self.supermesh.xyz_coord_v(i3)
            x = (x1 + x2 + x3) / 3
            # r = np.linalg.norm(np.cross(ez, x - x0))
            # if r <= r_max:
            if self.is_p_in_cylinder(x, x0, r_max, ez):
                V_add.update([i1, i2, i3])
                H_add.update(
                    [
                        h1,
                        h2,
                        h3,
                        self.supermesh.h_twin_h(h1),
                        self.supermesh.h_twin_h(h2),
                        self.supermesh.h_twin_h(h3),
                    ]
                )
                F_add.add(f)

        V_new = V_add - self.V
        H_new = H_add - self.H
        F_new = F_add - self.F
        self.V.update(V_new)
        self.H.update(H_new)
        self.F.update(F_new)
        self.h_right_B = self.find_h_right_B_from_VHF(
            V_check=set(), H_check=H_new + H_bdry, F_check=set()
        )
        return V_new, H_new, F_new

    def _move_boundary_towards_cylinder(self, x0, r_max, ez):
        """ """
        V_bdry = set()
        H_bdry = set()
        F_frontier = set()

        for h in self.generate_H_negative_bdry():
            v = self.v_origin_h(h)
            ht = self.h_twin_h(h)
            V_bdry.add(v)
            H_bdry.update([h, ht])
            F_frontier.update(self.supermesh.generate_F_incident_v_clockwise(v))
        V_frontier, H_frontier, F_frontier = self.supermesh.close_complex(
            set(), set(), F_frontier
        )
        F_min = self.F - F_frontier
        H_min = self.H - H_frontier
        V_min = self.V - V_frontier
        F_add_back = set()
        H_add_back = set()
        V_add_back = set()
        for f in F_frontier:
            h1 = self.supermesh.h_bound_f(f)
            h2 = self.supermesh.h_next_h(h1)
            h3 = self.supermesh.h_next_h(h2)
            i1 = self.supermesh.v_origin_h(h1)
            i2 = self.supermesh.v_origin_h(h2)
            i3 = self.supermesh.v_origin_h(h3)
            x1 = self.supermesh.xyz_coord_v(i1)
            x2 = self.supermesh.xyz_coord_v(i2)
            x3 = self.supermesh.xyz_coord_v(i3)
            x = (x1 + x2 + x3) / 3
            r = np.linalg.norm(np.cross(ez, x - x0))
            if r <= r_max:
                F_add_back.add(f)

        V_add_back, H_add_back, F_add_back = self.supermesh.closure(
            set(), set(), F_add_back
        )
        V_new = V - self.V
        H_new = H - self.H
        F_new = F - self.F
        self.V.update(V_new)
        self.H.update(H_new)
        self.F.update(F_new)
        self.h_right_B = self.find_h_right_B(F_need2check=F)
        return V_new, H_new, F_new

    def _expand_to_radius_perp(self, xyz_center, radius, vec):
        while True:
            V_new, H_new, F_new = self.expand_within_radius_perp(
                xyz_center, radius, vec
            )
            if not V_new:
                break


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
