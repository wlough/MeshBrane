from temp_python.src_python.half_edge_mesh import HalfEdgeMeshBase
import numpy as np


class HalfEdgeBoundary:
    """ """

    def __init__(self, supermesh, H):
        self.supermesh = supermesh
        self.H = H

    def v_origin_h(self, h):
        return self.supermesh.v_origin_h(h)

    def h_next_h(self, h):
        if h not in self.H:
            raise ValueError("Half-edge not in boundary.")
        n = self.supermesh.h_twin_h(h)
        v = self.supermesh.v_origin_h(n)
        for n in self.supermesh.generate_H_out_v_clockwise(v, h_start=n):
            if n in self.H:
                return n
        raise ValueError("Could not find next half-edge.")

    def h_prev_h(self, h):
        if h not in self.H:
            raise ValueError("Half-edge not in boundary.")
        v = self.supermesh.v_origin_h(h)
        for ht in self.supermesh.generate_H_out_v_clockwise(v, h_start=h):
            h = self.supermesh.h_twin_h(ht)
            if h in self.H:
                return h
        raise ValueError("Could not find previous half-edge.")

    def generate_H_next_h(self, h_start):
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def get_connected_components(self):
        H = self.H.copy()
        components = []
        while H:
            h = H.pop()
            c = set(self.generate_H_next_h(h))
            components.append(c)
            H.difference_update(c)
        return components

    @classmethod
    def from_faces(cls, supermesh, F):
        arrF = np.array(list(F), dtype=INT_TYPE)
        h_bound_F = supermesh.h_bound_f(arrF)
        next_h_bound_F = supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = supermesh.h_next_h(next_h_bound_F)
        H_interior = set(h_bound_F) | set(next_h_bound_F) | set(next_next_h_bound_F)
        arrH_interior = np.array(list(H_interior), dtype=INT_TYPE)
        arrH_twin = supermesh.h_twin_h(arrH_interior)
        H = set(arrH_twin) - H_interior
        return cls(supermesh, H)

    def generate_H_interior_wedge_h(self, h):
        while True:
            h = self.supermesh.h_twin_h(h)
            if h in self.H:
                break
            else:
                h = self.supermesh.h_next_h(h)
                yield h

    def get_interior_strips(self):
        H = self.H.copy()
        strips = []
        while H:
            h_start = H.pop()
            h = h_start
            Hs = set()
            while True:  # find the start of the strip
                ht = self.supermesh.h_twin_h(h)
                if ht in self.H:
                    if ht == h_start:
                        strips.append(Hs)
                        break
                    else:
                        h = self.supermesh.h_next_h(h)
                        H.discard(ht)
                else:
                    h = self.supermesh.h_next_h(ht)
                Hs.add(h)
        return strips

    # def get_interior_exterior_strips(self):
    #     H = self.H.copy()
    #     Se = []
    #     Si = []
    #     while H:
    #         h_start = H.pop()
    #         h = h_start
    #         He = set()
    #         Hi = set()
    #         h_is_interior = True
    #         while True:  # find the start of the strip
    #             ht = self.supermesh.h_twin_h(h)
    #             if ht in self.H:
    #                 if ht == h_start:
    #                     strips.append(Hs)
    #                     break
    #                 else:
    #                     h = self.supermesh.h_next_h(h)
    #                     H.discard(ht)
    #             else:
    #                 h = self.supermesh.h_next_h(ht)
    #             Hs.add(h)
    #     return strips

    def get_neighborhood(self):
        """ """
        H = self.H.copy()
        arrH = np.array(list(self.H), dtype=INT_TYPE)
        arrV = self.supermesh.v_origin_h(arrH)
        H_minus = np.array(
            list(self.H - set(self.supermesh.h_bound_f(F))), dtype=INT_TYPE
        )
        H_plus = self.supermesh.h_twin_h(H_minus)
        V = self.supermesh.v_origin_h(arrH_minus)
        return V, H_minus, H_plus


class MinimalPatch:
    """
    A submanifold of a HalfEdgeMesh.

    Parameters:
    ----------
    supermesh (HalfEdgeMesh): mesh containing the patch
    F (set): set of faces in the patch

    """

    def __init__(self, supermesh, F, H_boundary):
        self.supermesh = supermesh
        self.F = F

        ##########

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

    def get_H_bdry(self):
        arrF = np.array(list(F), dtype=INT_TYPE)
        # generators for each face boundary
        h_bound_F = self.supermesh.h_bound_f(arrF)
        next_h_bound_F = self.supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = self.supermesh.h_next_h(next_h_bound_F)
        # union of next cycles gives interior and positively oriented boundary half-edges
        H_closed_plus = set(h_bound_F) | set(next_h_bound_F) | set(next_next_h_bound_F)
        arrH_half_closed = np.array(list(H_half_closed), dtype=INT_TYPE)
        # image under twin map gives interior and negatively oriented boundary half-edges
        H_closed_minus = set(supermesh.h_twin_h(arrH_half_closed))
        H_boundary_plus = H_closed_plus - H_closed_minus
        H_interior = H_closed_plus - H_boundary_plus
        H_boundary_minus = H_closed_minus - H_interior

    def get_frontier(self):
        arrF = np.array(list(F), dtype=INT_TYPE)
        # generators for each face boundary
        h_bound_F = self.supermesh.h_bound_f(arrF)
        next_h_bound_F = self.supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = self.supermesh.h_next_h(next_h_bound_F)
        # union of next cycles gives interior and positively oriented boundary half-edges
        H_closed_plus = set(h_bound_F) | set(next_h_bound_F) | set(next_next_h_bound_F)
        arrH_half_closed = np.array(list(H_half_closed), dtype=INT_TYPE)
        # image under twin map gives interior and negatively oriented boundary half-edges
        H_closed_minus = set(supermesh.h_twin_h(arrH_half_closed))
        H_boundary_plus = H_closed_plus - H_closed_minus
        H_interior = H_closed_plus - H_boundary_plus
        H_boundary_minus = H_closed_minus - H_interior

        F_plus = set()
        F_minus = set()
        while H_boundary_plus:
            h_in_v = H_boundary_plus.pop()
            h_out_v = self.supermesh.h_twin_h(h_in_v)
            # for h in self.super

    #######################################################
    #######################################################
    #######################################################
    @classmethod
    def from_seed_vertex(cls, seed_vertex, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If seed_vertex is not in a boundary of supermesh, the patch will be a disk centered at seed_vertex. If seed_vertex is on a boundary, the patch will be wedge or sector of a disk.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = supermesh.closure(*supermesh.star_of_vertex(seed_vertex))
        self = cls(supermesh, V, H, F, seed_vertex=seed_vertex)

        return self

    @classmethod
    def from_seed_to_radius(cls, seed_vertex, supermesh, radius):
        """
        Create a patch consisting of the portion of supermesh contained within a sphere of the given radius centered at seed_vertex.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
            radius (float): radius of the patch
        """
        self = cls.from_seed_vertex(seed_vertex, supermesh)
        self.expand_to_radius(supermesh.xyz_coord_v(seed_vertex), radius)
        return self

    @classmethod
    def from_seed_to_cylinder(cls, seed_vertex, supermesh, p0, r_max, ez):
        """
        Create a patch consisting of the portion of supermesh contained a given cylinder. The cylinder is characterized by a radius r_max, a point p0 on its axis, and a unit vector ez parallel to the axis.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
            p0 (numpy.array): point on the axis of the cylinder
            r_max (float): radius of the cylinder
            ez (numpy.array): unit vector parallel to the axis of the cylinder
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
        h_right_B = set()
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
                        h_right_B.add(ht)
                    elif (not f_in_F) and ft_in_F:
                        h_right_B.add(h)
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
                    h_right_B.add(ht)
                elif (not f_in_F) and ft_in_F:
                    h_right_B.add(h)
                H_check.discard(h)
                H_check.discard(ht)

        while H_check:
            h = H_check.pop()
            ht = self.supermesh.h_twin_h(h)
            f_in_F = f in self.F
            ft_in_F = ft in self.F
            if f_in_F and (not ft_in_F):
                h_right_B.add(ht)
            elif (not f_in_F) and ft_in_F:
                h_right_B.add(h)
            H_check.discard(ht)

        h_right_B = []
        while h_right_B:
            h = h_right_B.pop()
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                h_right_B.discard(h)
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
        h_right_B = set()
        while H_frontier:
            h = H_frontier.pop()
            ht = self.supermesh.h_twin_h(h)
            H_frontier.discard(ht)
            f = self.supermesh.f_left_h(h)
            ft = self.supermesh.f_left_h(ht)
            f_in_F = f in self.F
            ft_in_F = ft in self.F
            if f_in_F and (not ft_in_F):
                h_right_B.add(ht)
            elif (not f_in_F) and ft_in_F:
                h_right_B.add(h)
            else:
                continue
            v = self.supermesh.v_origin_h(h)
            vt = self.supermesh.v_origin_h(ht)
            self.V.update([v, vt])
            self.H.update([h, ht])

        h_right_B = []
        while h_right_B:
            h = h_right_B.pop()
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                h_right_B.discard(h)
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


class HalfEdgeComplex:
    """An oriented simplicial subcomplex representing a submanifold of a HalfEdgeMesh."""

    def __init__(self, supermesh, V, H, F, close_complex=False):
        self.supermesh = supermesh
        self.V = V
        self.H = H
        self.F = F
        if close_complex:
            self.close_complex()

    #######################################################
    # Combinatorial maps ##################################
    #######################################################
    # def xyz_coord_v(self, v):
    #     return self.supermesh.xyz_coord_v(v)

    def h_out_v(self, v):
        for h in self.supermesh.generate_H_out_v_clockwise(v):
            if h in self.H:
                return h

    # def h_bound_f(self, f):
    #     return self._h_bound_F[f]

    # def v_origin_h(self, h):
    #     return self._v_origin_H[h]

    def f_left_h(self, h):
        return self._f_left_H[h]

    def h_next_h(self, h):
        return self._h_next_H[h]

    # def h_twin_h(self, h):
    #     return self._h_twin_H[h]

    def h_right_B(self, b):
        if b < 0:
            return self._h_right_B[-(b + 1)]
        return self._h_right_B[b]

    # Derived combinatorial maps
    def v_head_h(self, h):
        return self.v_origin_h(self.h_twin_h(h))

    def h_rotcw_h(self, h):
        return self.h_next_h(self.h_twin_h(h))

    def h_rotccw_h(self, h):
        return self.h_twin_h(self.h_prev_h(h))

    def h_prev_h(self, h):
        h_next = self.h_next_h(h)

        while h_next != h:
            h_prev = h_next
            h_next = self.h_next_h(h_prev)
        return h_prev

    def h_prev_h_by_rot(self, h):
        p_h = self.h_twin_h(h)
        n_h = self.h_next_h(p_h)
        while n_h != h:
            p_h = self.h_twin_h(n_h)
            n_h = self.h_next_h(p_h)
        return p_h

    @property
    def dim(self):
        if len(self.F) > 0:
            return 2
        elif len(self.H) > 0:
            return 1
        elif len(self.V) > 0:
            return 0
        else:
            return -1

    def close_faces(self):
        """
        Find simplicial closure of the complex in supermesh.
        """
        # next cycle of each face gets
        #   *interior half-edges
        #   *positive boundary half-edges
        V, H, F = set(), set(), self.F
        arrF = np.array(list(F), dtype=INT_TYPE)
        h_bound_F = self.supermesh.h_bound_f(arrF)
        next_h_bound_F = self.supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = self.supermesh.h_next_h(next_h_bound_F)
        H.update(h_bound_F)
        H.update(next_h_bound_F)
        H.update(next_next_h_bound_F)
        # twin of interior half-edges gets
        #   *negative boundary half-edges
        arrH = np.array(list(H), dtype=INT_TYPE)
        h_twin_H = self.supermesh.h_twin_h(arrH)
        H.update(h_twin_H)
        # origin of half-edges gets
        #  *vertices missing from V0
        arrH = np.array(list(H), dtype=INT_TYPE)
        v_origin_H = self.supermesh.v_origin_h(arrH)
        self.V.update(v_origin_H)
        self.H.update(H)

    def close_edges(self):
        """
        Find simplicial closure of 2d edge complex.
        """
        # twin of interior half-edges gets
        #   *negative boundary half-edges
        #   *any other twins missing from H0
        arrH = np.array(list(self.H), dtype=INT_TYPE)
        h_twin_H = self.supermesh.h_twin_h(arrH)
        self.H.update(h_twin_H)
        # origin of half-edges gets
        #  *vertices missing from V0
        arrH = np.array(list(self.H), dtype=INT_TYPE)
        v_origin_H = self.supermesh.v_origin_h(arrH)
        self.V.update(v_origin_H)

    def close_complex(self):
        self.close_edges()
        if self.dim == 2:
            self.close_faces()

    def get_boundary_of_faces(self):
        """
        Get the simplices of the complex.
        """
        arrF = np.array(list(F), dtype=INT_TYPE)
        h_bound_F = self.supermesh.h_bound_f(arrF)
        next_h_bound_F = self.supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = self.supermesh.h_next_h(next_h_bound_F)
        H_interior = set(h_bound_F) | set(next_h_bound_F) | set(next_next_h_bound_F)
        H_minus = self.H - H_interior
        h_next_H = np.zeros(len(H_minus), dtype=INT_TYPE)

        n = 0
        while H_minus:
            h = H_minus.pop()
            h_next_H[h] = self.supermesh.h_next_h(h)

        arrH_minus = np.array(list(H_minus), dtype=INT_TYPE)
        arrH_plus = self.supermesh.h_twin_h(arrH_minus)
        V = self.supermesh.v_origin_h(arrH_minus)

    def get_boundary_of_edges(self):
        """
        Get the simplices of the complex.
        """
        arrH = np.array(list(self.H), dtype=INT_TYPE)
        arrV = self.supermesh.v_origin_h(arrH)
        H_minus = np.array(
            list(self.H - set(self.supermesh.h_bound_f(F))), dtype=INT_TYPE
        )
        H_plus = self.supermesh.h_twin_h(H_minus)
        V = self.supermesh.v_origin_h(arrH_minus)


class HalfEdgePatch:
    """
    A submanifold of a HalfEdgeMesh.

    Parameters:
    ----------
    supermesh (HalfEdgeMesh): mesh containing the patch
    V (set): set of vertices in the patch
    H (set): set of half-edges in the patch
    F (set): set of faces in the patch


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
        close=False,
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
        if close:
            self.update_from_V()

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

    ##############################################
    @classmethod
    def from_seed_vertex(cls, seed_vertex, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If seed_vertex is not in a boundary of supermesh, the patch will be a disk centered at seed_vertex. If seed_vertex is on a boundary, the patch will be wedge or sector of a disk.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        V, H, F = supermesh.closure(*supermesh.star_of_vertex(seed_vertex))
        self = cls(supermesh, V, H, F, seed_vertex=seed_vertex)

        return self

    @classmethod
    def from_seed_to_radius(cls, seed_vertex, supermesh, radius):
        """
        Create a patch consisting of the portion of supermesh contained within a sphere of the given radius centered at seed_vertex.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
            radius (float): radius of the patch
        """
        self = cls.from_seed_vertex(seed_vertex, supermesh)
        self.expand_to_radius(supermesh.xyz_coord_v(seed_vertex), radius)
        return self

    @classmethod
    def from_seed_to_cylinder(cls, seed_vertex, supermesh, p0, r_max, ez):
        """
        Create a patch consisting of the portion of supermesh contained a given cylinder. The cylinder is characterized by a radius r_max, a point p0 on its axis, and a unit vector ez parallel to the axis.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
            p0 (numpy.array): point on the axis of the cylinder
            r_max (float): radius of the cylinder
            ez (numpy.array): unit vector parallel to the axis of the cylinder
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
        h_right_B = set()
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
                        h_right_B.add(ht)
                    elif (not f_in_F) and ft_in_F:
                        h_right_B.add(h)
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
                    h_right_B.add(ht)
                elif (not f_in_F) and ft_in_F:
                    h_right_B.add(h)
                H_check.discard(h)
                H_check.discard(ht)

        while H_check:
            h = H_check.pop()
            ht = self.supermesh.h_twin_h(h)
            f_in_F = f in self.F
            ft_in_F = ft in self.F
            if f_in_F and (not ft_in_F):
                h_right_B.add(ht)
            elif (not f_in_F) and ft_in_F:
                h_right_B.add(h)
            H_check.discard(ht)

        h_right_B = []
        while h_right_B:
            h = h_right_B.pop()
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                h_right_B.discard(h)
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
        h_right_B = set()
        while H_frontier:
            h = H_frontier.pop()
            ht = self.supermesh.h_twin_h(h)
            H_frontier.discard(ht)
            f = self.supermesh.f_left_h(h)
            ft = self.supermesh.f_left_h(ht)
            f_in_F = f in self.F
            ft_in_F = ft in self.F
            if f_in_F and (not ft_in_F):
                h_right_B.add(ht)
            elif (not f_in_F) and ft_in_F:
                h_right_B.add(h)
            else:
                continue
            v = self.supermesh.v_origin_h(h)
            vt = self.supermesh.v_origin_h(ht)
            self.V.update([v, vt])
            self.H.update([h, ht])

        h_right_B = []
        while h_right_B:
            h = h_right_B.pop()
            h_right_B.append(h)
            for h in self.generate_H_next_h(h):
                h_right_B.discard(h)
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


class HalfEdgeAtlas:
    def __init__(self, mesh=None, patches=None):
        self.mesh = mesh
        self.patches = patches
