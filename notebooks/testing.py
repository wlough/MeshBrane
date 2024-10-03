class HalfEdgeBoundary:
    """
    Boundary of a (sub)mesh

    Parameters
    ----------
    mesh : HalfEdgeMesh
        mesh containing the boundary
    H : set
        set of half-edges
    h_right_B : ndarray
        half-edge which generates each connected component of the positively oriented boundary
    """

    def __init__(self, supermesh, H=None, h_gen_B=None, *args, **kwargs):
        if H is None:
            H = set()
        if h_gen_B is None:
            h_gen_B = np.array([], dtype=INT_TYPE)
        self.supermesh = supermesh
        self.H = H
        self.h_gen_B = h_gen_B

    ######################################################
    # Initilization and update methods
    ######################################################
    @classmethod
    def from_mesh(cls, supermesh):
        arrH = supermesh.positive_boundary_contains_h(
            range(supermesh.num_half_edges)
        ).nonzero()[0]
        H = set(arrH)
        h_gen_B = supermesh.h_twin_h(supermesh.h_right_B)
        return cls(supermesh, H, h_gen_B)

    @classmethod
    def from_faces(cls, supermesh, F):
        arrF = np.array(list(F), dtype=INT_TYPE)
        # generators for each face boundary
        h_bound_F = supermesh.h_bound_f(arrF)
        next_h_bound_F = supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = supermesh.h_next_h(next_h_bound_F)
        # union of next cycles gives interior and positively oriented boundary half-edges
        H_closed_plus = set(h_bound_F) | set(next_h_bound_F) | set(next_next_h_bound_F)
        arrH_closed_plus = np.array(list(H_closed_plus), dtype=INT_TYPE)
        # image under twin map gives interior and negatively oriented boundary half-edges
        H_closed_minus = set(supermesh.h_twin_h(arrH_closed_plus))
        H_boundary_plus = H_closed_plus - H_closed_minus
        # H_interior = H_closed_plus - H_boundary_plus
        # H_boundary_minus = H_closed_minus - H_interior
        self = cls(supermesh, H=H_boundary_plus)
        self.refresh_h_gen_B()
        return self

    def initialize_complement(self):
        arrH = self.arrH
        twin_arrH = self.supermesh.h_twin_h(arrH)
        F_left_twin_arrH = self.supermesh.h_left_h(twin_arrH)
        H_complement = set(twin_arrH[F_left_twin_arrH >= 0])
        complement_boundary = HalfEdgeBoundary(self.supermesh, H=H_complement)
        complement_boundary.refresh_h_gen_B()
        return complement_boundary

    def refresh_h_gen_B(self):
        """
        Find half-edge which generates each connected component of the boundary
        """
        Hneed2visit = set(self.H)
        h_gen_B = []
        while Hneed2visit:
            h = Hneed2visit.pop()
            h_gen_B.append(h)
            for h in self.generate_H_next_h(h):
                Hneed2visit.discard(h)

        self.h_gen_B = np.array(h_gen_B, dtype=INT_TYPE)

    def update_from_faces(self, F):
        arrF = np.array(list(F), dtype=INT_TYPE)
        # generators for each face boundary
        h_bound_F = self.supermesh.h_bound_f(arrF)
        next_h_bound_F = self.supermesh.h_next_h(h_bound_F)
        next_next_h_bound_F = self.supermesh.h_next_h(next_h_bound_F)
        # union of next cycles gives interior and positively oriented boundary half-edges
        H_closed_plus = set(h_bound_F) | set(next_h_bound_F) | set(next_next_h_bound_F)
        arrH_closed_plus = np.array(list(H_closed_plus), dtype=INT_TYPE)
        # image under twin map gives interior and negatively oriented boundary half-edges
        H_closed_minus = set(self.supermesh.h_twin_h(arrH_closed_plus))
        H_boundary_plus = H_closed_plus - H_closed_minus
        self.H = H_boundary_plus
        self.refresh_h_gen_B()

    #######################################################
    # Fundamental accessors and properties
    ######################################################
    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        self._H = set(value)

    @property
    def h_gen_B(self):
        return self._h_gen_B

    @h_gen_B.setter
    def h_gen_B(self, value):
        self._h_gen_B = np.array(value, dtype=INT_TYPE)

    @property
    def arrH(self):
        return np.array(list(self.H), dtype=INT_TYPE)

    def h_gen_b(self, b):
        """get index of a half-edge contained in boundary b

        Args:
            b (int): boundary index

        Returns:
            int: half-edge index
        """
        return self._h_gen_B[b]

    def get_interior_faces(self):
        F_interior = set()
        F_frontier = set(
            self.supermesh.f_left_h(np.array(list(self.H), dtype=INT_TYPE))
        )
        while F_frontier:
            f = F_frontier.pop()
            for h in self.supermesh.generate_H_bound_f(f):
                if h in self.H:
                    continue
                ht = self.supermesh.h_twin_h(h)
                ft = self.supermesh.f_left_h(ht)
                if ft in F_interior:
                    continue
                F_frontier.add(ft)
            F_interior.add(f)
        return F_interior

    def get_frontier_faces(self):
        F_plus = set()
        F_minus = set()
        for h_start in self.H:
            # h_start = self.supermesh.h_twin_h(h_boundary)
            f_is_interior = True
            for h in self.supermesh.generate_H_rotcw_h(h_start):
                f = self.supermesh.f_left_h(h)
                if f < 0:
                    continue
                if f_is_interior:
                    F_plus.add(f)
                else:
                    F_minus.add(f)
                if h in self.H:
                    f_is_interior = not f_is_interior
        return F_plus, F_minus

    def h_next_h(self, h):
        """get index of the next half-edge after h in the face/boundary cycle
        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        n_start = self.supermesh.h_twin_h(h)
        for n in self.supermesh.generate_H_rotcw_h(n_start):
            if n in self.H:
                return n
        return h

    #######################################################
    # Generators ##########################################
    ######################################################
    def generate_H_next_h(self, h_start):
        h = h_start
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break

    def generate_interior_faces_cumulative(self):
        F_interior = set()
        F_frontier = set(
            self.supermesh.f_left_h(np.array(list(self.H), dtype=INT_TYPE))
        )
        while F_frontier:
            f = F_frontier.pop()
            for h in self.supermesh.generate_H_bound_f(f):
                if h in self.H:
                    continue
                ht = self.supermesh.h_twin_h(h)
                ft = self.supermesh.f_left_h(ht)
                if ft in F_interior:
                    continue
                F_frontier.add(ft)
            F_interior.add(f)
            yield F_interior, F_frontier

    def generate_H_interior_wedge_h(self, h):
        while True:
            ht = self.supermesh.h_twin_h(h)
            if ht in self.H:
                break
            else:
                h = self.supermesh.h_next_h(h)
                yield h

    def generate_H_exterior_wedge_h(self, h):
        while True:
            h = self.supermesh.h_twin_h(h)
            if h in self.H:
                break
            else:
                h = self.supermesh.h_next_h(h)
                yield h

    ######################################################
    # To be deprecated
    ######################################################
    def build_interior_mesh(self):
        F_interior = set()
        F_frontier = set(
            self.supermesh.f_left_h(np.array(list(self.H), dtype=INT_TYPE))
        )
        while F_frontier:
            f = F_frontier.pop()
            H_bound_f = set(self.supermesh.generate_H_bound_f(f))
            for h in H_bound_f:
                if h in self.H:
                    continue
                ht = self.supermesh.h_twin_h(h)
                ft = self.supermesh.f_left_h(ht)
                if ft in F_interior:
                    continue
                F_frontier.add(ft)
            F_interior.add(f)

        return HalfEdgeMesh.from_faces(self.supermesh, F_interior)

    def _get_F_adjacent_b(self, b):
        h_gen = self.h_gen_b(b)
        F = set()
        h = self.supermesh.h_next_h(h_gen)
        # while h != h_gen:
        #     while h not in self.H:
        #         f = self.supermesh.f_left_h(h)
        #         F.add(f)
        #         h = self.supermesh.h_rotcw_h(h)
        #     h = self.supermesh.h_next_h(h)
        while True:
            f = self.supermesh.f_left_h(h)
            F.add(f)
            h = self.supermesh.h_rotcw_h(h)
            if h in self.H:
                if h == h_gen:
                    return F
                else:
                    h = self.supermesh.h_next_h(h)
        return F

    def _xyz_coord_v(self, v):
        """
        get array of xyz coordinates of vertex v

        Args:
            v (int): vertex index

        Returns:
            numpy.array: xyz coordinates
        """
        return self.supermesh.xyz_coord_v(v)

    # def h_out_v(self, v):
    #     """
    #     get index of an non-boundary outgoing half-edge incident on vertex v

    #     Args:
    #         v (int): vertex index

    #     Returns:
    #         int: half-edge index
    #     """
    #     if v not in self.V:
    #         raise ValueError("Vertex not in patch.")
    #     for h in self.supermesh.generate_H_out_v_clockwise(v):
    #         if h not in self.H:
    #             continue
    #         elif self.supermesh.f_left_h(h) in self.F:
    #             return h

    def _h_next_h(self, h):
        """get index of the next half-edge after h in the face/boundary cycle
        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in boundary.")
        n = self.supermesh.h_next_h(h)
        while n not in self.H:
            n = self.supermesh.h_rotcw_h(n)
        return n

    def _h_twin_h(self, h):
        """get index of the half-edge anti-parallel to half-edge h

        Args:
            h (int): half-edge index

        Returns:
            int: half-edge index
        """
        if h not in self.H:
            raise ValueError("Half-edge not in patch.")
        return self.supermesh.h_twin_h(h)

    def _f_left_h(self, h):
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

    def _h_bound_f(self, f):
        """get index of a half-edge on the boundary of face f

        Args:
            f (int): face index

        Returns:
            int: half-edge index
        """
        if f not in self.F:
            raise ValueError("Face not in patch.")
        return self.supermesh.h_bound_f(f)

    def _generate_H_next_h(self, h):
        """Generate half-edges in the face/boundary cycle containing half-edge h"""
        h_start = h
        while True:
            yield h
            h = self.h_next_h(h)
            if h == h_start:
                break


class SubMesh:
    """
    A submanifold of a HalfEdgeMesh.

    Parameters:
    ----------
    supermesh (HalfEdgeMesh): mesh containing the patch
    F (set): set of faces in the patch

    """

    def __init__(self, supermesh, F=None, boundary=None, *args, **kwargs):
        self.supermesh = supermesh
        if F is None:
            F = set()
        if boundary is None:
            boundary = HalfEdgeBoundary(supermesh)
        self.F = F
        self.boundary = boundary

    ######################################################
    # Initilization and update methods
    ######################################################

    @classmethod
    def from_seed_vertex(cls, seed_vertex, supermesh):
        """
        Initialize a patch from a seed vertex by including taking the closure of all simplices in supermesh that contain the seed vertex. If seed_vertex is not in a boundary of supermesh, the patch will be a disk centered at seed_vertex. If seed_vertex is on a boundary, the patch will be wedge or sector of a disk.

        Parameters:
            seed_vertex (int): vertex index
            supermesh (HalfEdgeMesh): mesh from which the patch is extracted
        """
        F = set()
        h = supermesh.h_out_v(seed_vertex)
        for h in supermesh.generate_H_rotcw_h(h):
            f = supermesh.f_left_h(h)
            if f < 0:
                continue
            F.add(f)
        self = cls(supermesh, F)
        self.refresh_boundary()
        return self

    ######################################################
    # Fundamental accessors and properties
    ######################################################
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
    def num_faces(self):
        return len(self.F)

    def refresh_boundary(self):
        self.boundary.update_from_faces(self.F)


class TransitionFunction:
    def __init__(self, patch1, patch2, *args, **kwargs):
        pass
