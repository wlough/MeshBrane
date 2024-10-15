import numpy as np


class CombinatorialSimplex:
    """
    Oriented simplex with integer vertex set. Two simplices are equal if they have the same vertex set and the same parity.

    Properties
    ----------
    points : frozenset of int
        Set of vertex indices
    parity : bool
        Parity of the permutation that sorts the vertex indices
    point_list : list of int
        Representative of the simplex's equivalence class
    """

    def __init__(self, point_list, *args, **kwargs):
        self.parity = parity_of_argsort(point_list)
        self.points = frozenset(point_list)
        self.str = str(self.point_list)

    def __eq__(self, other):
        if isinstance(other, CombinatorialSimplex):
            return self.points == other.points and self.parity == other.parity
        return False

    def __hash__(self):
        return hash((self.points, self.parity))

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"simplex({self.point_list})"

    @property
    def sorted(self):
        return sorted(self.points)

    @property
    def point_list(self):
        if self.parity:
            return sorted(self.points)
        else:
            P = sorted(self.points)
            P[0], P[1] = P[1], P[0]
            return P

    @classmethod
    def from_points_and_parity(cls, points, parity, *args, **kwargs):
        if parity:
            return cls(sorted(points), *args, **kwargs)
        else:
            P = sorted(points)
            P[0], P[1] = P[1], P[0]
            return cls(P)

    def boundary(self):
        s = self.point_list
        return [
            CombinatorialSimplex.from_points_and_parity(
                (v for v in s if v != vi), i % 2 == 0
            )
            for i, vi in enumerate(s)
        ]


class SimplexBase:
    """An oriented simplex with integer vertex set.

    Parameters
    ----------
    points : iterable of int
        Set of vertex indices
    cached_sort : list of int
        Precomputed sorted vertex indices
    """

    def __init__(self, points, cached_sort=None, *args, **kwargs):
        self.points = points
        if cached_sort is None:
            self.cache_is_valid = False
        else:
            self.cached_sort = cached_sort

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = frozenset(value)
        self.cache_is_valid = False

    @property
    def cached_sort(self):
        return self._cached_sort

    @cached_sort.setter
    def cached_sort(self, value):
        if isinstance(value, list):
            self._cached_sort = value
        else:
            self._cached_sort = list(value)

    def sort(self):
        if self.cache_is_valid:
            return self.cached_sort
        else:
            self.cached_sort = sorted(self.points)
            self.cache_is_valid = True
        return self.cached_sort

    def __eq__(self, other):
        if isinstance(other, SimplexBase):
            return self.points == other.points
        return False

    def __hash__(self):
        return hash((self.points,))

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        if self.cache_is_valid:
            return f"simplex({self.cached_sort})"
        else:
            return f"simplex({self.sort()})"

    def __contains__(self, point):
        return point in self.points

    def __ge__(self, other):
        return self.points >= other.points

    def __gt__(self, other):
        return self.points > other.points

    def __le__(self, other):
        return self.points <= other.points

    def __lt__(self, other):
        return self.points < other.points


class SimpleChain:
    """
    A simplex with multiplicity or chain of the form

    multiplicity*simplex
    """

    def __init__(self, simplex, multiplicity, *args, **kwargs):
        self.simplex = simplex
        self.multiplicity = multiplicity

    @classmethod
    def from_points_and_multiplicity(cls, points, multiplicity, *args, **kwargs):
        simplex = SimplexBase(points)
        return cls(simplex, multiplicity, *args, **kwargs)

    @property
    def multiplicity(self):
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value):
        if isinstance(value, int):
            self._multiplicity = value
        else:
            self._multiplicity = int(value)

    @property
    def points(self):
        return self.simplex.points

    def __eq__(self, other):
        if isinstance(other, SimpleChain):
            return (
                self.simplex == other.simplex
                and self.multiplicity == other.multiplicity
            )
        return False

    def __hash__(self):
        return hash((self.simplex, self.multiplicity))

    def __len__(self):
        return len(self.simplex)

    def __repr__(self):
        if self.multiplicity == 1:
            return self.simplex.__repr__()
        else:
            return f"{self.multiplicity}*{self.simplex.__repr__()}"

    def simplex_eq(self, other):
        return self.simplex == other.simplex

    def __neg__(self):
        return SimpleChain(self.simplex, -self.multiplicity)

    def __add__(self, other):
        if self.simplex == other.simplex:
            return SimpleChain(self.simplex, self.multiplicity + other.multiplicity)
        else:
            raise ValueError("Not implemented")

    def __rmul__(self, value):
        return SimpleChain(self.simplex, value * self.multiplicity)

    def __mul__(self, value):
        return SimpleChain(self.simplex, value * self.multiplicity)

    def __ge__(self, other):
        return [self.simplex.sort(), self.multiplicity] >= [
            other.simplex.sort(),
            other.multiplicity,
        ]

    def __gt__(self, other):
        return [self.simplex.sort(), self.multiplicity] > [
            other.simplex.sort(),
            other.multiplicity,
        ]

    def __le__(self, other):
        return [self.simplex.sort(), self.multiplicity] <= [
            other.simplex.sort(),
            other.multiplicity,
        ]

    def __lt__(self, other):

        return [self.simplex.sort(), self.multiplicity] < [
            other.simplex.sort(),
            other.multiplicity,
        ]

    ########################################

    def boundary(self):
        s = self.simplex.sort()
        m = self.multiplicity
        return np.array(
            [
                SimpleChain.from_points_and_multiplicity(
                    (v for v in s if v != vi), (-1) ** i * m
                )
                for i, vi in enumerate(s)
            ],
            dtype=object,
        )
        # return [
        #     SimpleChain.from_points_and_multiplicity(
        #         set(v for v in s if v != vi), (-1) ** i * m
        #     )
        #     for i, vi in enumerate(s)
        # ]


class SimplicialChain:
    def __init__(self, simple_chains):
        self.simple_chains = simple_chains

    def __repr__(self):
        if self.is_zero_chain():
            return "0"
        c0 = self.simple_chains[0]
        r = c0.__repr__()
        for c in self.simple_chains[1:]:
            m = c.multiplicity
            if m > 0:
                r += "+"
            r += c.__repr__()
        return r

    def __neg__(self):
        return SimplicialChain(-self.simple_chains)

    def __add__(self, other):
        return SimplicialChain(
            [np.concatenate(self.simple_chains, other.simple_chains)]
        )

    def __rmul__(self, value):
        return SimplicialChain(value * self.simple_chains)

    def __mul__(self, value):
        return SimplicialChain(value * self.simple_chains)

    @property
    def simple_chains(self):
        return self._simple_chains

    @simple_chains.setter
    def simple_chains(self, value):
        self._simple_chains = np.array(value, dtype=object)

    def is_zero_chain(self):
        return len(self.simple_chains) == 0

    def sort_chain(self):
        self.simple_chains = sorted(self.simple_chains)
        return self

    # def simplify(self):
    # """broken..."""
    #     if self.is_zero_chain():
    #         return
    #     # simplices = {_.simplex: 0 for _ in self.simple_chains}
    #     self.simple_chains = sorted(self.simple_chains)
    #     c0 = self.simple_chains[0]
    #     s0 = c0.simplex
    #     simple_chains = [c0]
    #     for c in self.simple_chains[1:]:
    #         s, m = c.simplex, c.multiplicity
    #         if s == s0:
    #             c0 += c
    #         else:
    #             c0 = c
    #             s0 = c0.simplex
    #             simple_chains.append(c0)
    #     self.simple_chains = simple_chains
    #     return self
    def simplify(self):
        if self.is_zero_chain():
            return
        S = {_.simplex: 0 for _ in self.simple_chains}
        for x in self.simple_chains:
            S[x.simplex] += x.multiplicity
        self.simple_chains = [
            SimpleChain(simplex, multiplicity)
            for simplex, multiplicity in S.items()
            if multiplicity != 0
        ]
        return self

    def boundary(self):
        # c = self.simple_chains
        # dc = []
        # for _ in c:
        #     dc += _.boundary()
        # return dc
        return SimplicialChain(
            np.concatenate([_.boundary() for _ in self.simple_chains])
        )


class ZeroChain(SimplicialChain):
    def __init__(self):
        super().__init__([])

    def __repr__(self):
        return "0"

    def __neg__(self):
        return self

    def __add__(self, other):
        return other

    def __rmul__(self, value):
        return self

    def __mul__(self, value):
        return self

    def is_zero_chain(self):
        return True

    def sort_chain(self):
        return self

    def simplify(self):
        return self

    def boundary(self):
        return self


class _SignedSimplex:
    """
    Integer multiple of an oriented simplex.

    Properties
    ----------
    multiplicity : int
        Integer-valued coefficient
    """

    def __init__(self, point_list, multiplicity=1, *args, **kwargs):
        self.point_list = point_list
        self.parity = parity_of_argsort(point_list)
        parity_is_odd = not self.parity
        if parity_is_odd:
            self.multiplicity = -multiplicity
        else:
            self.multiplicity = multiplicity
        self.points = frozenset(point_list)
        # self.str = str(self.point_list)

    @classmethod
    def from_ordered_points(self, ordered_points):
        pass

    @classmethod
    def from_points_and_multiplicity(cls, points, multiplicity, *args, **kwargs):
        return cls(sorted(points), multiplicity, *args, **kwargs)

    @property
    def sorted(self):
        return sorted(self.points)

    @property
    def point_list(self):
        # if self.multiplicity >= 0:
        #     return sorted(self.points)
        # else:
        #     P = sorted(self.points)
        #     P[0], P[1] = P[1], P[0]
        #     return P
        # return sorted(self.points)
        return self._point_list

    @point_list.setter
    def point_list(self, value):
        self._point_list = list(value)

    def __eq__(self, other):
        if isinstance(other, SignedSimplex):
            return (
                self.points == other.points and self.multiplicity == other.multiplicity
            )
        return False

    def __hash__(self):
        return hash((self.points, self.multiplicity))

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        if self.multiplicity == 1:
            return f"simplex({self.sorted})"
        else:
            return f"{self.multiplicity}*simplex({self.sorted})"

    def unsigned_eq(self, other):
        return self.points == other.points

    def __neg__(self):
        return SignedSimplex(self.sorted, -self.multiplicity)

    def __add__(self, other):
        if self.points != other.points:
            raise ValueError("Incompatible point sets")
        else:
            return SignedSimplex(self.sorted, self.multiplicity + other.multiplicity)

    def __rmul__(self, value):
        return SignedSimplex(self.sorted, value * self.multiplicity)

    def __mul__(self, value):
        return SignedSimplex(self.sorted, value * self.multiplicity)

    ########################################
    # def __contains__(self, point):
    #     return point in self.points

    def __ge__(self, other):
        return [self.sorted, self.multiplicity] >= [other.sorted, other.multiplicity]

    def __gt__(self, other):
        return [self.sorted, self.multiplicity] > [other.sorted, other.multiplicity]

    def __le__(self, other):
        return [self.sorted, self.multiplicity] <= [other.sorted, other.multiplicity]

    def __lt__(self, other):

        return [self.sorted, self.multiplicity] < [other.sorted, other.multiplicity]

    ########################################

    def boundary(self):
        s = self.sorted
        m = self.multiplicity
        return [
            SignedSimplex.from_points_and_multiplicity(
                (v for v in s if v != vi), (-1) ** i * m
            )
            for i, vi in enumerate(s)
        ]


class _CombinatorialTetrahedron:
    """
    name : str
        Name of the cell.
    order : int
        Number of vertices in the cell.
    P : list
        List of point indices
    V : list
        List of 0-simplices (vertices).
    E : list
        List of 1-simplices (edges).
    F : list
        List of 2-simplices (faces).

    βi maps a dart to another dart with a different i-cell
    """

    def __init__(self):
        self.P = [0, 1, 2, 3]
        self.V = [
            CombinatorialSimplex([0]),
            CombinatorialSimplex([1]),
            CombinatorialSimplex([2]),
            CombinatorialSimplex([3]),
        ]
        self.E = [
            CombinatorialSimplex([0, 1]),
            CombinatorialSimplex([1, 0]),
            CombinatorialSimplex([0, 2]),
            CombinatorialSimplex([2, 0]),
            CombinatorialSimplex([0, 3]),
            CombinatorialSimplex([3, 0]),
            CombinatorialSimplex([1, 2]),
            CombinatorialSimplex([2, 1]),
            CombinatorialSimplex([1, 3]),
            CombinatorialSimplex([3, 1]),
            CombinatorialSimplex([2, 3]),
            CombinatorialSimplex([3, 2]),
        ]
        self.F = [
            CombinatorialSimplex([0, 2, 1]),
            CombinatorialSimplex([0, 1, 3]),
            CombinatorialSimplex([1, 2, 3]),
            CombinatorialSimplex([2, 0, 3]),
        ]
        self.C = [
            CombinatorialSimplex([0, 1, 2, 3]),
        ]
