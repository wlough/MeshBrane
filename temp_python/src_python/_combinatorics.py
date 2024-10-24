from dataclasses import dataclass
import numpy as np

###########################################################################################
# Finite groups
# -------------
# A bag of tricks for working with permutations, finite group actions/representations, and other combinatorial things
###########################################################################################

###################################################
# Symmetric group on Zn=[0,...,n-1]


def compose(P, Q):
    """
    Compose two permutations P*Q

    Parameters
    ----------
    P : list of int
        Permutation [0,...,N-1]
    Q : list of int
        Permutation [0,...,N-1]

    Returns
    -------
    list of int : composed permutation
    """
    return [P[q] for q in Q]


def inverse(P):
    """
    Invert a permutation

    Parameters
    ----------
    P : list of int
        Permutation [0,...,N-1]

    Returns
    -------
    list of int : inverted permutation
    """
    return [P.index(i) for i in range(len(P))]


def parity(P):
    """
    Compute parity of a permutation. Returns True/False for even/odd. The parity of a permutation is equal to the parity of cycle_len-cycle_num where cycle_num is the number of cycles in the permutation and cycle_len is sum of all cycle lengths.

    Parameters
    ----------
    P : list of int
        Permutation [0,...,len(P)-1]

    Returns
    -------
    bool : True for even permutation, False for odd for false
    """
    need2visit = set(P)
    # cycle_num = 0 # number of cycles
    # cycle_len = 0 # sum of cycle lengths
    parity = True
    while need2visit:
        # cycle_num += 1 # start a new cycle
        # parity = not parity * this cancels **
        i_start = need2visit.pop()
        # cycle_len += 1 # count first element in the cycle i=i_start
        # parity = not parity ** this cancels *
        i = P[i_start]
        while i != i_start:
            # cycle_len += 1 # count element i in the cycle
            parity = not parity
            need2visit.discard(i)  # remove visited element
            i = P[i]  # next element in the cycle
    return parity  # cycle_num - cycle_len % 2 == 0


def compute_cycles(P):
    """
    Compute cycles of a permutation

    Parameters
    ----------
    P : list of int
        Permutation [0,...,len(P)-1]

    Returns
    -------
    cycles : set of tuple of int
    parity : True for even permutation, False for odd for false
    """
    cycles = set()
    need2visit = set(P)
    if need2visit != set(range(len(P))):
        raise ValueError("P must be a permutation of [0,...,len(P)-1]")
    # cycle_num = 0 # number of cycles
    # cycle_len = 0 # sum of cycle lengths
    parity = True
    while need2visit:
        # cycle_num += 1 # start a new cycle
        # parity = not parity * this cancels **
        i_start = need2visit.pop()
        cycle = [i_start]
        # cycle_len += 1 # count first element in the cycle i=i_start
        # parity = not parity ** this cancels *
        i = P[i_start]
        while i != i_start:
            # cycle_len += 1 # count element i in the cycle
            cycle.append(i)
            parity = not parity
            need2visit.discard(i)  # remove visited element
            i = P[i]  # next element in the cycle
        cycles.add(tuple(cycle))
    return cycles, parity  # cycle_num - cycle_len % 2 == 0


def compute_nontrivial_cycles(P):
    """
    Compute cycles of a permutation (suppresses one-cycles)

    Parameters
    ----------
    P : list of int
        Permutation [0,...,len(P)-1]

    Returns
    -------
    cycles : set of tuple of int
    parity : True for even permutation, False for odd for false
    """
    cycles = set()
    need2visit = set(P)
    if need2visit != set(range(len(P))):
        raise ValueError("P must be a permutation of [0,...,len(P)-1]")
    # cycle_num = 0 # number of cycles
    # cycle_len = 0 # sum of cycle lengths
    parity = True
    while need2visit:
        # cycle_num += 1 # start a new cycle
        # parity = not parity * this cancels **
        i_start = need2visit.pop()
        cycle = [i_start]
        # cycle_len += 1 # count first element in the cycle i=i_start
        # parity = not parity ** this cancels *
        i = P[i_start]
        while i != i_start:
            # cycle_len += 1 # count element i in the cycle
            cycle.append(i)
            parity = not parity
            need2visit.discard(i)  # remove visited element
            i = P[i]  # next element in the cycle
        if len(cycle) > 1:
            cycles.add(tuple(cycle))
    return cycles, parity  # cycle_num - cycle_len % 2 == 0


def relative_parity(P, Q):
    """
    Equivalence relation on the set of lists of integers. Two lists are equivalent if they are related by an even permutation

    Parameters
    ----------
    P : list of int
        Permutation [0,...,N-1]
    Q : list of int
        Permutation [0,...,N-1]

    Returns
    -------
    bool : True if the lists are equivalent, False otherwise
    """

    return parity(compose(P, Q))


class PermutationBase:
    """
    An element of the symmetric group on [0,...,n-1] for some integer n.

    Attributes
    ----------
    P : list of int
        permutation [0,...,n-1]
    cycles : set of tuples
        cycle decomposition
    parity : bool
        True/False for even/odd
    """

    def __init__(self, P):
        self._P = P
        self._cycles, self._parity = self.compute_cycles()

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        assert set(value) == set(
            range(len(value))
        ), "P must be a permutation of [0,...,len(P)-1]"
        self._P = value
        self._cycles, self._parity = self.compute_cycles()

    @property
    def cycles(self):
        return self._cycles

    @property
    def parity(self):
        return self._parity

    def compute_cycles(self):
        """
        Compute cycles of a permutation

        Parameters
        ----------
        P : list of int
            Permutation [0,...,len(P)-1]

        Returns
        -------
        cycles : set of tuple of int
        parity : True for even permutation, False for odd for false
        """
        cycles = set()
        need2visit = set(self.P)
        if need2visit != set(range(len(self.P))):
            raise ValueError("P must be a permutation of [0,...,len(P)-1]")
        # cycle_num = 0 # number of cycles
        # cycle_len = 0 # sum of cycle lengths
        parity = True
        while need2visit:
            # cycle_num += 1 # start a new cycle
            # parity = not parity * this cancels **
            i_start = need2visit.pop()
            cycle = [i_start]
            # cycle_len += 1 # count first element in the cycle i=i_start
            # parity = not parity ** this cancels *
            i = self.P[i_start]
            while i != i_start:
                # cycle_len += 1 # count element i in the cycle
                cycle.append(i)
                parity = not parity
                need2visit.discard(i)  # remove visited element
                i = self.P[i]  # next element in the cycle
            cycles.add(tuple(cycle))
        return cycles, parity  # cycle_num - cycle_len % 2 == 0

    def __call__(self, i):
        return self._P[i]

    def __mul__(self, other):
        return PermutationBase([self.P[q] for q in other.P])

    def __pow__(self, other):
        if isinstance(other, int):
            X = [_ for _ in range(len(self.P))]
            if other < 0:
                P = [self.P.index(i) for i in range(len(self.P))]
                count = 0
                while count < abs(other):
                    X = [P[x] for x in X]
                    count += 1
            else:
                count = 0
                while count < other:
                    X = [self(x) for x in X]
                    count += 1
            return PermutationBase(X)
        else:
            raise TypeError("Unsupported type for exponentiation")

    def __invert__(self):
        return PermutationBase([self.P.index(i) for i in range(len(self.P))])

    def compute_parity(self):
        need2visit = set(self.P)
        parity = True
        while need2visit:
            i_start = need2visit.pop()
            i = self.P[i_start]
            while i != i_start:
                parity = not parity
                need2visit.discard(i)
                i = self.P[i]
        return parity

    def __eq__(self, other):
        return self.P == other.P

    def __ne__(self, other):
        return self.P != other.P

    def __hash__(self):
        return hash(tuple(self.P))

    def __repr__(self):
        return str(self.P)


# @dataclass
# class CycleRep:
#     cycles: list
class _Permutation:
    """
    An element of the symmetric group sortable things.

    Attributes
    ----------
    P : iterable of sortable
        e.g. permutation of [0,...,n-1]
    cycles : set of tuples
        cycle decomposition
    parity : bool
        True/False for even/odd
    """

    def __init__(self, cycles=None, refresh_cache=False):

        self.cycles = cycles
        if refresh_cache:
            self.refresh_cache()
            self.cache_is_valid = True
        else:
            self.cached_list = []
            self.cache_is_valid = False
        # self.cache_parity = cache_parity
        # self.cache_list = cache_list
        # self.cache_is_valid = False

    @property
    def cached_list(self):
        return self._cached_list

    @cached_list.setter
    def cached_list(self, value):
        self._cached_list = list(value)

    @property
    def cycles(self):
        return self._cycles

    def refresh_cache(self):
        pass

    def compute_cycles(self):
        """
        Compute cycles of a permutation

        Parameters
        ----------
        P : list of int
            Permutation [0,...,len(P)-1]

        Returns
        -------
        cycles : set of tuple of int
        parity : True for even permutation, False for odd for false
        """
        cycles = set()
        need2visit = set(self.P)
        if need2visit != set(range(len(self.P))):
            raise ValueError("P must be a permutation of [0,...,len(P)-1]")
        # cycle_num = 0 # number of cycles
        # cycle_len = 0 # sum of cycle lengths
        parity = True
        while need2visit:
            # cycle_num += 1 # start a new cycle
            # parity = not parity * this cancels **
            i_start = need2visit.pop()
            cycle = [i_start]
            # cycle_len += 1 # count first element in the cycle i=i_start
            # parity = not parity ** this cancels *
            i = self.P[i_start]
            while i != i_start:
                # cycle_len += 1 # count element i in the cycle
                cycle.append(i)
                parity = not parity
                need2visit.discard(i)  # remove visited element
                i = self.P[i]  # next element in the cycle
            cycles.add(tuple(cycle))
        return cycles, parity  # cycle_num - cycle_len % 2 == 0

    def __call__(self, i):
        return self._P[i]

    def __mul__(self, other):
        return PermutationBase([self.P[q] for q in other.P])

    def __pow__(self, other):
        if isinstance(other, int):
            X = [_ for _ in range(len(self.P))]
            if other < 0:
                P = [self.P.index(i) for i in range(len(self.P))]
                count = 0
                while count < abs(other):
                    X = [P[x] for x in X]
                    count += 1
            else:
                count = 0
                while count < other:
                    X = [self(x) for x in X]
                    count += 1
            return PermutationBase(X)
        else:
            raise TypeError("Unsupported type for exponentiation")

    def __invert__(self):
        return PermutationBase([self.P.index(i) for i in range(len(self.P))])

    def compute_parity(self):
        need2visit = set(self.P)
        parity = True
        while need2visit:
            i_start = need2visit.pop()
            i = self.P[i_start]
            while i != i_start:
                parity = not parity
                need2visit.discard(i)
                i = self.P[i]
        return parity

    def __eq__(self, other):
        return self.P == other.P

    def __ne__(self, other):
        return self.P != other.P

    def __hash__(self):
        return hash(tuple(self.P))

    def __repr__(self):
        return str(self.P)


##########################################
# Symmetric group on [...list of objects...]
def right_action(P, X):
    """
    Right action of a permutation P on a list X.

    Parameters
    ----------
    P : list of int
        Permutation [0,...,N-1]
    X : list of objects with len(X)=N

    Returns
    -------
    permutation of X
    """
    return [X[P[i]] for i in range(len(P))]


def left_action(P, X):
    """
    Left action of a permutation P on a list X. Equivalent to right_action(inverse(P), X).

    Parameters
    ----------
    P : list of int
        Permutation [0,...,N-1]
    X : list of objects with len(X)=N

    Returns
    -------
    permutation of X
    """
    return [X[P.index(i)] for i in range(len(P))]


def arg_right_action(Xsource, Xtarget):
    """
    Return permutation P that maps Xsource to Xtarget by right action:

        Xtarget[i] = (P*Xsource)[i]=Xsource[P[i]].

    Parameters
    ----------
    Xsource : list
        each element of Xsource must be unique (no duplicates)
    Xtarget : list
        permutation of Xsource

    Returns
    -------
    list of int : list of source indices in the order they appear in target

    Example
    -------
    Xsource=[a, b, c, d], Xtarget=[b, a, d, c]
            [0, 1, 2, 3]        P=[1, 0, 3, 2]

    Notes
    -----
    arg_right_action(Xsource, Xtarget) = [Xsource.index(x) for x in Xtarget]
                                       = [Xsource.index(Xtarget[i]) for i in Zn]
    """
    return [Xsource.index(x) for x in Xtarget]


def arg_left_action(Xsource, Xtarget):
    """
    Return permutation P that maps Xsource to Xtarget by left action (inverse right action):

        Xtarget[i] = (P*Xsource)[i]=Xsource[P^-1[i]].

    Parameters
    ----------
    Xsource : list
        each element of Xsource must be unique (no duplicates)
    Xtarget : list
        permutation of Xsource

    Returns
    -------
    list of int : list of target indices in the order they appear in source

    """
    return [Xtarget.index(x) for x in Xsource]


def argsort(X, reverse=False):
    """
    Returns permutation of the indices of X, that sorts X. Equivalent to arg_right_action(Xsource, Xtarget) with Xsource=X and Xtarget=sorted(X).

    sorted(X)= [X[i] for i in argsort(X)]

    Parameters
    ----------
    X : list of objects which have __lt__() method

    Returns
    -------
    list of int : list of indices
    """
    return sorted(range(len(X)), key=X.__getitem__, reverse=reverse)


def parity_of_argsort(X):
    """
    Determine parity of a permutation of U which sorts the

    Parameters
    ----------
    X : list of objects which have __lt__() method

    Returns
    -------
    bool : True for even permutation, False for odd for false
    """
    return parity(argsort(X))


##########################################
# Permutation group on Zn=[0,...,n-1]


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


##########################################
# alternative defs


def parity0(P):
    """
    Compute parity of a permutation. Returns True/False for even/odd. The parity of a permutation is equal to the parity of cycle_len-cycle_num where cycle_num is the number of cycles in the permutation and cycle_len is sum of all cycle lengths.

    Parameters
    ----------
    P : list of int
        Permutation [0,...,len(P)-1]

    Returns
    -------
    bool : True for even permutation, False for odd for false
    """
    visited = [False] * len(P)
    parity = True
    for i in range(len(P)):
        if visited[i]:
            continue
        visited[i] = True
        j = P[i]
        while j != i:
            parity = not parity
            visited[j] = True
            j = P[j]
    return parity


def parity_of_argsort0(U):
    """
    Determine parity of a permutation which sorts the list U using a cycle detection algorithm.

    parity of a permutation =
        - parity of the number of cycles of even length
        - (-1)^(n-c) where n sum of cycle lengths of the list and c is the number of cycles

    Parameters
    ----------
    U : list of objects which have __lt__() method

    Returns
    -------
    bool : True for even permutation, False for odd for false
    """
    # If V=sorted(U), then J=[U.index(v) for v in V]
    J = sorted(range(len(U)), key=U.__getitem__)
    visited = [False] * len(U)
    parity = True
    for i in range(len(U)):
        if visited[i]:
            continue
        visited[i] = True
        j = J[i]
        while j != i:
            parity = not parity
            visited[j] = True
            j = J[j]
    return parity


def relative_parity0(P1, P2):
    """Equivalence relation on the set of lists of integers. Two lists are equivalent if they are related by an even permutation

    Parameters
    ----------
    P1 : list of int
    P2 : list of int

    Returns
    -------
    bool : True if the lists are equivalent, False otherwise
    """
    J = sorted(range(len(P1)), key=lambda i: P2.index(P1[i]))

    return parity(J)


def relative_parity1(list1, list2):
    """Equivalence relation on the set of lists of integers. Two lists are equivalent if they are related by an even permutation

    Parameters
    ----------
    list1 : list of int
    list2 : list of int

    Returns
    -------
    bool : True if the lists are equivalent, False otherwise
    """
    if set(list1) == set(list2):
        return parity_of_argsort(list1) == parity_of_argsort(list2)

    return set(list1) == set(list2) and parity_of_argsort(list1) == parity_of_argsort(
        list2
    )


###########################################################################################
# Pairing and packing functions
# -------------
#
###########################################################################################
def szudzik_pairing(a, b):
    return a * a + a + b if a >= b else a + b * b


def inverse_szudzik_pairing(z):
    sqrt_z = int(z**0.5)
    remainder = z - sqrt_z * sqrt_z
    if remainder < sqrt_z:
        return remainder, sqrt_z
    else:
        return sqrt_z, remainder - sqrt_z


def szudzik_packing(*args):
    if len(args) > 2:
        return szudzik_pairing(args[0], szudzik_packing(*args[1:]))
    elif len(args) == 1:
        return args[0]
    else:
        return szudzik_pairing(*args)


def szudzik_unpacking(*args, target_len=2):
    while len(args) < target_len:
        args = args[:-1] + inverse_szudzik_pairing(args[-1])
    return args


def cantor_pairing(a, b):
    """
    Pair two integers a and b into a single integer using the Cantor pairing function.

    Parameters
    ----------
    a : int
    b : int

    Returns
    -------
    int : Cantor pairing of a and b
    """
    return (a + b) * (a + b + 1) // 2 + b


def inverse_cantor_pairing(z):
    """
    Inverse of the Cantor pairing function.

    Parameters
    ----------
    z : int
        Cantor pairing of a and b

    Returns
    -------
    tuple of int : a, b
    """
    w = int((8 * z + 1) ** 0.5)
    t = (w - 1) // 2
    y = z - t * (t + 1) // 2
    x = t - y
    return x, y


def cantor_packing_right2left(*args):
    """
    Pack a list of integers into a single integer using the Cantor pairing function.

    Parameters
    ----------
    *args : list of int

    Returns
    -------
    int : Cantor pairing of *args
    """
    if len(args) > 2:
        return cantor_pairing(args[0], cantor_packing_right2left(*args[1:]))
    else:
        return cantor_pairing(*args)


def cantor_packing_left2right(*args):
    """
    Pack a list of integers into a single integer using the Cantor pairing function.

    cantor_packing(a, b, c)=cantor_pairing(cantor_pairing(a, b), c)
    Parameters
    ----------
    *args : list of int

    Returns
    -------
    int : Cantor pairing of *args
    """
    if len(args) > 2:
        return cantor_pairing(cantor_packing_left2right(*args[:-1]), args[-1])
    else:
        return cantor_pairing(*args)
