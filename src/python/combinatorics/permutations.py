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
