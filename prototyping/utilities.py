import itertools


class KeyCounter:
    """
    Key counter for generating unique key that increments with each call

    Properties
    ----------
    count : int
        integer-valued key

    Methods
    -------
    __call__(self) -> int
        return key and increment count
    """

    def __init__(self, start=0):
        self._count = start

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

    def __call__(self):
        key = self._count
        self._count += 1
        return key


class KeyManager:
    """
    Key manager which assigns a KeyCounter to each new key_type, or returns the existing KeyCounter for that key_type if it already exists.

    Properties
    ----------
    key_counter_dict : dict of (key_type, KeyCounter) key-value pairs
        {key_type0: KeyCounter0, key_type1: KeyCounter1,...}

    Methods
    -------
    __call__(self, key_type) -> int
        return and increment integer-valued key for key_type

    __getitem__(self, key_type) -> KeyCounter
        return KeyCounter instance for key_type
    """

    def __init__(self, key_types=[]):
        self._key_counter_dict = {key_type: KeyCounter() for key_type in classes}

    def __call__(self, key_type):
        try:
            return self._key_counter_dict[key_type]()
        except KeyError:
            self._key_counter_dict[key_type] = KeyCounter()
            return self._key_counter_dict[key_type]()

    def __getitem__(self, key_type):
        try:
            return self._key_counter_dict[key_type]
        except KeyError:
            self._key_counter_dict[key_type] = KeyCounter()
            return self._key_counter_dict[key_type]


def argsort(seq):
    """
    Return the list of indices that would sort the list seq.

    Parameters
    ----------
    seq : list of objects which have __lt__() method

    Returns
    -------
    list of int : list of indices
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def parity_of_sort_permutation(U):
    """
    Determine parity of a permutation which sorts the list U using a cycle detection algorithm.

    parity of a permutation =
        - parity of the number of cycles of even length
        - (-1)^(n-c) where n is the length of the list and c is the number of cycles

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
