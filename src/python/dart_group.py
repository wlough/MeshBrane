from src.python.combinatorics import CombinatorialSimplex


class CombinatorialTetrahedron:
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


class PrimitivePatch:
    def __init__(self, valence, periodic=True):
        self.valence = valence
        self.periodic = periodic


class Dart:
    """
    Oriented
    """
