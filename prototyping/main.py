# from src.python.dart_group import Tetrahedron, CombinatorialSimplex
import random
from itertools import combinations, permutations
from src.python.combinatorics import (
    compose,
    parity,
    inverse,
    cycle_decomp,
    relative_parity,
    argsort,
    parity_of_argsort,
    relative_permutation,
    PermutationBase,
)
import numpy as np

Px = [2, 0, 1, 3]
Py = [3, 2, 0, 1]

X = PermutationBase(Px)
Y = PermutationBase(Py)
X.cycles
Y.cycles
Z = X * X


# %%
from plyfile import PlyData, PlyElement, PlyProperty, PlyListProperty
import numpy as np

# Define the vertex element
vertex = np.array(
    [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], dtype=[("x", "double"), ("y", "double"), ("z", "double")]
)

# face = np.array([[0, 1, 2], [3, 4, 5], [1, 2, 3]], dtype=[("vertex_indices", "int32", ("uint32",))])
vertex_element = PlyElement.describe(vertex, "vertex")

# Define the face element
face = np.array([([0, 1, 2],), ([0, 2, 3],)], dtype=[("vertex_indices", "int32", (3,))])

face_element = PlyElement.describe(face, "face")

# Create the PlyData object
ply_data = PlyData([vertex_element, face_element])


# Write the PLY file
ply_data.write("./output/output.ply")


"""
describe(data, name, len_types={}, val_types={}, comments=[])

Parameters
data: numpy.ndarray
Structured numpy array.

len_types: dict, optional
Mapping from list property names to type strings
(numpy-style like 'u1', 'f4', etc., or PLY-style like
'int8', 'float32', etc.), which will be used to encode
the length of the list in binary-format PLY files. Defaults
to 'u1' (8-bit integer) for all list properties.

val_types: dict, optional
Mapping from list property names to type strings as for
len_types, but is used to encode the list elements in
binary-format PLY files. Defaults to 'i4' (32-bit
integer) for all list properties.

comments: list of str
Comments between the "element" line and first property
definition in the header."""
