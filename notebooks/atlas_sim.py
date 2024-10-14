from src.python.half_edge_mesh import (
    HalfEdgeMeshBase,
    HalfEdgeComplex,
    # HalfEdgeSubComplex,
)
from src.python.combinatorics import CombinatorialSimplex, SimplexBase, SimpleChain, SimplicialChain, ZeroChain
import numpy as np

{_: 1 for _ in [1, 1, 2, 3]}
arr = np.random.randint(0, 55, 3)
l = []
for _ in arr:
    if _ in l:
        continue
    else:
        l.append(_)

z = ZeroChain()
s = SimplexBase(l)
sc = SimpleChain(s, -2)
# s0 = SimpleChain(s, 0)
s0 = SimplicialChain([SimpleChain(s, 0)])
s0.is_zero_chain()
3 * s0
c = SimplicialChain([sc])
dc = c.boundary()  # .simplify()
ddc = dc.boundary()
ddc.simplify()

s0 + sc == sc
