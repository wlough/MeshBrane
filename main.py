from src.python.ply_tools import SphereFactory, TorusFactory
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.sym_tools import UniformCoordinateInterval, Samples1D, IndexTransform, sym_sorted
import sympy as sp
import numpy as np

S = Samples1D.linspace(3, 5.1, 33)
S = Samples1D.geoseries(3, 5, 55)
S.apply_coord_transform(lambda _: _**2)


# %%
k, a, r, n = sp.symbols("k a r n")
# np.einsum("ij,kjm->ikm",[[1,2],[3,4]], 2*[2*[[1,2]]])
x = a * (1 - r**k) / (1 - r)
num_dict = {
    a: seq_start,
    r: seq_ratio,
    n: num,
}
self=Samples(x, k, num_dict, make_lamfun=False)
syms = [self.index] + sym_sorted(self.num_dict.keys())
if set(syms) > set(self.value_at_index.free_symbols):
    raise ValueError("Mismatched symbols")
set(syms)  set(self.value_at_index.free_symbols)
Samples.index= k
Samples.num_dict
# %%
set([1]) | set([2])
x, xx, y, z = sp.symbols("x xx y z")
np.linspace
isinstance(x, sp.Expr)
l = sym_sorted([z, x, y, xx], skip=[z, sp.Symbol("q")])
sp.Subs()
R = UniformCoordinateInterval(
    lower_bound=0,
    upper_bound=1,
    num_samples="2**p",
    dummy_index="k",
    subs={"p": 6},
    include_lower=False,
    include_upper=True,
)
Theta = UniformCoordinateInterval(
    lower_bound=0,
    upper_bound="pi",
    num_samples="2**p",
    dummy_index="k",
    subs={"p": 6},
    include_lower=False,
    include_upper=False,
)
Phi = UniformCoordinateInterval(
    lower_bound=0,
    upper_bound="2*pi",
    num_samples="2**p",
    dummy_index="k",
    subs={"p": 6},
    include_lower=True,
    include_upper=False,
)


R.numpy_samples - np.linspace(0, 1, 2**6 + 1)[1:]
# Theta.numpy_samples-np.linspace(0,np.pi,2**6+2)[1:-1]*********
Phi.numpy_samples - np.linspace(0, 2 * np.pi, 2**6 + 1)[:-1]
