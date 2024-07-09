from src.python.ply_tools import SphereFactory, TorusFactory
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.sym_tools import UniformCoordinateInterval, Samples1D, Samples2D, Samples, alphaseq_symbols
import sympy as sp
import numpy as np
from sympy.abc import a, b, c, i, j, k, l, m, n, s, t, u, v, w, x, y, z

a, b, n = 1.1, 22.6, 55
s = Samples1D.linspace(a, b, n)
# s.apply_index_transform(lambda i: i+1)
s(5)

s
S = [
    [Samples1D.linspacebc(a, b, include_lower=bool(i), include_upper=bool(j), num=n) for j in range(2)]
    for i in range(2)
]
s00 = Samples1D.linspacebc(a, b, include_lower=bool(0), include_upper=bool(0), num=n)
s01 = Samples1D.linspacebc(a, b, include_lower=bool(0), include_upper=bool(1), num=n)
s00 = Samples1D.linspacebc(a, b, include_lower=bool(1), include_upper=bool(0), num=n)
s01 = Samples1D.linspacebc(a, b, include_lower=bool(1), include_upper=bool(1), num=n)

S[0][0]

# %%

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
# %%
