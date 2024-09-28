# from src.python.half_edge_base_brane import Brane
# from src.python.half_edge_base_viewer import MeshViewer
from src.python.stetch_sim import StretchSim  # , Spindle, SPB, Envelope, ParamManager

from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np

# from src.python.pretty_pictures import RGBA_DICT

yaml_path = "./data/parameters.yaml"

sim = StretchSim.from_parameters_file(yaml_path, overwrite_output_dir=True)
# sim.run()
# sim.update(patch=True, force=True, pretty=True)
# sim.plot(save=False, show=True, title="")
# sim.evolve_for_DT(1e-2, 1e-3)
# %%

m = sim.envelope
mv = sim.mesh_viewer
mv.plot()
# %%
from src.python.half_edge_base_brane import Brane
from src.python.half_edge_base_viewer import MeshViewer
from src.python.half_edge_base_patch import HalfEdgePatch
import numpy as np
from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
m = Brane.load(ply_path=ply_path)
v = 1322
p1 = HalfEdgePatch.from_seed_to_radius(v, m, 0.3)
p2 = HalfEdgePatch.from_seed_to_radius(v, m, 0.2)
p0 = HalfEdgePatch.from_seed_to_radius(v, m, 0.1)
F = p2.F - p0.F
H = p2.H - p0.H
V = p2.V - p0.V
V0, H0, F0 = m.closure(V.copy(), H.copy(), F.copy())
V1, H1, F1 = m.closure1(V.copy(), H.copy(), F.copy())
V2, H2, F2 = m.closure2(V.copy(), H.copy(), F.copy())
[
    V1.symmetric_difference(V0),
    V2.symmetric_difference(V1),
    V0.symmetric_difference(V2),
    H1.symmetric_difference(H0),
    H2.symmetric_difference(H1),
    H0.symmetric_difference(H2),
    F1.symmetric_difference(F0),
    F2.symmetric_difference(F1),
    F0.symmetric_difference(F2),
]

# %timeit m.closure(V.copy(), H.copy(), F.copy())
# %timeit m.closure1(V.copy(), H.copy(), F.copy())
# %timeit m.closure2(V.copy(), H.copy(), F.copy())


# %%
from src.python.half_edge_base_brane import Brane

from src.python.half_edge_base_viewer import MeshViewer

# from src.python.half_edge_base_patch import HalfEdgePatch
from src.python.half_edge_mesh import HalfEdgeMeshBase, HalfEdgeBoundary
import numpy as np

# from mayavi import mlab

ply_path = "./data/half_edge_base/ply/unit_sphere_005120_he.ply"
ply_path = "./data/half_edge_base/ply/neovius_he.ply"
m = HalfEdgeMeshBase.load(ply_path=ply_path)

mesh = m
# bigH = np.arange(mesh.num_half_edges)
H_minus = mesh.negative_boundary_contains_h(range(mesh.num_half_edges)).nonzero()[0]
nextH_minus = mesh.h_next_h(H_minus)
H_plus = mesh.h_twin_h(nextH_minus)
nextH_plus = mesh.h_twin_h(H_minus)

mv = MeshViewer(m, show_half_edges=True)
# mv.update(rgba_half_edge=(0, 0, 0, 0))
mv.update_rgba_H((0, 0, 0, 0))
mv.update_rgba_H(np.array([1, 0, 0, 1]), H_plus)
mv.plot()
# %%
from src.python.combinatorics import arg_right_action

P = arg_right_action(list(H_plus), list(nextH_plus))
nextH_plus - H_plus[P]


def arg_right_action0(Xsource, Xtarget):
    """
    Return permutation P that maps Xsource to Xtarget by right action:

        Xtarget[i] = (P*Xsource)[i]=Xsource[P[i]].

    Parameters
    ----------
    Xsource : array-like
        each element of Xsource must be unique (no duplicates)
    Xtarget : array-like
        permutation of Xsource

    Returns
    -------
    numpy.ndarray : array of source indices in the order they appear in target

    Example
    -------
    Xsource=[a, b, c, d], Xtarget=[b, a, d, c]
            [0, 1, 2, 3]        P=[1, 0, 3, 2]

    Notes
    -----
    arg_right_action(Xsource, Xtarget) = [Xsource.index(x) for x in Xtarget]
                                       = [Xsource.index(Xtarget[i]) for i in Zn]
    """
    Xsource = np.array(Xsource)
    Xtarget = np.array(Xtarget)

    # Create an array of indices for Xsource
    indices = np.argsort(Xsource)

    # Map Xtarget to the indices of Xsource
    return indices[np.searchsorted(Xsource, Xtarget, sorter=indices)]


P0 = list(arg_right_action0(H_plus, nextH_plus))

P0 - P
