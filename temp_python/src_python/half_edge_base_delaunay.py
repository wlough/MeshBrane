import numpy as np
from scipy.spatial import Delaunay
from python.half_edge_mesh import HalfEdgeMeshBase
from temp_python.src_python.half_edge_base_viewer import MeshViewer
from temp_python.src_python.jit_utils import (
    check_vf_list_orientation,
    fib_sphere,
    uniform_sphere,
)
from numba import jit


@jit
def extract_surface_faces(tetrahedra):
    """
    Extract the 2D triangular faces on the surface from the 3D tetrahedra.

    Parameters:
    tetrahedra (ndarray): An array of tetrahedra, each defined by four vertex indices.

    Returns:
    list: A list of unique surface faces, each defined by three vertex indices.
    """
    face_count = {}

    def add_face(face):
        face = sorted(face)
        face = (face[0], face[1], face[2])
        if face in face_count:
            face_count[face] += 1
        else:
            face_count[face] = 1

    # List all faces of each tetrahedron
    for tet in tetrahedra:
        add_face([tet[0], tet[1], tet[2]])
        add_face([tet[0], tet[1], tet[3]])
        add_face([tet[0], tet[2], tet[3]])
        add_face([tet[1], tet[2], tet[3]])

    # Extract faces that appear exactly once
    surface_faces = [face for face, count in face_count.items() if count == 1]

    return np.array(surface_faces)


# from numba.typed import List, Dict, Tuple
@jit
def jit_extract_surface_faces(tetrahedra):
    """
    Extract the 2D triangular faces on the surface from the 3D tetrahedra.

    Parameters:
    tetrahedra (ndarray): An array of tetrahedra, each defined by four vertex indices.

    Returns:
    list: A list of unique surface faces, each defined by three vertex indices.
    """
    # Ntets = len(tetrahedra)
    # Fcount = np.zeros(4 * Ntets)
    face_count = {}

    def add_face(face):
        face = tuple(sorted(face))
        if face in face_count:
            face_count[face] += 1
        else:
            face_count[face] = 1

    # List all faces of each tetrahedron
    for tet in tetrahedra:
        faces = [
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ]
        for face in faces:
            if face in face_count:
                face_count[face] += 1
            else:
                face_count[face] = 1

    # Extract faces that appear exactly once
    surface_faces = [face for face, count in face_count.items() if count == 1]

    return np.array(surface_faces)


V = fib_sphere(10)
# V = fib_sphere(10).tolist()
tets = Delaunay(V).simplices
F1 = extract_surface_faces(tets)
F2 = jit_extract_surface_faces(tets)
sorted(set(F2.ravel()))
F1 = check_vf_list_orientation(V, F1)
m1 = HalfEdgeMeshBase.from_vf_data(V, F1)
m2 = HalfEdgeMeshBase.from_vf_data(V, F2)
# %%
num_refine = 6
Nv = np.array([10 * 4**k + 2 for k in range(num_refine)], dtype=np.int32)
newNv = Nv[1:] - Nv[:-1]
maxV = fib_sphere(Nv[-1])
_bigV = [maxV[:nv] for nv in Nv]

bigD = [Delaunay(_, qhull_options="QJ") for _ in _bigV]
bigT = [_.simplices for _ in bigD]


_bigF = [np.array(extract_surface_faces(tets)) for tets in bigT]
_bigVind = [np.array(sorted(set(_F.ravel()))) for _F in _bigF]
bigV = [_V[_Vind] for _V, _Vind in zip(_bigV, _bigVind)]
bigVsubs = [{vold: vnew for vnew, vold in enumerate(_Vind)} for _Vind in _bigVind]
_bigF = [
    np.array([[Vsubs[vold] for vold in face] for face in _F], dtype=np.int32)
    for Vsubs, _F in zip(bigVsubs, _bigF)
]
bigF = [check_vf_list_orientation(V, _F) for V, _F in zip(bigV, _bigF)]

Mfib = [HalfEdgeMeshBase.from_vert_face_list(V, F) for V, F in zip(bigV, bigF)]
Nflipsfib = [m.flip_non_delaunay() for m in Mfib]

plys = [f"./data/ply/binary/unit_sphere_{N:06d}_he.ply" for N in Nv]
Mico = [HalfEdgeMeshBase.from_half_edge_ply(ply) for ply in plys]
nv = 0
mfib = Mfib[nv]
mico = Mico[nv]

# %%
import numpy as np
from scipy.spatial import Delaunay
from temp_python.src_python.half_edge_mesh import HalfEdgeMesh
from temp_python.src_python.mesh_viewer import MeshViewer
from temp_python.src_python.jit_utils import (
    check_vf_list_orientation,
    fib_sphere,
    uniform_sphere,
)

V = uniform_sphere(10)
V = uniform_sphere(10)
V = uniform_sphere(10).tolist()
F = Delaunay(V).simplices[:, 1:].tolist()

num_refine = 5
Nv = np.array([10 * 4**k + 2 for k in range(num_refine)], dtype=np.int32)
newNv = Nv[1:] - Nv[:-1]
maxV = uniform_sphere(Nv[-1])
_bigV = [maxV[:nv] for nv in Nv]

bigD = [Delaunay(_, qhull_options="QJ") for _ in _bigV]
bigT = [_.simplices for _ in bigD]


_bigF = [np.array(extract_surface_faces(tets)) for tets in bigT]
_bigVind = [np.array(sorted(set(_F.ravel()))) for _F in _bigF]
bigV = [_V[_Vind] for _V, _Vind in zip(_bigV, _bigVind)]
bigVsubs = [{vold: vnew for vnew, vold in enumerate(_Vind)} for _Vind in _bigVind]
_bigF = [
    np.array([[Vsubs[vold] for vold in face] for face in _F], dtype=np.int32)
    for Vsubs, _F in zip(bigVsubs, _bigF)
]
bigF = [check_vf_list_orientation(V, _F) for V, _F in zip(bigV, _bigF)]

Muni = [HalfEdgeMesh.from_vert_face_list(V, F) for V, F in zip(bigV, bigF)]
Nflipsuni = [m.flip_non_delaunay() for m in Muni]

plys = [f"./data/ply/binary/unit_sphere_{N:06d}_he.ply" for N in Nv]
Mico = [HalfEdgeMesh.from_half_edge_ply(ply) for ply in plys]
# %%

nv = 4
muni = Muni[nv]
mico = Mico[nv]
m = muni
mv = MeshViewer(*m.data_lists)
mv.plot()
# %%
self = m
m.flip_edge(13)
flip_count = 0
for h in self.Hkeys:
    if not self.h_is_locally_delaunay(h):
        print(h)
        if self.h_is_flippable(h):
            self.flip_edge(h)
            flip_count += 1


# _F = np.array(surface_faces)
# _Vind = np.array(sorted(set(_F.ravel())))
# V = points[_Vind]
# Vsubs = {vold: vnew for vnew, vold in enumerate(_Vind)}
# _F = np.array([[Vsubs[vold] for vold in face] for face in _F], dtype=np.int32)

# V, F = check_vf_list_orientation(V, _F)
# m = HalfEdgeMesh.from_vert_face_list(V, F)

# num_flips = m.flip_non_delaunay()
# mv = MeshViewer(*m.data_lists)
# mv.plot()
