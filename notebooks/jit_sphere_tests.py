from src.python.sphere_builder import (
    load_refine_icososphere_and_save_output,
    SphereFactory,
    cotan_laplacian,
    belkin_laplacian,
    jit_vf_samples_to_he_samples,
)
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
import numpy as np
from timeit import default_timer


SF = load_refine_icososphere_and_save_output(
    9,
    output_dir="./output/sphere_builder",
    save_name="refinement",
)
SF[-2].num_vertices()
# M = [HalfEdgeMesh(*sf.HE()) for sf in SF]
# %%
sf = SF[4]
Nv = sf.num_vertices()
s = 4 * np.pi * sf.radius**2 / Nv
Q = sf.VF()[0]
lapQbelkin = belkin_laplacian(Q, s, *sf.HE())
lapQcotan = cotan_laplacian(Q, *sf.HE())
np.linalg.norm(lapQbelkin - lapQcotan, axis=-1)
np.linalg.norm(lapQcotan, axis=-1)
# %timeit belkin_laplacian(Q, s, *sf.HE())
# %timeit cotan_laplacian(Q, *sf.HE())
#
# 1.28/(218e-6)
# %%
from numba.experimental import jitclass
from numba import typeof, from_dtype
from numba.typed import (
    List,
    Dict,
    Tuple,
)
from numba.types import (
    unicode_type,  # str
    boolean,
    int32,
    int64,
    float64,
    ListType,
    DictType,
    Array,
)
import numpy as np
from src.python.sphere_builder import (
    load_refine_icososphere_and_save_output,
    SphereFactory,
    cotan_laplacian,
    belkin_laplacian,
    jit_vf_samples_to_he_samples,
)
import os


# os.environ['NUMBA_DEBUG'] = '1'


sf_spec = [
    ("V0", float64[:, :]),
    ("F0", int32[:, :]),
    ("V", float64[:, :]),
    ("F", ListType(Array(int32, 2, "C"))),
    ("_name", unicode_type),
    ("h_out_V", ListType(int32[:])),
    ("v_origin_H", ListType(int32[:])),
    ("h_next_H", ListType(int32[:])),
    ("h_twin_H", ListType(int32[:])),
    ("f_left_H", ListType(int32[:])),
    ("h_bound_F", ListType(int32[:])),
    ("_num_vertices", ListType(int32[:])),
    ("t_refine", ListType(float64)),
    ("refine_level", int32),
]


@jitclass(sf_spec)
class jitSphereFactory:
    """
    subdivides icosahedron (20 triangles; 12 vertices) to create meshes of unit sphere

    after k refinements we have

    |V|=10*4^k+2
    |E|=30*4^k
    |F|=20*4^k
    """

    def __init__(self, name):
        phi = (1.0 + np.sqrt(5.0)) * 0.5  # golden ratio
        a = 1.0
        b = 1.0 / phi
        a = a / np.sqrt(a**2 + b**2)
        b = b / np.sqrt(a**2 + b**2)
        V0 = np.array(
            [
                [0.0, b, -a],
                [b, a, 0.0],
                [-b, a, 0.0],
                [0.0, b, a],
                [0.0, -b, a],
                [-a, 0.0, b],
                [0.0, -b, -a],
                [a, 0.0, -b],
                [a, 0.0, b],
                [-a, 0.0, -b],
                [b, -a, 0.0],
                [-b, -a, 0.0],
            ],
            dtype=np.float64,
        )
        F0 = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
                [5, 4, 3],
                [4, 8, 3],
                [7, 6, 0],
                [6, 9, 0],
                [11, 10, 4],
                [10, 11, 6],
                [9, 5, 2],
                [5, 9, 11],
                [8, 7, 1],
                [7, 8, 10],
                [2, 5, 3],
                [8, 1, 3],
                [9, 2, 0],
                [1, 7, 0],
                [11, 9, 6],
                [7, 10, 6],
                [5, 11, 4],
                [10, 8, 4],
            ],
            dtype=np.int32,
        )

        (
            V,
            _h_out_V,
            _v_origin_H,
            _h_next_H,
            _h_twin_H,
            _f_left_H,
            _h_bound_F,
        ) = jit_vf_samples_to_he_samples(V0, F0)
        #############################################
        self.V0 = V0
        self.F0 = F0
        self.V = V
        self.F = List([F0])

        self._name = name
        # self.h_out_V = List([_h_out_V])

    #     self.v_origin_H = [_v_origin_H]
    #     self.h_next_H = [_h_next_H]
    #     self.h_twin_H = [_h_twin_H]
    #     self.f_left_H = [_f_left_H]
    #     self.h_bound_F = [_h_bound_F]
    #     self._num_vertices = [len(self.V)]
    #     self.t_refine = []
    #     self.refine_level = 0
    #
    @property
    def name(self):
        return self._name

    #
    # def VF(self, level=-1):
    #     return self.V[: self.num_vertices(level)], self.F[level]
    #
    # def HE(self, level=-1):
    #     return (
    #         self.V[: self.num_vertices(level)],
    #         self.h_out_V[level],
    #         self.v_origin_H[level],
    #         self.h_next_H[level],
    #         self.h_twin_H[level],
    #         self.f_left_H[level],
    #         self.h_bound_F[level],
    #     )
    #
    # def num_vertices(self, level=-1):
    #     return self._num_vertices[level]
    #
    # def next_num_vertices(self):
    #     level = len(self._num_vertices)
    #     # return self._NUM_VERTICES_[level]
    #     return 10 * 4**level + 2


sf = jitSphereFactory("jit_sphere")
F0 = sf.F0
V0 = sf.V0
V_data = np.array(
    [tuple(v) for v in V0],
    dtype=[("x", "double"), ("y", "double"), ("z", "double")],
)
F_data = np.empty(len(F0), dtype=[("vertex_indices", "int32", (3,))])
V_type = np.dtype([("x", "double"), ("y", "double"), ("z", "double")])
F_type = np.dtype([("vertex_indices", "int32", (3,))])

vertex_index_numba_type = int32
halfedge_index_numba_type = int32
face_index_numba_type = int32
xyz_numba_type = Array(
    from_dtype(np.dtype([("x", "float64"), ("y", "float64"), ("z", "float64")])), 1, "C"
)
face_numba_type = Array(
    from_dtype(np.dtype([("vertex_indices", "int32", (3,))])), 1, "C"
)
hedge_numba_type = Array(
    from_dtype(np.dtype([("vertex_indices", "int32", (2,))])), 1, "C"
)

xyz_coord_V_numba_type = DictType(vertex_index_numba_type, xyz_numba_type)
h_out_V_numba_type = DictType(vertex_index_numba_type, halfedge_index_numba_type)
v_origin_H_numba_type = DictType(halfedge_index_numba_type, vertex_index_numba_type)
h_next_H_numba_type = DictType(halfedge_index_numba_type, halfedge_index_numba_type)
h_twin_H_numba_type = DictType(halfedge_index_numba_type, halfedge_index_numba_type)
f_left_H_numba_type = DictType(halfedge_index_numba_type, face_index_numba_type)
h_bound_F_numba_type = DictType(face_index_numba_type, halfedge_index_numba_type)


xyz_coord_V = Dict.empty(
    key_type=vertex_index_numba_type,
    value_type=xyz_numba_type,
)

# %%
#
#
phi = (1.0 + np.sqrt(5.0)) * 0.5  # golden ratio
a = 1.0
b = 1.0 / phi
a = a / np.sqrt(a**2 + b**2)
b = b / np.sqrt(a**2 + b**2)
V0 = np.array(
    [
        [0.0, b, -a],
        [b, a, 0.0],
        [-b, a, 0.0],
        [0.0, b, a],
        [0.0, -b, a],
        [-a, 0.0, b],
        [0.0, -b, -a],
        [a, 0.0, -b],
        [a, 0.0, b],
        [-a, 0.0, -b],
        [b, -a, 0.0],
        [-b, -a, 0.0],
    ],
    dtype=np.float64,
)
F0 = np.array(
    [
        [2, 1, 0],
        [1, 2, 3],
        [5, 4, 3],
        [4, 8, 3],
        [7, 6, 0],
        [6, 9, 0],
        [11, 10, 4],
        [10, 11, 6],
        [9, 5, 2],
        [5, 9, 11],
        [8, 7, 1],
        [7, 8, 10],
        [2, 5, 3],
        [8, 1, 3],
        [9, 2, 0],
        [1, 7, 0],
        [11, 9, 6],
        [7, 10, 6],
        [5, 11, 4],
        [10, 8, 4],
    ],
    dtype=np.int32,
)

(
    V,
    _h_out_V,
    _v_origin_H,
    _h_next_H,
    _h_twin_H,
    _f_left_H,
    _h_bound_F,
) = jit_vf_samples_to_he_samples(V0, F0)
typeof(V)
typeof(_h_out_V)
#
#
#
