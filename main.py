from src.python.ply_tools import VertTri2HalfEdgeConverter, SphereBuilder
from src.python.half_edge_mesh import HalfEdgeMesh
from src.python.mesh_viewer import MeshViewer
from src.python.half_edge_ops import CotanLaplaceOperator
from src.python.rigid_body import exp_so3

ply = "/home/wlough/Downloads/torus-coarse.ply"
m = HalfEdgeMesh.from_vertex_face_ply(ply)
# sb = SphereBuilder()
#
# sb._name = "test_sphere"
# sb.write_plys()
# sb.divide_faces()
import numpy as np


def f_surf(x, y, z):
    # return (x**2 + y**2 + z**2 + rad_big**2 - rad_small**2) ** 2 - 4 * rad_big**2 * (x**2 + y**2)
    return (np.sqrt(x**2 + y**2) - rad_big) ** 2 + z**2 - rad_small**2


def Df_surf(x, y, z):
    # xyz = np.array([x, y, z])
    # return 4 * (x**2 + y**2 + z**2 + rad_big**2 - rad_small**2) * xyz - 8 * rad_big**2 * np.array([x, y, 0.0])
    return 2 * (np.sqrt(x**2 + y**2) - rad_big) * np.array([x, y, 0]) / np.sqrt(x**2 + y**2) + np.array(
        [0, 0, 2 * z]
    )


def closest_point(xyz, iterations=18):
    """
    input
    -----
    r = [...,[x[s],y[s],z[s]],...]

    phi*Dphi/normDphi**2 ~ dist_S(r)
    """
    for _ in range(iterations):
        f = f_surf(*xyz)
        Df = Df_surf(*xyz)
        xyz -= f * Df / (Df @ Df)
    return xyz


def divide_faces(V0, F0):
    F = []
    V = V0.copy()
    v_midpt_vv = dict()
    for tri in F0:
        v0, v1, v2 = tri
        v01 = v_midpt_vv.get((v0, v1))
        v12 = v_midpt_vv.get((v1, v2))
        v20 = v_midpt_vv.get((v2, v0))
        p0, p1, p2 = V[v0], V[v1], V[v2]
        n = np.cross(p0, p1) + np.cross(p1, p2) + np.cross(p2, p0)
        n /= np.linalg.norm(n)
        if v01 is None:
            v01 = len(V)
            xyz01 = (V[v0] + V[v1]) / 2 + 0.1 * n
            xyz01 = closest_point(xyz01)
            V.append(xyz01)
            v_midpt_vv[(v0, v1)] = v01
            v_midpt_vv[(v1, v0)] = v01
        if v12 is None:
            v12 = len(V)
            xyz12 = (V[v1] + V[v2]) / 2 + 0.1 * n
            xyz12 = closest_point(xyz12)
            V.append(xyz12)
            v_midpt_vv[(v1, v2)] = v12
            v_midpt_vv[(v2, v1)] = v12
        if v20 is None:
            v20 = len(V)
            xyz20 = (V[v2] + V[v0]) / 2 + 0.1 * n
            xyz20 = closest_point(xyz20)
            V.append(xyz20)
            v_midpt_vv[(v2, v0)] = v20
            v_midpt_vv[(v0, v2)] = v20
        F.append([v0, v01, v20])
        F.append([v01, v1, v12])
        F.append([v20, v12, v2])
        F.append([v01, v12, v20])
    return V, F


R = 1
r = 1 / 3
Nphi = 4
Ntheta = 3
phi = np.array([k * 2 * np.pi / Nphi for k in range(Nphi)])
theta = np.array([k * 2 * np.pi / Ntheta for k in range(Ntheta)])
Ex, Ey, Ez = np.eye(3)
X = r * np.cos(theta)
Y = r * np.sin(theta)
Z = 0 * X


o0, ex0, ey0, ez0 = R * Ex, -Ex, Ez, -Ey
V0 = [o0 + r * np.cos(th) * ex0 + r * np.sin(th) * ey0 for th in theta]

o1, ex1, ey1, ez1 = R * Ey, Ey, -Ez, -Ex
V1 = [o1 + r * np.cos(th) * ex1 + r * np.sin(th) * ey1 for th in theta]

o2, ex2, ey2, ez2 = -R * Ex, Ex, Ez, -Ey
V2 = [o2 + r * np.cos(th) * ex2 + r * np.sin(th) * ey2 for th in theta]

o3, ex3, ey3, ez3 = -R * Ey, -Ey, -Ez, Ex
V3 = [o3 + r * np.cos(th) * ex3 + r * np.sin(th) * ey3 for th in theta]


V0 = [*V0, *V1, *V2, *V3]
F0 = []
for k in range(4):
    am, bm, cm = 3 * k, 3 * k + 1, 3 * k + 2
    ap, bp, cp = (am + 3) % 12, (bm + 3) % 12, (cm + 3) % 12
    F0.extend([[ap, bm, cm], [am, bp, cm], [am, bm, cp], [am, cp, bp], [ap, cp, bm], [ap, cm, bp]])

V, F = divide_faces(V0, F0)

m = HalfEdgeMesh.from_vert_face_list(V, F)

mv = MeshViewer(*m.data_lists, show_vertices=True)

mv.plot()
# %%
rad_big = 1
rad_small = 1 / 3
N_big = 4
N_small = 3
phi_big = np.array([k * 2 * np.pi / N_big for k in range(N_big)])
phi_small = np.array([k * 2 * np.pi / N_small for k in range(N_small)])
Ex, Ey, Ez = np.eye(3)
V_csec = [np.array([rad_small * np.cos(phi), rad_small * np.sin(phi), 0]) for phi in phi_small]
Drot_small = exp_so3(0, 0)

O = [np.array([rad_big * np.cos(phi), rad_big * np.sin(phi), 0]) for phi in phi_big]


o0, ex0, ey0, ez0 = R * Ex, -Ex, Ez, -Ey
V0 = [o0 + r * np.cos(th) * ex0 + r * np.sin(th) * ey0 for th in theta]

o1, ex1, ey1, ez1 = R * Ey, Ey, -Ez, -Ex
V1 = [o1 + r * np.cos(th) * ex1 + r * np.sin(th) * ey1 for th in theta]

o2, ex2, ey2, ez2 = -R * Ex, Ex, Ez, -Ey
V2 = [o2 + r * np.cos(th) * ex2 + r * np.sin(th) * ey2 for th in theta]

o3, ex3, ey3, ez3 = -R * Ey, -Ey, -Ez, Ex
V3 = [o3 + r * np.cos(th) * ex3 + r * np.sin(th) * ey3 for th in theta]


V = [*V0, *V1, *V2, *V3]
F = []
for k in range(4):
    am, bm, cm = 3 * k, 3 * k + 1, 3 * k + 2
    ap, bp, cp = (am + 3) % 12, (bm + 3) % 12, (cm + 3) % 12
    F.extend([[ap, bm, cm], [am, bp, cm], [am, bm, cp], [am, cp, bp], [ap, cp, bm], [ap, cm, bp]])

m = HalfEdgeMesh.from_vert_face_list(V, F)
mv = MeshViewer(*m.data_lists, show_vertices=True)
mv.plot()
np.mean(m.xyz_array, axis=0)


# %%
R = 1
r = 1 / 3
Nphi = 4
Ntheta = 3
phi = np.array([k * 2 * np.pi / Nphi for k in range(Nphi)])
theta = np.array([k * 2 * np.pi / Ntheta for k in range(Ntheta)])
Ex, Ey, Ez = np.eye(3)
X = r * np.cos(theta) - R
Y = r * np.sin(theta)
Z = 0 * X

Rot0 = np.array([-Ex, Ez, Ey]).T
Rot_z = exp_so3(0, 0, np.pi / 2)
Rot_y = exp_so3(0, np.pi, 0)
Rot1 = np.round(Rot_z @ Rot_y @ Rot0)
Rot2 = np.round(Rot_z @ Rot_z @ Rot0)
Rot3 = np.round(Rot_z @ Rot_z @ Rot1)
O = np.array([[R * np.cos(_phi), R * np.sin(_phi), 0] for _phi in phi])
Rot = [Rot0, Rot1, Rot2, Rot3]

V = [rot @ [x, y, z] for x, y, z in zip(X, Y, Z) for rot in Rot]
F = []
for k in range(4):
    am, bm, cm = 3 * k, 3 * k + 1, 3 * k + 2
    ap, bp, cp = (am + 3) % 12, (bm + 3) % 12, (cm + 3) % 12
    F.extend([[ap, bm, cm], [am, bp, cm], [am, bm, cp], [am, cp, bp], [ap, cp, bm], [ap, cm, bp]])

m = HalfEdgeMesh.from_vert_face_list(V, F)
mv = MeshViewer(*m.data_lists, show_vertices=True)
mv.plot()
# %%
import math

# Input parameters
st = 8  # number of times we draw a ring
sl = 15  # number of subdivisions of the ring
innerR = 1 / 3  # inner radius
outerR = 1  # outer radius

# Initialize angles
phi = 0.0
theta = 0.0
dp = (2 * math.pi) / sl  # delta phi
dt = (2 * math.pi) / st  # delta theta

vertices = []
for stack in range(st + 1):  # Including the last element for a closed loop
    theta = dt * stack
    for slice in range(sl + 1):  # Including the last element for a closed loop
        phi = dp * slice
        x = math.cos(theta) * (outerR + math.cos(phi) * innerR)
        y = math.sin(theta) * (outerR + math.cos(phi) * innerR)
        z = math.sin(phi) * innerR
        vertices.append((x, y, z))

tris = []
for stack in range(st):
    for slice in range(sl):
        i1 = stack * (sl + 1) + slice
        i2 = (stack + 1) * (sl + 1) + slice
        i3 = stack * (sl + 1) + (slice + 1)
        i4 = (stack + 1) * (sl + 1) + (slice + 1)
        tris.append((i1, i3, i4))
        tris.append((i1, i4, i2))

# vertices now contains the vertex positions
# tris now contains the indices that form triangles
V = vertices
F = tris
# len(V)
m = HalfEdgeMesh.from_vert_face_list(V, F)

mv = MeshViewer(*m.data_lists, show_vertices=True)

mv.plot()
# %%
import math

# Input parameters
# st = 8  # number of times we draw a ring
# sl = 15  # number of subdivisions of the ring
# innerR = 1  # inner radius
# outerR = 5  # outer radius
rad_big = 1
rad_small = 1 / 3
N_big = 8 + 1
N_small = 15 + 1
phi_big = np.array([k * 2 * np.pi / N_big for k in range(N_big)])
phi_small = np.array([k * 2 * np.pi / N_small for k in range(N_small)])

V = [
    np.array(
        [
            math.cos(phi_b) * (rad_big + math.cos(phi_s) * rad_small),
            math.sin(phi_b) * (rad_big + math.cos(phi_s) * rad_small),
            math.sin(phi_s) * rad_small,
        ]
    )
    for phi_s in phi_small
    for phi_b in phi_big
]
len(V)
F = []

for stack in range(st):
    for slice in range(sl):
        i1 = stack * (sl + 1) + slice
        i2 = (stack + 1) * (sl + 1) + slice
        i3 = stack * (sl + 1) + (slice + 1)
        i4 = (stack + 1) * (sl + 1) + (slice + 1)
        tris.append((i1, i3, i4))
        tris.append((i1, i4, i2))

# vertices now contains the vertex positions
# tris now contains the indices that form triangles
V = vertices
F = tris

m = HalfEdgeMesh.from_vert_face_list(V, F)

mv = MeshViewer(*m.data_lists, show_vertices=True)

mv.plot()
