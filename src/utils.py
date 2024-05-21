import numpy as np
import sympy as sp
from skimage.measure import marching_cubes
from plyfile import PlyData, PlyElement
from numba import njit
import os

# from src.numdiff import jitcross, jitdot, jitnorm, index_of_nested, transpose_csr
# from src.model import Brane
# import multiprocessing as mu


###########################
# mesh regularization and helper functions
@njit
def index_of_nested(array, element):
    """
    gets the index of subarray in array
    """

    for i, x in enumerate(array):
        if (x == element).all():
            return i

    msg = "element is not in array"
    raise ValueError(msg)


@njit
def get_halfedges(vertices, faces):
    """Builds halfedges from vertices and faces and determines whether the halfedge is
    contained in the boundary of the mesh."""
    halfedges = []
    H_isboundary = []
    ####################
    # save and label halfedges
    h = 0
    for face in faces:
        N_v_of_f = len(face)
        for _ in range(N_v_of_f):
            # index shift to get next
            _next = (_ + 1) % N_v_of_f
            v0 = face[_]  #
            v1 = face[_next]
            hedge = [v0, v1]
            halfedges.append(hedge)
            H_isboundary.append(False)
            h += 1

    for hedge in halfedges:
        v0, v1 = hedge
        hedge_twin = [v1, v0]
        try:
            halfedges.index(hedge_twin)
        except Exception:
            halfedges.append(hedge_twin)
            H_isboundary.append(True)
            h += 1

    return (
        np.array(halfedges, dtype=np.int32),
        np.array(H_isboundary),
    )


@njit
def get_combinatorial_mesh_data(vertices_in, faces_in):
    """builds half edge mesh from vertex+face lists"""
    vertices = vertices_in.copy()
    faces = faces_in.copy()
    Nvertices = len(vertices)
    Nfaces = len(faces)
    halfedges, H_isboundary = get_halfedges(vertices, faces)
    Nhalfedges = len(halfedges)

    #
    # H_isboundary = self.H_isboundary
    # faces = self.faces.copy()
    ####################
    # vertices
    V_hedge = -np.ones(Nvertices, dtype=np.int32)  # outgoing halfedge
    ####################
    # faces
    F_hedge = -np.ones(Nfaces, dtype=np.int32)  # one of the halfedges bounding it
    ####################
    # halfedges
    H_vertex = -np.ones(Nhalfedges, dtype=np.int32)  # vertex it points to
    H_face = -np.ones_like(H_vertex)  # face it belongs to
    # next/previous halfedge inside the face (ordered counter-clockwise)
    H_next = -np.ones_like(H_vertex)
    H_prev = -np.ones_like(H_vertex)
    H_twin = -np.ones_like(H_vertex)  # opposite halfedge
    ####################

    # assign each face a halfedge
    # assign each interior halfedge previous/next halfedge
    # assign each interior halfedge a face
    # assign each halfedge a twin halfedge
    for f in range(Nfaces):
        face = faces[f]
        N_v_of_f = len(face)
        hedge0 = np.array([face[0], face[1]])
        h0 = index_of_nested(halfedges, hedge0)
        F_hedge[f] = h0  # assign each face a halfedge
        for _ in range(N_v_of_f):
            # for each vertex in face, get the indices of the
            # previous/next vertex
            _p1 = (_ + 1) % N_v_of_f
            _m1 = (_ - 1) % N_v_of_f
            vm1 = face[_m1]
            v0 = face[_]
            vp1 = face[_p1]
            # get outgoing halfedge
            hedge = np.array([v0, vp1])
            h = index_of_nested(halfedges, hedge)
            # get incident halfedge
            hedge_prev = np.array([vm1, v0])
            h_prev = index_of_nested(halfedges, hedge_prev)
            # assign previous/next halfedge
            H_prev[h] = h_prev
            H_next[h_prev] = h
            # assign face to halfedge
            H_face[h] = f

            hedge_twin = np.array([vp1, v0])
            h_t = index_of_nested(halfedges, hedge_twin)
            H_twin[h] = h_t
            H_twin[h_t] = h

    # assign each halfedge a vertex
    # assign each vertex a halfedge
    # assign each boundary halfedge previous/next halfedge
    for h in range(Nhalfedges):
        v0, v1 = halfedges[h]
        H_vertex[h] = v1
        if V_hedge[v0] == -1:
            V_hedge[v0] = h
        if H_isboundary[h]:
            h_next = H_twin[h]
            while True:
                h_next = H_twin[H_prev[h_next]]
                if H_isboundary[h_next]:
                    break
            H_next[h] = h_next
            H_prev[h_next] = h

    return (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    )


@njit
def valence(
    v,
    V_hedge,
    H_prev,
    H_twin,
):
    """valence of vertex in halfedge mesh"""
    h_start = V_hedge[v]
    val = 0
    h = h_start
    while True:
        val += 1
        h = H_prev[h]
        h = H_twin[h]
        if h == h_start:
            break
    return val


@njit
def flip_helps_valence(
    h,
    V_hedge,
    H_vertex,
    H_next,
    H_prev,
    H_twin,
):
    r"""
    Returns True if flipping edge 'h' decreases variance
    of the valence of the four vertices illustrated below.
    \sum_{i=1}^4 valence(i)^2/4 - (\sum_{i=1}^4 valence(i))^2/4
    This favors vertices with valence=6 and is equivalent
    returning True if the flip decreases the energy
    \sum_{i in all vertices} (valence(i)-6)^2.
      v2             v2
     / \            /|\
   v3---v1  |---> v3 | v1
     \ /            \|/
      v4             v4
    """

    ht = H_twin[h]
    h1 = H_next[h]
    h3 = H_next[ht]
    v1 = H_vertex[h]
    v2 = H_vertex[h1]
    v3 = H_vertex[ht]
    v4 = H_vertex[h3]

    val1 = valence(
        v1,
        V_hedge,
        H_prev,
        H_twin,
    )
    val2 = valence(
        v2,
        V_hedge,
        H_prev,
        H_twin,
    )
    val3 = valence(
        v3,
        V_hedge,
        H_prev,
        H_twin,
    )
    val4 = valence(
        v4,
        V_hedge,
        H_prev,
        H_twin,
    )

    flip_it = val1 - val2 + val3 - val4 > 2

    return flip_it


@njit
def regularize_by_flips(
    V_hedge,
    halfedges,
    H_vertex,
    H_face,
    H_next,
    H_prev,
    H_twin,
    faces,
    F_hedge,
):
    """flips halfedges in mesh when flip_helps_valence(h)==True"""
    Nh = len(halfedges)
    Nflips = 0
    for h in range(Nh):
        flip_it = flip_helps_valence(
            h,
            V_hedge,
            H_vertex,
            H_next,
            H_prev,
            H_twin,
        )
        if flip_it:
            ht = H_twin[h]
            h1 = H_next[h]
            h2 = H_prev[h]
            h3 = H_next[ht]
            h4 = H_prev[ht]
            f1 = H_face[h]
            f2 = H_face[ht]
            v1 = H_vertex[h4]
            v2 = H_vertex[h1]
            v3 = H_vertex[h2]
            v4 = H_vertex[h3]

            faces[f1] = np.array([v2, v3, v4])
            faces[f2] = np.array([v4, v1, v2])
            # and halfedge referenced by new faces
            F_hedge[f1] = h2
            F_hedge[f2] = h4

            halfedges[h] = np.array([v4, v2])
            halfedges[ht] = np.array([v2, v4])
            # update next/prev halfedge
            H_next[h] = h2
            H_prev[h2] = h

            H_next[h2] = h3
            H_prev[h3] = h2

            H_next[h3] = h
            H_prev[h] = h3

            H_next[ht] = h4
            H_prev[h4] = ht

            H_next[h4] = h1
            H_prev[h1] = h4

            H_next[h1] = ht
            H_prev[ht] = h1

            H_face[h3] = f1

            H_face[h1] = f2

            H_vertex[h] = v2
            H_vertex[ht] = v4

            V_hedge[v3] = h3
            V_hedge[v1] = h1
            V_hedge[v2] = h2
            V_hedge[v4] = h4
            Nflips += 1
    return (
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        faces,
        F_hedge,
        Nflips,
    )


@njit
def regularize_by_shifts(
    weight,
    vertices,
    V_hedge,
    H_vertex,
    H_prev,
    H_twin,
):
    """shifts vertices towards center of mass of their neighbors"""
    Nv = len(vertices)
    for v in range(Nv):
        h = V_hedge[v]
        h_start = h
        r = np.zeros(3)
        val = 0
        while True:
            vb = H_vertex[h]
            r += vertices[vb]
            val += 1
            h = H_prev[h]
            h = H_twin[h]
            if h == h_start:
                break
        r /= val
        vertices[v] = weight * r + (1 - weight) * vertices[v]
    return vertices


#################################################################
# implicit function --> vertex/face lists --> .ply files
def make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz):
    """
    uses marching cubes to get vertex/face list from implicit function

    xyz_minmax = [-3.0, 3.0, -3.0, 3.0, -3.0, 3.0]
    Nxyz = [60j, 60j, 60j]
    implicit_fun_str = (
        "1.0*(y**2 + z**2 + (x - 8)**2 - 1)*(y**2 + z**2 + (x + 8)**2 - 1) - 4200.0"
    )"""
    xyz = sp.Array(sp.symbols("x y z"))
    implicit_fun_sym = sp.sympify(implicit_fun_str)
    implicit_fun = sp.lambdify(xyz, implicit_fun_sym)

    x0, x1, y0, y1, z0, z1 = xyz_minmax
    Nx, Ny, Nz = Nxyz
    xyz_grid = np.mgrid[x0:x1:Nx, y0:y1:Ny, z0:z1:Nz]
    x, y, z = xyz_grid

    dx = x[1, 0, 0] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dz = z[0, 0, 1] - z[0, 0, 0]
    vol = implicit_fun(x, y, z)

    iso_val = 0.0
    verts, faces, normals, values = marching_cubes(vol, iso_val, spacing=(dx, dy, dz))

    verts[:, 0] += x[0, 0, 0]
    verts[:, 1] += y[0, 0, 0]
    verts[:, 2] += z[0, 0, 0]
    normals = -normals.astype(np.float64)
    return verts, faces, normals


def make_trisurface_patch(Nfaces=5):
    """makes a triangle mesh patch around a vertex"""
    N = 1 * Nfaces
    dr = 0.25
    dz = 0.1
    theta = np.array([2 * np.pi * _ / N for _ in range(N)])
    r = np.random.rand(N)
    r *= dr * np.max(r)
    r += 1 - dr
    z = dz * np.cos(theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    vertices = np.array([x, y, z]).T
    vertices = np.array([[0.0, 0.0, dz], *vertices])
    faces = [[0, i, i + 1] for i in range(1, N)]
    faces = np.array([*faces, [0, N, 1]], dtype=np.int32)
    return vertices, faces


def make_quadsurface_patch(Nfaces=3):
    """makes a quad mesh patch around a vertex"""
    N = 2 * Nfaces
    dr = 0.25
    dz = 0.1
    theta = np.array([2 * np.pi * _ / N for _ in range(N)])
    r = np.random.rand(N)
    r *= dr * np.max(r)
    r += 1 - dr
    z = dz * np.cos(theta)
    x, y = r * np.cos(theta), r * np.sin(theta)
    vertices = np.array([x, y, z]).T
    vertices = np.array([[0.0, 0.0, dz], *vertices])
    faces = [[0, i, i + 1, i + 2] for i in range(1, N - 2, 2)]
    faces = np.array([*faces, [0, N - 1, N, 1]], dtype=np.int32)
    return vertices, faces


def make_sample_mesh(surface_name, Nxyz=None, xyz_minmax=None):
    """makes meshes for some cool surfaces"""
    surface_names = [
        "dumbbell",
        "dumbbell2",
        "torus",
        "double_torus",
        "triple_torus",
        "neovius",
        "sphere",
        "oblate",
        "prolate",
        "transverse_tori",
        "pyramid3",
        "pyramid4",
    ]
    if Nxyz is None:
        Nxyz = [30j, 30j, 30j]
    if xyz_minmax is None:
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    if surface_name not in surface_names:
        msg = f"{surface_name} undefined.\n"
        msg += "surface_name must be one of: "
        for _ in surface_names:
            msg += f"{_}, "
        raise ValueError(msg)
    elif surface_name == "dumbbell":
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        implicit_fun_str = "9*x**2 + 9*y**2 - 9*(z**2 - 1)*(cos(3*pi*z/4) - 1.25)/4"
    elif surface_name == "dumbbell2":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [30j, 30j, 30j]
        implicit_fun_str = (
            "(144*y**2 + 144*z**2 + (12*x - 8)**2 - 1)*(144*y**2 + 144*z**2 + (12*x + 8)**2 - 1) - 4200"
        )
    elif surface_name == "torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        R = 0.7  # big radius
        r = 0.7 / 3.0  # small radius
        implicit_fun_str = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
    elif surface_name == "double_torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [30j, 30j, 30j]
        # implicit_fun_str = "z**2 + (x**2*(x - 1)*(x + 1) + y**2)**2 - 0.01"
        implicit_fun_str = "(z/0.2)**2 + (x**2*(x - 0.7)*(x + 0.7)/0.05 + y**2/0.05)**2 - 1"
    elif surface_name == "triple_torus":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [40j, 40j, 40j]
        # implicit_fun_str = "0.16*z**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
        implicit_fun_str = "1.69*z**2*(1 - 0.769*cos(pi*Abs(y**2 + (x + 0.2)**2)/4))**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
    elif surface_name == "neovius":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        implicit_fun_str = (
            "4*cos(3*x + 3)*cos(3*y + 3)*cos(3*z + 3) + 3*cos(3*x + 3) + 3*cos(3*y + 3) + 3*cos(3*z + 3)"
        )
    elif surface_name == "sphere":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        R = 0.9
        implicit_fun_str = f"x**2+y**2+z**2-{R**2}"
    elif surface_name == "oblate":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        Rxy = 0.9
        Rz = 0.6
        implicit_fun_str = f"(x/{Rxy})**2+(y/{Rxy})**2+(z/{Rz})**2-1"
    elif surface_name == "prolate":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [20j, 20j, 20j]
        Rxy = 0.6
        Rz = 0.9
        implicit_fun_str = f"(x/{Rxy})**2+(y/{Rxy})**2+(z/{Rz})**2-1"
    elif surface_name == "transverse_tori":
        # xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # # Nxyz = [60j, 60j, 60j]
        # Nxyz = [20j, 20j, 20j]
        # R = 0.7  # big radius
        # r = 0.7 / 3.0  # small radius
        # b = (0.5 * r) ** 2
        # xyz_minmax = [-1.3, 1.3, -1.3, 1.3, -1.3, 1.3]
        # Nxyz = [60j, 60j, 60j]
        # Nxyz = [30j, 30j, 30j]
        R = 1.0  # big radius
        r = 0.2  # small radius
        b = 0.01
        # F1 = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
        # F2 = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (y**2 + z**2)"
        # F3 = f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (z**2 + x**2)"
        F1 = (
            f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((x/.7)**2 + (y/.7)**2)"
        )
        F2 = (
            f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((y/.7)**2 + (z/.7)**2)"
        )
        F3 = (
            f"((x/.7)**2 + (y/.7)**2 + (z/.7)**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * ((z/.7)**2 + (x/.7)**2)"
        )
        implicit_fun_str = f"({F1})*({F2})*({F3})-{b}"
    elif surface_name == "pyramid3":
        implicit_fun_str = None
        verts = np.zeros((4, 3))
        faces = np.zeros((4, 3), dtype=np.int32)
        xyz_top = np.array([0.0, 0.0, 1.0])
        v_top = 0
        xyz_base1 = np.array([1.0, 0.0, -1.0])
        v_base1 = 1
        xyz_base2 = np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3), -1.0])
        v_base2 = 2
        xyz_base3 = np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3), -1.0])
        v_base3 = 3
        f_base = np.array([v_base3, v_base2, v_base1])
        f_side1 = np.array([v_base1, v_base2, v_top])
        f_side2 = np.array([v_base2, v_base3, v_top])
        f_side3 = np.array([v_base3, v_base1, v_top])
        verts[v_top] = xyz_top
        verts[v_base1] = xyz_base1
        verts[v_base2] = xyz_base2
        verts[v_base3] = xyz_base3
        faces = np.array([f_base, f_side1, f_side2, f_side3], dtype=np.int32)
        verts *= 0.9
    elif surface_name == "pyramid4":
        implicit_fun_str = None
        verts = np.zeros((5, 3))
        faces = np.zeros((5, 3), dtype=np.int32)
        xyz_top = np.array([0.0, 0.0, 1.0])
        v_top = 0
        xyz_base1 = np.array([1.0, 0.0, -1.0])
        v_base1 = 1
        xyz_base2 = np.array([np.cos(2 * np.pi / 4), np.sin(2 * np.pi / 4), -1.0])
        v_base2 = 2
        xyz_base3 = np.array([np.cos(4 * np.pi / 4), np.sin(4 * np.pi / 4), -1.0])
        v_base3 = 3
        xyz_base4 = np.array([np.cos(6 * np.pi / 4), np.sin(6 * np.pi / 4), -1.0])
        v_base4 = 4

        f_base1 = np.array([v_base4, v_base3, v_base2])
        f_base2 = np.array([v_base2, v_base1, v_base4])
        f_side1 = np.array([v_base1, v_base2, v_top])
        f_side2 = np.array([v_base2, v_base3, v_top])
        f_side3 = np.array([v_base3, v_base4, v_top])
        f_side4 = np.array([v_base4, v_base1, v_top])
        verts[v_top] = xyz_top
        verts[v_base1] = xyz_base1
        verts[v_base2] = xyz_base2
        verts[v_base3] = xyz_base3
        verts[v_base4] = xyz_base4
        faces = np.array([f_base1, f_base2, f_side1, f_side2, f_side3, f_side4], dtype=np.int32)
        verts *= 0.9
    if implicit_fun_str is None:
        pass
    else:
        verts, faces, normals = make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz)
    # normal_norms = np.linalg.norm(normals, axis=1)
    # normals = np.array([n / np.linalg.norm(n) for n in normals])

    # surf_dict = {"vertices": verts, "faces": faces, "normals": normals}
    # with open(f"./scratch/{surface_name}_dict.pickle", "wb") as _f:
    #     dill.dump(surf_dict, _f, recurse=True)

    return verts, faces


def save_mesh_to_ply(vertices, faces, file_path):
    """saves vertex+face list to a .ply file"""
    # Create the vertex data
    vertex_data = np.array([tuple(v) for v in vertices], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])

    # Create the face data
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    # Create the PlyElements
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    face_element = PlyElement.describe(face_data, "face")

    # Write to a .ply file
    PlyData([vertex_element, face_element], text=True).write(file_path)


def load_mesh_from_ply(file_path):
    """loads vertex+face list from a .ply file"""
    # Read the ply file
    plydata = PlyData.read(file_path)

    # Extract the vertex and face data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    faces = np.vstack(plydata["face"]["vertex_indices"])

    if not isinstance(vertices[0], np.float64):
        # faces = [tuple(f) for f in faces]
        vertices = vertices.astype(np.float64)
    if not isinstance(faces[0], np.int32):
        # faces = [tuple(f) for f in faces]
        faces = faces.astype(np.int32)

    return vertices, faces


def save_sample_mesh_to_ply(surface_name, file_path=None, Nxyz=None, xyz_minmax=None):
    """saves one of the meshes made by make_sample_mesh() to a .ply file"""
    if file_path is None:
        file_path = f"./data/ply_files/{surface_name}.ply"
    vertices, faces = make_sample_mesh(surface_name, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
    save_mesh_to_ply(vertices, faces, file_path)


def save_all_sample_meshes():
    """saves all sample meshes generated by make_sample_mesh() to .ply files"""
    surface_names = [
        "dumbbell",
        "dumbbell2",
        "torus",
        "double_torus",
        "triple_torus",
        "neovius",
        "sphere",
        "oblate",
        "prolate",
        "transverse_tori",
        "pyramid3",
        "pyramid4",
    ]
    print("saving\n")
    for _ in surface_names:
        surface_name = _
        Nxyz = [30j, 30j, 30j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        file_path = f"./data/ply_files/{surface_name}.ply"
        print(file_path + 15 * " ", end="\r")
        save_sample_mesh_to_ply(surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
        print(file_path + " -done" + 9 * " ", end="\n")
    for _ in surface_names:
        surface_name = _
        Nxyz = [60j, 60j, 60j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        file_path = f"./data/ply_files/{surface_name}_fine.ply"
        print(file_path + 15 * " ", end="\r")
        save_sample_mesh_to_ply(surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
        print(file_path + " -done" + 9 * " ", end="\n")
    for _ in surface_names:
        surface_name = _
        Nxyz = [20j, 20j, 20j]
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        print(file_path + 15 * " ", end="\r")
        file_path = f"./data/ply_files/{surface_name}_coarse.ply"
        save_sample_mesh_to_ply(surface_name, file_path=file_path, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
        print(file_path + " -done" + 9 * " ", end="\n")
    print("\ndone")


def save_halfedge_mesh_from_ply(ply_path, output_directory):
    """loads vertex/face lists from .ply file, builds halfedge mesh data, and
    saves as numpy arrays"""
    os.system(f"rm -r {output_directory}")
    os.system(f"mkdir {output_directory}")
    plydata = PlyData.read(ply_path)

    # Extract the vertex and face data
    vertices = np.vstack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]]).T
    faces = np.vstack(plydata["face"]["vertex_indices"])

    if not isinstance(vertices[0], np.float64):
        # faces = [tuple(f) for f in faces]
        vertices = vertices.astype(np.float64)
    if not isinstance(faces[0], np.int32):
        # faces = [tuple(f) for f in faces]
        faces = faces.astype(np.int32)

    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = get_combinatorial_mesh_data(vertices, faces)
    np.save(f"{output_directory}/vertices.npy", vertices)
    np.save(f"{output_directory}/V_hedge.npy", V_hedge)
    np.save(f"{output_directory}/halfedges.npy", halfedges)
    np.save(f"{output_directory}/H_vertex.npy", H_vertex)
    np.save(f"{output_directory}/H_face.npy", H_face)
    np.save(f"{output_directory}/H_next.npy", H_next)
    np.save(f"{output_directory}/H_prev.npy", H_prev)
    np.save(f"{output_directory}/H_twin.npy", H_twin)
    np.save(f"{output_directory}/faces.npy", faces)
    np.save(f"{output_directory}/F_hedge.npy", F_hedge)


def load_halfedge_mesh_data(mesh_directory):
    """loads halfedge mesh numpy arrays"""
    mesh_data = {}
    mesh_data["vertices"] = np.load(f"{mesh_directory}/vertices.npy")
    mesh_data["V_hedge"] = np.load(f"{mesh_directory}/V_hedge.npy")
    mesh_data["halfedges"] = np.load(f"{mesh_directory}/halfedges.npy")
    mesh_data["H_vertex"] = np.load(f"{mesh_directory}/H_vertex.npy")
    mesh_data["H_face"] = np.load(f"{mesh_directory}/H_face.npy")
    mesh_data["H_next"] = np.load(f"{mesh_directory}/H_next.npy")
    mesh_data["H_prev"] = np.load(f"{mesh_directory}/H_prev.npy")
    mesh_data["H_twin"] = np.load(f"{mesh_directory}/H_twin.npy")
    mesh_data["faces"] = np.load(f"{mesh_directory}/faces.npy")
    mesh_data["F_hedge"] = np.load(f"{mesh_directory}/F_hedge.npy")
    return mesh_data


def save_all_halfedge_meshes():
    """applies save_halfedge_mesh_from_ply() to selected sample meshes"""
    surface_names = [
        "sphere",
        "oblate",
        "prolate",
        "dumbbell",
        "torus",
        "double_torus",
        "triple_torus",
        "transverse_tori",
    ]
    print("saving...\n")
    for surface_name in surface_names:
        ply_path = f"./data/ply_files/{surface_name}_coarse.ply"
        mesh_directory = f"./data/halfedge_meshes/{surface_name}_coarse"
        print(mesh_directory + 15 * " ", end="\r")
        save_halfedge_mesh_from_ply(ply_path, mesh_directory)
        print(mesh_directory + " -done" + 9 * " ", end="\n")
    for surface_name in surface_names:
        ply_path = f"./data/ply_files/{surface_name}.ply"
        mesh_directory = f"./data/halfedge_meshes/{surface_name}"
        print(mesh_directory + 15 * " ", end="\r")
        save_halfedge_mesh_from_ply(ply_path, mesh_directory)
        print(mesh_directory + " -done" + 9 * " ", end="\n")
    for surface_name in surface_names:
        ply_path = f"./data/ply_files/{surface_name}_fine.ply"
        mesh_directory = f"./data/halfedge_meshes/{surface_name}_fine"
        print(mesh_directory + 15 * " ", end="\r")
        save_halfedge_mesh_from_ply(ply_path, mesh_directory)
        print(mesh_directory + " -done" + 9 * " ", end="\n")
    print("\ndone")


def regularize_halfedge_mesh_data(
    vertices,
    V_hedge,
    halfedges,
    H_vertex,
    H_face,
    H_next,
    H_prev,
    H_twin,
    faces,
    F_hedge,
):
    iters = 20
    weight = 0.1
    Nflips = -1
    for iter in range(iters):
        print(
            f" regularizing halfedge mesh data iteration {iter} of {iters-1}    ",
            end="\r",
        )

        (
            V_hedge,
            halfedges,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            faces,
            F_hedge,
            Nflips,
        ) = regularize_by_flips(
            V_hedge,
            halfedges,
            H_vertex,
            H_face,
            H_next,
            H_prev,
            H_twin,
            faces,
            F_hedge,
        )

        vertices = regularize_by_shifts(
            weight,
            vertices,
            V_hedge,
            H_vertex,
            H_prev,
            H_twin,
        )
    return (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    )


def sphere_oblate_torus_dumbbell_mesh_inputs(reg=True, coarse=False, fine=False, ultrafine=False):
    """Gets list of inputs required to make .ply files and halfedge mesh
    data for sphere, oblate spheroid, torus, and dumbbell

    sphere={
        "implicit_fun_str": "x**2+y**2+z**2-0.9",
        "Nxyz": [30j, 30j, 30j],
        "xyz_minmax": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "ply_path": "./data/ply_files/sphere.ply",
        "mesh_directory": "./data/halfedge_meshes/sphere",
    }"""
    Nxyz = [30j, 30j, 30j]
    Nxyz_coarse = [20j, 20j, 20j]
    Nxyz_fine = [60j, 60j, 60j]
    Nxyz_ultrafine = [200j, 200j, 200j]
    xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    surfs_dict = {
        "sphere": "x**2+y**2+z**2-0.9",
        "oblate": "(x/0.9)**2+(y/0.9)**2+(z/0.6)**2-1",
        "torus": "(x**2 + y**2 + z**2 + 0.7**2 - 0.24**2) ** 2 - 4 * 0.7**2 * (x**2 + y**2)",
        "dumbbell": "9*x**2 + 9*y**2 - 9*(z**2 - 1)*(cos(3*pi*z/4) - 1.25)/4",
    }
    reg_list = []
    coarse_list = []
    fine_list = []
    ultrafine_list = []
    for surface_name, implicit_fun_str in surfs_dict.items():
        reg_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}",
                "Nxyz": Nxyz,
                "xyz_minmax": xyz_minmax,
            }
        )
        coarse_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}_coarse.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}_coarse",
                "Nxyz": Nxyz_coarse,
                "xyz_minmax": xyz_minmax,
            }
        )
        fine_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}_fine.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}_fine",
                "Nxyz": Nxyz_fine,
                "xyz_minmax": xyz_minmax,
            }
        )
        ultrafine_list.append(
            {
                "implicit_fun_str": implicit_fun_str,
                "ply_path": f"./data/ply_files/{surface_name}_ultrafine.ply",
                "mesh_directory": f"./data/halfedge_meshes/{surface_name}_ultrafine",
                "Nxyz": Nxyz_ultrafine,
                "xyz_minmax": xyz_minmax,
            }
        )

    surfaces = []
    if coarse:
        surfaces.extend(coarse_list)
    if reg:
        surfaces.extend(reg_list)
    if fine:
        surfaces.extend(fine_list)
    if ultrafine:
        surfaces.extend(ultrafine_list)
    return surfaces


def generate_regularize_save_mesh(implicit_fun_str, Nxyz, xyz_minmax, ply_path, mesh_directory):
    """Uses marching cubes to generate vertex/face list for the
    zero level set 'implicit_fun_str', builds halfedge mesh data, and regularizes
    mesh. Saves vertex/face list .ply file to 'ply_path' and saves
    halfedge mesh data as numpy arrays to 'mesh_directory'.
    """
    print(f"{implicit_fun_str}=0")
    print(f" ply_path: {ply_path}")
    print(f" mesh_directory: {mesh_directory}")
    os.system(f"rm -r {mesh_directory}")
    os.system(f"mkdir {mesh_directory}")
    print(" generating .ply file", end="\r")
    vertices_in, faces_in, normals = make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz)
    save_mesh_to_ply(vertices_in, faces_in, ply_path)
    print(" generating .ply file -done")
    print(" generating halfedge mesh data", end="\r")
    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = get_combinatorial_mesh_data(vertices_in, faces_in)
    print(" generating halfedge mesh data -done")
    print(" regularizing halfedge mesh data", end="\r")
    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = regularize_halfedge_mesh_data(
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    )
    print(" regularizing halfedge mesh data -done" + 15 * " ")
    print(" saving", end="\r")
    np.save(f"{mesh_directory}/vertices.npy", vertices)
    np.save(f"{mesh_directory}/V_hedge.npy", V_hedge)
    np.save(f"{mesh_directory}/halfedges.npy", halfedges)
    np.save(f"{mesh_directory}/H_vertex.npy", H_vertex)
    np.save(f"{mesh_directory}/H_face.npy", H_face)
    np.save(f"{mesh_directory}/H_next.npy", H_next)
    np.save(f"{mesh_directory}/H_prev.npy", H_prev)
    np.save(f"{mesh_directory}/H_twin.npy", H_twin)
    np.save(f"{mesh_directory}/faces.npy", faces)
    np.save(f"{mesh_directory}/F_hedge.npy", F_hedge)
    save_mesh_to_ply(vertices, faces, ply_path)

    print(" saving -done")


def generate_sphere_oblate_torus_dumbbell(reg=True, coarse=False, fine=False, ultrafine=False):
    surfaces = sphere_oblate_torus_dumbbell_mesh_inputs(reg=reg, coarse=coarse, fine=fine, ultrafine=ultrafine)
    for surf in surfaces:
        print(55 * "-")
        generate_regularize_save_mesh(**surf)

    print(55 * "-")


def time_mesh_inputs():
    """Gets list of inputs required to make .ply files and halfedge mesh
    data for sphere, oblate spheroid, torus, and dumbbell

    sphere={
        "implicit_fun_str": "x**2+y**2+z**2-0.9",
        "Nxyz": [30j, 30j, 30j],
        "xyz_minmax": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "ply_path": "./data/ply_files/sphere.ply",
        "mesh_directory": "./data/halfedge_meshes/sphere",
    }"""
    Nxyz16 = [16j, 16j, 16j]
    Nxyz32 = [32j, 32j, 32j]
    Nxyz64 = [64j, 64j, 64j]
    xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    surfaces = []
    implicit_fun_str = "x**2+y**2+z**2-0.9"
    surface_name = "time_sphere"
    surfaces.append(
        {
            "implicit_fun_str": implicit_fun_str,
            "ply_path": f"./data/ply_files/{surface_name}16.ply",
            "mesh_directory": f"./data/halfedge_meshes/{surface_name}16",
            "Nxyz": Nxyz16,
            "xyz_minmax": xyz_minmax,
        }
    )
    surfaces.append(
        {
            "implicit_fun_str": implicit_fun_str,
            "ply_path": f"./data/ply_files/{surface_name}32.ply",
            "mesh_directory": f"./data/halfedge_meshes/{surface_name}32",
            "Nxyz": Nxyz32,
            "xyz_minmax": xyz_minmax,
        }
    )
    surfaces.append(
        {
            "implicit_fun_str": implicit_fun_str,
            "ply_path": f"./data/ply_files/{surface_name}64.ply",
            "mesh_directory": f"./data/halfedge_meshes/{surface_name}64",
            "Nxyz": Nxyz64,
            "xyz_minmax": xyz_minmax,
        }
    )

    return surfaces


def time_generate_regularize_save_mesh(implicit_fun_str, Nxyz, xyz_minmax, ply_path, mesh_directory):
    """Uses marching cubes to generate vertex/face list for the
    zero level set 'implicit_fun_str', builds halfedge mesh data, and regularizes
    mesh. Saves vertex/face list .ply file to 'ply_path' and saves
    halfedge mesh data as numpy arrays to 'mesh_directory'.
    """
    from time import time

    t0 = time()
    tdict = {}
    print(f"{implicit_fun_str}=0")
    print(f" ply_path: {ply_path}")
    print(f" mesh_directory: {mesh_directory}")
    os.system(f"rm -r {mesh_directory}")
    os.system(f"mkdir {mesh_directory}")
    print(" generating .ply file", end="\r")
    tdict["make_implicit_surface_mesh"] = time()
    vertices_in, faces_in, normals = make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz)
    tdict["make_implicit_surface_mesh"] = time() - tdict["make_implicit_surface_mesh"]
    tdict["save_mesh_to_ply0"] = time()
    save_mesh_to_ply(vertices_in, faces_in, ply_path)
    tdict["save_mesh_to_ply0"] = time() - tdict["save_mesh_to_ply0"]
    print(" generating .ply file -done")
    print(" generating halfedge mesh data", end="\r")
    tdict["get_combinatorial_mesh_data"] = time()
    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = get_combinatorial_mesh_data(vertices_in, faces_in)
    tdict["get_combinatorial_mesh_data"] = time() - tdict["get_combinatorial_mesh_data"]
    print(" generating halfedge mesh data -done")
    print(" regularizing halfedge mesh data", end="\r")
    tdict["regularize_halfedge_mesh_data"] = time()
    (
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    ) = regularize_halfedge_mesh_data(
        vertices,
        V_hedge,
        halfedges,
        H_vertex,
        H_face,
        H_next,
        H_prev,
        H_twin,
        faces,
        F_hedge,
    )
    tdict["regularize_halfedge_mesh_data"] = time() - tdict["regularize_halfedge_mesh_data"]
    print(" regularizing halfedge mesh data -done" + 15 * " ")
    print(" saving", end="\r")
    tdict["save_halfedge_mesh_data"] = time()
    np.save(f"{mesh_directory}/vertices.npy", vertices)
    np.save(f"{mesh_directory}/V_hedge.npy", V_hedge)
    np.save(f"{mesh_directory}/halfedges.npy", halfedges)
    np.save(f"{mesh_directory}/H_vertex.npy", H_vertex)
    np.save(f"{mesh_directory}/H_face.npy", H_face)
    np.save(f"{mesh_directory}/H_next.npy", H_next)
    np.save(f"{mesh_directory}/H_prev.npy", H_prev)
    np.save(f"{mesh_directory}/H_twin.npy", H_twin)
    np.save(f"{mesh_directory}/faces.npy", faces)
    np.save(f"{mesh_directory}/F_hedge.npy", F_hedge)
    tdict["save_halfedge_mesh_data"] = time() - tdict["save_halfedge_mesh_data"]
    tdict["save_mesh_to_ply"] = time()
    save_mesh_to_ply(vertices, faces, ply_path)
    tdict["save_mesh_to_ply"] = time() - tdict["save_mesh_to_ply"]
    tdict["total"] = time() - t0
    print(" saving -done")
    return tdict


def time_order():
    import dill

    # surf16, sphere32, sphere64 = time_mesh_inputs()
    Tdict = {
        "make_implicit_surface_mesh": np.zeros(3),
        "save_mesh_to_ply0": np.zeros(3),
        "get_combinatorial_mesh_data": np.zeros(3),
        "regularize_halfedge_mesh_data": np.zeros(3),
        "save_halfedge_mesh_data": np.zeros(3),
        "save_mesh_to_ply": np.zeros(3),
        "total": np.zeros(3),
        "Nxyz": np.array([16, 32, 64]),
    }
    Nxyz16 = [16j, 16j, 16j]
    Nxyz32 = [32j, 32j, 32j]
    Nxyz64 = [64j, 64j, 64j]
    xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    surfaces = []
    implicit_fun_str = "x**2+y**2+z**2-0.9"
    surface_name = "time_sphere"

    surf16 = {
        "implicit_fun_str": implicit_fun_str,
        "ply_path": f"./data/ply_files/{surface_name}16.ply",
        "mesh_directory": f"./data/halfedge_meshes/{surface_name}16",
        "Nxyz": Nxyz16,
        "xyz_minmax": xyz_minmax,
    }

    surf32 = {
        "implicit_fun_str": implicit_fun_str,
        "ply_path": f"./data/ply_files/{surface_name}32.ply",
        "mesh_directory": f"./data/halfedge_meshes/{surface_name}32",
        "Nxyz": Nxyz32,
        "xyz_minmax": xyz_minmax,
    }

    surf64 = {
        "implicit_fun_str": implicit_fun_str,
        "ply_path": f"./data/ply_files/{surface_name}64.ply",
        "mesh_directory": f"./data/halfedge_meshes/{surface_name}64",
        "Nxyz": Nxyz64,
        "xyz_minmax": xyz_minmax,
    }

    tdict16 = time_generate_regularize_save_mesh(
        surf16["implicit_fun_str"],
        surf16["Nxyz"],
        surf16["xyz_minmax"],
        surf16["ply_path"],
        surf16["mesh_directory"],
    )
    tdict32 = time_generate_regularize_save_mesh(
        surf32["implicit_fun_str"],
        surf32["Nxyz"],
        surf32["xyz_minmax"],
        surf32["ply_path"],
        surf32["mesh_directory"],
    )
    tdict64 = time_generate_regularize_save_mesh(
        surf64["implicit_fun_str"],
        surf64["Nxyz"],
        surf64["xyz_minmax"],
        surf64["ply_path"],
        surf64["mesh_directory"],
    )
    # tdict16 = time_generate_regularize_save_mesh(
    #     surf32["implicit_fun_str"],
    #     surf32["Nxyz"],
    #     surf32["xyz_minmax"],
    #     surf32["ply_path"],
    #     surf32["mesh_directory"],
    # )
    # tdict32 = time_generate_regularize_save_mesh(
    #     surf32["implicit_fun_str"],
    #     surf32["Nxyz"],
    #     surf32["xyz_minmax"],
    #     surf32["ply_path"],
    #     surf32["mesh_directory"],
    # )
    # tdict64 = time_generate_regularize_save_mesh(
    #     surf32["implicit_fun_str"],
    #     surf32["Nxyz"],
    #     surf32["xyz_minmax"],
    #     surf32["ply_path"],
    #     surf32["mesh_directory"],
    # )
    for n, tdict in enumerate([tdict16, tdict32, tdict64]):
        for key, val in tdict.items():
            Tdict[key][n] = val

    with open("./scratch/Tdict.pickle", "wb") as f:
        dill.dump(Tdict, f, recurse=True)
    return Tdict


# Tdict = time_order()
# with open("./scratch/Tdict.pickle", "rb") as f:
#     brr = dill.load(f)
# %%
# generate_sphere_oblate_torus_dumbbell(
#     reg=True, coarse=True, fine=False, ultrafine=False
# )
# generate_sphere_oblate_torus_dumbbell(
#     reg=False, coarse=False, fine=True, ultrafine=False
# )
# %%

#################################################################
# mesh regularization stuff
# def regularize_and_resave_ply(surface_name, backup=True):
#     """Loads vertex/face list from .ply file, uses the
#     Brane.regularize_by_flips() and Brane.regularize_by_shifts()
#     methods to regularize the mesh, and resaves the .ply file.
#
#     does not work for surface with boundary!!"""
#     iters = 20
#     weight = 0.1
#     file_path = f"./data/ply_files/{surface_name}.ply"
#     if backup:
#         backup_file_path = f"./data/ply_files/{surface_name}_backup.ply"
#         os.system(f"cp {file_path} {backup_file_path}")
#
#     vertices0, faces0 = load_mesh_from_ply(file_path)
#     # brane_init_data = {
#     #     "vertices": vertices0,
#     #     "faces": faces0,
#     #     "length_reg_stiffness": 1.0,
#     #     "area_reg_stiffness": 1.0,
#     #     "bending_modulus": 1.0,
#     #     "splay_modulus": 1.0,
#     #     "linear_drag_coeff": 1.0,
#     # }
#     brane_init_data = {
#         "vertices": vertices0,
#         "faces": faces0,
#         "length_reg_stiffness": 1.0,
#         "area_reg_stiffness": 1.0,
#         "volume_reg_stiffness": 1.0,
#         "bending_modulus": 1.0,
#         "splay_modulus": 1.0,
#         "linear_drag_coeff": 1.0,
#         "spontaneous_curvature": 0.0,
#     }
#
#     b = Brane(**brane_init_data)
#     for iter in range(iters):
#         Nflips = b.regularize_by_flips()
#         b.regularize_by_shifts(weight)
#         # print(f"iter={iter+1} of {iters}, Nflips={Nflips}            ", end="\r")
#     vertices = b.V_pq[:, :3]
#     faces = b.faces
#     save_mesh_to_ply(vertices, faces, file_path)
#
#
# def regularize_all_sample_meshes():
#     """applies regularize_and_resave_ply() to all the sample (boundary-less) meshes"""
#     surface_names = [
#         "dumbbell",
#         "dumbbell2",
#         "torus",
#         "double_torus",
#         "triple_torus",
#         "sphere",
#         "oblate",
#         "prolate",
#         "transverse_tori",
#         "dumbbell_coarse",
#         "dumbbell2_coarse",
#         "torus_coarse",
#         "double_torus_coarse",
#         "triple_torus_coarse",
#         "sphere_coarse",
#         "oblate_coarse",
#         "prolate_coarse",
#         "transverse_tori_coarse",
#         "dumbbell_fine",
#         "dumbbell2_fine",
#         "torus_fine",
#         "double_torus_fine",
#         "triple_torus_fine",
#         "sphere_fine",
#         "oblate_fine",
#         "prolate_fine",
#         "transverse_tori_fine",
#     ]
#     print("saving...\n")
#     for _ in surface_names:
#         surface_name = _
#         file_path = f"./data/ply_files/{surface_name}.ply"
#         print(file_path + 10 * " ", end="\n")
#         regularize_and_resave_ply(surface_name, backup=True)
#         print(30 * " ")
#     print("\ndone")
#
#     # surf_dict = {"vertices": verts, "faces": faces, "normals": normals}
#     # with open(f"./scratch/{surface_name}_dict.pickle", "wb") as _f:
#     #     dill.dump(surf_dict, _f, recurse=True)


#################################################################
#################################################################
#################################################################
##########################
# @njit
# def jit_boundary_ops_csr_data(vertices, faces):
#     """
#     Computes edges and boundary operators from vertices and faces
#     """
#     Nvertices = len(vertices)
#     Nfaces = len(faces)
#     edges_list = []
#
#     # Afe ###############
#     Afe_data_list = []  # [-1,1,...]
#     Afe_indices_list = []  #
#     Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)
#
#     # Aev ###############
#     # Aev_data_list = []  # [-1,1,...]
#     Aev_indices_list = []  # vertex indices
#     # Aev_indptr = []  # [0,2,4,...]
#
#     #######################
#     Afv_indices = faces.ravel()
#     Afv_indptr = [3 * f for f in range(Nfaces + 1)]
#     Afv_data = np.ones(3 * Nfaces)
#
#     for f in range(Nfaces):
#         face = faces[f]
#         for _v in range(3):
#             vm = face[_v]
#             vp = face[np.mod(_v + 1, 3)]
#             edge_p = [vm, vp]
#             edge_m = [vp, vm]
#             try:  # is negative edge already in edges?
#                 e = edges_list.index(edge_m)
#                 fe_orientation = -1
#             except Exception:
#                 try:  # is positive edge already in edges?
#                     e = edges_list.index(edge_p)
#                     fe_orientation = 1
#                 except Exception:  # if neither, add positive edge to edges
#                     edges_list.append(edge_p)
#                     e = len(edges_list) - 1
#                     fe_orientation = 1
#                     Aev_indices_list.append(vm)
#                     Aev_indices_list.append(vp)
#
#             Afe_indices_list.append(e)
#             Afe_data_list.append(fe_orientation)
#
#     Afe_data = np.array(Afe_data_list, dtype=np.int32)
#     Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
#     Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
#     edges = np.array(edges_list, dtype=np.int32)
#     Nedges = len(edges)
#     Aev_data = np.array(Nedges * [-1, 1], dtype=np.int32)
#     Aev_indptr = np.array(
#         [2 * _ for _ in range(Nedges + 1)], dtype=np.int32
#     )  # [0,2,4,...]
#     return (
#         Afv_indices,
#         Afv_indptr,
#         Afv_data,
#         Afe_data,
#         Afe_indices,
#         Afe_indptr,
#         Aev_data,
#         Aev_indices,
#         Aev_indptr,
#         edges,
#     )
#
#
# @njit
# def jit_face_data(vertices, faces, surface_com):
#     """
#     computes what are hopefully outward pointing unit normal vectors
#     and directed area vectors of the faces. Reorders vertices of each face to
#     match unit normal direction.
#     """
#     # Nvertices = len(vertices)
#     # surface_com = np.einsum("vx->x", vertices)
#
#     Nfaces = len(faces)
#     face_normals = np.zeros((Nfaces, 3))
#     face_areas = np.zeros((Nfaces, 3))
#     face_centroids = np.zeros((Nfaces, 3))
#     for f in range(Nfaces):
#         fv0, fv1, fv2 = faces[f]
#         v0_xyz = vertices[fv0]
#         v1_xyz = vertices[fv1]
#         v2_xyz = vertices[fv2]
#
#         # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
#         f_normal = (
#             jitcross(v0_xyz, v1_xyz)
#             + jitcross(v1_xyz, v2_xyz)
#             + jitcross(v2_xyz, v0_xyz)
#         )
#         f_area = 0.5 * f_normal
#
#         f_normal /= np.sqrt(f_normal @ f_normal)
#         face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
#         face_centroids[f] = face_com
#         n_dot_dr = f_normal @ (face_com - surface_com)
#         if n_dot_dr > 0:
#             # faces[f, :] = np.array([fv0, fv1, fv2])
#             face_normals[f, :] = f_normal
#             face_areas[f, :] = f_area
#         else:
#             faces[f, :] = np.array([fv1, fv0, fv2])
#             face_normals[f, :] = -f_normal
#             face_areas[f, :] = -f_area
#     return faces, face_centroids, face_normals, face_areas
#
#
# @njit
# def jit_area_weighted_vertex_normals(vertices, Avf_indices, Avf_indptr, face_areas):
#     """
#     computes unit normal vectors at vertices.
#     Avf_indices, Avf_indptr
#     """
#     Nvertices = len(vertices)
#     vertex_normals = np.zeros((Nvertices, 3))
#     for v in range(Nvertices):
#         # Nfaces_of_v = len(faces_of_vertices[v])
#         # Nfaces_of_v = max([1,len(faces_of_vertices[v])])
#         faces_of_v = Avf_indices[Avf_indptr[v] : Avf_indptr[v + 1]]
#         for f in faces_of_v:
#             vertex_normals[v] += face_areas[f]
#         normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
#         if normal_norm > 0:
#             vertex_normals[v] /= normal_norm
#     return vertex_normals
#
#
# @njit
# def jit_y_of_x_csr(Axy_indices, Axy_indptr):
#     # indices, indptr = Axy_csr.indices, Axy_csr.indptr
#     Nx = len(Axy_indptr) - 1
#     x_of_y = []
#     for nx in range(Nx):
#         x_of_y.append(Axy_indices[Axy_indptr[nx] : Axy_indptr[nx + 1]])
#
#     return x_of_y
#
#
# @njit
# def check_faces(vertices, faces, keep_face=0):
#     """
#     something doesn't work!
#     """
#     (
#         Afv_indices,
#         Afv_indptr,
#         Afv_data,
#         Afe_data,
#         Afe_indices,
#         Afe_indptr,
#         Aev_data,
#         Aev_indices,
#         Aev_indptr,
#         edges,
#     ) = jit_boundary_ops_csr_data(vertices, faces)
#     Nfaces, Nedges, Nvertices = len(faces), len(edges), len(vertices)
#     Aef_data, Aef_indices, Aef_indptr = transpose_csr(Afe_data, Afe_indices, Afe_indptr)
#     E_of_F = jit_y_of_x_csr(Afe_indices, Afe_indptr)
#     F_of_E = jit_y_of_x_csr(Aef_indices, Aef_indptr)
#
#     F_of_F = np.zeros((Nfaces, 3), dtype=np.int32)
#     for f in range(Nfaces):
#         for n_e in range(3):
#             e = E_of_F[f][n_e]
#             for n_f in range(2):
#                 _f = F_of_E[e][n_f]
#                 if _f != f:
#                     F_of_F[f][n_e] = _f
#
#     checked = np.zeros(Nfaces, dtype=np.int32)
#     flipped = np.zeros(Nfaces, dtype=np.int32)
#     checked[keep_face] = 1
#     flipped[keep_face] = 1
#     for f in range(Nfaces):
#         ###########
#         F_of_f = F_of_F[f]
#         ###########
#         v1, v2, v3 = faces[f]
#         r1, r2, r3 = vertices[v1], vertices[v2], vertices[v3]
#         Af = jitcross(r1, r2) + jitcross(r2, r3) + jitcross(r3, r1)
#         flip_all_neighbors = True
#         for ff in F_of_f:
#             if checked[ff] == 1:
#                 flip_all_neighbors = False
#                 v1ff, v2ff, v3ff = faces[ff]
#                 r1ff, r2ff, r3ff = vertices[v1ff], vertices[v2ff], vertices[v3ff]
#                 Aff = jitcross(r1ff, r2ff) + jitcross(r2ff, r3ff) + jitcross(r3ff, r1ff)
#                 AfdotAff = jitdot(Af, Aff)
#                 if AfdotAff < 0:
#                     print(f"flipping f={f}")
#                     # faces[f] = np.array([v2, v1, v3], dtype=np.int32)
#                     faces[f] = np.flip(faces[f])
#                     flipped[f] = 1
#                 checked[f] = 1
#                 break
#         if flip_all_neighbors:
#             for ff in F_of_f:
#                 v1ff, v2ff, v3ff = faces[ff]
#                 r1ff, r2ff, r3ff = vertices[v1ff], vertices[v2ff], vertices[v3ff]
#                 Aff = jitcross(r1ff, r2ff) + jitcross(r2ff, r3ff) + jitcross(r3ff, r1ff)
#                 AfdotAff = jitdot(Af, Aff)
#                 if AfdotAff < 0:
#                     print(f"flipping ff={ff}")
#                     # faces[ff] = np.array([v2ff, v1ff, v3ff], dtype=np.int32)
#                     faces[ff] = np.flip(faces[ff])
#                     flipped[ff] = 1
#                 checked[ff] = 1
#     return faces, flipped, checked
#
#
# def save_flipped_mesh(surface_name, file_path=None, Nxyz=None, xyz_minmax=None):
#     if file_path is None:
#         file_path = f"./data/ply_files/{surface_name}_flipped.ply"
#     vertices, faces = make_sample_mesh(surface_name, Nxyz=Nxyz, xyz_minmax=xyz_minmax)
#     faces[0] = np.flip(faces[0])
#     faces, flipped, checked = check_faces(vertices, faces, keep_face=0)
#     save_mesh_to_ply(vertices, faces, file_path)
#
#     # surf_dict = {"vertices": verts, "faces": faces, "normals": normals}
#     # with open(f"./scratch/{surface_name}_dict.pickle", "wb") as _f:
#     #     dill.dump(surf_dict, _f, recurse=True)


###########################################
# def get_face_data(vertices, faces, surface_com=None):
#     """
#     computes what are hopefully outward pointing unit normal vectors
#     and directed area vectors of the faces. Reorders vertices of each face to
#     match unit normal direction.
#     """
#     # faces = faces_old
#
#     Nfaces = len(faces)
#
#     if surface_com is None:
#         Nvertices = len(vertices)
#         surface_com = np.zeros(3)
#         for v in vertices:
#             surface_com += v
#         surface_com /= Nvertices
#     face_normals = np.zeros((Nfaces, 3))
#     face_areas = np.zeros((Nfaces, 3))
#     face_centroids = np.zeros((Nfaces, 3))
#     for f in range(Nfaces):
#         fv0, fv1, fv2 = faces[f]
#         v0_xyz = vertices[fv0]
#         v1_xyz = vertices[fv1]
#         v2_xyz = vertices[fv2]
#
#         # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
#         f_normal = (
#             jitcross(v0_xyz, v1_xyz)
#             + jitcross(v1_xyz, v2_xyz)
#             + jitcross(v2_xyz, v0_xyz)
#         )
#         f_area = 0.5 * f_normal
#
#         f_normal /= np.sqrt(f_normal @ f_normal)
#         face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
#         face_centroids[f] = face_com
#         n_dot_dr = f_normal @ (face_com - surface_com)
#         if n_dot_dr > 0:
#             face_normals[f, :] = f_normal
#             face_areas[f, :] = f_area
#         else:
#             faces[f, :] = np.array([fv1, fv0, fv2])
#             face_normals[f, :] = -f_normal
#             face_areas[f, :] = -f_area
#     return faces, face_centroids, face_normals, face_areas
#
#
# def get_area_weighted_vertex_normals(vertices, faces_of_vertices, face_areas):
#     """
#     computes unit normal vectors at vertices.
#     """
#     Nvertices = len(vertices)
#     vertex_normals = np.zeros((Nvertices, 3))
#     for v in range(Nvertices):
#         # Nfaces_of_v = len(faces_of_vertices[v])
#         # Nfaces_of_v = max([1,len(faces_of_vertices[v])])
#         for f in faces_of_vertices[v]:
#             vertex_normals[v] += face_areas[f]
#         normal_norm = np.sqrt(vertex_normals[v] @ vertex_normals[v])
#         if normal_norm > 0:
#             vertex_normals[v] /= normal_norm
#     return vertex_normals
#
#
# def _label_vertices_and_faces(vertices, faces):
#     """assigns integers to vertices and faces"""
#     Nvertices = len(vertices)
#     Nfaces = len(faces)
#     V_label = np.array([_ for _ in range(Nvertices)], dtype=np.int32)
#     F_label = np.array([_ for _ in range(Nfaces)], dtype=np.int32)
#
#     return V_label, F_label
#
#
# def _label_halfedges(vertices, faces):
#     """Builds halfedges from vertices and faces, assigns an integer-valued
#     label/index to each halfedge, and determines whether the halfedge is
#     contained in the boundary of the mesh."""
#     halfedges = []
#     H_isboundary = []
#     H_label = []
#     ####################
#     # save and label halfedges
#     h = 0
#     for face in faces:
#         # face = faces[f]
#         N_v_of_f = len(face)
#         for _ in range(N_v_of_f):
#             # index shift to get next
#             _next = (_ + 1) % N_v_of_f
#             v0 = face[_]  #
#             v1 = face[_next]
#             hedge = [v0, v1]
#             halfedges.append(hedge)
#             H_isboundary.append(False)
#             H_label.append(h)
#             h += 1
#
#     for hedge in halfedges:
#         v0, v1 = hedge
#         hedge_twin = [v1, v0]
#         try:
#             halfedges.index(hedge_twin)
#         except Exception:
#             halfedges.append(hedge_twin)
#             H_isboundary.append(True)
#             H_label.append(h)
#             h += 1
#
#     return (
#         np.array(halfedges, dtype=np.int32),
#         np.array(H_label, dtype=np.int32),
#         np.array(H_isboundary),
#     )
#
#
# def _get_combinatorial_mesh_data(
#     V_label, H_label, F_label, halfedges, H_isboundary, faces
# ):
#     """."""
#     # V_label = self.V_label
#     # H_label = self.H_label
#     # F_label = self.F_label
#     # halfedges = self.halfedges
#     #
#     # H_isboundary = self.H_isboundary
#     # faces = self.faces.copy()
#     ####################
#     # vertices
#     V_hedge = -np.ones_like(V_label)  # outgoing halfedge
#     ####################
#     # faces
#     F_hedge = -np.ones_like(F_label)  # one of the halfedges bounding it
#     ####################
#     # halfedges
#     H_vertex = -np.ones_like(H_label)  # vertex it points to
#     H_face = -np.ones_like(H_label)  # face it belongs to
#     # next/previous halfedge inside the face (ordered counter-clockwise)
#     H_next = -np.ones_like(H_label)
#     H_prev = -np.ones_like(H_label)
#     H_twin = -np.ones_like(H_label)  # opposite halfedge
#     ####################
#
#     # assign each face a halfedge
#     # assign each interior halfedge previous/next halfedge
#     # assign each interior halfedge a face
#     # assign each halfedge a twin halfedge
#     for f in F_label:
#         face = faces[f]
#         N_v_of_f = len(face)
#         hedge0 = np.array([face[0], face[1]])
#         h0 = index_of_nested(halfedges, hedge0)
#         F_hedge[f] = h0  # assign each face a halfedge
#         for _ in range(N_v_of_f):
#             # for each vertex in face, get the indices of the
#             # previous/next vertex
#             _p1 = (_ + 1) % N_v_of_f
#             _m1 = (_ - 1) % N_v_of_f
#             vm1 = face[_m1]
#             v0 = face[_]
#             vp1 = face[_p1]
#             # get outgoing halfedge
#             hedge = np.array([v0, vp1])
#             h = index_of_nested(halfedges, hedge)
#             # get incident halfedge
#             hedge_prev = np.array([vm1, v0])
#             h_prev = index_of_nested(halfedges, hedge_prev)
#             # assign previous/next halfedge
#             H_prev[h] = h_prev
#             H_next[h_prev] = h
#             # assign face to halfedge
#             H_face[h] = f
#
#             hedge_twin = np.array([vp1, v0])
#             h_t = index_of_nested(halfedges, hedge_twin)
#             H_twin[h] = h_t
#             H_twin[h_t] = h
#
#     # assign each halfedge a vertex
#     # assign each vertex a halfedge
#     # assign each boundary halfedge previous/next halfedge
#     for h in H_label:
#         v0, v1 = halfedges[h]
#         H_vertex[h] = v1
#         if V_hedge[v0] == -1:
#             V_hedge[v0] = h
#         if H_isboundary[h]:
#             h_next = H_twin[h]
#             while True:
#                 h_next = H_twin[H_prev[h_next]]
#                 if H_isboundary[h_next]:
#                     break
#             H_next[h] = h_next
#             H_prev[h_next] = h
#
#     return (
#         V_hedge,
#         H_vertex,
#         H_face,
#         H_next,
#         H_prev,
#         H_twin,
#         F_hedge,
#     )
#
#
# def _f_adjacent_to_v(v, V_hedge, H_face, H_prev, H_twin):
#     """
#     gets faces adjacent to v in counterclockwise order
#     """
#     h_start = V_hedge[v]
#     neighbors = []
#
#     h = h_start
#     while True:
#         neighbors.append(H_face[h])
#         h = H_prev[h]
#         h = H_twin[h]
#         if h == h_start:
#             break
#
#     return neighbors


# # @njit
# def get_halfedges_from_face(face):
#     Nhedges_per_face = len(face)
#     halfedge = np.zeros((Nhedges_per_face, 2), dtype=np.int32)
#     for i in range(Nhedges_per_face):
#         halfedge[i, 0] = face[i]
#         halfedge[i, 1] = face[(i + 1) % Nhedges_per_face]
#     return halfedge
#
#
# # @njit
# def get_halfedges_from_triangle(face):
#     return np.array(
#         [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]], dtype=np.int32
#     )
#
#
# def get_halfedges_from_faces(F):
#     """gets halfedges from faces with Nhedges_per_face"""
#     Nfaces, Nhedges_per_face = F.shape
#     Nhedges = Nfaces * Nhedges_per_face
#     label_type = np.int32
#     H = np.zeros((Nhedges, 2), dtype=label_type)
#     H_label = np.zeros(Nhedges, dtype=label_type)
#     for f in range(Nfaces):
#         face = F[f]
#         for i in range(Nhedges_per_face):
#             h = Nhedges_per_face * f + i
#             H_label[h] = Nhedges_per_face * f + i
#             H[h, 0] = face[i]
#             H[h, 1] = face[(i + 1) % Nhedges_per_face]
#
#     # H_label = np.array([h for h in range(Nhedges)])
#     return H, H_label
#
#
# def get_halfedges_from_triangles(F):
#     H = np.concatenate([[face[:2], face[1:], [face[2], face[0]]] for face in F])
#     H_label = np.array([h for h, _ in enumerate(H)], dtype=np.int32)
#     return H, H_label
#
#
# def get_halfedges_from_faces_parallel(F):
#     N_cpu = mu.cpu_count()
#     with mu.Pool(processes=N_cpu) as p:
#         H = np.concatenate(p.map(get_halfedges_from_face, F))
#     H_label = np.array([h for h, _ in enumerate(H)], dtype=np.int32)
#     return H, H_label
#
#
# def get_halfedges_from_triangles_parallel(F):
#     N_cpu = mu.cpu_count()
#     with mu.Pool(processes=N_cpu) as p:
#         H = np.concatenate(p.map(get_halfedges_from_triangle, F))
#     H_label = np.array([h for h, _ in enumerate(H)], dtype=np.int32)
#     return H, H_label


# %%
