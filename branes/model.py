from numba import float64, int32, njit
from numba.experimental import jitclass
import numpy as np
import sympy as sp
from skimage.measure import marching_cubes
from numdiff import (
    jitcross,
    # matrix_to_quaternion,
    quaternion_to_matrix,
    # exp_so3,
    # log_so3,
    # exp_quaternion,
    # log_quaternion,
)
from scipy.sparse import csr_matrix
from mayavi import mlab


@njit
def transpose_csr(data, indices, indptr):
    # Compute the number of non-zero entries and the number of columns
    # n = len(data)
    m = np.max(indices) + 1

    # Initialize the data, indices, and indptr for the transpose
    data_T = np.empty_like(data)
    indices_T = np.empty_like(indices)
    # indptr_T = np.zeros(m + 1, dtype=indptr.dtype)
    _indptr_T = np.zeros(m + 1, dtype=indptr.dtype)
    indptr_T = np.zeros(m + 1, dtype=np.int32)
    # _indptr_T[0] = 0
    # _indptr_T[j+1] = indptr_T[j]

    # Compute the column counts
    for index in indices:
        # indptr_T[index + 1] += 1
        _indptr_T[index + 1] += 1

    # Compute the column pointers
    # indptr_T = np.cumsum(indptr_T)
    _indptr_T = np.cumsum(_indptr_T)

    for i in range(len(indptr) - 1):
        # For each non-zero in the row...
        for data_index in range(indptr[i], indptr[i + 1]):
            # Get the column index
            j = indices[data_index]

            # Get the insertion index
            insert_index = _indptr_T[j]
            # _insert_index = _indptr_T[j + 1]

            # Insert the data and row index
            data_T[insert_index] = data[data_index]
            indices_T[insert_index] = i

            # Increment the column pointer
            _indptr_T[j] += 1
            # _indptr_T[j + 1] += 1

    indptr_T[1:] = _indptr_T[:-1]
    return data_T, indices_T, indptr_T


# @njit
# def _transpose_csr(data, indices, indptr):
#     # Compute the number of non-zero entries and the number of columns
#     # n = len(data)
#     m = np.max(indices) + 1
#
#     # Initialize the data, indices, and indptr for the transpose
#     data_T = np.empty_like(data)
#     indices_T = np.empty_like(indices)
#     indptr_T = np.zeros(m + 1, dtype=indptr.dtype)
#
#     # Compute the column counts
#     for index in indices:
#         indptr_T[index + 1] += 1
#
#     # Compute the column pointers
#     indptr_T = np.cumsum(indptr_T)
#
#     # For each row...
#     for i in range(len(indptr) - 1):
#         # For each non-zero in the row...
#         for data_index in range(indptr[i], indptr[i + 1]):
#             # Get the column index
#             j = indices[data_index]
#
#             # Get the insertion index
#             insert_index = indptr_T[j]
#
#             # Insert the data and row index
#             data_T[insert_index] = data[data_index]
#             indices_T[insert_index] = i
#
#             # Increment the column pointer
#             indptr_T[j] += 1
#
#     return data_T, indices_T, indptr_T
#


def mayavi_mesh_plot(
    vertices,
    faces,
    edges=None,
    frames=None,
    vector_field_data=None,
    submesh_data=None,
    show=True,
    save=False,
    fig_path=None,
    plot_vertices=True,
    plot_edges=True,
    plot_faces=True,
):
    face_color = (0.0, 0.2667, 0.1059)
    edge_color = (1.0, 0.498, 0.0)
    vertex_color = (0.7057, 0.0156, 0.1502)
    frame_color = (0.2298, 0.2987, 0.7537)

    # figsize = (2180, 2180)
    figsize = (720, 720)
    ##########################################################
    if show:
        mlab.options.offscreen = False
    else:
        mlab.options.offscreen = True

    # bgcolor = (1.0, 1.0, 1.0)
    # fgcolor = (0.0, 0.0, 0.0)
    figsize = (2180, 2180)
    title = "Membrane mesh"
    # , bgcolor=bgcolor, fgcolor=fgcolor)
    fig = mlab.figure(title, size=figsize)

    if plot_edges:
        # mem_edges = #
        mlab.triangular_mesh(
            *vertices.T,
            faces,
            # opacity=0.4,
            color=edge_color,
            # representation="wireframe"
            representation="mesh",
            # representation="surface"
            # representation="fancymesh",
            tube_radius=0.002,
            tube_sides=3,
        )
    if plot_faces:
        # mem_faces = #
        mlab.triangular_mesh(
            *vertices.T,
            faces,
            opacity=0.4,
            color=face_color,
            # representation="wireframe"
            # representation="mesh",
            representation="surface"
            # representation="fancymesh",
            # tube_radius=None
        )
    if plot_vertices:
        # mem_vertices = #
        mlab.points3d(
            *vertices.T, mode="sphere", scale_factor=0.015, color=vertex_color
        )
    if frames is not None:
        # tangents1 = #
        mlab.quiver3d(
            *vertices.T,
            *frames[:, :, 0].T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=0.075,
            color=frame_color,
        )
        # tangents2 =
        # mlab.quiver3d(
        #     *vertices.T,
        #     *frames[:, :, 1].T,
        #     # line_width=5,
        #     mode="arrow",
        #     # scale_mode="vector",
        #     scale_factor=0.075,
        #     color=frame_color,
        # )
        # normals =
        mlab.quiver3d(
            *vertices.T,
            *frames[:, :, 2].T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=0.075,
            color=frame_color,
        )

    if vector_field_data is not None:
        vectors = vector_field_data["vectors"]
        vector_positions = vector_field_data["positions"]
        vector_color = vector_field_data["color"]
        # vecs =
        mlab.quiver3d(
            *vector_positions.T,
            *vectors.T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=1,
            color=vector_color,
        )
    if submesh_data is not None:
        vectors = vector_field_data["vectors"]
        vector_positions = vector_field_data["positions"]
        vector_color = vector_field_data["color"]
        # vecs =
        mlab.quiver3d(
            *vector_positions.T,
            *vectors.T,
            # line_width=5,
            mode="arrow",
            # scale_mode="vector",
            scale_factor=1,
            color=vector_color,
        )

    if show:
        # mlab.axes()
        mlab.orientation_axes()
        mlab.show()
    if save:
        mlab.savefig(fig_path, figure=fig, size=figsize)
    mlab.close(all=True)


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Quaternion(
            self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
        )

    def __sub__(self, other):
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __mul__(self, other):
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def __normalize__(self, other):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        w = self.w / norm
        x = self.x / norm
        y = self.y / norm
        z = self.z / norm
        return Quaternion(w, x, y, z)

    def __str__(self):
        return f"({self.w}, {self.x}, {self.y}, {self.z})"


def make_implicit_surface_mesh(implicit_fun_str, xyz_minmax, Nxyz):
    """
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


def make_sample_mesh(surface_name):
    if surface_name == "dumbbell":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        Nxyz = [20j, 20j, 20j]
        implicit_fun_str = "(144*y**2 + 144*z**2 + (12*x - 8)**2 - 1)*(144*y**2 + 144*z**2 + (12*x + 8)**2 - 1) - 4200"
    elif surface_name == "dumbbell2":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [20j, 20j, 20j]
        implicit_fun_str = "9*x**2 + 9*y**2 - 9*(z**2 - 1)*(cos(3*pi*z/4) - 1.25)/4"
    elif surface_name == "torus":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        # Nxyz = [60j, 60j, 60j]
        Nxyz = [20j, 20j, 20j]
        R = 0.7  # big radius
        r = 0.7 / 3.0  # small radius
        implicit_fun_str = (
            f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
        )
    elif surface_name == "double torus":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [30j, 30j, 30j]
        # implicit_fun_str = "z**2 + (x**2*(x - 1)*(x + 1) + y**2)**2 - 0.01"
        implicit_fun_str = (
            "(z/0.2)**2 + (x**2*(x - 0.7)*(x + 0.7)/0.05 + y**2/0.05)**2 - 1"
        )
    elif surface_name == "triple torus":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [40j, 40j, 40j]
        # implicit_fun_str = "0.16*z**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
        implicit_fun_str = "1.69*z**2*(1 - 0.769*cos(pi*Abs(y**2 + (x + 0.2)**2)/4))**2 + (-(x + 0.2)*(-3*y**2 + (x + 0.2)**2) + (y**2 + (x + 0.2)**2)**2)**2 - 0.008"
    elif surface_name == "neovius":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [20j, 20j, 20j]
        implicit_fun_str = "4*cos(3*x + 3)*cos(3*y + 3)*cos(3*z + 3) + 3*cos(3*x + 3) + 3*cos(3*y + 3) + 3*cos(3*z + 3)"
    elif surface_name == "sphere":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [20j, 20j, 20j]
        R = 0.9
        implicit_fun_str = f"x**2+y**2+z**2-{R**2}"
    elif surface_name == "oblate":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [20j, 20j, 20j]
        Rxy = 0.9
        Rz = 0.6
        implicit_fun_str = f"(x/{Rxy})**2+(y/{Rxy})**2+(z/{Rz})**2-1"
    elif surface_name == "prolate":
        xyz_minmax = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        Nxyz = [20j, 20j, 20j]
        Rxy = 0.6
        Rz = 0.9
        implicit_fun_str = f"(x/{Rxy})**2+(y/{Rxy})**2+(z/{Rz})**2-1"
    verts, faces, normals = make_implicit_surface_mesh(
        implicit_fun_str, xyz_minmax, Nxyz
    )
    return verts, faces, normals


def get_face_data(vertices, faces, surface_com):
    """
    computes what are hopefully outward pointing unit normal vectors
    and directed area vectors of the faces. Reorders vertices of each face to
    match unit normal direction.
    """
    # faces = faces_old
    Nfaces = len(faces)
    face_normals = np.zeros((Nfaces, 3))
    face_areas = np.zeros((Nfaces, 3))
    face_centroids = np.zeros((Nfaces, 3))
    for f in range(Nfaces):
        fv0, fv1, fv2 = faces[f]
        v0_xyz = vertices[fv0]
        v1_xyz = vertices[fv1]
        v2_xyz = vertices[fv2]

        # this is just (v1_xyz-v0_xyz) x (v2_xyz-v1_xyz)
        f_normal = (
            jitcross(v0_xyz, v1_xyz)
            + jitcross(v1_xyz, v2_xyz)
            + jitcross(v2_xyz, v0_xyz)
        )
        f_area = 0.5 * f_normal

        f_normal /= np.sqrt(f_normal @ f_normal)
        face_com = (v0_xyz + v1_xyz + v2_xyz) / 3.0
        face_centroids[f] = face_com
        n_dot_dr = f_normal @ (face_com - surface_com)
        if n_dot_dr > 0:
            face_normals[f, :] = f_normal
            face_areas[f, :] = f_area
        else:
            faces[f, :] = np.array([fv1, fv0, fv2])
            face_normals[f, :] = -f_normal
            face_areas[f, :] = -f_area
    return faces, face_centroids, face_normals, face_areas


framed_brane_spec = [
    # ("vertices", float64[:, :]),
    # ("normals", float64[:, :]),
    ("faces", int32[:, :]),
    ("edges", int32[:, :]),
    ("framed_vertices", float64[:, :]),
    ("Afe_data", int32[:]),
    ("Afe_indices", int32[:]),
    ("Afe_indptr", int32[:]),
    ("Aev_data", int32[:]),
    ("Aev_indices", int32[:]),
    ("Aev_indptr", int32[:]),
    ("Afv_data", int32[:]),
    ("Afv_indices", int32[:]),
    ("Afv_indptr", int32[:]),
    ("Aef_data", int32[:]),
    ("Aef_indices", int32[:]),
    ("Aef_indptr", int32[:]),
    ("Ave_data", int32[:]),
    ("Ave_indices", int32[:]),
    ("Ave_indptr", int32[:]),
    ("Avf_data", int32[:]),
    ("Avf_indices", int32[:]),
    ("Avf_indptr", int32[:]),
]


@jitclass(framed_brane_spec)
class FramedBrane:
    def __init__(self, vertices, faces, normals):
        self.faces = faces
        self.framed_vertices = self.frame_the_mesh(vertices, normals)
        # self.edges = self.get_edges()

        Nfaces = len(faces)
        self.Afv_indices = faces.ravel()
        self.Afv_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)
        self.Afv_data = np.ones(3 * Nfaces, dtype=np.int32)

        (
            self.edges,
            self.Afe_data,
            self.Afe_indices,
            self.Afe_indptr,
            self.Aev_data,
            self.Aev_indices,
            self.Aev_indptr,
        ) = self.get_adjacency_edges_and_adjacency()
        self.Aef_data, self.Aef_indices, self.Aef_indptr = transpose_csr(
            self.Afe_data, self.Afe_indices, self.Afe_indptr
        )
        self.Ave_data, self.Ave_indices, self.Ave_indptr = transpose_csr(
            self.Aev_data, self.Aev_indices, self.Aev_indptr
        )
        self.Avf_data, self.Avf_indices, self.Avf_indptr = transpose_csr(
            self.Afv_data, self.Afv_indices, self.Afv_indptr
        )

    def frame_the_mesh(self, vertices, normals):
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        # ez = np.array([0.0, 0.0, 1.0])
        Nverts = len(vertices)
        # framed_vertices = np.zeros((Nverts, 7))
        # matrices = np.zeros((Nverts, 3, 3))
        framed_vertices = np.zeros((Nverts, 7))
        for i in range(Nverts):
            cross_with_ey = np.sqrt(normals[i, 2] ** 2 + normals[i, 0] ** 2) > 1e-6
            if cross_with_ey:
                e1 = jitcross(ey, normals[i])
            else:
                e1 = jitcross(ex, normals[i])
            e1 /= np.sqrt(e1[0] ** 2 + e1[1] ** 2 + e1[2] ** 2)
            e2 = jitcross(normals[i], e1)

            R = np.zeros((3, 3))
            R[:, 0] = e1
            R[:, 1] = e2
            R[:, 2] = normals[i]
            framed_vertices[i, 3] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
            framed_vertices[i, 4] = (R[2, 1] - R[1, 2]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 5] = (R[0, 2] - R[2, 0]) / (4 * framed_vertices[i, 3])
            framed_vertices[i, 6] = (R[1, 0] - R[0, 1]) / (4 * framed_vertices[i, 3])

            framed_vertices[i, :3] = vertices[i, :]
        return framed_vertices

    # def get_edges(self):
    #     # vertices = self.framed_vertices
    #     faces = self.faces
    #     # Nvertices = len(vertices)
    #     Nfaces = len(faces)
    #     edges_list = []
    #
    #     # Afe ###############
    #     Afe_data_list = []  # [-1,1,...]
    #     Afe_indices_list = []  #
    #     # Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)
    #
    #     # Aev ###############
    #     # Aev_data_list = []  # [-1,1,...]
    #     Aev_indices_list = []  # vertex indices
    #     # Aev_indptr = []  # [0,2,4,...]
    #
    #     for f in range(Nfaces):
    #         face = faces[f]
    #         for _v in range(3):
    #             vm = face[_v]
    #             vp = face[np.mod(_v + 1, 3)]
    #             edge_p = [vm, vp]
    #             edge_m = [vp, vm]
    #             try:  # is negative edge already in edges?
    #                 edges_list.index(edge_m)
    #             except Exception:  # if not, then add it
    #                 edges_list.append(edge_m)
    #                 e = len(edges_list) - 1
    #                 Afe_indices_list.append(e)
    #                 Afe_data_list.append(-1)
    #                 Aev_indices_list.append(vp)
    #                 Aev_indices_list.append(vm)
    #             try:  # is positive edge already in edges?
    #                 edges_list.index(edge_p)
    #             except Exception:  # if neither, add positive edge to edges
    #                 edges_list.append(edge_p)
    #                 e = len(edges_list) - 1
    #                 Afe_indices_list.append(e)
    #                 Afe_data_list.append(1)
    #                 Aev_indices_list.append(vm)
    #                 Aev_indices_list.append(vp)
    #
    #     Afe_data = np.array(Afe_data_list, dtype=np.int32)
    #     Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
    #     Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
    #     edges = np.array(edges_list, dtype=np.int32)
    #
    #     return edges

    def get_adjacency_edges_and_adjacency(self):
        # vertices = self.framed_vertices
        faces = self.faces
        # Nvertices = len(vertices)
        Nfaces = len(faces)
        edges_list = []

        # Afe ###############
        Afe_data_list = []  # [-1,1,...]
        Afe_indices_list = []  #
        # Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

        # Aev ###############
        # Aev_data_list = []  # [-1,1,...]
        Aev_indices_list = []  # vertex indices
        # Aev_indptr = []  # [0,2,4,...]

        for f in range(Nfaces):
            face = faces[f]
            for _v in range(3):
                vm = face[_v]
                vp = face[np.mod(_v + 1, 3)]
                edge_p = [vm, vp]
                edge_m = [vp, vm]
                try:  # is negative edge already in edges?
                    edges_list.index(edge_m)
                except Exception:  # if not, then add it
                    edges_list.append(edge_m)
                    e = len(edges_list) - 1
                    Afe_indices_list.append(e)
                    Afe_data_list.append(-1)
                    Aev_indices_list.append(vp)
                    Aev_indices_list.append(vm)
                try:  # is positive edge already in edges?
                    edges_list.index(edge_p)
                except Exception:  # if neither, add positive edge to edges
                    edges_list.append(edge_p)
                    e = len(edges_list) - 1
                    Afe_indices_list.append(e)
                    Afe_data_list.append(1)
                    Aev_indices_list.append(vm)
                    Aev_indices_list.append(vp)

        Afe_data = np.array(Afe_data_list, dtype=np.int32)
        Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
        Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

        edges = np.array(edges_list, dtype=np.int32)
        Nedges = len(edges)

        Aev_data = np.array(Nedges * [-1, 1], dtype=np.int32)  # [-1,1,...]
        Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
        Aev_indptr = np.array(
            [2 * _ for _ in range(Nedges + 1)], dtype=np.int32
        )  # [0,2,4,...]

        return (
            edges,
            Afe_data,
            Afe_indices,
            Afe_indptr,
            Aev_data,
            Aev_indices,
            Aev_indptr,
        )

    def position_vectors(self):
        return self.framed_vertices[:, :3]

    def unit_quaternions(self):
        return self.framed_vertices[:, 3:]

    def orthogonal_matrices(self):
        Q = self.framed_vertices[:, 3:]
        Nv = len(Q)
        R = np.zeros((Nv, 3, 3))
        for v in range(Nv):
            q = Q[v]
            R[v] = quaternion_to_matrix(q)
        return R

    def get_y_of_x_csr(self, Axy_indices, Axy_indptr):
        # indices, indptr = Axy_csr.indices, Axy_csr.indptr
        Nx = len(Axy_indptr) - 1
        x_of_y = []
        for nx in range(Nx):
            x_of_y.append(Axy_indices[Axy_indptr[nx] : Axy_indptr[nx + 1]])

        return x_of_y

    def get_edges_of_faces(self):
        """e_of_f, v_of_e"""
        # indices, indptr = Axy_csr.indices, Axy_csr.indptr
        Afe_indices, Afe_indptr = self.Afe_indices, self.Afe_indptr
        Nfaces = len(Afe_indptr) - 1
        e_of_f = []
        for f in range(Nfaces):
            e_of_f.append(Afe_indices[Afe_indptr[f] : Afe_indptr[f + 1]])

        return e_of_f

    def edges_of_face(self, face):
        """face = index of face"""
        return self.Afe_indices[self.Afe_indptr[face] : self.Afe_indptr[face + 1]]

    def faces_of_edge(self, edge):
        """face = index of face"""
        return self.Aef_indices[self.Aef_indptr[edge] : self.Aef_indptr[edge + 1]]

    def faces_of_vertex(self, vertex):
        """face = index of face"""
        return self.Avf_indices[self.Avf_indptr[vertex] : self.Avf_indptr[vertex + 1]]

    def vertices_of_face(self, face):
        """face = index of face"""
        return self.Afv_indices[self.Afv_indptr[face] : self.Afv_indptr[face + 1]]

    def edges_of_vertex(self, vertex):
        """face = index of face"""
        return self.Ave_indices[self.Ave_indptr[vertex] : self.Ave_indptr[vertex + 1]]

    def vertices_of_edge(self, edge):
        """face = index of face"""
        return self.Aev_indices[self.Aev_indptr[edge] : self.Aev_indptr[edge + 1]]

    def vertices_adjacent_to_vertex(self, vertex):
        """face = index of face"""

        e_of_v = self.edges_of_vertex(vertex)
        v_of_e_of_v = []  # [b.edges[_] for _ in e_of_v]

        for e in e_of_v:
            edge = self.edges[e]
            for v in edge:
                add_v = (not v in v_of_e_of_v) and v != vertex
                if add_v:
                    v_of_e_of_v.append(v)
        return v_of_e_of_v

    def star(self, v):
        V = np.array([v], dtype=np.int32)
        E = self.edges_of_vertex(v)
        F = self.faces_of_vertex(v)
        return V, E, F

    # def closure(self, Vin, Ein, Fin):
    #     faces = self.faces
    #     V, E, F = [], [], []
    #     V.appen(V[0])
    #     for f in Fin:
    #         face = faces[f]
    #         for _v in face:
    #             add_v = not _v in V
    #             if add_v:
    #                 V.append(_v)

    # return 0

    # def link(self, v):
    #     return v

    # def closure_star(self, v):
    #     faces = self.faces
    #     # V, E, F = [], [], []
    #     F = self.faces_of_vertex(v)
    #     V = [v]
    #     for f in F:
    #         face = faces[f]
    #         for _v in face:
    #             add_v = not _v in V
    #             if add_v:
    #                 V.append(_v)
    def mini_mesh(self, v):
        faces = self.faces
        vertices = self.position_vectors()
        # V, E, F = [], [], []
        F = self.faces_of_vertex(v)
        V = [v]
        for f in F:
            face = faces[f]
            for _v in face:
                add_v = not _v in V
                if add_v:
                    V.append(_v)

        Nfaces = len(F)
        Nvertices = len(V)
        mini_faces = np.zeros((Nfaces, 3), dtype=np.int32)
        mini_vertices = np.zeros((Nvertices, 3))
        for _ in range(Nvertices):
            _v = V[_]
            mini_vertices[_] = vertices[_v]
        for _ in range(Nfaces):
            f = F[_]
            face = faces[f]
            mini_faces[_, 0] = V.index(face[0])
            mini_faces[_, 1] = V.index(face[1])
            mini_faces[_, 2] = V.index(face[2])
        return mini_vertices, mini_faces


class brane(object):
    """ """

    def __init__(
        self,
        vertices=None,
        faces=None,
        rotations=None,
        implicit_fun_str=None,
        sample_surface_name=None,
    ):
        ###################################################
        if sample_surface_name is not None:
            vertices, faces, normals = make_sample_mesh(sample_surface_name)
            self.vertices = vertices
            self.faces = faces
            self.normals = normals
        else:
            self.vertices = vertices
            self.faces = faces
            self.rotations = rotations
            self.implicit_fun_str = implicit_fun_str

    def plot_mesh_mayavi(
        self,
        show=True,
        save=False,
        fig_path=None,
        plot_vertices=True,
        plot_edges=True,
        plot_faces=True,
        plot_face_normals=False,
        plot_vertex_normals=False,
    ):
        vertices = self.vertices
        faces = self.faces
        face_centroids = self.face_centroids
        face_normals = self.face_normals
        vertex_normals = self.vertex_normals
        face_color = self.face_color
        edge_color = self.edge_color
        vertex_color = self.vertex_color
        # normal_color = self.normal_color
        # figsize = (2180, 2180)
        figsize = (720, 720)
        ##########################################################
        if show:
            mlab.options.offscreen = False
        else:
            mlab.options.offscreen = True

        vertex_X, vertex_Y, vertex_Z = vertices.T
        face_X, face_Y, face_Z = face_centroids.T
        face_nX, face_nY, face_nZ = face_normals.T
        vertex_nX, vertex_nY, vertex_nZ = vertex_normals.T
        # bgcolor = (1.0, 1.0, 1.0)
        # fgcolor = (0.0, 0.0, 0.0)
        figsize = (2180, 2180)
        title = "Membrane mesh"
        # , bgcolor=bgcolor, fgcolor=fgcolor)
        fig = mlab.figure(title, size=figsize)
        if plot_edges:
            # mem_edges =
            mlab.triangular_mesh(
                vertex_X,
                vertex_Y,
                vertex_Z,
                faces,
                # opacity=0.4,
                color=edge_color,
                # representation="wireframe"
                representation="mesh",
                # representation="surface"
                # representation="fancymesh",
                tube_radius=0.02,
                tube_sides=3,
            )
        if plot_faces:
            # mem_faces =
            mlab.triangular_mesh(
                vertex_X,
                vertex_Y,
                vertex_Z,
                faces,
                opacity=0.4,
                color=face_color,
                # representation="wireframe"
                # representation="mesh",
                representation="surface"
                # representation="fancymesh",
                # tube_radius=None
            )
        if plot_vertices:
            # mem_vertices =
            mlab.points3d(
                *vertices.T, mode="sphere", scale_factor=0.075, color=vertex_color
            )

        if plot_face_normals:
            # f_normals =
            mlab.quiver3d(
                face_X, face_Y, face_Z, face_nX, face_nY, face_nZ, color=face_color
            )

        if plot_vertex_normals:
            # v_normals =
            mlab.quiver3d(
                vertex_X,
                vertex_Y,
                vertex_Z,
                vertex_nX,
                vertex_nY,
                vertex_nZ,
                color=vertex_color,
            )

        if show:
            mlab.show()
        if save:
            mlab.savefig(fig_path, figure=fig, size=figsize)
        mlab.close(all=True)


def get_boundary_ops_csr(vertices, faces):
    Nvertices = len(vertices)
    (
        Afe_data,
        Afe_indices,
        Aev_indices,
        edges,
    ) = get_boundary_ops_csr_data(vertices, faces)
    # edges = np.array(edges, dtype=np.int32)
    # Afe ###############
    Nfaces = len(faces)
    # Afe_data = np.array(Afe_data_list, dtype=np.int32)  # [-1,1,...]
    # Afe_indices = np.array(Afe_indices_list, dtype=np.int32)  #
    Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

    # Aev ###############
    Nedges = len(edges)
    Aev_data = np.array(Nedges * [-1, 1], dtype=np.int32)  # [-1,1,...]
    # Aev_indices = np.array(Aev_indices_list, dtype=np.int32)  # vertex indices
    Aev_indptr = np.array(
        [2 * _ for _ in range(Nedges + 1)], dtype=np.int32
    )  # [0,2,4,...]

    Afe = csr_matrix((Afe_data, Afe_indices, Afe_indptr), shape=(Nfaces, Nedges))
    Aev = csr_matrix((Aev_data, Aev_indices, Aev_indptr), shape=(Nedges, Nvertices))

    return Afe, Aev, edges


def get_boundary_ops_csr_data(vertices, faces):
    """
    Computes edges and boundary operators from vertices and faces
    """
    # Nvertices = len(vertices)
    Nfaces = len(faces)
    edges_list = []

    # Afe ###############
    Afe_data_list = []  # [-1,1,...]
    Afe_indices_list = []  #
    # Afe_indptr = np.array([3 * f for f in range(Nfaces + 1)], dtype=np.int32)

    # Aev ###############
    # Aev_data_list = []  # [-1,1,...]
    Aev_indices_list = []  # vertex indices
    # Aev_indptr = []  # [0,2,4,...]

    for f in range(Nfaces):
        face = faces[f]
        for _v in range(3):
            vm = face[_v]
            vp = face[np.mod(_v + 1, 3)]
            edge_p = [vm, vp]
            edge_m = [vp, vm]
            try:  # is negative edge already in edges?
                e = edges_list.index(edge_m)
                fe_orientation = -1
            except Exception:
                try:  # is positive edge already in edges?
                    e = edges_list.index(edge_p)
                    fe_orientation = 1
                except Exception:  # if neither, add positive edge to edges
                    edges_list.append(edge_p)
                    e = len(edges_list) - 1
                    fe_orientation = 1
                    Aev_indices_list.append(vm)
                    Aev_indices_list.append(vp)

            Afe_indices_list.append(e)
            Afe_data_list.append(fe_orientation)

    Afe_data = np.array(Afe_data_list, dtype=np.int32)
    Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
    Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
    edges = np.array(edges_list, dtype=np.int32)
    return Afe_data, Afe_indices, Aev_indices, edges
    return Afe_data, Afe_indices, Aev_indices, edges
