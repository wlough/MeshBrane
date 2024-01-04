from numba import float64, int32
from numba.experimental import jitclass
from numba.types import ListType
import numpy as np
import sympy as sp
from skimage.measure import marching_cubes
from numdiff import (
    jitcross,
    matrix_to_quaternion,
    quaternion_to_matrix,
    exp_so3,
    log_so3,
    exp_quaternion,
    log_quaternion,
)
from scipy.sparse import csr_matrix


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
    # X, Y, Z = (
    #     verts[:, 0] + x[0, 0, 0],
    #     verts[:, 1] + y[0, 0, 0],
    #     verts[:, 2] + z[0, 0, 0],
    # )

    verts[:, 0] += x[0, 0, 0]
    verts[:, 1] += y[0, 0, 0]
    verts[:, 2] += z[0, 0, 0]
    normals = normals.astype(np.float64)
    return verts, faces, normals


def make_sample_mesh(surface_name):
    if surface_name == "dumbbell":
        xyz_minmax = [-12.0, 12.0, -5.0, 5.0, -5.0, 5.0]
        Nxyz = [60j, 60j, 60j]
        implicit_fun_str = (
            "1.0*(y**2 + z**2 + (x - 8)**2 - 1)*(y**2 + z**2 + (x + 8)**2 - 1) - 4200.0"
        )
    elif surface_name == "torus":
        xyz_minmax = [-6.0, 6.0, -6.0, 6.0, -6.0, 6.0]
        # Nxyz = [60j, 60j, 60j]
        Nxyz = [20j, 20j, 20j]
        R = 4
        r = 4.0 / 3.0
        implicit_fun_str = (
            f"(x**2 + y**2 + z**2 + {R}**2 - {r}**2) ** 2 - 4 * {R}**2 * (x**2 + y**2)"
        )

    elif surface_name == "neovius":
        xyz_minmax = [0.0, 6.0, 0.0, 6.0, 0.0, 6.0]
        Nxyz = [60j, 60j, 60j]
        R = 4
        r = 2.0 / 3.0
        # implicit_fun_str = "3 * (sp.cos(x) + sp.cos(y) + sp.cos(z)) + 4 * sp.cos(x) * sp.cos(y) * sp.cos(z)"
        implicit_fun_str = (
            "3 * (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z)"
        )

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
]


@jitclass(framed_brane_spec)
class FramedBrane:
    def __init__(self, vertices, faces, normals):
        self.faces = faces
        self.framed_vertices = self.frame_the_mesh(vertices, normals)
        self.edges = self.get_edges()

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

    def get_edges(self):
        vertices = self.framed_vertices
        faces = self.faces
        Nvertices = len(vertices)
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
                    e_m = edges_list.index(edge_m)
                except Exception:  # if not, then add it
                    edges_list.append(edge_m)
                    e = len(edges_list) - 1
                    Afe_indices_list.append(e)
                    Afe_data_list.append(-1)
                    Aev_indices_list.append(vp)
                    Aev_indices_list.append(vm)
                try:  # is positive edge already in edges?
                    e_p = edges_list.index(edge_p)
                except Exception:  # if neither, add positive edge to edges
                    edges_list.append(edge_p)
                    e = len(edges_list) - 1
                    Afe_indices_list.append(e)
                    Afe_data_list.append(1)
                    Aev_indices_list.append(vm)
                    Aev_indices_list.append(vp)

        Afe_data = np.array(Afe_data_list, dtype=np.int32)
        Afe_indices = np.array(Afe_indices_list, dtype=np.int32)
        Aev_indices = np.array(Aev_indices_list, dtype=np.int32)
        edges = np.array(edges_list, dtype=np.int32)

        return edges

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
        title = f"Membrane mesh"
        fig = mlab.figure(title, size=figsize)  # , bgcolor=bgcolor, fgcolor=fgcolor)
        if plot_edges:
            mem_edges = mlab.triangular_mesh(
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
            mem_faces = mlab.triangular_mesh(
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
            mem_vertices = mlab.points3d(
                *vertices.T, mode="sphere", scale_factor=0.075, color=vertex_color
            )

        if plot_face_normals:
            f_normals = mlab.quiver3d(
                face_X, face_Y, face_Z, face_nX, face_nY, face_nZ, color=face_color
            )

        if plot_vertex_normals:
            v_normals = mlab.quiver3d(
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
    Nvertices = len(vertices)
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
